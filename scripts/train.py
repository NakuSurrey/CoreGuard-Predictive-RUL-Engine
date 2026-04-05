"""
Master Training Script — CoreGuard Predictive RUL Engine.

Runs the full pipeline from raw data to trained, evaluated models.
This is the ONE command that builds everything:

    python scripts/train.py

Pipeline flow:
    Step 1 → Load raw C-MAPSS data
    Step 2 → Preprocess training data (clean, normalize, engineer features)
    Step 3 → Preprocess test data (using training scaler)
    Step 4 → Train XGBoost model
    Step 5 → Train LSTM model
    Step 6 → Evaluate both models on test data
    Step 7 → Compare models and declare winner
"""

import sys
import os
import time

# Add project root to Python path so imports work correctly.
# When we run "python scripts/train.py", Python's working directory
# might not include the project root. This line fixes that.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    LSTM_SEQUENCE_LENGTH,
)
from src.data.loader import load_train_data, load_test_data, download_cmapss
from src.data.preprocessor import (
    preprocess_training_data,
    preprocess_test_data,
    create_lstm_sequences,
    get_feature_columns,
)
from src.models.xgboost_model import train_xgboost, predict_xgboost
from src.models.lstm_model import train_lstm, predict_lstm
from src.models.evaluate import evaluate_model, compare_models


def main() -> None:
    """Run the complete training and evaluation pipeline."""

    total_start: float = time.time()

    # ------------------------------------------------------------------
    # STEP 1: Load raw data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1/7: Loading raw data")
    print("=" * 60)

    download_cmapss()
    train_df = load_train_data()
    test_df, rul_df = load_test_data()

    # ------------------------------------------------------------------
    # STEP 2: Preprocess training data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2/7: Preprocessing training data")
    print("=" * 60)

    processed_train, scaler = preprocess_training_data(train_df)

    # ------------------------------------------------------------------
    # STEP 3: Preprocess test data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3/7: Preprocessing test data")
    print("=" * 60)

    processed_test = preprocess_test_data(test_df, rul_df, scaler)

    # ------------------------------------------------------------------
    # STEP 4: Train XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4/7: Training XGBoost model")
    print("=" * 60)

    xgb_start: float = time.time()
    xgb_model, xgb_val_metrics = train_xgboost(processed_train, validation_split=True)
    xgb_time: float = time.time() - xgb_start
    print(f"[xgboost] Training time: {xgb_time:.1f} seconds")

    # ------------------------------------------------------------------
    # STEP 5: Train LSTM
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5/7: Training LSTM model")
    print("=" * 60)

    lstm_start: float = time.time()

    # Create sequences for LSTM
    X_all, y_all = create_lstm_sequences(processed_train)

    # Split sequences into train and validation
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_all, y_all,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
    )

    lstm_model, lstm_train_metrics = train_lstm(
        X_train_lstm, y_train_lstm,
        X_val_lstm, y_val_lstm,
    )
    lstm_time: float = time.time() - lstm_start
    print(f"[lstm] Training time: {lstm_time:.1f} seconds")

    # ------------------------------------------------------------------
    # STEP 6: Evaluate both models on TEST data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6/7: Evaluating models on test data")
    print("=" * 60)

    # Get true RUL values from test set
    y_true: np.ndarray = processed_test["rul"].values

    # XGBoost predictions on test data
    xgb_predictions: np.ndarray = predict_xgboost(xgb_model, processed_test)
    xgb_metrics = evaluate_model(y_true, xgb_predictions, "XGBoost")

    # LSTM predictions on test data
    # For LSTM, we need sequences. The test set only has the LAST row per engine.
    # We need to rebuild sequences from the full test data.
    # Since we preprocessed test data to only keep last cycles, we use a
    # different approach: create sequences from the full preprocessed test data
    # before it was reduced to last-cycles-only.
    #
    # For now, we re-run preprocessing on test_df to get full sequences,
    # then take only the last sequence per engine.
    from src.data.preprocessor import (
        select_features,
        normalize_sensors,
        add_rolling_features,
        add_lag_features,
    )

    # Rebuild full test data with all cycles (not just last)
    test_full = select_features(test_df.copy())
    test_full, _ = normalize_sensors(test_full, scaler=scaler, fit=False)
    test_full = add_rolling_features(test_full)
    test_full = add_lag_features(test_full)

    # Create sequences and take only the last sequence per engine
    feature_cols = get_feature_columns(test_full)
    lstm_test_sequences = []

    for unit_id in sorted(test_full["unit_id"].unique()):
        engine_data = test_full[test_full["unit_id"] == unit_id]
        features = engine_data[feature_cols].values

        if len(features) >= LSTM_SEQUENCE_LENGTH:
            # Take the last sequence_length rows as one sequence
            lstm_test_sequences.append(features[-LSTM_SEQUENCE_LENGTH:])
        else:
            # Engine has fewer cycles than sequence length — pad with zeros at the start
            padded = np.zeros((LSTM_SEQUENCE_LENGTH, len(feature_cols)))
            padded[-len(features):] = features
            lstm_test_sequences.append(padded)

    X_test_lstm: np.ndarray = np.array(lstm_test_sequences)
    print(f"[lstm] Test sequences shape: {X_test_lstm.shape}")

    lstm_predictions: np.ndarray = predict_lstm(lstm_model, X_test_lstm)
    lstm_metrics = evaluate_model(y_true, lstm_predictions, "LSTM")

    # ------------------------------------------------------------------
    # STEP 7: Compare models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7/7: Model comparison")
    print("=" * 60)

    winner = compare_models(xgb_metrics, lstm_metrics)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    total_time: float = time.time() - total_start

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:        {total_time:.1f} seconds")
    print(f"  XGBoost time:      {xgb_time:.1f} seconds")
    print(f"  LSTM time:         {lstm_time:.1f} seconds")
    print(f"  Best model:        {winner}")
    print(f"  XGBoost RMSE:      {xgb_metrics['rmse']:.4f}")
    print(f"  LSTM RMSE:         {lstm_metrics['rmse']:.4f}")
    print(f"  Test engines:      {len(y_true)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
