"""
XGBoost Model for RUL Prediction.

This is the BASELINE model. It takes one row of preprocessed sensor data
and predicts how many cycles remain until the engine fails.

XGBoost stands for Extreme Gradient Boosting. It builds many small
decision trees one after another. Each new tree tries to fix the mistakes
the previous trees made. The final prediction is the sum of all trees.

Why XGBoost is the baseline:
    - Works very well on tabular data (rows and columns)
    - Fast to train (seconds, not minutes)
    - Easy to explain with SHAP
    - Easy to tune and debug
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.config import (
    XGBOOST_PARAMS,
    XGBOOST_MODEL_PATH,
    TEST_SPLIT_RATIO,
    RANDOM_SEED,
)
from src.data.preprocessor import get_feature_columns


def train_xgboost(
    df: pd.DataFrame,
    validation_split: bool = True,
) -> Tuple[XGBRegressor, Optional[dict]]:
    """
    Train an XGBoost model on preprocessed training data.

    Steps:
        1. Separate input features (X) from target label (y = RUL)
        2. Optionally split into train/validation sets
        3. Create XGBRegressor with parameters from config
        4. Fit the model on training data
        5. If validation split is on, evaluate on validation set
        6. Save the trained model to disk

    Args:
        df: Preprocessed training DataFrame (output of preprocess_training_data).
        validation_split: If True, hold out 20% of data for validation during training.

    Returns:
        Tuple of:
            - Trained XGBRegressor model
            - Validation metrics dict (or None if validation_split is False)
    """
    feature_cols: list[str] = get_feature_columns(df)

    # Separate features and target
    X: pd.DataFrame = df[feature_cols]
    y: pd.Series = df["rul"]

    print(f"[xgboost] Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"[xgboost] Target range: {y.min()} to {y.max()}")

    validation_metrics: Optional[dict] = None

    if validation_split:
        # Split: 80% train, 20% validation
        # We split by rows, not by engines. This is acceptable for XGBoost
        # because it treats each row independently (no sequence dependency).
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED,
        )
        print(f"[xgboost] Train split: {X_train.shape[0]} rows")
        print(f"[xgboost] Validation split: {X_val.shape[0]} rows")
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Create and train the model
    model: XGBRegressor = XGBRegressor(**XGBOOST_PARAMS)

    print("[xgboost] Training started...")

    if validation_split and X_val is not None:
        # early_stopping_rounds: if validation score does not improve for
        # 20 consecutive rounds, stop training early to prevent overfitting.
        # Overfitting means the model memorizes training data instead of
        # learning general patterns.
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Calculate validation metrics
        y_val_pred: np.ndarray = model.predict(X_val)
        rmse: float = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
        mae: float = np.mean(np.abs(y_val - y_val_pred))

        validation_metrics = {
            "val_rmse": round(rmse, 4),
            "val_mae": round(mae, 4),
            "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else XGBOOST_PARAMS["n_estimators"],
        }

        print(f"[xgboost] Validation RMSE: {rmse:.4f}")
        print(f"[xgboost] Validation MAE: {mae:.4f}")
    else:
        model.fit(X_train, y_train, verbose=False)

    # Save the trained model to disk
    XGBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XGBOOST_MODEL_PATH)
    print(f"[xgboost] Model saved to {XGBOOST_MODEL_PATH}")

    return model, validation_metrics


def predict_xgboost(
    model: XGBRegressor,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Make RUL predictions using a trained XGBoost model.

    Args:
        model: Trained XGBRegressor.
        df: Preprocessed DataFrame (must have same feature columns as training data).

    Returns:
        Numpy array of predicted RUL values.
    """
    feature_cols: list[str] = get_feature_columns(df)
    X: pd.DataFrame = df[feature_cols]

    predictions: np.ndarray = model.predict(X)

    # Clip predictions to valid range: RUL cannot be negative
    predictions = np.clip(predictions, 0, None)

    return predictions


def load_xgboost_model() -> XGBRegressor:
    """
    Load a previously saved XGBoost model from disk.

    Returns:
        Trained XGBRegressor model.
    """
    model: XGBRegressor = joblib.load(XGBOOST_MODEL_PATH)
    print(f"[xgboost] Model loaded from {XGBOOST_MODEL_PATH}")
    return model
