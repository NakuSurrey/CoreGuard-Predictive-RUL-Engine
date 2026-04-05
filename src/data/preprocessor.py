"""
Data Preprocessor for the CoreGuard Predictive RUL Engine.

Takes raw C-MAPSS DataFrames and transforms them into clean,
feature-engineered datasets ready for model training.

Pipeline steps:
    1. Calculate RUL labels for each row (training data only)
    2. Cap RUL at MAX_RUL (so the model focuses on near-failure patterns)
    3. Drop useless sensors (constant or noisy)
    4. Normalize sensor values (scale to 0-1 range)
    5. Engineer rolling average features (smooth out noise)
    6. Engineer lag features (capture rate of change)
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.config import (
    MAX_RUL,
    ROLLING_WINDOW,
    USEFUL_SENSORS,
    USEFUL_SETTINGS,
    SCALER_PATH,
    DATA_PROCESSED_DIR,
    LSTM_SEQUENCE_LENGTH,
)


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Remaining Useful Life (RUL) for every row in the training data.

    How RUL is calculated:
        For each engine, the last cycle is the failure point (RUL = 0).
        Every row before that counts backwards from the max cycle.

        Example for engine with 200 cycles:
            Cycle 1   -> RUL = 199  (199 cycles until failure)
            Cycle 2   -> RUL = 198
            ...
            Cycle 199 -> RUL = 1
            Cycle 200 -> RUL = 0    (this is the failure cycle)

    Args:
        df: Raw training DataFrame with columns unit_id and cycle.

    Returns:
        Same DataFrame with a new column 'rul' appended.
    """
    # For each engine, find the maximum cycle number (= the failure cycle)
    max_cycles: pd.DataFrame = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    # Merge max_cycle back onto every row
    df = df.merge(max_cycles, on="unit_id", how="left")

    # RUL = max_cycle - current_cycle
    # At cycle 1 of a 200-cycle engine: RUL = 200 - 1 = 199
    # At cycle 200 (failure):           RUL = 200 - 200 = 0
    df["rul"] = df["max_cycle"] - df["cycle"]

    # Drop the helper column
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def cap_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Cap RUL values at a maximum threshold.

    Why we do this:
        An engine at RUL=300 and one at RUL=200 are both "healthy."
        The model wastes effort trying to distinguish between them.
        By capping at 125, we tell the model: "Anything above 125, just call it 125.
        Focus your learning on the degradation zone (RUL < 125)."

    Args:
        df: DataFrame with a 'rul' column.
        max_rul: Maximum allowed RUL value.

    Returns:
        Same DataFrame with RUL values clipped.
    """
    df["rul"] = df["rul"].clip(upper=max_rul)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that carry useful signal for prediction.

    Drops sensors that are nearly constant across all cycles (they have
    near-zero variance and add noise, not signal).

    The kept columns are:
        - unit_id, cycle (identifiers)
        - setting_1, setting_2 (operational context)
        - 14 useful sensors (identified during EDA)
        - rul (if present — training data has it, test data may not)

    Args:
        df: DataFrame with all 26 original columns.

    Returns:
        DataFrame with only useful columns.
    """
    keep_cols: list[str] = ["unit_id", "cycle"] + USEFUL_SETTINGS + USEFUL_SENSORS

    # Include 'rul' column if it exists (training data)
    if "rul" in df.columns:
        keep_cols.append("rul")

    df = df[keep_cols].copy()
    return df


def normalize_sensors(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale all sensor and setting values to the range [0, 1].

    Why we normalize:
        Sensor values have different scales. Temperature might be 500-600.
        Pressure might be 30-50. If we feed these raw values to a model,
        the model thinks temperature is "more important" just because the
        numbers are bigger. Normalization puts every sensor on the same
        0-to-1 scale so the model treats them fairly.

    Important:
        During TRAINING, we fit the scaler (learn the min/max from training data)
        AND transform the data.
        During INFERENCE (live predictions), we only transform using the
        already-fitted scaler. We NEVER fit on test/live data.

    Args:
        df: DataFrame with sensor columns.
        scaler: Pre-fitted scaler (for inference). None means create a new one.
        fit: If True, fit the scaler on this data. If False, only transform.

    Returns:
        Tuple of (transformed DataFrame, fitted scaler).
    """
    # Columns to normalize: settings + sensors (not unit_id, cycle, or rul)
    feature_cols: list[str] = USEFUL_SETTINGS + USEFUL_SENSORS

    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        # Learn min/max from this data AND apply the transformation
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        # Apply transformation using previously learned min/max
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features for each sensor.

    What a rolling average does:
        For each row, it calculates the average of the current value
        and the previous (ROLLING_WINDOW - 1) values.

        Example with window=5 for sensor_2:
            Cycle 1: sensor_2 = 0.5  -> rolling_sensor_2 = 0.5  (only 1 value)
            Cycle 2: sensor_2 = 0.6  -> rolling_sensor_2 = 0.55
            Cycle 3: sensor_2 = 0.7  -> rolling_sensor_2 = 0.6
            Cycle 4: sensor_2 = 0.8  -> rolling_sensor_2 = 0.65
            Cycle 5: sensor_2 = 0.9  -> rolling_sensor_2 = 0.7  (avg of 5 values)

    Why this helps:
        Individual sensor readings are noisy. A single spike might be
        measurement error, not real degradation. The rolling average
        smooths out these random spikes so the model sees the TRUE trend.

    Args:
        df: DataFrame with normalized sensor columns.

    Returns:
        Same DataFrame with new rolling_sensor_* columns added.
    """
    for sensor in USEFUL_SENSORS:
        col_name: str = f"rolling_{sensor}"
        # Group by engine (unit_id) so we don't mix data across engines.
        # min_periods=1 means: for the first few rows where we don't have
        # enough history, just average whatever we have.
        df[col_name] = (
            df.groupby("unit_id")[sensor]
            .rolling(window=ROLLING_WINDOW, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def add_lag_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Add lag features showing the CHANGE in sensor values over time.

    What a lag feature is:
        The difference between the current sensor reading and the reading
        N cycles ago. If lag=1, it's the change from the previous cycle.

        Example for sensor_2 with lag=1:
            Cycle 1: sensor_2 = 0.5  -> diff_sensor_2 = NaN  (no previous)
            Cycle 2: sensor_2 = 0.6  -> diff_sensor_2 = 0.1  (0.6 - 0.5)
            Cycle 3: sensor_2 = 0.5  -> diff_sensor_2 = -0.1 (0.5 - 0.6)

    Why this helps:
        The raw sensor value tells the model WHERE the sensor is.
        The lag feature tells the model which DIRECTION it's moving.
        A sensor going up fast (big positive diff) is more urgent than
        a sensor that's high but stable.

    Args:
        df: DataFrame with normalized sensor columns.
        lag: Number of cycles to look back.

    Returns:
        Same DataFrame with new diff_sensor_* columns added.
    """
    for sensor in USEFUL_SENSORS:
        col_name: str = f"diff_{sensor}"
        # diff(lag) within each engine group
        df[col_name] = df.groupby("unit_id")[sensor].diff(periods=lag)

    # Fill NaN values in lag features with 0.
    # The first row of each engine has no previous cycle, so diff is NaN.
    # We fill with 0 because "no change" is the safest assumption.
    lag_cols: list[str] = [f"diff_{s}" for s in USEFUL_SENSORS]
    df[lag_cols] = df[lag_cols].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of all feature column names (everything the model sees).

    This excludes: unit_id (identifier), cycle (identifier), rul (target label).

    Returns:
        List of column names that are input features.
    """
    exclude: set = {"unit_id", "cycle", "rul"}
    return [col for col in df.columns if col not in exclude]


def preprocess_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Run the full preprocessing pipeline on training data.

    Pipeline order:
        Step 1 -> Add RUL labels
        Step 2 -> Cap RUL at MAX_RUL
        Step 3 -> Select useful features
        Step 4 -> Normalize sensors (fit scaler)
        Step 5 -> Add rolling average features
        Step 6 -> Add lag features
        Step 7 -> Save processed data and scaler

    Args:
        df: Raw training DataFrame from loader.load_train_data().

    Returns:
        Tuple of (processed DataFrame, fitted scaler).
    """
    print("[preprocessor] Starting training data preprocessing...")

    # Step 1: Add RUL labels
    df = add_rul_labels(df)
    print(f"  Step 1/7: RUL labels added. Range: {df['rul'].min()} to {df['rul'].max()}")

    # Step 2: Cap RUL
    df = cap_rul(df)
    print(f"  Step 2/7: RUL capped at {MAX_RUL}. New range: {df['rul'].min()} to {df['rul'].max()}")

    # Step 3: Select useful features
    df = select_features(df)
    print(f"  Step 3/7: Feature selection done. {len(df.columns)} columns kept.")

    # Step 4: Normalize
    df, scaler = normalize_sensors(df, fit=True)
    print(f"  Step 4/7: Sensors normalized to [0, 1].")

    # Step 5: Rolling averages
    df = add_rolling_features(df)
    print(f"  Step 5/7: Rolling averages added (window={ROLLING_WINDOW}).")

    # Step 6: Lag features
    df = add_lag_features(df)
    print(f"  Step 6/7: Lag features added.")

    # Step 7: Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED_DIR / "train_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"  Step 7/7: Saved processed training data to {processed_path}")

    # Save the scaler for later use during inference
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved to {SCALER_PATH}")

    print(f"[preprocessor] Training preprocessing complete. "
          f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df, scaler


def preprocess_test_data(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
) -> pd.DataFrame:
    """
    Run the preprocessing pipeline on test data.

    Key difference from training:
        - RUL labels come from the separate RUL file (not calculated)
        - The scaler is NOT re-fitted — we use the one fitted on training data
        - We only keep the LAST cycle of each engine (that's the prediction point)

    Pipeline order:
        Step 1 -> Select useful features
        Step 2 -> Normalize sensors (using training scaler)
        Step 3 -> Add rolling average features
        Step 4 -> Add lag features
        Step 5 -> Keep only the last cycle per engine
        Step 6 -> Attach ground truth RUL labels

    Args:
        test_df: Raw test DataFrame from loader.load_test_data().
        rul_df: Ground truth RUL DataFrame from loader.load_test_data().
        scaler: Fitted scaler from training preprocessing.

    Returns:
        Processed test DataFrame with one row per engine and true RUL labels.
    """
    print("[preprocessor] Starting test data preprocessing...")

    # Load saved scaler if none provided
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
        print("  Loaded saved scaler from disk.")

    # Step 1: Select useful features
    test_df = select_features(test_df)

    # Step 2: Normalize using TRAINING scaler (fit=False is critical)
    test_df, _ = normalize_sensors(test_df, scaler=scaler, fit=False)

    # Step 3: Rolling averages
    test_df = add_rolling_features(test_df)

    # Step 4: Lag features
    test_df = add_lag_features(test_df)

    # Step 5: Keep only the last cycle per engine
    # The test data is cut off at a random point. The last row of each
    # engine is the "current state" — that's what we predict from.
    last_cycles: pd.DataFrame = test_df.groupby("unit_id").last().reset_index()

    # Step 6: Attach ground truth RUL
    # rul_df has one row per engine, in the same order as the engine IDs
    last_cycles["rul"] = rul_df["rul"].values

    # Cap RUL to match training data distribution
    last_cycles = cap_rul(last_cycles)

    # Save
    processed_path = DATA_PROCESSED_DIR / "test_processed.csv"
    last_cycles.to_csv(processed_path, index=False)
    print(f"[preprocessor] Test preprocessing complete. "
          f"Final shape: {last_cycles.shape[0]} rows x {last_cycles.shape[1]} columns")

    return last_cycles


def create_lstm_sequences(
    df: pd.DataFrame,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data into 3D sequences for LSTM input.

    LSTM expects input shaped as: (num_samples, sequence_length, num_features)

    What this function does:
        For each engine, it creates sliding windows of consecutive cycles.

        Example with sequence_length=3 and 2 features:
            Engine data (5 cycles):
                Cycle 1: [0.1, 0.5]
                Cycle 2: [0.2, 0.6]
                Cycle 3: [0.3, 0.7]
                Cycle 4: [0.4, 0.8]
                Cycle 5: [0.5, 0.9]

            Sequences produced:
                Sequence 1: [[0.1,0.5], [0.2,0.6], [0.3,0.7]] -> RUL at cycle 3
                Sequence 2: [[0.2,0.6], [0.3,0.7], [0.4,0.8]] -> RUL at cycle 4
                Sequence 3: [[0.3,0.7], [0.4,0.8], [0.5,0.9]] -> RUL at cycle 5

    Args:
        df: Preprocessed training DataFrame.
        sequence_length: Number of consecutive cycles per sequence.

    Returns:
        Tuple of:
            - X: numpy array of shape (num_samples, sequence_length, num_features)
            - y: numpy array of shape (num_samples,) containing RUL targets
    """
    feature_cols: list[str] = get_feature_columns(df)
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    # Process each engine separately
    for unit_id in df["unit_id"].unique():
        engine_data: pd.DataFrame = df[df["unit_id"] == unit_id]
        features: np.ndarray = engine_data[feature_cols].values
        rul_values: np.ndarray = engine_data["rul"].values

        # Create sliding windows
        # We need at least sequence_length rows to create one sequence
        if len(features) < sequence_length:
            continue

        for i in range(sequence_length, len(features) + 1):
            # Window: rows from (i - sequence_length) to i
            sequences.append(features[i - sequence_length : i])
            # Target: RUL at the LAST cycle of this window
            targets.append(rul_values[i - 1])

    X: np.ndarray = np.array(sequences)
    y: np.ndarray = np.array(targets)

    print(f"[preprocessor] LSTM sequences created. X shape: {X.shape}, y shape: {y.shape}")

    return X, y
