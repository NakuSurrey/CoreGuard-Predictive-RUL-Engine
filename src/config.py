"""
Central configuration for the CoreGuard Predictive RUL Engine.

All file paths, model parameters, column names, and constants live here.
Every other module imports from this file instead of hardcoding values.
This is the single source of truth for the entire project.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# Path.resolve() converts any relative path to an absolute path.
# We go up two levels from this file: src/config.py -> src/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# DIRECTORY PATHS
# ---------------------------------------------------------------------------
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SHAP_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "shap"

# ---------------------------------------------------------------------------
# NASA C-MAPSS DATASET CONFIGURATION
# ---------------------------------------------------------------------------
# The C-MAPSS dataset has 4 subsets: FD001, FD002, FD003, FD004.
# FD001 is the simplest: 1 operating condition, 1 fault mode.
# We start with FD001 to keep things focused and explainable.
DATASET_ID: str = "FD001"

# The raw data files have no column headers. These are the column names
# defined by NASA's documentation.
# Column layout:
#   - unit_id: which engine (1, 2, 3, ...)
#   - cycle: which time step (1, 2, 3, ... until failure)
#   - setting_1, setting_2, setting_3: operational settings (throttle, altitude, etc.)
#   - sensor_1 through sensor_21: the 21 sensor readings per cycle
COLUMN_NAMES: list[str] = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING PARAMETERS
# ---------------------------------------------------------------------------
# Rolling window size: how many past cycles to average over.
# A window of 5 means: for each row, calculate the average of the current
# value and the 4 previous values. This smooths out noise.
ROLLING_WINDOW: int = 5

# Maximum RUL cap: in the raw data, early cycles have very high RUL values
# (e.g., 300 cycles remaining). The model learns better when we cap RUL
# at a maximum value. Research shows 125 is a good cap for C-MAPSS.
# Why: an engine at 300 cycles remaining and one at 200 cycles remaining
# are both "healthy" — the model does not need to distinguish between them.
MAX_RUL: int = 125

# Sensors that carry useful degradation signal (not constant or noisy).
# These were identified through exploratory data analysis.
# Sensors not listed here are nearly constant across all cycles and
# add no predictive value — they just add noise.
USEFUL_SENSORS: list[str] = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]

# Operational settings to keep as features
USEFUL_SETTINGS: list[str] = ["setting_1", "setting_2"]

# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------------
# Train/test split ratio: 80% of engines for training, 20% for testing.
TEST_SPLIT_RATIO: float = 0.2

# Random seed: ensures reproducibility. Every random operation in the
# project uses this seed so results are identical every time.
RANDOM_SEED: int = 42

# LSTM sequence length: how many consecutive cycles the LSTM looks at
# to make one prediction. 30 means it reads the last 30 cycles of
# sensor data as one input sequence.
LSTM_SEQUENCE_LENGTH: int = 30

# XGBoost hyperparameters (starting values — can be tuned later)
XGBOOST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
}

# LSTM hyperparameters
LSTM_PARAMS: dict = {
    "epochs": 50,
    "batch_size": 256,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
}

# ---------------------------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# SAVED MODEL FILE PATHS
# ---------------------------------------------------------------------------
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_rul.pkl"
LSTM_MODEL_PATH: Path = MODELS_DIR / "lstm_rul.h5"
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
