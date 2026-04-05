"""
Data Loader for NASA C-MAPSS Dataset.

Downloads the C-MAPSS dataset (if not already present) and loads the
training and test files into pandas DataFrames.

The raw data files have NO column headers and are space-separated.
This module handles both of those quirks.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from src.config import (
    COLUMN_NAMES,
    DATA_RAW_DIR,
    DATASET_ID,
)

# ---------------------------------------------------------------------------
# DATASET SOURCE URL
# ---------------------------------------------------------------------------
# NASA's direct download link no longer works (redirects to HTML page).
# We use a GitHub-hosted copy of the original NASA C-MAPSS dataset.
# The data is identical — same files, same format, same content.
CMAPSS_BASE_URL: str = (
    "https://raw.githubusercontent.com/hankroark/"
    "Turbofan-Engine-Degradation/master/CMAPSSData"
)

# The three files we need for the FD001 subset:
#   train_FD001.txt — run-to-failure data (100 engines)
#   test_FD001.txt  — partial run data (100 engines, cut off before failure)
#   RUL_FD001.txt   — ground truth RUL at each test engine's cutoff point
REQUIRED_FILES: list[str] = [
    f"train_{DATASET_ID}.txt",
    f"test_{DATASET_ID}.txt",
    f"RUL_{DATASET_ID}.txt",
]


def download_cmapss(target_dir: Path = DATA_RAW_DIR) -> Path:
    """
    Download the C-MAPSS dataset files if they are not already present.

    Steps:
        1. Check if all three required files exist in target_dir.
        2. If any file is missing, download it from GitHub.
        3. Return the path to the data directory.

    Args:
        target_dir: Directory where raw data files will be stored.

    Returns:
        Path to the directory containing the data files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if all files already exist
    all_exist: bool = all((target_dir / f).exists() for f in REQUIRED_FILES)
    if all_exist:
        print(f"[loader] Data already exists at {target_dir}. Skipping download.")
        return target_dir

    # Download each missing file
    print(f"[loader] Downloading C-MAPSS dataset ({DATASET_ID})...")

    for filename in REQUIRED_FILES:
        file_path: Path = target_dir / filename

        if file_path.exists():
            print(f"  {filename}: already exists, skipping.")
            continue

        url: str = f"{CMAPSS_BASE_URL}/{filename}"
        print(f"  Downloading {filename}...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"  {filename}: downloaded ({len(response.content)} bytes)")

        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download {filename} from {url}. Error: {e}\n"
                f"Please download manually from NASA's Prognostics Data Repository "
                f"and place the files in {target_dir}"
            )

    print(f"[loader] Dataset downloaded to {target_dir}")
    return target_dir


def load_train_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Load the training data file for the configured dataset.

    The training file contains run-to-failure data. Each engine runs
    from cycle 1 until it fails. The last row for each engine is the
    failure point.

    Args:
        data_dir: Directory containing the raw .txt data files.

    Returns:
        DataFrame with columns: unit_id, cycle, setting_1..3, sensor_1..21
    """
    file_path: Path = data_dir / f"train_{DATASET_ID}.txt"

    if not file_path.exists():
        print("[loader] Training data not found. Attempting download...")
        download_cmapss(data_dir)

    # Read the space-separated file with no header row.
    # sep=r"\s+" handles any amount of whitespace between columns.
    # engine="python" is needed for regex separators.
    df: pd.DataFrame = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    print(f"[loader] Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns, "
          f"{df['unit_id'].nunique()} engines")

    return df


def load_test_data(data_dir: Path = DATA_RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the test data file AND the ground truth RUL values.

    The test file is DIFFERENT from the training file:
    - Engines are cut off at some random point BEFORE failure.
    - A separate RUL file tells us the true remaining life at the cutoff point.

    This lets us measure: "given sensor data up to the cutoff, can the model
    predict how many cycles are left?"

    Args:
        data_dir: Directory containing the raw .txt data files.

    Returns:
        Tuple of:
            - test_df: DataFrame of sensor readings (cut off before failure)
            - rul_df: DataFrame with one row per engine, column "rul" = true remaining life
    """
    test_path: Path = data_dir / f"test_{DATASET_ID}.txt"
    rul_path: Path = data_dir / f"RUL_{DATASET_ID}.txt"

    if not test_path.exists():
        print("[loader] Test data not found. Attempting download...")
        download_cmapss(data_dir)

    # Load test sensor data
    test_df: pd.DataFrame = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    # Load ground truth RUL values
    # This file has one number per line — one for each engine in the test set.
    rul_df: pd.DataFrame = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )

    print(f"[loader] Loaded test data: {test_df.shape[0]} rows, "
          f"{test_df['unit_id'].nunique()} engines")
    print(f"[loader] Loaded RUL ground truth: {rul_df.shape[0]} entries")

    return test_df, rul_df
