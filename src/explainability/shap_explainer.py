"""
SHAP Explainer for the CoreGuard Predictive RUL Engine.

This module answers the question: WHY did the model predict this RUL value?

SHAP stands for SHapley Additive exPlanations.
It comes from game theory. The idea:
    - A prediction is a "game" where each feature (sensor) is a "player."
    - SHAP calculates how much each player (sensor) contributed to the
      final prediction.
    - A positive SHAP value means that sensor PUSHED the prediction higher
      (more remaining life).
    - A negative SHAP value means that sensor PUSHED the prediction lower
      (closer to failure).

This module provides three types of explanations:

    1. Global Feature Importance (summary plot)
       → Which sensors matter most ACROSS ALL engines?
       → Used by engineers to know which sensors to monitor closely.

    2. Single Engine Waterfall Plot
       → For ONE specific engine, which sensors drove THIS prediction?
       → Used when an engineer asks: "Why does the model say engine 45
         will fail in 23 cycles?"

    3. Single Engine Force Plot
       → Same information as waterfall but in a horizontal bar format.
       → Shows the push/pull of each sensor visually.

All plots are saved as image files so the dashboard (Phase 5) can display them.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
# using Agg backend — no GUI needed, just saving images to files
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    XGBOOST_MODEL_PATH,
    SHAP_OUTPUT_DIR,
    DATA_PROCESSED_DIR,
)
from src.data.preprocessor import get_feature_columns


def load_model_and_data() -> Tuple:
    """
    Load the trained XGBoost model and the processed training data.

    The training data is needed because SHAP uses it as "background data."
    SHAP needs to know the typical range of each feature so it can calculate
    how much each feature's actual value deviates from what is normal.

    Returns:
        Tuple of:
            - model: trained XGBoost model
            - train_df: processed training DataFrame
            - feature_cols: list of feature column names
    """
    # loading the trained XGBoost model from disk
    model = joblib.load(XGBOOST_MODEL_PATH)
    print(f"[shap] Loaded XGBoost model from {XGBOOST_MODEL_PATH}")

    # loading the processed training data — SHAP needs this as background
    train_path = DATA_PROCESSED_DIR / "train_processed.csv"
    train_df = pd.read_csv(train_path)
    print(f"[shap] Loaded training data: {train_df.shape[0]} rows")

    feature_cols = get_feature_columns(train_df)
    print(f"[shap] Feature columns: {len(feature_cols)}")

    return model, train_df, feature_cols


def compute_shap_values(
    model,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = 500,
) -> Tuple[shap.Explainer, np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a sample of the training data.

    Why we sample:
        Computing SHAP values for all 20,631 rows takes a long time.
        A random sample of 500 rows gives the same global picture
        but finishes in seconds instead of minutes.

    How TreeExplainer works:
        For tree-based models like XGBoost, SHAP has a fast exact algorithm.
        It traces every possible path through every decision tree and
        calculates each feature's exact contribution. No approximation needed.

    Steps:
        1. Take a random sample of training rows
        2. Create a TreeExplainer using the XGBoost model
        3. Calculate SHAP values for every row in the sample
        4. Return the explainer, SHAP values, and the sample data

    Args:
        model: trained XGBoost model
        train_df: full processed training DataFrame
        feature_cols: list of feature column names
        sample_size: how many rows to explain (default 500)

    Returns:
        Tuple of:
            - explainer: SHAP TreeExplainer object
            - shap_values: numpy array of shape (sample_size, num_features)
            - X_sample: the sampled feature DataFrame
    """
    # take a random sample to keep computation fast
    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = train_df.copy()

    X_sample = sample_df[feature_cols]

    print(f"[shap] Computing SHAP values for {len(X_sample)} samples...")

    # TreeExplainer is the fast, exact method for tree-based models
    explainer = shap.TreeExplainer(model)

    # this returns one SHAP value per feature per row
    # shape: (sample_size, num_features)
    shap_values = explainer.shap_values(X_sample)

    print(f"[shap] SHAP values computed. Shape: {shap_values.shape}")

    return explainer, shap_values, X_sample


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Create a SHAP summary plot — global feature importance.

    What this plot shows:
        - Y-axis: each feature (sensor), sorted by importance (most important at top)
        - X-axis: SHAP value (how much that feature pushed the prediction)
        - Each dot is one engine/row from the sample
        - Dot color: red = high feature value, blue = low feature value
        - Dot position: left = pushed prediction DOWN (closer to failure),
                        right = pushed prediction UP (more life remaining)

    How to read it:
        If sensor_7 has many red dots on the LEFT side, that means:
        "When sensor_7 has a HIGH reading, the model predicts LESS remaining life."
        This tells the engineer: watch sensor_7 — high values mean trouble.

    Args:
        shap_values: SHAP values array from compute_shap_values()
        X_sample: feature DataFrame matching the SHAP values
        save_path: where to save the plot image (default: SHAP_OUTPUT_DIR)

    Returns:
        Path to the saved plot image.
    """
    if save_path is None:
        SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHAP_OUTPUT_DIR / "shap_summary.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        plot_size=(12, 8),
    )
    plt.title("Global Feature Importance — Which Sensors Drive RUL Predictions?", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[shap] Summary plot saved to {save_path}")
    return save_path


def plot_bar_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Create a SHAP bar plot — average absolute feature importance.

    What this plot shows:
        - Y-axis: each feature (sensor), sorted by importance
        - X-axis: mean absolute SHAP value (average impact on prediction)
        - This is simpler than the summary plot — just shows which features
          matter most overall, without showing the direction

    When to use this vs summary plot:
        - Bar plot: quick overview for non-technical audience
        - Summary plot: detailed view for engineers who want direction info

    Args:
        shap_values: SHAP values array from compute_shap_values()
        X_sample: feature DataFrame matching the SHAP values
        save_path: where to save the plot image

    Returns:
        Path to the saved plot image.
    """
    if save_path is None:
        SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHAP_OUTPUT_DIR / "shap_bar_importance.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        plot_size=(10, 8),
    )
    plt.title("Average Sensor Impact on RUL Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[shap] Bar importance plot saved to {save_path}")
    return save_path


def plot_waterfall(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    engine_index: int = 0,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Create a SHAP waterfall plot for ONE specific engine.

    What this plot shows:
        - Starts with the base value (average prediction across all engines)
        - Each row adds or subtracts one feature's contribution
        - Red bars: this feature INCREASED the predicted RUL
        - Blue bars: this feature DECREASED the predicted RUL
        - The final value at the top is the model's actual prediction for this engine

    Example reading:
        Base value = 80 cycles (average prediction)
        sensor_7 = -15 cycles (this sensor says engine is degrading)
        sensor_12 = -8 cycles (this sensor also says degrading)
        rolling_sensor_3 = +5 cycles (this smoothed sensor looks OK)
        Final prediction = 62 cycles

    Args:
        explainer: SHAP TreeExplainer from compute_shap_values()
        X_sample: feature DataFrame
        engine_index: which row in X_sample to explain (0 = first row)
        save_path: where to save the plot image

    Returns:
        Path to the saved plot image.
    """
    if save_path is None:
        SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHAP_OUTPUT_DIR / f"shap_waterfall_engine_{engine_index}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # compute SHAP explanation for just this one row
    single_row = X_sample.iloc[[engine_index]]
    shap_explanation = explainer(single_row)

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap_explanation[0],
        show=False,
        max_display=15,
    )
    plt.title(f"Prediction Breakdown — Sample Engine #{engine_index}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[shap] Waterfall plot saved to {save_path}")
    return save_path


def plot_force(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    engine_index: int = 0,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Create a SHAP force plot for ONE specific engine.

    What this plot shows:
        - A horizontal bar centered on the base value
        - Red sections: features pushing the prediction HIGHER (more life)
        - Blue sections: features pushing the prediction LOWER (less life)
        - The length of each section shows how strong that feature's push is
        - The bold number is the final prediction

    This is the same information as the waterfall plot but in a different
    visual layout. Some people find it easier to read.

    Args:
        explainer: SHAP TreeExplainer from compute_shap_values()
        X_sample: feature DataFrame
        engine_index: which row in X_sample to explain
        save_path: where to save the plot image

    Returns:
        Path to the saved plot image.
    """
    if save_path is None:
        SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHAP_OUTPUT_DIR / f"shap_force_engine_{engine_index}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # compute SHAP explanation for this one row
    single_row = X_sample.iloc[[engine_index]]
    shap_explanation = explainer(single_row)

    # force_plot returns a visualization object — save it as image using matplotlib
    shap.force_plot(
        shap_explanation.base_values[0],
        shap_explanation.values[0],
        single_row.iloc[0],
        matplotlib=True,
        show=False,
    )
    plt.title(f"Force Plot — Sample Engine #{engine_index}", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[shap] Force plot saved to {save_path}")
    return save_path


def explain_single_prediction(
    model,
    feature_values: pd.DataFrame,
    feature_cols: list[str],
    train_background: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Explain one prediction without saving any plots.

    This function is designed for the API (Phase 4) and dashboard (Phase 5).
    Given one row of feature data, it returns a dictionary with:
        - The predicted RUL
        - Each feature's SHAP value (how much it pushed the prediction)
        - The base value (average prediction)

    The dashboard will use this data to build interactive plots.

    Args:
        model: trained XGBoost model
        feature_values: single-row DataFrame with feature columns
        feature_cols: list of feature column names
        train_background: background data for the explainer (optional, loads from disk if None)

    Returns:
        Dict with keys: prediction, base_value, feature_contributions
        feature_contributions is a list of dicts, each with: feature, value, shap_value
    """
    X = feature_values[feature_cols]

    # get the model's prediction for this row
    prediction = float(model.predict(X)[0])

    # create explainer
    explainer = shap.TreeExplainer(model)

    # compute SHAP values for this single row
    shap_explanation = explainer(X)

    base_value = float(shap_explanation.base_values[0])
    shap_vals = shap_explanation.values[0]

    # build a list of each feature's contribution, sorted by absolute impact
    contributions = []
    for i, col in enumerate(feature_cols):
        contributions.append({
            "feature": col,
            "value": float(X.iloc[0][col]),
            "shap_value": float(shap_vals[i]),
        })

    # sort by absolute SHAP value — most impactful feature first
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    result = {
        "prediction": round(prediction, 2),
        "base_value": round(base_value, 2),
        "feature_contributions": contributions,
    }

    return result


def run_full_explanation(sample_size: int = 500) -> dict:
    """
    Run the complete SHAP explanation pipeline.

    This is the main entry point. It:
        1. Loads the model and data
        2. Computes SHAP values for a sample
        3. Generates and saves all four plot types
        4. Returns paths to all saved plots

    Args:
        sample_size: number of rows to use for SHAP computation

    Returns:
        Dict with paths to all generated plots.
    """
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY PIPELINE")
    print("=" * 60)

    # Step 1: Load model and data
    model, train_df, feature_cols = load_model_and_data()

    # Step 2: Compute SHAP values
    explainer, shap_values, X_sample = compute_shap_values(
        model, train_df, feature_cols, sample_size
    )

    # Step 3: Generate all plots
    print("\n[shap] Generating plots...")

    summary_path = plot_summary(shap_values, X_sample)
    bar_path = plot_bar_importance(shap_values, X_sample)
    waterfall_path = plot_waterfall(explainer, X_sample, engine_index=0)
    force_path = plot_force(explainer, X_sample, engine_index=0)

    # Step 4: Also explain the first sample engine as a demo
    single_explanation = explain_single_prediction(
        model,
        X_sample.iloc[[0]],
        feature_cols,
    )

    print(f"\n[shap] Demo single prediction explanation:")
    print(f"  Predicted RUL: {single_explanation['prediction']} cycles")
    print(f"  Base value: {single_explanation['base_value']} cycles")
    print(f"  Top 5 contributing sensors:")
    for contrib in single_explanation["feature_contributions"][:5]:
        direction = "+" if contrib["shap_value"] > 0 else ""
        print(f"    {contrib['feature']}: {direction}{contrib['shap_value']:.2f} cycles")

    results = {
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "waterfall_plot": str(waterfall_path),
        "force_plot": str(force_path),
        "sample_explanation": single_explanation,
        "num_samples": len(X_sample),
        "num_features": len(feature_cols),
    }

    print(f"\n[shap] All plots saved to {SHAP_OUTPUT_DIR}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # running this file directly triggers the full explanation pipeline
    run_full_explanation()
