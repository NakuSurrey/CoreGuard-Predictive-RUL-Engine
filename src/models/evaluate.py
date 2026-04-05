"""
Model Evaluation and Comparison.

Takes trained XGBoost and LSTM models, runs them on test data,
and compares their performance using standard metrics.

Metrics used:
    - RMSE (Root Mean Squared Error): average prediction error in cycles.
      Lower is better. Penalizes large errors more than small ones.
    - MAE (Mean Absolute Error): average absolute prediction error in cycles.
      Lower is better. Treats all errors equally.
    - Score (NASA's custom scoring function): penalizes late predictions
      (predicting failure AFTER it actually happens) much more heavily than
      early predictions. In real life, a late prediction means the engine
      already failed and people could be hurt.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.data.preprocessor import get_feature_columns


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Formula:
        RMSE = sqrt( mean( (actual - predicted)^2 ) )

    Example:
        actual    = [10, 20, 30]
        predicted = [12, 18, 35]
        errors    = [-2, +2, -5]
        squared   = [4, 4, 25]
        mean      = 11.0
        RMSE      = sqrt(11.0) = 3.32

    This means: on average, the prediction is off by about 3.32 cycles.

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.

    Returns:
        RMSE value (float).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Formula:
        MAE = mean( |actual - predicted| )

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.

    Returns:
        MAE value (float).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate NASA's custom asymmetric scoring function for RUL prediction.

    This score is ASYMMETRIC — it punishes differently depending on
    whether the prediction is early or late:

        If prediction is EARLY (predicted RUL > actual RUL):
            The model said "50 cycles left" but only 30 were left.
            Penalty: e^(-d/13) - 1   (lighter penalty)
            Why lighter: early prediction means we replace the part
            sooner than needed — wasteful but SAFE.

        If prediction is LATE (predicted RUL < actual RUL):
            The model said "50 cycles left" but 70 were actually left.
            Wait — this seems backwards. Let me clarify:
            Actually: d = predicted - actual.
            If d < 0: we predicted LESS remaining life than reality.
                      This is a CONSERVATIVE (early) prediction. Lighter penalty.
            If d > 0: we predicted MORE remaining life than reality.
                      This is a DANGEROUS (late) prediction. Heavier penalty.

    The formula uses d = predicted - actual:
        d < 0 (early/conservative): score += e^(-d/13) - 1
        d >= 0 (late/dangerous):    score += e^(d/10) - 1

    Lower total score is better.

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Total NASA score (float). Lower is better.
    """
    d: np.ndarray = y_pred - y_true

    score: float = 0.0
    for di in d:
        if di < 0:
            # Early prediction (conservative) — lighter penalty
            score += np.exp(-di / 13.0) - 1
        else:
            # Late prediction (dangerous) — heavier penalty
            score += np.exp(di / 10.0) - 1

    return float(score)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """
    Evaluate a model's predictions using all three metrics.

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.
        model_name: Name of the model (for printing).

    Returns:
        Dict with keys: rmse, mae, nasa_score.
    """
    rmse: float = calculate_rmse(y_true, y_pred)
    mae: float = calculate_mae(y_true, y_pred)
    nasa_score: float = calculate_nasa_score(y_true, y_pred)

    metrics: Dict[str, float] = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "nasa_score": round(nasa_score, 4),
    }

    print(f"\n[evaluate] {model_name} Results:")
    print(f"  RMSE:        {metrics['rmse']:.4f} cycles")
    print(f"  MAE:         {metrics['mae']:.4f} cycles")
    print(f"  NASA Score:  {metrics['nasa_score']:.4f} (lower is better)")

    return metrics


def compare_models(
    xgb_metrics: Dict[str, float],
    lstm_metrics: Dict[str, float],
) -> str:
    """
    Compare XGBoost and LSTM metrics side by side and declare a winner.

    Args:
        xgb_metrics: XGBoost evaluation metrics dict.
        lstm_metrics: LSTM evaluation metrics dict.

    Returns:
        Name of the winning model ("XGBoost" or "LSTM").
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'XGBoost':>12} {'LSTM':>12} {'Winner':>12}")
    print("-" * 60)

    winners: list[str] = []

    for metric in ["rmse", "mae", "nasa_score"]:
        xgb_val: float = xgb_metrics[metric]
        lstm_val: float = lstm_metrics[metric]

        # For all three metrics, lower is better
        if xgb_val < lstm_val:
            winner = "XGBoost"
        elif lstm_val < xgb_val:
            winner = "LSTM"
        else:
            winner = "Tie"

        winners.append(winner)
        print(f"{metric:<15} {xgb_val:>12.4f} {lstm_val:>12.4f} {winner:>12}")

    print("-" * 60)

    # Overall winner: whichever model wins more metrics
    xgb_wins: int = winners.count("XGBoost")
    lstm_wins: int = winners.count("LSTM")

    if xgb_wins > lstm_wins:
        overall_winner = "XGBoost"
    elif lstm_wins > xgb_wins:
        overall_winner = "LSTM"
    else:
        # Tie-breaker: use RMSE (most standard metric)
        overall_winner = "XGBoost" if xgb_metrics["rmse"] <= lstm_metrics["rmse"] else "LSTM"

    print(f"\nOverall Winner: {overall_winner}")
    print("=" * 60)

    return overall_winner
