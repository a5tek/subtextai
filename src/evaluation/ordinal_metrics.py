"""
Ordinal Severity Evaluation Metrics

Provides specialized metrics that respect the natural order of distress severity:
  Control (0) < Low Stress (1) < Moderate Distress (2) < Severe Crisis (3)

Metrics:
  - Mean Absolute Error (MAE): Severity step distance penalty
  - Mean Squared Error (MSE): Quadratic severity penalty
  - Adjacent Accuracy (<= 1 step off): Triage safety metric
  - Catastrophic Error Rate (>= 2 steps off): Dangerous misclassification rate
  - Kendall's Tau: Non-parametric rank correlation
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_ordinal_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Computes ordinal severity metrics comparing predicted ranks to true severity ranks.

    Args:
        y_true: Array of true integer ordinal labels [0, K-1].
        y_pred: Array of predicted integer ordinal labels [0, K-1].
        class_names: Optional class names (default: ['Control', 'Low Stress', 'Moderate Distress', 'Severe Crisis']).

    Returns:
        Dictionary of computed ordinal metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "kendall_tau": 0.0,
            "exact_accuracy": 0.0,
            "adjacent_accuracy": 0.0,
            "catastrophic_error_rate": 0.0,
        }

    abs_diff = np.abs(y_true - y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))

    # Rank correlation: Kendall's Tau-b
    tau, p_val = stats.kendalltau(y_true, y_pred)
    if np.isnan(tau):
        tau = 0.0

    # Distance-based rate breakdown
    exact_acc = float(np.mean(abs_diff == 0))
    off_by_one = float(np.mean(abs_diff == 1))
    adjacent_acc = float(np.mean(abs_diff <= 1))
    catastrophic_errors = float(np.mean(abs_diff >= 2))

    # Per-class MAE: reveals which severity tier suffers the highest distance error
    default_names = ["Control", "Low Stress", "Moderate Distress", "Severe Crisis"]
    names = class_names or default_names
    per_class_mae: Dict[str, float] = {}

    unique_classes = np.unique(y_true)
    for c in unique_classes:
        mask = (y_true == c)
        c_mae = float(np.mean(abs_diff[mask]))
        name_str = names[c] if c < len(names) else f"Class_{c}"
        per_class_mae[name_str] = round(c_mae, 4)

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "kendall_tau": round(float(tau), 4),
        "exact_accuracy": round(exact_acc, 4),
        "off_by_one_rate": round(off_by_one, 4),
        "adjacent_accuracy": round(adjacent_acc, 4),
        "catastrophic_error_rate": round(catastrophic_errors, 4),
        "per_class_mae": per_class_mae,
    }
