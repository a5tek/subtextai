"""
Evaluation Metrics Module

Computes primary and fine-grained classification metrics for distress severity detection,
prioritizing Macro F1 and severe-class sensitivity/precision trade-offs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


LABEL_NAMES = {
    0: "Control",
    1: "Low Stress",
    2: "Moderate Distress",
    3: "Severe Crisis",
}


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[np.ndarray] = None,
    severe_class_idx: int = 3,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted discrete labels.
        y_prob: Optional predicted probability distribution (N, num_classes).
        severe_class_idx: Index of the highest risk severe crisis category (default: 3).
        
    Returns:
        Dictionary of scalar metrics and per-class reports.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    classes = sorted(list(np.unique(np.concatenate([y_true_arr, y_pred_arr]))))

    # Primary metrics
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    # Per-class metrics
    per_class_precision = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_class_recall = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)

    per_class_dict = {}
    for i, c in enumerate(classes):
        c_name = LABEL_NAMES.get(c, f"Class {c}")
        per_class_dict[c_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int((y_true_arr == c).sum()),
        }

    # Safety-critical Severe Crisis metrics
    severe_name = LABEL_NAMES.get(severe_class_idx, f"Class {severe_class_idx}")
    severe_recall = per_class_dict.get(severe_name, {}).get("recall", 0.0)
    severe_precision = per_class_dict.get(severe_name, {}).get("precision", 0.0)
    severe_f1 = per_class_dict.get(severe_name, {}).get("f1", 0.0)

    metrics: Dict[str, Any] = {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "severe_crisis_recall": round(severe_recall, 4),
        "severe_crisis_precision": round(severe_precision, 4),
        "severe_crisis_f1": round(severe_f1, 4),
        "per_class": per_class_dict,
    }

    # Multiclass AUROC if probabilities are provided
    if y_prob is not None:
        try:
            if y_prob.shape[1] > 2:
                macro_auc = float(roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro"))
                metrics["macro_auroc"] = round(macro_auc, 4)
        except Exception:
            metrics["macro_auroc"] = None

    return metrics


def save_metrics_report(
    metrics: Dict[str, Any],
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Saves metrics to JSON and outputs a formatted classification report text file.
    
    Args:
        metrics: Computed metrics dictionary.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_dir: Destination directory.
        
    Returns:
        Tuple of (metrics_json_path, report_txt_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Save Classification Report
    target_names = [LABEL_NAMES.get(c, str(c)) for c in sorted(list(np.unique(y_true)))]
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    txt_path = out_dir / "classification_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SUBTEXT CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Primary Metric (Macro F1): {metrics['macro_f1']:.4f}\n")
        f.write(f"Severe Crisis Sensitivity: {metrics['severe_crisis_recall']:.4f}\n")
        f.write(f"Severe Crisis Precision:   {metrics['severe_crisis_precision']:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)

    return json_path, txt_path
