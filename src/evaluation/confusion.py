"""
Confusion Matrix Visualization Module

Plots normalized and count-annotated confusion matrices to expose failure patterns,
specifically distinguishing adjacent misclassifications from distant errors.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> Path:
    """
    Renders and exports a dual-annotated confusion matrix (normalized + raw counts).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class display names.
        output_path: File path to save PNG figure.
        title: Figure title.
        cmap: Matplotlib colormap.
        
    Returns:
        Path to saved figure.
    """
    if class_names is None:
        class_names = ["Control", "Low Stress", "Moderate", "Severe"]

    cm_raw = confusion_matrix(y_true, y_pred)
    # Compute row-normalized proportions (recall per class)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_norm = cm_raw.astype("float") / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom", fontsize=10)

    n_classes = len(class_names)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Severity",
        xlabel="Predicted Severity",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate cells with both normalized proportion and raw count
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            norm_val = cm_norm[i, j]
            raw_val = cm_raw[i, j]
            text = f"{norm_val:.2f}\n({raw_val})"
            color = "white" if norm_val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.grid(False)
    plt.tight_layout()

    if output_path is None:
        output_path = Path("experiments/baseline/confusion_matrix.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close(fig)

    return out_p
