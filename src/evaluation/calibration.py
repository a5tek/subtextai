"""
Probability Calibration and Temperature Scaling (Milestone 8)

Implements:
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Multiclass Brier Score
  - Post-training Temperature Scaling (Guo et al., ICML 2017)
  - Reliability diagrams comparing raw softmax vs calibrated confidence
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def compute_ece(
    probs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.

    Args:
        probs: Array of predicted probabilities of shape (N, num_classes).
        labels: Ground truth integer class labels of shape (N,).
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Dictionary with ece, mce, brier_score, bin_accuracies, bin_confidences, and bin_counts.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    N = len(labels)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0}

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    # Multiclass Brier Score: 1/N sum_i sum_k (p_ik - y_ik)^2
    num_classes = probs.shape[-1]
    one_hot = np.zeros((N, num_classes), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if 0 <= lbl < num_classes:
            one_hot[i, lbl] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            gap = abs(avg_acc - avg_conf)

            ece += (bin_size / N) * gap
            mce = max(mce, gap)

            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_counts.append(int(bin_size))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier_score), 4),
        "bin_accuracies": [round(x, 4) for x in bin_accs],
        "bin_confidences": [round(x, 4) for x in bin_confs],
        "bin_counts": bin_counts,
    }


class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling module (Guo et al., 2017).
    
    Learns a single scalar temperature T > 0 on validation set logits:
        p_calibrated = softmax(logits / T)
    Preserves top-1 accuracy while correcting overconfidence.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature."""
        return logits / self.temperature

    def fit(
        self,
        val_logits: Union[np.ndarray, torch.Tensor],
        val_labels: Union[np.ndarray, torch.Tensor],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Learns optimal temperature T by minimizing Negative Log-Likelihood (NLL) on validation data.
        """
        if isinstance(val_logits, np.ndarray):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.tensor(val_labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            # Constrain T > 0.01
            with torch.no_grad():
                self.temperature.clamp_(min=0.01)
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        return float(self.temperature.item())

    def calibrate(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Applies learned temperature scaling and returns normalized probabilities."""
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            scaled = self.forward(logits)
            probs = torch.softmax(scaled, dim=-1).cpu().numpy()
        return probs


def plot_reliability_diagram(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> Path:
    """
    Renders side-by-side reliability diagrams comparing raw vs temperature-scaled confidence.
    """
    uncal_res = compute_ece(uncalibrated_probs, labels, n_bins=n_bins)
    cal_res = compute_ece(calibrated_probs, labels, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    width = 0.08

    # Uncalibrated Plot
    ax1.bar(
        bin_centers,
        uncal_res["bin_accuracies"],
        width=width,
        edgecolor="black",
        alpha=0.7,
        color="#C44E52",
        label="Accuracy",
    )
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(f"Uncalibrated (ECE: {uncal_res['ece']:.4f}, MCE: {uncal_res['mce']:.4f})", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Calibrated Plot
    ax2.bar(
        bin_centers,
        cal_res["bin_accuracies"],
        width=width,
        edgecolor="black",
        alpha=0.7,
        color="#4C72B0",
        label="Accuracy",
    )
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"Calibrated (ECE: {cal_res['ece']:.4f}, MCE: {cal_res['mce']:.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
