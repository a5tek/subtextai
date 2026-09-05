"""
Unit Tests for Evaluation Metrics and Reporting
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    save_metrics_report,
)


def test_compute_classification_metrics_perfect():
    """Verify metrics calculation when predictions perfectly match ground truth."""
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 3]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["severe_crisis_recall"] == 1.0
    assert metrics["severe_crisis_precision"] == 1.0


def test_compute_classification_metrics_imperfect():
    """Verify metrics calculation on realistic mismatched predictions."""
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 3, 2]  # Moderate/Severe confused

    metrics = compute_classification_metrics(y_true, y_pred)
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert metrics["severe_crisis_recall"] == 0.5  # 1 of 2 correct
    assert metrics["severe_crisis_precision"] == 0.5


def test_compute_classification_metrics_with_probabilities():
    """Verify AUROC calculation when probability distributions are provided."""
    y_true = np.array([0, 1, 2, 3])
    # Near-perfect probabilities
    y_prob = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.05, 0.85, 0.05, 0.05],
        [0.02, 0.08, 0.8, 0.1],
        [0.01, 0.01, 0.08, 0.9],
    ])
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)
    assert "macro_auroc" in metrics
    assert metrics["macro_auroc"] > 0.90


def test_save_metrics_report():
    """Verify that metrics JSON and classification report text are saved properly."""
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    metrics = compute_classification_metrics(y_true, y_pred)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_p, txt_p = save_metrics_report(metrics, y_true, y_pred, tmpdir)
        assert json_p.exists()
        assert txt_p.exists()

        content = txt_p.read_text(encoding="utf-8")
        assert "Primary Metric (Macro F1): 1.0000" in content
