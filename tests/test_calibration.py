"""
Unit Tests for Probability Calibration and Temperature Scaling
"""

import pytest
import numpy as np
import torch

from src.evaluation.calibration import (
    compute_ece,
    TemperatureScaler,
)


def test_compute_ece_overconfident():
    # 10 samples: model is 100% confident on class 0, but only 5 are actually class 0
    probs = np.zeros((10, 4))
    probs[:, 0] = 1.0  # Conf = 1.0
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3])  # 5 correct, 5 incorrect -> Acc = 0.50

    ece_res = compute_ece(probs, labels, n_bins=10)
    assert ece_res["ece"] == pytest.approx(0.5, abs=1e-3)
    assert ece_res["mce"] == pytest.approx(0.5, abs=1e-3)
    assert ece_res["brier_score"] > 0


def test_compute_ece_perfect():
    # Perfect calibration: 100% confident and 100% correct
    probs = np.zeros((10, 4))
    probs[:, 0] = 1.0
    labels = np.zeros(10, dtype=int)

    ece_res = compute_ece(probs, labels, n_bins=10)
    assert ece_res["ece"] == 0.0
    assert ece_res["mce"] == 0.0


def test_temperature_scaler_optimization():
    # Generate overconfident validation logits
    torch.manual_seed(42)
    val_logits = torch.randn(40, 4) * 5.0  # High scale -> overconfident
    val_labels = torch.randint(0, 4, (40,))

    scaler = TemperatureScaler()
    initial_t = scaler.temperature.item()
    fitted_t = scaler.fit(val_logits, val_labels)

    # Temperature should have adapted and must be strictly positive
    assert fitted_t > 0.0
    assert isinstance(fitted_t, float)

    # Calibrate returns valid probabilities
    calibrated_probs = scaler.calibrate(val_logits)
    assert calibrated_probs.shape == (40, 4)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0)
