"""
Automated Unit Tests for Custom Severity Loss Functions (E1 - E5)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import (
    compute_class_weights,
    WeightedCrossEntropyLoss,
    FocalLoss,
    OrdinalCoralLoss,
    get_loss_function,
)


def test_compute_class_weights():
    # Imbalanced labels: 0 (10), 1 (10), 2 (5), 3 (2)
    labels = [0] * 10 + [1] * 10 + [2] * 5 + [3] * 2
    weights = compute_class_weights(labels, num_classes=4, method="inverse_frequency")

    assert weights.shape == (4,)
    # Mean of weights should be normalized to 1.0
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    # Less frequent classes should have higher weights: w[3] > w[2] > w[0] == w[1]
    assert weights[3] > weights[2] > weights[0]
    assert torch.isclose(weights[0], weights[1], atol=1e-5)


def test_compute_class_weights_with_severe_boost():
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    base_weights = compute_class_weights(labels, num_classes=4, severe_boost=1.0)
    boosted_weights = compute_class_weights(labels, num_classes=4, severe_boost=2.5)

    # In balanced distribution, base weights are all 1.0
    assert torch.allclose(base_weights, torch.tensor([1.0, 1.0, 1.0, 1.0]))
    # Boosted weights should emphasize severe class (index 3)
    assert boosted_weights[3] > boosted_weights[0]


def test_weighted_cross_entropy_loss():
    weights = torch.tensor([1.0, 1.0, 1.5, 3.0])
    loss_fn = WeightedCrossEntropyLoss(weight=weights)

    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0.0

    # Backpropagation verification
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_focal_loss_equivalence_to_ce_at_gamma_zero():
    """When gamma=0 and alpha=None, Focal Loss must equal standard Cross Entropy."""
    focal_fn = FocalLoss(gamma=0.0, alpha=None)
    ce_fn = nn.CrossEntropyLoss()

    logits = torch.tensor([[2.0, -1.0, 0.5, -0.2], [-0.5, 3.0, 1.2, 0.1]])
    targets = torch.tensor([0, 1])

    focal_val = focal_fn(logits, targets)
    ce_val = ce_fn(logits, targets)

    assert torch.isclose(focal_val, ce_val, atol=1e-5)


def test_focal_loss_modulating_downweighting():
    """Focal loss should down-weight well-classified easy samples compared to standard CE."""
    focal_fn = FocalLoss(gamma=2.0)
    ce_fn = nn.CrossEntropyLoss(reduction="none")

    # Sample 1: Easy/Confident (logit 10 for true class 0)
    # Sample 2: Hard/Ambiguous (logits close for all classes)
    logits = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 1.1, 0.9, 0.8],
    ])
    targets = torch.tensor([0, 0])

    ce_losses = ce_fn(logits, targets)
    focal_losses = FocalLoss(gamma=2.0, reduction="none")(logits, targets)

    # For easy sample, focal loss should be vastly smaller relative to its CE loss
    ratio_easy = focal_losses[0] / ce_losses[0]
    ratio_hard = focal_losses[1] / ce_losses[1]

    # Modulating factor (1 - p_t)^2 is much smaller on the easy sample
    assert ratio_easy < ratio_hard
    assert ratio_easy < 0.01  # Highly confident example should be suppressed


def test_weighted_focal_loss_backward():
    alpha = torch.tensor([1.0, 1.2, 1.5, 2.5])
    focal_fn = FocalLoss(gamma=2.0, alpha=alpha)

    logits = torch.randn(6, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 2, 3])

    loss = focal_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_ordinal_coral_targets():
    coral_fn = OrdinalCoralLoss(num_classes=4)
    targets = torch.tensor([0, 1, 2, 3])
    coral_targets = coral_fn.label_to_coral_targets(targets)

    # Expected:
    # 0 -> [0, 0, 0]
    # 1 -> [1, 0, 0]
    # 2 -> [1, 1, 0]
    # 3 -> [1, 1, 1]
    expected = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    assert torch.allclose(coral_targets, expected)


def test_ordinal_coral_predictions_and_probs():
    coral_fn = OrdinalCoralLoss(num_classes=4)

    # Extreme logits: clear class 3 (all cutoffs high), clear class 0 (all cutoffs low)
    logits = torch.tensor([
        [-10.0, -10.0, -10.0],  # Class 0
        [10.0, 10.0, 10.0],    # Class 3
    ])

    preds = coral_fn.logits_to_ordinal_preds(logits)
    assert torch.equal(preds, torch.tensor([0, 3]))

    probs = coral_fn.logits_to_class_probs(logits)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)
    assert probs[0, 0] > 0.99
    assert probs[1, 3] > 0.99


def test_get_loss_function_factory():
    dummy_labels = [0, 1, 2, 3] * 5

    # 1. Standard Cross Entropy
    l1 = get_loss_function({"loss": {"type": "cross_entropy"}})
    assert isinstance(l1, nn.CrossEntropyLoss)

    # 2. Weighted Cross Entropy
    l2 = get_loss_function({"loss": {"type": "weighted_cross_entropy", "severe_boost_factor": 2.0}}, train_labels=dummy_labels)
    assert isinstance(l2, WeightedCrossEntropyLoss)

    # 3. Focal Loss
    l3 = get_loss_function({"loss": {"type": "focal_loss", "gamma": 2.5}})
    assert isinstance(l3, FocalLoss)
    assert l3.gamma == 2.5

    # 4. Weighted Focal Loss
    l4 = get_loss_function({"loss": {"type": "weighted_focal_loss", "gamma": 1.5}}, train_labels=dummy_labels)
    assert isinstance(l4, FocalLoss)
    assert l4.gamma == 1.5
    assert l4.alpha is not None

    # 5. Ordinal Coral Loss
    l5 = get_loss_function({"loss": {"type": "ordinal_coral"}})
    assert isinstance(l5, OrdinalCoralLoss)
