"""
Unit Tests for Ordinal Modeling and Evaluation Metrics
"""

import pytest
import numpy as np
import torch
from transformers import RobertaConfig

from src.evaluation.ordinal_metrics import compute_ordinal_metrics
from src.models.ordinal import CoralOrdinalHead, RoBERTaOrdinalDistressClassifier


def test_compute_ordinal_metrics_perfect():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    metrics = compute_ordinal_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["adjacent_accuracy"] == 1.0
    assert metrics["catastrophic_error_rate"] == 0.0
    assert metrics["kendall_tau"] == 1.0


def test_compute_ordinal_metrics_adjacent_vs_catastrophic():
    # True: [0, 0, 0, 3]
    # Pred: [1, 1, 1, 0]  (three off-by-1, one off-by-3)
    y_true = np.array([0, 0, 0, 3])
    y_pred = np.array([1, 1, 1, 0])

    metrics = compute_ordinal_metrics(y_true, y_pred)
    assert metrics["exact_accuracy"] == 0.0
    assert metrics["off_by_one_rate"] == 0.75
    assert metrics["adjacent_accuracy"] == 0.75
    assert metrics["catastrophic_error_rate"] == 0.25
    assert metrics["mae"] == 1.5  # (1 + 1 + 1 + 3) / 4


def test_coral_ordinal_head():
    head = CoralOrdinalHead(in_features=64, num_classes=4)
    x = torch.randn(8, 64)
    out = head(x)

    assert out.shape == (8, 3)
    # Check shared weight and 3 cutoff biases
    assert head.fc.weight.shape == (1, 64)
    assert head.biases.shape == (3,)


def test_roberta_ordinal_classifier_forward():
    # Lightweight offline config
    cfg = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = RoBERTaOrdinalDistressClassifier(config=cfg, num_classes=4)

    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert "logits" in outputs
    assert outputs["logits"].shape == (4, 3)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0

    # Backpropagation test
    outputs["loss"].backward()
    assert model.coral_head.fc.weight.grad is not None

    # Prediction test
    ranks = model.predict_rank(input_ids, attention_mask)
    assert ranks.shape == (4,)
    assert torch.all((ranks >= 0) & (ranks < 4))

    probs = model.predict_proba(input_ids, attention_mask)
    assert probs.shape == (4, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)
