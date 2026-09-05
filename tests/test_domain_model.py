"""
Unit Tests for Domain-Adapted Transformer Classifier (MentalBERT)
"""

import pytest
import torch
from transformers import RobertaConfig

from src.models.domain_model import DomainAdaptedDistressClassifier
from src.training.losses import WeightedCrossEntropyLoss


def test_domain_model_forward():
    cfg = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    loss_fn = WeightedCrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.5, 3.0]))
    model = DomainAdaptedDistressClassifier(config=cfg, loss_fn=loss_fn, num_labels=4)

    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert "logits" in outputs
    assert outputs["logits"].shape == (4, 4)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0

    outputs["loss"].backward()
    assert model.transformer.classifier.out_proj.weight.grad is not None

    probs = model.predict_proba(input_ids, attention_mask)
    assert probs.shape == (4, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)
