"""
Unit Tests for RoBERTa Sequence Classification Model Architecture
"""

import pytest
import torch
from transformers import RobertaConfig

from src.models.roberta import RoBERTaDistressClassifier


@pytest.fixture(scope="module")
def tiny_roberta_model():
    """Instantiates a lightweight RoBERTa model from config for instant offline testing."""
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=128,
        num_labels=4,
    )
    return RoBERTaDistressClassifier(config=config, num_labels=4)


def test_roberta_forward_pass_shapes(tiny_roberta_model):
    """Verify RoBERTa forward pass yields correct logits shape (B, num_labels)."""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 500, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.tensor([0, 3], dtype=torch.long)

    outputs = tiny_roberta_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, 4)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0.0


def test_roberta_predict_proba_sums(tiny_roberta_model):
    """Verify predict_proba outputs valid softmax distributions summing to 1.0."""
    batch_size = 3
    seq_len = 16
    input_ids = torch.randint(0, 500, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    probs = tiny_roberta_model.predict_proba(input_ids, attention_mask)
    assert probs.shape == (batch_size, 4)

    row_sums = probs.sum(dim=-1).detach().cpu().numpy()
    import numpy as np
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_roberta_save_and_from_pretrained(tiny_roberta_model, tmp_path):
    """Verify model save_pretrained and reload."""
    save_dir = tmp_path / "saved_roberta"
    tiny_roberta_model.save_pretrained(str(save_dir))
    assert (save_dir / "config.json").exists()

    reloaded = RoBERTaDistressClassifier.from_pretrained(str(save_dir), num_labels=4)
    assert reloaded.num_labels == 4
