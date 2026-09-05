"""
Unit Tests for Interpretability and Attribution Modules
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.interpretability.attention import (
    compute_attention_rollout,
    plot_attention_heatmap,
)
from src.interpretability.attribution import (
    TokenOcclusionAttributor,
    plot_token_attributions,
    compare_attention_vs_attribution,
)


def test_compute_attention_rollout():
    # 2 layers, 1 batch, 4 heads, seq_len 5
    layer1 = torch.softmax(torch.randn(1, 4, 5, 5), dim=-1)
    layer2 = torch.softmax(torch.randn(1, 4, 5, 5), dim=-1)

    rollout = compute_attention_rollout((layer1, layer2), discard_ratio=0.1)
    assert rollout.shape == (5, 5)
    assert np.all(np.isfinite(rollout))
    assert np.all(rollout >= 0.0)


def test_compare_attention_vs_attribution():
    attn = np.array([0.1, 0.4, 0.3, 0.2])
    attr = np.array([0.05, 0.5, 0.25, 0.2])

    res = compare_attention_vs_attribution(attn, attr)
    assert "spearman_correlation" in res
    assert -1.0 <= res["spearman_correlation"] <= 1.0


def test_plot_visualizations(tmp_path: Path):
    tokens = ["[CLS]", "exam", "is", "killing", "me", "[SEP]"]
    attn_matrix = np.random.rand(6, 6)
    attr_scores = np.array([0.0, 0.1, -0.05, 0.4, 0.2, 0.0])

    attn_plot = plot_attention_heatmap(tokens, attn_matrix, tmp_path / "attn.png")
    assert attn_plot.exists()

    attr_plot = plot_token_attributions(tokens, attr_scores, tmp_path / "attr.png")
    assert attr_plot.exists()


class DummyTokenizer:
    mask_token_id = 103
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[101, 10, 20, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]", "tok1", "tok2", "[SEP]"]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask):
        # Return deterministic logits based on input_ids sum
        val = input_ids.float().sum() * 0.01
        logits = torch.tensor([[val, 1.0, 0.0, -1.0]])
        return {"logits": logits}


def test_token_occlusion_attributor():
    model = DummyModel()
    tok = DummyTokenizer()
    attributor = TokenOcclusionAttributor(model, tok)

    res = attributor.attribute("dummy text", target_class=1)
    assert len(res["tokens"]) == 4
    assert len(res["attributions"]) == 4
    # Special tokens [CLS] and [SEP] should have attribution 0.0
    assert res["attributions"][0] == 0.0
    assert res["attributions"][3] == 0.0
