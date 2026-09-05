"""
Transformer Attention Introspection Module (Milestone 9)

Extracts self-attention patterns across layers and heads for introspection:
  - Per-head and layer-averaged attention extraction
  - Attention Rollout (Abnar & Zuidema, ACL 2020)
  - Token-level attention heatmaps

RESEARCH BOUNDARY NOTE:
  Attention matrices reflect internal model routing, NOT causal explanations.
  Do not equate high attention with definitive proof of human causal significance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def extract_attention_weights(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Runs inference with attention extraction enabled.

    Args:
        model: Transformer classifier supporting output_attentions=True.
        tokenizer: Hugging Face tokenizer.
        text: Input text string to introspect.
        device: Torch device.
        max_length: Maximum sequence length.

    Returns:
        Dictionary with 'tokens', 'attentions' (tuple of layer tensors), and 'logits'.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(dev)
    attention_mask = encoding["attention_mask"].to(dev)

    # Save original attention setting and ensure output_attentions=True
    orig_setting = getattr(model, "output_attentions", False)
    model.output_attentions = True

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    model.output_attentions = orig_setting

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    attentions = outputs.get("attentions", None)

    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": outputs["logits"],
        "attentions": attentions,
    }


def compute_attention_rollout(
    attentions: Tuple[torch.Tensor, ...],
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """
    Computes Attention Rollout (Abnar & Zuidema, 2020) tracking multi-layer information flow.

    Args:
        attentions: Tuple of length L (num_layers), each tensor of shape (1, num_heads, seq_len, seq_len).
        discard_ratio: Proportion of lowest attention weights to prune for noise reduction.

    Returns:
        2D numpy array of shape (seq_len, seq_len) representing cumulative information flow.
    """
    if attentions is None or len(attentions) == 0:
        raise ValueError("No attention tensors provided.")

    seq_len = attentions[0].shape[-1]
    result = np.eye(seq_len, dtype=np.float32)

    for layer_attn in attentions:
        # Average across attention heads: shape (seq_len, seq_len)
        attn_avg = layer_attn[0].mean(dim=0).cpu().numpy()

        if discard_ratio > 0.0:
            flat = attn_avg.flatten()
            thresh = np.quantile(flat, discard_ratio)
            attn_avg[attn_avg < thresh] = 0.0
            # Renormalize rows
            row_sums = attn_avg.sum(axis=-1, keepdims=True)
            attn_avg = np.divide(attn_avg, row_sums, out=np.zeros_like(attn_avg), where=row_sums > 0)

        # Residual connection identity matrix addition: A_hat = 0.5 * (A + I)
        attn_with_identity = 0.5 * (attn_avg + np.eye(seq_len))
        # Matrix multiplication across layers
        result = np.matmul(attn_with_identity, result)

    return result


def plot_attention_heatmap(
    tokens: List[str],
    attention_matrix: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Self-Attention Introspection Map",
) -> Path:
    """
    Renders token-to-token attention heatmap.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    clean_tokens = [t.replace("Ġ", "").replace(" ", "") for t in tokens]
    seq_len = len(clean_tokens)

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.4), max(5, seq_len * 0.35)))
    cax = ax.matshow(attention_matrix, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(clean_tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(clean_tokens, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_p, dpi=200)
    plt.close(fig)
    return out_p
