"""
Token-Level Feature Attribution Module (Milestone 9)

Implements:
  - Token Occlusion / Perturbation Attribution
  - Integrated Gradients (Sundararajan et al., ICML 2017)
  - Comparative Analysis: Attention Weights vs Attribution Scores
  - Diverging token attribution bar charts
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn


class TokenOcclusionAttributor:
    """
    Measures feature importance by systematically masking individual tokens
    and recording the resulting change in target class probability or logit.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, mask_token_id: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id or getattr(tokenizer, "mask_token_id", 0)

    def attribute(
        self,
        text: str,
        target_class: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
    ) -> Dict[str, Any]:
        """
        Computes token occlusion importance scores.
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(dev)
        attention_mask = encoding["attention_mask"].to(dev)

        with torch.no_grad():
            base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs["logits"][0]
            base_probs = torch.softmax(base_logits, dim=-1)

        if target_class is None:
            target_class = int(torch.argmax(base_probs).item())

        base_score = float(base_probs[target_class].item())

        seq_len = input_ids.size(1)
        scores = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        special_tokens = {
            getattr(self.tokenizer, "cls_token_id", None),
            getattr(self.tokenizer, "sep_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        }

        for pos in range(seq_len):
            tok_id = input_ids[0, pos].item()
            if tok_id in special_tokens:
                scores.append(0.0)
                continue

            # Clone and mask token at position
            perturbed_ids = input_ids.clone()
            perturbed_ids[0, pos] = self.mask_token_id

            with torch.no_grad():
                pert_outputs = self.model(input_ids=perturbed_ids, attention_mask=attention_mask)
                pert_probs = torch.softmax(pert_outputs["logits"][0], dim=-1)
                pert_score = float(pert_probs[target_class].item())

            # Impact: drop in target probability when token is omitted
            drop = base_score - pert_score
            scores.append(drop)

        return {
            "tokens": tokens,
            "attributions": np.array(scores, dtype=np.float32),
            "target_class": target_class,
            "base_probability": base_score,
        }


class IntegratedGradientsAttributor:
    """
    Axiomatic Feature Attribution via Integrated Gradients (Sundararajan et al., 2017).
    """

    def __init__(self, model: nn.Module, tokenizer: Any, embedding_layer: Optional[nn.Module] = None):
        self.model = model
        self.tokenizer = tokenizer
        # Extract word embeddings module
        if embedding_layer is not None:
            self.embeddings = embedding_layer
        elif hasattr(model, "transformer") and hasattr(model.transformer, "roberta"):
            self.embeddings = model.transformer.roberta.embeddings.word_embeddings
        elif hasattr(model, "roberta"):
            self.embeddings = model.roberta.embeddings.word_embeddings
        else:
            self.embeddings = None

    def attribute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        steps: int = 20,
    ) -> np.ndarray:
        """
        Approximates path integral along linear interpolation from zero baseline to input embeddings.
        """
        self.model.eval()
        if self.embeddings is None:
            raise ValueError("Word embeddings module not found on model.")

        # Baseline: all zero embeddings
        input_embeds = self.embeddings(input_ids).detach()
        baseline_embeds = torch.zeros_like(input_embeds)

        # Scale interpolation: alpha from 0 to 1
        alphas = torch.linspace(0.0, 1.0, steps, device=input_ids.device)
        accumulated_grads = torch.zeros_like(input_embeds)

        for alpha in alphas:
            interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interp_embeds.requires_grad_(True)

            # Forward through model using inputs_embeds if supported, or via hook
            if hasattr(self.model, "transformer"):
                outputs = self.model.transformer(
                    inputs_embeds=interp_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                logits = outputs.logits
            else:
                outputs = self.model(inputs_embeds=interp_embeds, attention_mask=attention_mask)
                logits = outputs["logits"]

            target_logit = logits[0, target_class]
            grads = torch.autograd.grad(target_logit, interp_embeds)[0]
            accumulated_grads += grads

        avg_grads = accumulated_grads / float(steps)
        integrated_grads = (input_embeds - baseline_embeds) * avg_grads
        # Sum across hidden dimension
        token_attributions = integrated_grads.sum(dim=-1)[0].detach().cpu().numpy()
        return token_attributions


def plot_token_attributions(
    tokens: List[str],
    attributions: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Token Importance Attribution",
) -> Path:
    """
    Renders horizontal diverging bar chart of token attributions.
    Positive (red/coral): pushes toward predicted severity class.
    Negative (blue/teal): pushes away from predicted severity class.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    clean_tokens = [t.replace("Ġ", "").replace(" ", "") for t in tokens]
    y_pos = np.arange(len(clean_tokens))

    colors = ["#C44E52" if val >= 0 else "#4C72B0" for val in attributions]

    fig, ax = plt.subplots(figsize=(8, max(4.5, len(clean_tokens) * 0.3)))
    ax.barh(y_pos, attributions, align="center", color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_tokens, fontsize=9)
    ax.invert_yaxis()  # Read top-to-bottom
    ax.set_xlabel("Attribution Score (Drop in P(class))", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p, dpi=200)
    plt.close(fig)
    return out_p


def compare_attention_vs_attribution(
    attention_vector: np.ndarray,
    attribution_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Quantifies rank correlation between attention weights and attribution scores.
    Tests the empirical hypothesis of whether attention can be treated as faithful explanation.
    """
    if len(attention_vector) != len(attribution_scores):
        min_len = min(len(attention_vector), len(attribution_scores))
        attention_vector = attention_vector[:min_len]
        attribution_scores = attribution_scores[:min_len]

    spearman_corr, p_val = stats.spearmanr(attention_vector, np.abs(attribution_scores))
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    return {
        "spearman_correlation": round(float(spearman_corr), 4),
        "p_value": round(float(p_val), 4),
    }
