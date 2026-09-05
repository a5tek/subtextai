"""
Consistent Rank Logits (CORAL) Ordinal Transformer Architecture

Implements ordinal classification over RoBERTa backbone for ordered distress severity:
  Control (0) < Low Stress (1) < Moderate Distress (2) < Severe Crisis (3)

Architecture:
  - Shared feature extractor (RoBERTa transformer)
  - Dense projection with activation and dropout
  - CORAL head: Single weight vector w with K-1 independent cutoff thresholds b_k:
      g_k(x) = w^T phi(x) + b_k
  - Guarantees parallel decision boundaries and rank monotonicity.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.training.losses import OrdinalCoralLoss


class CoralOrdinalHead(nn.Module):
    """
    Consistent Rank Logits (CORAL) classification head.
    
    Given representation vector phi(x) in R^d:
    Computes K-1 ordinal cutoffs sharing weight vector w:
        g_k(x) = w^T phi(x) + b_k, for k in {0, ..., K-2}
    """

    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutoffs = num_classes - 1

        # Shared weight vector (in_features -> 1)
        self.fc = nn.Linear(in_features, 1, bias=False)
        # K-1 independent cutoff bias parameters
        self.biases = nn.Parameter(torch.zeros(self.num_cutoffs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor of shape (B, in_features).
            
        Returns:
            Cutoff logits of shape (B, num_cutoffs).
        """
        # Linear projection: shape (B, 1)
        proj = self.fc(x)
        # Broadcast addition with biases: shape (B, num_cutoffs)
        logits = proj + self.biases
        return logits


class RoBERTaOrdinalDistressClassifier(nn.Module):
    """
    Contextual Transformer with CORAL Ordinal Head for Severity Modeling.
    """

    def __init__(
        self,
        pretrained_model_name: Optional[str] = "roberta-base",
        num_classes: int = 4,
        classifier_dropout: float = 0.2,
        output_attentions: bool = False,
        config: Optional[AutoConfig] = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        self.num_labels = num_classes
        self.output_attentions = output_attentions

        if config is not None:
            self.config = config
            self.config.output_attentions = output_attentions
            self.transformer = AutoModel.from_config(self.config)
        else:
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name,
                output_attentions=output_attentions,
            )
            self.transformer = AutoModel.from_pretrained(
                pretrained_model_name,
                config=self.config,
            )

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.coral_head = CoralOrdinalHead(hidden_size, num_classes=num_classes)
        self.loss_fn = OrdinalCoralLoss(num_classes=num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass producing ordinal cutoff logits and loss.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            return_dict=True,
        )

        # Extract sequence pooled representation (first token / [CLS])
        cls_rep = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_rep)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)

        cutoff_logits = self.coral_head(x)  # Shape: (B, K-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(cutoff_logits, labels)

        result = {
            "logits": cutoff_logits,
            "loss": loss,
        }
        if self.output_attentions and hasattr(outputs, "attentions"):
            result["attentions"] = outputs.attentions

        return result

    def predict_rank(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predicts integer ordinal ranks [0, K-1]."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            ranks = OrdinalCoralLoss.logits_to_ordinal_preds(outputs["logits"], threshold=threshold)
        return ranks

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Predicts calibrated class probability distribution across all K classes."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            scaled_logits = outputs["logits"] / max(temperature, 1e-4)
            probs = OrdinalCoralLoss.logits_to_class_probs(scaled_logits)
        return probs

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and backbone config."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(save_dir)
        torch.save(
            {
                "dense": self.dense.state_dict(),
                "coral_head": self.coral_head.state_dict(),
                "num_classes": self.num_classes,
            },
            save_dir / "coral_head.pt",
        )
