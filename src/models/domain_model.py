"""
Domain-Adapted Transformer Architecture (MentalBERT / MentalRoBERTa)

Wraps models pretrained on online mental health communities (Ji et al., 2022)
to evaluate domain adaptation gains over generic RoBERTa:
  - mental/mental-roberta-base
  - mental/mental-bert-base-uncased
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class DomainAdaptedDistressClassifier(nn.Module):
    """
    Domain-adapted transformer sequence classifier for distress severity detection.
    """

    def __init__(
        self,
        pretrained_model_name: str = "mental/mental-roberta-base",
        fallback_model_name: str = "roberta-base",
        num_labels: int = 4,
        classifier_dropout: float = 0.2,
        loss_fn: Optional[nn.Module] = None,
        output_attentions: bool = False,
        config: Optional[AutoConfig] = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.fallback_model_name = fallback_model_name
        self.num_labels = num_labels
        self.output_attentions = output_attentions

        if config is not None:
            self.config = config
            self.config.num_labels = num_labels
            self.config.output_attentions = output_attentions
            self.transformer = AutoModelForSequenceClassification.from_config(self.config)
        else:
            try:
                self.config = AutoConfig.from_pretrained(
                    pretrained_model_name,
                    num_labels=num_labels,
                    classifier_dropout=classifier_dropout,
                    output_attentions=output_attentions,
                )
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name,
                    config=self.config,
                )
            except Exception:
                # Graceful fallback to generic backbone if domain model is unavailable or offline
                self.config = AutoConfig.from_pretrained(
                    fallback_model_name,
                    num_labels=num_labels,
                    classifier_dropout=classifier_dropout,
                    output_attentions=output_attentions,
                )
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    fallback_model_name,
                    config=self.config,
                )

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass handling optional token_type_ids for BERT-based backbones.
        """
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": self.output_attentions,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**kwargs)
        logits = outputs.logits
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        res = {
            "logits": logits,
            "loss": loss,
        }
        if self.output_attentions and hasattr(outputs, "attentions"):
            res["attentions"] = outputs.attentions

        return res

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts softmax probabilities with optional temperature scaling."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs["logits"] / max(temperature, 1e-4)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and configuration."""
        self.transformer.save_pretrained(str(save_directory))
