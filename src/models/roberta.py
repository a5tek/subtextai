"""
RoBERTa Sequence Classification Model Architecture

Wraps pretrained RoBERTa backbone with sequence classification head,
supporting custom loss functions, self-attention extraction, and uncertainty output.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class RoBERTaDistressClassifier(nn.Module):
    """
    Contextual Transformer Classifier for 4-Class Distress Severity Detection.
    """

    def __init__(
        self,
        pretrained_model_name: Optional[str] = "roberta-base",
        num_labels: int = 4,
        classifier_dropout: float = 0.2,
        loss_fn: Optional[nn.Module] = None,
        output_attentions: bool = False,
        config: Optional[AutoConfig] = None,
    ):
        """
        Args:
            pretrained_model_name: Hugging Face model identifier (e.g. 'roberta-base').
            num_labels: Number of output severity classes (4).
            classifier_dropout: Dropout probability before linear projection.
            loss_fn: Optional custom loss criterion (defaults to standard CrossEntropyLoss).
            output_attentions: If True, model returns self-attention matrices.
            config: Optional pre-built AutoConfig for offline testing.
        """
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.output_attentions = output_attentions

        if config is not None:
            self.config = config
            self.config.num_labels = num_labels
            self.config.output_attentions = output_attentions
            self.transformer = AutoModelForSequenceClassification.from_config(self.config)
        else:
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

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token IDs, shape (B, max_length).
            attention_mask: Tensor of binary attention masks, shape (B, max_length).
            labels: Optional ground-truth severity labels, shape (B,).
            
        Returns:
            Dict containing:
                - 'logits': Tensor of shape (B, num_labels).
                - 'loss': Scalar loss if labels provided, else None.
                - 'attentions': Tuple of attention tensors if output_attentions=True.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            return_dict=True,
        )

        logits = outputs.logits  # Shape: (B, num_labels)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        result = {
            "logits": logits,
            "loss": loss,
        }

        if self.output_attentions and hasattr(outputs, "attentions"):
            result["attentions"] = outputs.attentions

        return result

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates softmax class probabilities with optional temperature scaling.
        
        Args:
            input_ids: Token ID tensor.
            attention_mask: Attention mask tensor.
            temperature: Positive scaling factor (T > 0).
            
        Returns:
            Softmax probabilities, shape (B, num_labels).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"] / temperature
            from src.training.losses import OrdinalCoralLoss
            if isinstance(getattr(self, "loss_fn", None), OrdinalCoralLoss):
                probs = OrdinalCoralLoss.logits_to_class_probs(logits[:, : self.num_labels - 1])
            else:
                probs = torch.softmax(logits, dim=-1)
        return probs

    def save_pretrained(self, save_directory: str) -> None:
        """Saves model weights and configuration."""
        self.transformer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        num_labels: int = 4,
        loss_fn: Optional[nn.Module] = None,
        output_attentions: bool = False,
    ) -> "RoBERTaDistressClassifier":
        """Loads model weights and configuration from a saved directory."""
        instance = cls.__new__(cls)
        super(RoBERTaDistressClassifier, instance).__init__()
        instance.pretrained_model_name = save_directory
        instance.num_labels = num_labels
        instance.output_attentions = output_attentions
        instance.config = AutoConfig.from_pretrained(save_directory, num_labels=num_labels)
        instance.transformer = AutoModelForSequenceClassification.from_pretrained(save_directory, config=instance.config)
        instance.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        return instance
