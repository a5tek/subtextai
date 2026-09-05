"""
PyTorch Transformer Trainer Module

Handles training loop execution, AdamW optimization, linear learning rate warmup,
validation monitoring, early stopping, and checkpoint selection based on validation Macro F1.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import compute_classification_metrics
from src.training.losses import OrdinalCoralLoss
from src.utils.logging import setup_logger


class TransformerTrainer:
    """
    Standardized trainer for fine-tuning transformer classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 4,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 2,
        checkpoint_dir: Union[str, Path] = "experiments/roberta/checkpoints",
        device: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("trainer")

        # Set up AdamW optimizer with weight decay exclusion on bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "val_severe_recall": [],
        }

    def train_epoch(self, epoch: int) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def evaluate(self, data_loader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates model on provided data loader.
        
        Returns:
            Tuple of (metrics_dict, y_true, y_pred, y_prob).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                if loss is not None:
                    total_loss += loss.item()

                logits = outputs["logits"]
                if isinstance(getattr(self.model, "loss_fn", None), OrdinalCoralLoss):
                    num_cutoffs = self.model.loss_fn.num_cutoffs
                    cutoff_logits = logits[:, :num_cutoffs]
                    probs = OrdinalCoralLoss.logits_to_class_probs(cutoff_logits).cpu().numpy()
                    preds = OrdinalCoralLoss.logits_to_ordinal_preds(cutoff_logits).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        y_pred = np.concatenate(all_preds) if all_preds else np.array([])
        y_prob = np.concatenate(all_probs) if all_probs else np.array([])
        avg_loss = total_loss / max(1, len(data_loader))

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = round(avg_loss, 4)
        return metrics, y_true, y_pred, y_prob

    def fit(self) -> Dict[str, Any]:
        """
        Runs the full training loop with early stopping on validation Macro F1.
        
        Returns:
            Dictionary containing best validation metrics and path to saved best checkpoint.
        """
        best_val_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

        self.logger.info(f"Training on device: {self.device} for {self.num_epochs} epochs...")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics, _, _, _ = self.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]
            val_f1 = val_metrics["macro_f1"]
            val_sev_rec = val_metrics["severe_crisis_recall"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(val_f1)
            self.history["val_severe_recall"].append(val_sev_rec)

            self.logger.info(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f} | "
                f"Severe Recall: {val_sev_rec:.4f}"
            )

            # Checkpoint selection based on validation Macro F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                self.logger.info(f"  -> Best model saved to: {best_checkpoint_path} (Val Macro F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs (patience={self.early_stopping_patience}).")
                    break

        # Load best checkpoint weights
        if best_checkpoint_path.exists():
            self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))
            self.logger.info(f"Restored best weights from epoch {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "best_checkpoint_path": str(best_checkpoint_path),
            "history": self.history,
        }
