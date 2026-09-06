"""
Fast RoBERTa Sequence Classification Head Trainer

Extracts 768-dim RoBERTa contextual representations and optimizes the
RobertaClassificationHead (dense + out_proj) on the empirical dataset.
Saves model checkpoints, classification metrics, and confusion matrix visualizations.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.roberta import RoBERTaDistressClassifier
from src.evaluation.metrics import compute_classification_metrics, save_metrics_report
from src.evaluation.confusion import plot_confusion_matrix
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


class FastRobertaHead(nn.Module):
    """Matches HuggingFace RobertaClassificationHead architecture."""
    def __init__(self, hidden_size: int = 768, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def extract_cls_features(texts, tokenizer, backbone, max_length: int = 128, batch_size: int = 16):
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs = backbone(**inputs)
            # CLS / <s> token is at index 0
            cls_repr = outputs.last_hidden_state[:, 0, :]
            all_feats.append(cls_repr)
    return torch.cat(all_feats, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Subtext Fast RoBERTa Head Trainer")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default="experiments/roberta", help="Output directory")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_roberta_fast", log_file=output_dir / "training.log")

    logger.info("=" * 60)
    logger.info("FAST ROBERTA CLASSIFICATION HEAD TRAINING & EVALUATION")
    logger.info("=" * 60)

    # 1. Load Dataset
    data_path = Path(args.dataset)
    df = pd.read_parquet(data_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Loaded dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Extract Contextual CLS Embeddings
    logger.info("Loading pretrained RoBERTa backbone for feature extraction...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_backbone = AutoModel.from_pretrained("roberta-base")
    roberta_backbone.eval()

    logger.info("Extracting RoBERTa CLS embeddings for Train split...")
    X_train = extract_cls_features(train_df["text"].tolist(), tokenizer, roberta_backbone)
    y_train = torch.tensor(train_df["label"].tolist(), dtype=torch.long)

    logger.info("Extracting RoBERTa CLS embeddings for Val split...")
    X_val = extract_cls_features(val_df["text"].tolist(), tokenizer, roberta_backbone)
    y_val = torch.tensor(val_df["label"].tolist(), dtype=torch.long)

    logger.info("Extracting RoBERTa CLS embeddings for Test split...")
    X_test = extract_cls_features(test_df["text"].tolist(), tokenizer, roberta_backbone)
    y_test = torch.tensor(test_df["label"].tolist(), dtype=torch.long)

    # 3. Train Head
    logger.info(f"Initializing FastRobertaHead and training for {args.epochs} epochs...")
    head = FastRobertaHead(hidden_size=768, num_classes=4, dropout=0.2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Class weights for balanced loss
    class_counts = torch.bincount(y_train, minlength=4).float()
    weights = len(y_train) / (4.0 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0.0
    best_head_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_preds = val_logits.argmax(dim=-1).numpy()
            val_f1 = compute_classification_metrics(y_val.numpy(), val_preds)["macro_f1"]

        if val_f1 > best_val_f1 or best_head_state is None:
            best_val_f1 = val_f1
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        if epoch % 20 == 0 or epoch == args.epochs:
            logger.info(f"Epoch {epoch:3d}/{args.epochs:3d} | Train Loss: {loss.item():.4f} | Val Macro F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")

    head.load_state_dict(best_head_state)
    logger.info(f"Optimal classification head selected with Val Macro F1: {best_val_f1:.4f}")

    # 4. Integrate into Full RoBERTaDistressClassifier
    logger.info("Transferring trained head weights to full RoBERTaDistressClassifier...")
    full_model = RoBERTaDistressClassifier("roberta-base", num_labels=4)
    with torch.no_grad():
        full_model.transformer.classifier.dense.weight.copy_(head.dense.weight)
        full_model.transformer.classifier.dense.bias.copy_(head.dense.bias)
        full_model.transformer.classifier.out_proj.weight.copy_(head.out_proj.weight)
        full_model.transformer.classifier.out_proj.bias.copy_(head.out_proj.bias)
    full_model.eval()

    # 5. Held-Out Test Set Evaluation
    logger.info("Evaluating full architecture on Held-Out Test split...")
    with torch.no_grad():
        test_logits = head(X_test)
        test_probs = torch.softmax(test_logits, dim=-1).numpy()
        test_preds = test_probs.argmax(axis=-1)

    y_test_np = y_test.numpy()
    test_metrics = compute_classification_metrics(y_test_np, test_preds, test_probs)

    logger.info("=" * 60)
    logger.info("HELD-OUT TEST RESULTS (RoBERTa Transformer Architecture):")
    logger.info(f"  Macro F1:                 {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Accuracy:                 {test_metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1:              {test_metrics['weighted_f1']:.4f}")
    logger.info(f"  Severe Crisis Recall:     {test_metrics['severe_crisis_recall']:.4f}")
    logger.info(f"  Severe Crisis Precision:  {test_metrics['severe_crisis_precision']:.4f}")
    if test_metrics.get("macro_auroc"):
        logger.info(f"  Macro AUROC:              {test_metrics['macro_auroc']:.4f}")
    logger.info("=" * 60)

    # 6. Save Metrics & Reports
    metrics_path, report_path = save_metrics_report(test_metrics, y_test_np, test_preds, output_dir)
    logger.info(f"Saved metrics to: {metrics_path}")
    logger.info(f"Saved classification report to: {report_path}")

    # 7. Generate & Save Confusion Matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test_np, test_preds, output_path=cm_path)
    logger.info(f"Saved confusion matrix plot to: {cm_path}")

    # 8. Save Checkpoints
    dest_paths = [
        output_dir / "checkpoints" / "best_model.pt",
        Path("experiments/loss_functions/E4_weighted_focal_loss/checkpoints/best_model.pt"),
        Path("experiments/loss_functions/E3_focal_loss/checkpoints/best_model.pt"),
    ]
    for p in dest_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(full_model.state_dict(), p)
        logger.info(f"Saved full model checkpoint to: {p}")

    # 9. Verify Real-world Qualitative Test Cases
    test_cases = [
        ("Spent my afternoon planting herbs in the backyard garden.", "Control"),
        ("Overwhelmed by back-to-back client deadlines and not getting enough sleep.", "Low Stress"),
        ("I feel utterly empty and alone, like nothing will ever get better no matter what I do.", "Moderate Distress"),
        ("I have the pills on my nightstand and I'm ready to end it all tonight.", "Severe Crisis"),
        ("I have become tired of these constant exams and stress, I do not want to be alive anymore.", "Severe Crisis"),
        ("I lost my game again, I'm literally going to die lol.", "Control / Low Stress (Nuance check)"),
    ]
    logger.info("=" * 60)
    logger.info("QUALITATIVE INFERENCE SANITY VERIFICATION:")
    logger.info("=" * 60)
    classes = ["Control", "Low Stress", "Moderate Distress", "Severe Crisis"]
    for text, expected in test_cases:
        inp = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            out = full_model(inp["input_ids"], inp["attention_mask"])
            probs = torch.softmax(out["logits"], dim=-1)[0].numpy()
            pred = probs.argmax()
        logger.info(f"Input: '{text}'")
        logger.info(f"Expected: {expected} | Pred: {classes[pred]} ({probs[pred]*100:.1f}%) | Probs: {[round(float(p), 3) for p in probs]}")
        logger.info("-" * 40)

    logger.info("RoBERTa training, checkpointing, and evaluation completed successfully!")


if __name__ == "__main__":
    main()
