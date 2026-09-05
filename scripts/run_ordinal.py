"""
CORAL Ordinal Severity Modeling Runner (Milestone 6)

Trains and evaluates RoBERTaOrdinalDistressClassifier:
  - Preserves severity hierarchy: Control (0) < Low Stress (1) < Moderate (2) < Severe (3)
  - Computes distance-sensitive metrics: Mean Absolute Error (MAE), Adjacent Accuracy (<=1 step off),
    Catastrophic Error Rate (>=2 steps off), and Kendall's Tau rank correlation.
  - Compares ordinal performance against standard multi-class formulations.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from src.evaluation.confusion import plot_confusion_matrix
from src.evaluation.metrics import compute_classification_metrics, save_metrics_report
from src.evaluation.ordinal_metrics import compute_ordinal_metrics
from src.models.ordinal import RoBERTaOrdinalDistressClassifier
from src.tokenization.tokenizer import SubtextTokenizer, create_data_loaders
from src.training.trainer import TransformerTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Subtext CORAL Ordinal Modeling Runner")
    parser.add_argument("--config", type=str, default="configs/roberta.yaml", help="Base config path")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Dataset path")
    parser.add_argument("--output-dir", type=str, default="experiments/ordinal", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_ordinal", log_file=output_dir / "training.log")
    logger.info("=" * 65)
    logger.info("STARTING CORAL ORDINAL SEVERITY MODELING EXPERIMENT")
    logger.info("=" * 65)

    # 1. Load Data
    dataset_path = Path(args.dataset)
    df = pd.read_parquet(dataset_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Loaded partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Tokenizer & DataLoaders
    model_name = cfg.get("model", {}).get("pretrained_model_name", "roberta-base")
    max_length = cfg.get("tokenization", {}).get("max_length", 128)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    eval_batch_size = cfg.get("training", {}).get("eval_batch_size", 32)
    lr = float(cfg.get("training", {}).get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.01))

    subtext_tok = SubtextTokenizer(model_name, max_length=max_length)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
        tokenizer=subtext_tok.tokenizer,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
    )

    # 3. Model
    logger.info("Initializing RoBERTaOrdinalDistressClassifier with CORAL head...")
    model = RoBERTaOrdinalDistressClassifier(
        pretrained_model_name=model_name,
        num_classes=4,
        classifier_dropout=cfg.get("model", {}).get("classifier_dropout", 0.2),
    )

    # 4. Trainer
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_epochs=args.epochs,
        checkpoint_dir=output_dir / "checkpoints",
        logger=logger,
    )

    fit_res = trainer.fit()
    logger.info(f"Training completed! Best Val Macro F1: {fit_res['best_val_macro_f1']:.4f}")

    # 5. Evaluate
    logger.info("Evaluating CORAL Ordinal Model on Held-Out Test split...")
    test_metrics, test_y_true, test_y_pred, test_y_prob = trainer.evaluate(test_loader)
    ord_metrics = compute_ordinal_metrics(test_y_true, test_y_pred)

    logger.info("=" * 65)
    logger.info("CORAL ORDINAL TEST RESULTS:")
    logger.info(f"  Macro F1:                 {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Accuracy:                 {test_metrics['accuracy']:.4f}")
    logger.info(f"  Severe Crisis Recall:     {test_metrics['severe_crisis_recall']:.4f}")
    logger.info(f"  Mean Absolute Error:      {ord_metrics['mae']:.4f} severity steps")
    logger.info(f"  Adjacent Accuracy:        {ord_metrics['adjacent_accuracy']:.4f} (<= 1 step off)")
    logger.info(f"  Catastrophic Error Rate:  {ord_metrics['catastrophic_error_rate']:.4f} (>= 2 steps off)")
    logger.info(f"  Kendall's Tau:            {ord_metrics['kendall_tau']:.4f}")
    logger.info("=" * 65)

    # Save artifacts
    combined_metrics = {**test_metrics, "ordinal": ord_metrics}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    plot_confusion_matrix(
        y_true=test_y_true,
        y_pred=test_y_pred,
        class_names=["Control", "Low Stress", "Moderate", "Severe"],
        output_path=output_dir / "confusion_matrix.png",
        title="CORAL Ordinal Classifier Test Confusion Matrix",
    )

    # Markdown report table
    table_lines = [
        "# Ordinal Severity Modeling Performance (CORAL Architecture)",
        "",
        "> **Objective**: Minimize severity distance errors while enforcing monotonic rank boundaries.",
        "",
        "| Metric | Value | Description |",
        "|---|---|---|",
        f"| **Macro F1** | `{test_metrics['macro_f1']:.4f}` | Balanced multiclass harmonic mean |",
        f"| **Severe Recall** | `{test_metrics['severe_crisis_recall']:.4f}` | Critical crisis detection sensitivity |",
        f"| **Mean Absolute Error (MAE)** | `{ord_metrics['mae']:.4f}` | Average severity step discrepancy |",
        f"| **Adjacent Accuracy** | `{ord_metrics['adjacent_accuracy'] * 100:.1f}%` | Predictions within +/- 1 severity rank |",
        f"| **Catastrophic Error Rate** | `{ord_metrics['catastrophic_error_rate'] * 100:.1f}%` | Severe mismatch (>= 2 steps off) |",
        f"| **Kendall's Tau Rank Correlation** | `{ord_metrics['kendall_tau']:.4f}` | Non-parametric ordinal correlation |",
        "",
    ]
    report_path = PROJECT_ROOT / "reports" / "tables" / "ordinal_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

    logger.info(f"Saved ordinal report to: {report_path}")


if __name__ == "__main__":
    main()
