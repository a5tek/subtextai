"""
RoBERTa Sequence Classification Training and Evaluation Runner

Fine-tunes RoBERTa-base on the training split, tracks validation Macro F1,
selects the best model checkpoint, evaluates on the held-out test split,
and exports metrics, reports, training curves, and confusion matrix plots.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.roberta import RoBERTaDistressClassifier
from src.tokenization.tokenizer import SubtextTokenizer, create_data_loaders
from src.training.trainer import TransformerTrainer
from src.evaluation.metrics import compute_classification_metrics, save_metrics_report
from src.evaluation.confusion import plot_confusion_matrix
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def plot_training_curves(history: dict, output_path: Path) -> Path:
    """Plots train vs val loss and validation Macro F1 progression."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax_loss.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4C72B0")
    ax_loss.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#C44E52")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax_loss.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.6)

    # Macro F1 plot
    ax_f1.plot(epochs, history["val_macro_f1"], "o-", label="Val Macro F1", color="#55A868")
    ax_f1.plot(epochs, history["val_severe_recall"], "^--", label="Val Severe Recall", color="#8172B3")
    ax_f1.set_xlabel("Epoch", fontsize=11)
    ax_f1.set_ylabel("Score", fontsize=11)
    ax_f1.set_title("Validation Macro F1 & Severe Recall", fontsize=12, fontweight="bold")
    ax_f1.legend()
    ax_f1.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Subtext RoBERTa Baseline Trainer")
    parser.add_argument("--config", type=str, default="configs/roberta.yaml", help="Path to roberta YAML config")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model and metrics")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir or cfg.get("evaluation", {}).get("output_dir", "experiments/roberta"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_roberta", log_file=output_dir / "training.log")
    logger.info("=" * 60)
    logger.info(f"STARTING ROBERTA EXPERIMENT: {cfg.get('experiment_name', 'roberta')}")
    logger.info("=" * 60)

    # 1. Load Processed Dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at: {dataset_path}. Run scripts/run_eda.py first.")
        sys.exit(1)

    logger.info(f"Loading partitioned dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Partitions -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Tokenizer & DataLoaders
    model_cfg = cfg.get("model", {})
    tok_cfg = cfg.get("tokenization", {})
    train_cfg = cfg.get("training", {})

    model_name = model_cfg.get("pretrained_model_name", "roberta-base")
    max_length = tok_cfg.get("max_length", 256)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    eval_batch_size = train_cfg.get("eval_batch_size", 32)
    num_epochs = args.epochs or train_cfg.get("num_epochs", 4)
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    logger.info(f"Initializing tokenizer for: {model_name} (max_length={max_length})")
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

    # 3. Model Initialization
    logger.info(f"Loading pretrained architecture: {model_name}...")
    model = RoBERTaDistressClassifier(
        pretrained_model_name=model_name,
        num_labels=cfg.get("data", {}).get("num_classes", 4),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.2),
    )

    # 4. Trainer Initialization & Training
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 2),
        checkpoint_dir=output_dir / "checkpoints",
        logger=logger,
    )

    fit_results = trainer.fit()
    logger.info(f"Training completed! Best Val Macro F1: {fit_results['best_val_macro_f1']:.4f} at epoch {fit_results['best_epoch']}")

    # 5. Plot and Save Training Curves
    curve_path = output_dir / "training_curve.png"
    plot_training_curves(fit_results["history"], curve_path)
    logger.info(f"Saved training curves to: {curve_path}")

    # 6. Evaluate on Held-Out Test Split
    logger.info("Evaluating best model on Held-Out Test split...")
    test_metrics, test_y_true, test_y_pred, test_y_prob = trainer.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("HELD-OUT TEST RESULTS (RoBERTa Transformer Baseline):")
    logger.info(f"  Macro F1:                 {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Accuracy:                 {test_metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1:              {test_metrics['weighted_f1']:.4f}")
    logger.info(f"  Severe Crisis Recall:     {test_metrics['severe_crisis_recall']:.4f}")
    logger.info(f"  Severe Crisis Precision:  {test_metrics['severe_crisis_precision']:.4f}")
    if test_metrics.get("macro_auroc"):
        logger.info(f"  Macro AUROC:              {test_metrics['macro_auroc']:.4f}")
    logger.info("=" * 60)

    logger.info("Per-Class Metrics:")
    for cls_name, p_dict in test_metrics["per_class"].items():
        logger.info(f"  {cls_name:22s} -> Precision: {p_dict['precision']:.4f}, Recall: {p_dict['recall']:.4f}, F1: {p_dict['f1']:.4f} (Support: {p_dict['support']})")

    # 7. Save Standardized Artifacts
    json_path, txt_path = save_metrics_report(
        test_metrics,
        test_y_true,
        test_y_pred,
        output_dir,
    )
    logger.info(f"Saved metrics to: {json_path}")
    logger.info(f"Saved classification report to: {txt_path}")

    # 8. Render and Save Confusion Matrix
    class_names = ["Control", "Low Stress", "Moderate", "Severe"]
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_y_true,
        y_pred=test_y_pred,
        class_names=class_names,
        output_path=cm_path,
        title="RoBERTa Baseline Test Confusion Matrix",
    )
    logger.info(f"Saved confusion matrix plot to: {cm_path}")
    logger.info("RoBERTa training and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
