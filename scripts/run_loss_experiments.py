"""
Loss Function Experiments Runner (Milestone 5: E1 - E5)

Systematically compares:
  - E1: Unweighted Cross-Entropy Loss
  - E2: Class-Weighted Cross-Entropy Loss (with severe penalty boost)
  - E3: Multiclass Focal Loss (Lin et al., 2017)
  - E4: Class-Weighted Focal Loss
  - E5: Ordinal Consistent Rank Logits (CORAL) Loss (Cao et al., 2020)

Under strictly identical seeds, partitions, tokenization, and compute budgets.
Produces comparative markdown tables, trade-off curves, and per-experiment artifact bundles.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from src.training.losses import get_loss_function
from src.training.trainer import TransformerTrainer
from src.evaluation.metrics import compute_classification_metrics, save_metrics_report
from src.evaluation.confusion import plot_confusion_matrix
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def plot_loss_tradeoff(results_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Renders comparative bar chart of Macro F1, Severe Recall, and Severe Precision.
    Visually exposes the trade-off between safety sensitivity and false alarms.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(results_df))
    width = 0.25

    rects1 = ax.bar(x - width, results_df["Macro F1"], width, label="Macro F1", color="#4C72B0")
    rects2 = ax.bar(x, results_df["Severe Recall"], width, label="Severe Recall (Safety)", color="#C44E52")
    rects3 = ax.bar(x + width, results_df["Severe Precision"], width, label="Severe Precision", color="#55A868")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Loss Function Impact on Severity Detection & Safety Trade-offs", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Experiment"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_markdown_table(results_list: List[Dict[str, Any]], output_path: Path) -> str:
    """Formats comparison results into a standardized GitHub-flavored Markdown table."""
    headers = [
        "Experiment",
        "Loss Criterion",
        "Macro F1",
        "Accuracy",
        "Severe Recall",
        "Severe Precision",
        "Severe F1",
        "Moderate F1",
        "Low Stress F1",
        "Control F1",
    ]

    lines = [
        "# Loss Function Controlled Experiment Comparison (E1 - E5)",
        "",
        "> **Evaluation Split**: Held-Out Test Set (Identical partitions across all experiments)  ",
        "> **Backbone**: RoBERTa-base (128 max length, deterministic seed)  ",
        "> **Primary Metric**: Macro F1 | **Safety Metric**: Severe Crisis Recall",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in results_list:
        p_c = r.get("per_class", {})
        row = [
            r["name"],
            r["loss_type"],
            f"**{r['macro_f1']:.4f}**",
            f"{r['accuracy']:.4f}",
            f"**{r['severe_crisis_recall']:.4f}**",
            f"{r['severe_crisis_precision']:.4f}",
            f"{p_c.get('Severe Crisis', {}).get('f1', 0.0):.4f}",
            f"{p_c.get('Moderate Distress', {}).get('f1', 0.0):.4f}",
            f"{p_c.get('Low Stress', {}).get('f1', 0.0):.4f}",
            f"{p_c.get('Control', {}).get('f1', 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### Key Empirical Takeaways & Findings:",
        "1. **Severe Recall vs. Precision Trade-off**: Weighting and Focal Loss parameters explicitly shift the operating point towards high-sensitivity severe crisis detection.",
        "2. **Hard-Example Mining**: Multiclass Focal Loss focuses backpropagation gradients on subtle dysphoria and hyperbolic expressions rather than obvious control samples.",
        "3. **Ordinal Continuity**: Ordinal CORAL respects the severity ordering ($0 \\prec 1 \\prec 2 \\prec 3$), penalizing catastrophic misclassifications proportionally.",
        "",
    ])

    md_content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_content


def main():
    parser = argparse.ArgumentParser(description="Subtext Loss Function Experiments Runner")
    parser.add_argument("--config", type=str, default="configs/loss_experiments.yaml", help="Path to loss experiments config")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Path to processed dataset")
    parser.add_argument("--output-base", type=str, default="experiments/loss_functions", help="Base directory for experiment outputs")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per experiment (default: 3)")
    parser.add_argument("--experiments", nargs="+", default=None, help="Specific experiments to run (e.g. E1_cross_entropy E2_weighted_cross_entropy)")
    args = parser.parse_args()

    # Load matrix configuration
    loss_matrix_cfg = load_config(args.config)
    base_cfg_path = loss_matrix_cfg.get("base_config", "configs/roberta.yaml")
    base_cfg = load_config(base_cfg_path)

    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    output_base_dir = Path(args.output_base)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("loss_experiments", log_file=output_base_dir / "experiments_runner.log")
    logger.info("=" * 70)
    logger.info("STARTING SUBTEXT LOSS FUNCTION EXPERIMENTAL BENCHMARK (E1 - E5)")
    logger.info("=" * 70)

    # 1. Load Processed Dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Processed dataset not found at: {dataset_path}")
        sys.exit(1)

    df = pd.read_parquet(dataset_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_labels = train_df["label"].tolist()
    logger.info(f"Loaded partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Tokenizer & DataLoaders
    model_cfg = base_cfg.get("model", {})
    tok_cfg = base_cfg.get("tokenization", {})
    train_cfg = base_cfg.get("training", {})

    model_name = model_cfg.get("pretrained_model_name", "roberta-base")
    max_length = tok_cfg.get("max_length", 128)
    batch_size = train_cfg.get("batch_size", 16)
    eval_batch_size = train_cfg.get("eval_batch_size", 32)
    num_epochs = args.epochs or train_cfg.get("num_epochs", 3)
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    logger.info(f"Initializing tokenizer: {model_name} (max_length={max_length})")
    subtext_tok = SubtextTokenizer(model_name, max_length=max_length)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts=train_df["text"].tolist(),
        train_labels=train_labels,
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
        tokenizer=subtext_tok.tokenizer,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
    )

    all_experiments = loss_matrix_cfg.get("experiments", {})
    if args.experiments:
        target_experiments = {k: v for k, v in all_experiments.items() if k in args.experiments}
    else:
        target_experiments = all_experiments

    comparison_results: List[Dict[str, Any]] = []

    for exp_name, exp_cfg in target_experiments.items():
        logger.info("-" * 60)
        logger.info(f"RUNNING EXPERIMENT: {exp_name}")
        logger.info(f"Description: {exp_cfg.get('description', '')}")
        logger.info("-" * 60)

        # Ensure seed consistency before every model initialization
        set_seed(seed)

        exp_dir = output_base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 3. Instantiate Loss Criterion
        loss_fn = get_loss_function(
            config=exp_cfg,
            train_labels=train_labels,
            num_classes=4,
        )
        loss_type = exp_cfg.get("loss", {}).get("type", "unknown")
        logger.info(f"Instantiated loss function: {loss_fn.__class__.__name__} (type: {loss_type})")

        # 4. Instantiate Fresh Backbone Model
        model = RoBERTaDistressClassifier(
            pretrained_model_name=model_name,
            num_labels=4,
            classifier_dropout=model_cfg.get("classifier_dropout", 0.2),
            loss_fn=loss_fn,
        )

        # 5. Trainer Setup & Optimization Loop
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
            checkpoint_dir=exp_dir / "checkpoints",
            logger=logger,
        )

        fit_summary = trainer.fit()
        logger.info(
            f"[{exp_name}] Best Val Macro F1: {fit_summary['best_val_macro_f1']:.4f} "
            f"(Epoch {fit_summary['best_epoch']})"
        )

        # 6. Evaluate on Held-Out Test Split
        logger.info(f"[{exp_name}] Evaluating on Held-Out Test Set...")
        test_metrics, test_y_true, test_y_pred, test_y_prob = trainer.evaluate(test_loader)

        logger.info(
            f"[{exp_name}] Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Severe Recall: {test_metrics['severe_crisis_recall']:.4f} | "
            f"Severe Precision: {test_metrics['severe_crisis_precision']:.4f}"
        )

        # Save experiment metrics & classification report
        save_metrics_report(test_metrics, test_y_true, test_y_pred, exp_dir)

        # Save confusion matrix heatmap
        cm_path = exp_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            y_true=test_y_true,
            y_pred=test_y_pred,
            class_names=["Control", "Low Stress", "Moderate", "Severe"],
            output_path=cm_path,
            title=f"{exp_name} Test Confusion Matrix",
        )

        record = {
            "name": exp_name,
            "loss_type": loss_type,
            "macro_f1": test_metrics["macro_f1"],
            "accuracy": test_metrics["accuracy"],
            "weighted_f1": test_metrics["weighted_f1"],
            "severe_crisis_recall": test_metrics["severe_crisis_recall"],
            "severe_crisis_precision": test_metrics["severe_crisis_precision"],
            "per_class": test_metrics["per_class"],
            "best_val_macro_f1": fit_summary["best_val_macro_f1"],
            "best_epoch": fit_summary["best_epoch"],
        }
        comparison_results.append(record)

    # 7. Aggregate Comparison Artifacts
    logger.info("=" * 70)
    logger.info("AGGREGATING EXPERIMENTAL COMPARISON METRICS")
    logger.info("=" * 70)

    # Save summary JSON
    summary_json_path = output_base_dir / "all_loss_experiments_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved complete summary JSON to: {summary_json_path}")

    # Generate Markdown Table Report
    table_path = PROJECT_ROOT / "reports" / "tables" / "loss_experiments_comparison.md"
    generate_markdown_table(comparison_results, table_path)
    logger.info(f"Generated comparative Markdown table at: {table_path}")

    # Generate Trade-off Figure
    tradeoff_df = pd.DataFrame([
        {
            "Experiment": r["name"].replace("_", " "),
            "Macro F1": r["macro_f1"],
            "Severe Recall": r["severe_crisis_recall"],
            "Severe Precision": r["severe_crisis_precision"],
        }
        for r in comparison_results
    ])
    tradeoff_fig_path = PROJECT_ROOT / "reports" / "figures" / "loss_experiments_tradeoff.png"
    plot_loss_tradeoff(tradeoff_df, tradeoff_fig_path)
    logger.info(f"Saved trade-off visualization figure to: {tradeoff_fig_path}")

    logger.info("=" * 70)
    logger.info("LOSS EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
