"""
Classical Baseline Training and Evaluation Runner

Trains a TF-IDF + Logistic Regression (or Linear SVM) model on the training split,
evaluates on validation and held-out test splits, and exports standardized metrics,
classification reports, and confusion matrix visualizations.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.baseline import ClassicalBaseline
from src.evaluation.metrics import compute_classification_metrics, save_metrics_report
from src.evaluation.confusion import plot_confusion_matrix
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Subtext Classical Baseline Trainer")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to baseline YAML config")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model and metrics")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir or cfg.get("evaluation", {}).get("output_dir", "experiments/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_baseline", log_file=output_dir / "training.log")
    logger.info("=" * 60)
    logger.info(f"STARTING BASELINE EXPERIMENT: {cfg.get('experiment_name', 'baseline')}")
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

    logger.info(f"Partition sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Instantiate Model from Config
    model_cfg = cfg.get("model", {})
    vec_cfg = model_cfg.get("vectorizer", {})
    clf_cfg = model_cfg.get("classifier", {})

    baseline = ClassicalBaseline(
        classifier_type=clf_cfg.get("type", "logistic_regression"),
        max_features=vec_cfg.get("max_features", 10000),
        ngram_range=vec_cfg.get("ngram_range", (1, 2)),
        sublinear_tf=vec_cfg.get("sublinear_tf", True),
        min_df=vec_cfg.get("min_df", 2),
        C=clf_cfg.get("C", 1.0),
        class_weight=clf_cfg.get("class_weight", "balanced"),
        random_state=seed,
    )

    # 3. Fit Pipeline Strictly on Training Split
    logger.info("Fitting TF-IDF Vectorizer and Classifier strictly on Train split...")
    baseline.fit(train_df["text"].values, train_df["label"].values)
    logger.info(f"Vocabulary size learned: {len(baseline.vectorizer.vocabulary_):,} features")

    # 4. Save Checkpoint
    checkpoint_path = output_dir / "baseline_model.joblib"
    baseline.save(checkpoint_path)
    logger.info(f"Saved baseline model checkpoint to: {checkpoint_path}")

    # 5. Evaluate on Validation Split
    logger.info("Evaluating on Validation split...")
    val_preds = baseline.predict(val_df["text"].values)
    val_probs = baseline.predict_proba(val_df["text"].values)
    val_metrics = compute_classification_metrics(val_df["label"].values, val_preds, val_probs)
    logger.info(f"Validation Macro F1: {val_metrics['macro_f1']:.4f} | Severe Recall: {val_metrics['severe_crisis_recall']:.4f}")

    # 6. Evaluate on Held-Out Test Split (Primary Reporting Benchmark)
    logger.info("Evaluating on Held-Out Test split...")
    test_preds = baseline.predict(test_df["text"].values)
    test_probs = baseline.predict_proba(test_df["text"].values)
    test_metrics = compute_classification_metrics(test_df["label"].values, test_preds, test_probs)

    logger.info("=" * 60)
    logger.info(f"HELD-OUT TEST RESULTS (Empirical Baseline):")
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
        test_df["label"].values,
        test_preds,
        output_dir,
    )
    logger.info(f"Saved metrics to: {json_path}")
    logger.info(f"Saved classification report to: {txt_path}")

    # 8. Render and Save Confusion Matrix
    class_names = ["Control", "Low Stress", "Moderate", "Severe"]
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_df["label"].values,
        y_pred=test_preds,
        class_names=class_names,
        output_path=cm_path,
        title="Baseline (TF-IDF + LogReg) Test Confusion Matrix",
    )
    logger.info(f"Saved confusion matrix plot to: {cm_path}")
    logger.info("Baseline training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
