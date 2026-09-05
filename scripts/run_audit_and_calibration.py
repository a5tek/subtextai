"""
Model Calibration, Error Audit, and Interpretability Pipeline

Executes:
  1. Systematic Error Analysis (identifying and categorizing false negatives and positives)
  2. Post-hoc Temperature Scaling Calibration & Reliability Diagrams
  3. Feature Attribution and Attention Rollout on challenging edge-case samples
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

from src.evaluation.calibration import (
    TemperatureScaler,
    compute_ece,
    plot_reliability_diagram,
)
from src.evaluation.error_analysis import DistressErrorAnalyzer
from src.interpretability.attention import (
    compute_attention_rollout,
    extract_attention_weights,
    plot_attention_heatmap,
)
from src.interpretability.attribution import (
    TokenOcclusionAttributor,
    compare_attention_vs_attribution,
    plot_token_attributions,
)
from src.models.roberta import RoBERTaDistressClassifier
from src.tokenization.tokenizer import SubtextTokenizer, create_data_loaders
from src.training.trainer import TransformerTrainer
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Subtext Audit and Calibration Runner")
    parser.add_argument("--model-ckpt", type=str, default=None, help="Path to checkpoint directory or model weights")
    parser.add_argument("--dataset", type=str, default="data/processed/distress_severity.parquet", help="Path to processed data")
    parser.add_argument("--output-dir", type=str, default="reports", help="Base reports directory")
    args = parser.parse_args()

    set_seed(42)
    output_dir = Path(args.output_dir)
    logger = setup_logger("audit_calibration")
    logger.info("=" * 65)
    logger.info("STARTING POST-TRAINING AUDIT, CALIBRATION & INTERPRETABILITY")
    logger.info("=" * 65)

    # 1. Load Dataset
    df = pd.read_parquet(args.dataset)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # 2. Tokenizer & DataLoaders
    subtext_tok = SubtextTokenizer("roberta-base", max_length=128)
    _, val_loader, test_loader = create_data_loaders(
        train_texts=val_df["text"].tolist(),
        train_labels=val_df["label"].tolist(),
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
        tokenizer=subtext_tok.tokenizer,
        batch_size=16,
        eval_batch_size=32,
        max_length=128,
    )

    # 3. Load Model Checkpoint
    ckpt_path = None
    if args.model_ckpt:
        ckpt_path = Path(args.model_ckpt)
    else:
        # Search available checkpoints
        candidates = [
            PROJECT_ROOT / "experiments" / "loss_functions" / "E3_focal_loss" / "checkpoints" / "best_model.pt",
            PROJECT_ROOT / "experiments" / "loss_functions" / "E1_cross_entropy" / "checkpoints" / "best_model.pt",
            PROJECT_ROOT / "experiments" / "roberta" / "checkpoints" / "best_model.pt",
        ]
        for cand in candidates:
            if cand.exists():
                ckpt_path = cand
                break

    model = RoBERTaDistressClassifier(pretrained_model_name="roberta-base", num_labels=4)
    if ckpt_path and ckpt_path.exists():
        logger.info(f"Loading weights from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    else:
        logger.warning("No checkpoint found; running with initialized weights.")

    model.eval()

    # 4. Extract Logits & Predictions
    trainer = TransformerTrainer(model=model, train_loader=val_loader, val_loader=val_loader)
    logger.info("Extracting validation predictions for temperature scaling...")
    _, val_y_true, _, val_probs = trainer.evaluate(val_loader)

    # Re-extract validation raw logits for calibration
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            val_logits_list.append(out["logits"].cpu().numpy())
    val_logits = np.concatenate(val_logits_list)

    logger.info("Extracting test predictions for evaluation and error audit...")
    test_metrics, test_y_true, test_y_pred, test_probs = trainer.evaluate(test_loader)

    test_logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            test_logits_list.append(out["logits"].cpu().numpy())
    test_logits = np.concatenate(test_logits_list)

    # 5. Fit Temperature Scaling Calibration
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits, val_y_true)
    logger.info(f"Fitted Temperature: T = {opt_temp:.4f}")

    calibrated_test_probs = scaler.calibrate(test_logits)

    uncal_ece = compute_ece(test_probs, test_y_true)
    cal_ece = compute_ece(calibrated_test_probs, test_y_true)
    logger.info(f"Uncalibrated ECE: {uncal_ece['ece']:.4f} | Calibrated ECE: {cal_ece['ece']:.4f}")

    # Plot Reliability Diagram
    rel_path = output_dir / "figures" / "reliability_diagram.png"
    plot_reliability_diagram(test_probs, calibrated_test_probs, test_y_true, rel_path)
    logger.info(f"Saved reliability diagram to: {rel_path}")

    # 6. Systematic Error Analysis
    error_analyzer = DistressErrorAnalyzer(
        texts=test_df["text"].tolist(),
        y_true=test_y_true,
        y_pred=test_y_pred,
        y_prob=calibrated_test_probs,
    )
    error_report_path = error_analyzer.export_error_report(output_dir / "tables", "Systematic Error Analysis")
    logger.info(f"Saved error analysis report to: {error_report_path}")

    # 7. Interpretability Case Studies
    sample_texts = [
        "This exam is literally killing me right now, so stressed out!",
        "I feel completely empty inside and cannot find reasons to live.",
    ]

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    attributor = TokenOcclusionAttributor(model, subtext_tok.tokenizer)

    for idx, sample_text in enumerate(sample_texts):
        logger.info(f"Generating interpretability artifacts for Case {idx + 1}: '{sample_text}'")
        # Attention
        attn_data = extract_attention_weights(model, subtext_tok.tokenizer, sample_text)
        if attn_data["attentions"]:
            rollout = compute_attention_rollout(attn_data["attentions"], discard_ratio=0.1)
            attn_fig_path = fig_dir / f"attention_case_{idx + 1}.png"
            plot_attention_heatmap(attn_data["tokens"], rollout, attn_fig_path, title=f"Case {idx + 1} Attention Rollout")
            logger.info(f"  Saved attention map to: {attn_fig_path}")

        # Feature Attribution
        attr_data = attributor.attribute(sample_text)
        attr_fig_path = fig_dir / f"attribution_case_{idx + 1}.png"
        plot_token_attributions(attr_data["tokens"], attr_data["attributions"], attr_fig_path, title=f"Case {idx + 1} Token Attribution")
        logger.info(f"  Saved attribution chart to: {attr_fig_path}")

    logger.info("AUDIT, CALIBRATION AND INTERPRETABILITY COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
