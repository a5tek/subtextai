"""
EDA and Stratification CLI Runner

Loads the sanitized interim corpus, executes stratified train/val/test splitting
with leakage verification, exports processed parquet, and generates distribution reports & plots.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.splitting import stratified_split, save_processed_dataset
from src.data.eda import (
    analyze_class_distribution,
    analyze_sequence_lengths,
    analyze_lexical_diversity,
    generate_eda_plots,
    generate_markdown_summary,
)
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Subtext Stratification & EDA Runner")
    parser.add_argument("--interim-file", type=str, default="data/interim/sanitized_benchmark.parquet", help="Path to sanitized interim data")
    parser.add_argument("--output-processed", type=str, default="data/processed/distress_severity.parquet", help="Processed parquet path")
    parser.add_argument("--output-table", type=str, default="reports/tables/data_summary.md", help="Markdown summary table path")
    parser.add_argument("--output-figures", type=str, default="reports/figures", help="Figures output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger = setup_logger("eda_runner")
    set_seed(args.seed)

    # 1. Load Interim Data
    interim_path = Path(args.interim_file)
    if not interim_path.exists():
        logger.error(f"Interim file not found at: {interim_path}. Run scripts/run_data_prep.py first.")
        sys.exit(1)

    logger.info(f"Loading sanitized interim dataset from: {interim_path}")
    interim_df = pd.read_parquet(interim_path)
    logger.info(f"Loaded {len(interim_df)} samples.")

    # 2. Stratified Splitting with Leakage Verification
    logger.info("Performing stratified 70/15/15 splitting with data leakage checks...")
    split_df = stratified_split(
        interim_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        deduplicate_text=True,
        prevent_leakage=True,
    )

    # 3. Save Processed Dataset
    saved_path = save_processed_dataset(split_df, args.output_processed)
    logger.info(f"Saved processed partitioned dataset to: {saved_path}")

    # 4. Perform Exploratory Data Analysis
    logger.info("Analyzing class distributions...")
    class_dist = analyze_class_distribution(split_df)
    logger.info(f"\n{class_dist}")

    logger.info("Computing sequence length percentiles...")
    len_stats = analyze_sequence_lengths(split_df)
    w = len_stats["word_stats"]
    logger.info(f"Word length: mean={w['mean']:.1f}, median={w['median']}, 95th={w['p95']:.1f}, max={w['max']}")

    logger.info("Evaluating lexical diversity...")
    lex_div = analyze_lexical_diversity(split_df)
    logger.info(f"\n{lex_div.to_string(index=False)}")

    # 5. Generate Visualizations
    fig_dir = Path(args.output_figures)
    logger.info(f"Generating distribution plots in: {fig_dir}...")
    plots = generate_eda_plots(split_df, fig_dir)
    for name, p in plots.items():
        logger.info(f"  Generated {name}: {p}")

    # 6. Generate Markdown Report
    summary_path = Path(args.output_table)
    generate_markdown_summary(split_df, class_dist, len_stats, lex_div, summary_path)
    logger.info(f"Saved markdown summary to: {summary_path}")
    logger.info("EDA and dataset stratification complete!")


if __name__ == "__main__":
    main()
