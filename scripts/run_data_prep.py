"""
Data Preparation CLI Runner

Generates the benchmark corpus (or ingests external raw data),
applies the ethical PII sanitization pipeline, and saves the clean interim dataset.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.ingestion import generate_synthetic_benchmark, load_raw_dataset
from src.data.preprocessing import preprocess_corpus
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Subtext Data Ingestion & PII Sanitization Pipeline")
    parser.add_argument("--raw-file", type=str, default=None, help="Path to raw dataset (CSV, Parquet, JSONL)")
    parser.add_argument("--num-samples", type=int, default=600, help="Number of synthetic samples to generate if no raw file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-raw", type=str, default="data/raw/synthetic_benchmark.parquet", help="Raw dataset output path")
    parser.add_argument("--output-interim", type=str, default="data/interim/sanitized_benchmark.parquet", help="Sanitized interim dataset output path")
    args = parser.parse_args()

    logger = setup_logger("data_prep")
    set_seed(args.seed)

    # 1. Ingestion / Generation
    if args.raw_file and Path(args.raw_file).exists():
        logger.info(f"Loading external raw dataset from: {args.raw_file}")
        raw_df = load_raw_dataset(args.raw_file)
    else:
        logger.info(f"Generating synthetic benchmark corpus ({args.num_samples} samples, seed={args.seed})...")
        raw_df = generate_synthetic_benchmark(num_samples=args.num_samples, seed=args.seed, inject_pii=True)
        raw_out = Path(args.output_raw)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_parquet(raw_out, index=False)
        logger.info(f"Saved raw benchmark to: {raw_out}")

    logger.info(f"Raw dataset shape: {raw_df.shape}")
    logger.info(f"Raw class distribution:\n{raw_df['label'].value_counts().sort_index()}")

    # 2. Ethical Sanitization Pipeline
    logger.info("Executing ethical PII sanitization pipeline...")
    sanitized_df, audit = preprocess_corpus(raw_df, show_progress=False)

    interim_out = Path(args.output_interim)
    interim_out.parent.mkdir(parents=True, exist_ok=True)
    sanitized_df.to_parquet(interim_out, index=False)
    logger.info(f"Saved sanitized interim dataset to: {interim_out}")

    # 3. Print Audit Report
    logger.info("=" * 55)
    logger.info("ETHICAL SANITIZATION AUDIT REPORT")
    logger.info("=" * 55)
    for k, v in audit.items():
        logger.info(f"  {k:32s}: {v}")
    logger.info("=" * 55)
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
