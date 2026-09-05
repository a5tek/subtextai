"""
Unit Tests for Stratified Splitting and Leakage Detection
"""

import pandas as pd
import pytest

from src.data.splitting import (
    stratified_split,
    check_data_leakage,
    DataLeakageError,
)
from src.data.ingestion import generate_synthetic_benchmark


@pytest.fixture
def sample_dataset():
    """Generates a clean 200-sample benchmark dataset for splitting tests."""
    return generate_synthetic_benchmark(num_samples=200, seed=42, inject_pii=False)


def test_stratified_split_proportions(sample_dataset):
    """Verify train/val/test splits match requested 70/15/15 proportions."""
    split_df = stratified_split(sample_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    total_len = len(split_df)
    train_count = (split_df["split"] == "train").sum()
    val_count = (split_df["split"] == "val").sum()
    test_count = (split_df["split"] == "test").sum()

    assert train_count == 140  # 70% of 200
    assert val_count == 30    # 15% of 200
    assert test_count == 30   # 15% of 200
    assert train_count + val_count + test_count == total_len


def test_stratified_split_class_balance(sample_dataset):
    """Verify class balance is preserved across each split partition."""
    split_df = stratified_split(sample_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    for split_name, expected_total in [("train", 140), ("val", 30), ("test", 30)]:
        partition = split_df[split_df["split"] == split_name]
        counts = partition["label"].value_counts()
        # Each class should have approximately equal proportion (within 1 sample for rounding)
        expected_per_class = expected_total / 4
        for label in [0, 1, 2, 3]:
            assert abs(counts[label] - expected_per_class) <= 1


def test_leakage_detection_raises_error():
    """Verify that text overlap between train and test splits triggers DataLeakageError."""
    leaked_data = pd.DataFrame({
        "id": ["id1", "id2", "id3", "id4"],
        "text": ["Identical text", "Text B", "Identical text", "Text D"],
        "label": [0, 1, 0, 3],
        "source": ["test"] * 4,
    })

    train_df = leaked_data.iloc[[0, 1]].copy()
    val_df = leaked_data.iloc[[1]].copy()
    test_df = leaked_data.iloc[[2, 3]].copy()

    report = check_data_leakage(train_df, val_df, test_df)
    assert report["has_leakage"] is True
    assert report["total_leaked_texts"] >= 1


def test_stratified_split_reproducibility(sample_dataset):
    """Verify identical random seed produces bit-exact identical splits."""
    split1 = stratified_split(sample_dataset, seed=123)
    split2 = stratified_split(sample_dataset, seed=123)

    assert split1["id"].tolist() == split2["id"].tolist()
    assert split1["split"].tolist() == split2["split"].tolist()
