"""
Unit Tests for Dataset Ingestion and Preprocessing Pipeline
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data.ingestion import (
    validate_dataset,
    load_raw_dataset,
    generate_synthetic_benchmark,
    DatasetValidationError,
)
from src.data.preprocessing import preprocess_corpus


def test_validate_dataset_valid():
    """Verify compliant dataframe passes validation without errors."""
    data = {
        "id": ["1", "2", "3", "4"],
        "text": ["Control text", "Low stress text", "Moderate distress text", "Severe crisis text"],
        "label": [0, 1, 2, 3],
        "source": ["test_source"] * 4,
        "split": ["train", "train", "val", "test"],
    }
    df = pd.DataFrame(data)
    is_valid, errors = validate_dataset(df, require_split=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataset_missing_columns():
    """Verify missing columns trigger validation errors."""
    df = pd.DataFrame({"id": ["1"], "text": ["hello"]})
    is_valid, errors = validate_dataset(df)
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_dataset_invalid_labels():
    """Verify out-of-range labels (e.g. 5, -1) trigger validation failure."""
    df = pd.DataFrame({
        "id": ["1", "2"],
        "text": ["text a", "text b"],
        "label": [0, 99],
        "source": ["src", "src"],
        "split": ["train", "test"],
    })
    is_valid, errors = validate_dataset(df)
    assert is_valid is False
    assert any("invalid labels" in e for e in errors)


def test_validate_dataset_empty_text():
    """Verify empty or whitespace-only text entries are rejected."""
    df = pd.DataFrame({
        "id": ["1", "2"],
        "text": ["Valid text", "    "],
        "label": [0, 1],
        "source": ["src", "src"],
        "split": ["train", "test"],
    })
    is_valid, errors = validate_dataset(df)
    assert is_valid is False
    assert any("empty or whitespace-only" in e for e in errors)


def test_generate_synthetic_benchmark():
    """Verify synthetic generator produces balanced 4-class data conforming to contract."""
    df = generate_synthetic_benchmark(num_samples=100, seed=42)
    assert len(df) == 100
    assert set(df["label"].unique()) == {0, 1, 2, 3}
    # Check exact balance: 25 per class
    counts = df["label"].value_counts()
    for c in [0, 1, 2, 3]:
        assert counts[c] == 25
    is_valid, errors = validate_dataset(df, require_split=False)
    assert is_valid, f"Validation errors: {errors}"


def test_load_raw_dataset_with_column_mapping():
    """Verify loading custom-column CSV and normalizing schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "custom_data.csv"
        raw_df = pd.DataFrame({
            "post_content": ["I am feeling sad", "I am happy"],
            "severity_level": [2, 0],
            "dataset_origin": ["reddit", "reddit"],
        })
        raw_df.to_csv(csv_path, index=False)

        loaded = load_raw_dataset(
            csv_path,
            column_mapping={"post_content": "text", "severity_level": "label", "dataset_origin": "source"},
        )
        assert "text" in loaded.columns
        assert "label" in loaded.columns
        assert "id" in loaded.columns  # Auto-generated deterministic ID
        assert len(loaded) == 2


def test_preprocess_corpus_scrubbing_and_audit():
    """Verify preprocessing correctly redacts PII and tracks aggregate audit counts."""
    raw_df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "text": [
            "Normal text with no PII",
            "Contact @moderator or visit https://subtext.org for info",
            "Email user@test.com or call 555-123-4567",
        ],
        "label": [0, 1, 2],
        "source": ["test", "test", "test"],
    })

    clean_df, audit = preprocess_corpus(raw_df)
    assert audit["rows_with_pii_redacted"] == 2
    assert audit["total_user_handles_redacted"] == 1
    assert audit["total_urls_redacted"] == 1
    assert audit["total_emails_redacted"] == 1
    assert audit["total_phones_redacted"] >= 1
    assert "[USER]" in clean_df.iloc[1]["text"]
    assert "[URL]" in clean_df.iloc[1]["text"]
    assert "[EMAIL]" in clean_df.iloc[2]["text"]
    assert "[PHONE]" in clean_df.iloc[2]["text"]
