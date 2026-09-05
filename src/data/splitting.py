"""
Stratified Splitting and Data Leakage Prevention Module

Partitions datasets into train, validation, and test splits with strict class stratification
and verifies that zero data leakage occurs between partitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.ingestion import validate_dataset, DatasetValidationError


class DataLeakageError(Exception):
    """Raised when data leakage (identical or near-duplicate samples) is detected between splits."""
    pass


def check_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "id",
) -> Dict[str, Union[bool, int, List[str]]]:
    """
    Checks for exact text or ID overlap between train, validation, and test splits.
    
    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        text_col: Name of the text column.
        id_col: Name of the unique identifier column.
        
    Returns:
        Dictionary reporting leakage metrics and overlap details.
    """
    train_ids: Set[str] = set(train_df[id_col].astype(str))
    val_ids: Set[str] = set(val_df[id_col].astype(str))
    test_ids: Set[str] = set(test_df[id_col].astype(str))

    train_texts: Set[str] = set(train_df[text_col].astype(str).str.strip())
    val_texts: Set[str] = set(val_df[text_col].astype(str).str.strip())
    test_texts: Set[str] = set(test_df[text_col].astype(str).str.strip())

    id_overlap_train_val = train_ids.intersection(val_ids)
    id_overlap_train_test = train_ids.intersection(test_ids)
    id_overlap_val_test = val_ids.intersection(test_ids)

    text_overlap_train_val = train_texts.intersection(val_texts)
    text_overlap_train_test = train_texts.intersection(test_texts)
    text_overlap_val_test = val_texts.intersection(test_texts)

    has_id_leakage = bool(id_overlap_train_val or id_overlap_train_test or id_overlap_val_test)
    has_text_leakage = bool(text_overlap_train_val or text_overlap_train_test or text_overlap_val_test)

    total_leaked_texts = len(text_overlap_train_val) + len(text_overlap_train_test) + len(text_overlap_val_test)
    total_leaked_ids = len(id_overlap_train_val) + len(id_overlap_train_test) + len(id_overlap_val_test)

    return {
        "has_leakage": has_id_leakage or has_text_leakage,
        "total_leaked_ids": total_leaked_ids,
        "total_leaked_texts": total_leaked_texts,
        "id_overlaps": {
            "train_val": list(id_overlap_train_val),
            "train_test": list(id_overlap_train_test),
            "val_test": list(id_overlap_val_test),
        },
        "text_overlaps": {
            "train_val": list(text_overlap_train_val)[:10],  # Sample first 10 for inspection
            "train_test": list(text_overlap_train_test)[:10],
            "val_test": list(text_overlap_val_test)[:10],
        },
    }


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    deduplicate_text: bool = True,
    prevent_leakage: bool = True,
) -> pd.DataFrame:
    """
    Splits a DataFrame into stratified train, val, and test subsets.
    
    Args:
        df: Input DataFrame containing 'text', 'label', 'id'.
        train_ratio: Proportion of dataset allocated to training.
        val_ratio: Proportion allocated to validation.
        test_ratio: Proportion allocated to testing.
        seed: Random seed for deterministic reproducibility.
        deduplicate_text: If True, drops duplicate texts before splitting to prevent leakage.
        prevent_leakage: If True, raises DataLeakageError if overlaps exist.
        
    Returns:
        DataFrame with updated 'split' column ('train', 'val', 'test').
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

    if "label" not in df.columns or "text" not in df.columns:
        raise DatasetValidationError("DataFrame must contain 'label' and 'text' columns for splitting.")

    clean_df = df.copy().reset_index(drop=True)

    if deduplicate_text:
        clean_df = clean_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # First split: Separate Train from (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    splitter_train = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_test_ratio,
        random_state=seed,
    )
    
    train_idx, val_test_idx = next(splitter_train.split(clean_df, clean_df["label"]))
    train_df = clean_df.iloc[train_idx].copy()
    val_test_df = clean_df.iloc[val_test_idx].copy().reset_index(drop=True)

    # Second split: Separate Val from Test
    # Proportion of test within the (val + test) partition
    relative_test_ratio = test_ratio / val_test_ratio
    splitter_val_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=seed,
    )

    val_idx, test_idx = next(splitter_val_test.split(val_test_df, val_test_df["label"]))
    val_df = val_test_df.iloc[val_idx].copy()
    test_df = val_test_df.iloc[test_idx].copy()

    # Assign split tags
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Leakage verification
    leakage_report = check_data_leakage(train_df, val_df, test_df)
    if leakage_report["has_leakage"]:
        msg = (
            f"Data leakage detected! "
            f"Overlapping IDs: {leakage_report['total_leaked_ids']}, "
            f"Overlapping Texts: {leakage_report['total_leaked_texts']}"
        )
        if prevent_leakage:
            raise DataLeakageError(msg)

    # Combine back into a single standardized DataFrame
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Validate final dataset against schema contract
    is_valid, errors = validate_dataset(combined_df, require_split=True)
    if not is_valid:
        raise DatasetValidationError(f"Split dataset failed schema validation: {errors}")

    return combined_df


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "data/processed/distress_severity.parquet",
) -> Path:
    """Saves the final split dataset to parquet format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
