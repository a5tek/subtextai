"""
Dataset Preprocessing Pipeline

Combines text sanitization, PII redaction, quality filtering,
and audit reporting into a reproducible pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from tqdm import tqdm

from src.data.sanitization import TextSanitizer, SanitizationAudit
from src.data.ingestion import validate_dataset, DatasetValidationError


def preprocess_corpus(
    df: pd.DataFrame,
    sanitizer: Optional[TextSanitizer] = None,
    drop_empty: bool = True,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end preprocessing and ethical PII redaction on a dataset.
    
    Args:
        df: Input DataFrame conforming to at least 'id', 'text', 'label', 'source'.
        sanitizer: Optional custom TextSanitizer instance.
        drop_empty: If True, drops rows whose text becomes empty after sanitization.
        show_progress: If True, displays a tqdm progress bar.
        
    Returns:
        Tuple of (sanitized_df, audit_metrics_dict).
    """
    if "text" not in df.columns:
        raise DatasetValidationError("DataFrame missing required 'text' column.")

    if sanitizer is None:
        sanitizer = TextSanitizer()

    sanitized_df = df.copy()
    total_samples = len(sanitized_df)

    audit_summary: Dict[str, int] = {
        "total_input_rows": total_samples,
        "rows_with_pii_redacted": 0,
        "total_emails_redacted": 0,
        "total_urls_redacted": 0,
        "total_user_handles_redacted": 0,
        "total_subreddits_redacted": 0,
        "total_phones_redacted": 0,
        "total_ips_redacted": 0,
        "empty_rows_dropped": 0,
    }

    sanitized_texts = []
    iterator = tqdm(sanitized_df["text"], desc="Sanitizing text") if show_progress else sanitized_df["text"]

    for raw_text in iterator:
        clean_text, audit = sanitizer.sanitize_with_audit(str(raw_text))
        sanitized_texts.append(clean_text)

        if audit.has_modifications:
            audit_summary["rows_with_pii_redacted"] += 1

        audit_summary["total_emails_redacted"] += audit.replacements.get("email", 0)
        audit_summary["total_urls_redacted"] += audit.replacements.get("url", 0)
        audit_summary["total_user_handles_redacted"] += audit.replacements.get("user_handle", 0)
        audit_summary["total_subreddits_redacted"] += audit.replacements.get("subreddit", 0)
        audit_summary["total_phones_redacted"] += audit.replacements.get("phone", 0)
        audit_summary["total_ips_redacted"] += audit.replacements.get("ip", 0)

    sanitized_df["text"] = sanitized_texts

    # Drop empty or whitespace-only texts after sanitization
    if drop_empty:
        non_empty_mask = sanitized_df["text"].str.strip().ne("")
        dropped_count = (~non_empty_mask).sum()
        audit_summary["empty_rows_dropped"] = int(dropped_count)
        sanitized_df = sanitized_df[non_empty_mask].reset_index(drop=True)

    audit_summary["final_clean_rows"] = len(sanitized_df)
    return sanitized_df, audit_summary


def run_pipeline(
    raw_file_path: Union[str, Path],
    output_path: Union[str, Path],
    source_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level convenience runner: loads raw data, sanitizes it, and saves interim parquet.
    
    Args:
        raw_file_path: Path to raw input file.
        output_path: Path to save sanitized parquet.
        source_name: Optional corpus source name.
        
    Returns:
        Tuple of (clean_df, audit_summary).
    """
    from src.data.ingestion import load_raw_dataset

    raw_df = load_raw_dataset(raw_file_path, source_name=source_name)
    clean_df, audit = preprocess_corpus(raw_df)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_p, index=False)

    return clean_df, audit
