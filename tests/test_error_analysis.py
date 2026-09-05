"""
Unit Tests for Systematic Error Analysis Module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.error_analysis import (
    classify_error_taxonomy,
    DistressErrorAnalyzer,
)


def test_classify_error_taxonomy_hyperbole():
    text = "This exam is literally dying and killing me."
    cat = classify_error_taxonomy(text, y_true=0, y_pred=2)
    assert cat == "FIGURATIVE_HYPERBOLE"


def test_classify_error_taxonomy_negation():
    text = "I am not planning to kill myself, things are getting better."
    cat = classify_error_taxonomy(text, y_true=1, y_pred=3)
    assert cat == "NEGATION_FAILURE"


def test_classify_error_taxonomy_implicit_distress():
    text = "Everything is completely numb and hollow. I just want the darkness to take over."
    cat = classify_error_taxonomy(text, y_true=3, y_pred=1)
    assert cat == "IMPLICIT_DISTRESS"


def test_classify_error_taxonomy_dark_humor_mask():
    text = "Lol wishing my car would swerve off the bridge, classic me haha."
    cat = classify_error_taxonomy(text, y_true=3, y_pred=0)
    assert cat == "DARK_HUMOR_MASK"


def test_classify_error_taxonomy_ambiguous_intent():
    text = "I'm done and can't take this anymore, lost 5 matches in a row."
    cat = classify_error_taxonomy(text, y_true=0, y_pred=2)
    assert cat == "AMBIGUOUS_INTENT_FALSE_ALARM"


def test_classify_error_taxonomy_borderline_and_catastrophic():
    text = "Feeling tired after working late."
    assert classify_error_taxonomy(text, y_true=1, y_pred=2) == "BORDERLINE_ADJACENT"
    assert classify_error_taxonomy(text, y_true=0, y_pred=3) == "CATASTROPHIC_ERROR"


def test_distress_error_analyzer(tmp_path: Path):
    texts = [
        "Normal sunny day",
        "Midterm is killing me literally",
        "I want to end it all tonight",
        "I am fine now, stopped feeling suicidal",
    ]
    y_true = np.array([0, 0, 3, 0])
    y_pred = np.array([0, 2, 1, 3])  # 1 correct, 3 errors
    probs = np.ones((4, 4)) * 0.25

    analyzer = DistressErrorAnalyzer(texts, y_true, y_pred, y_prob=probs)
    errors_df = analyzer.get_errors()

    assert len(errors_df) == 3
    sev_fn = analyzer.get_severe_false_negatives()
    assert len(sev_fn) == 1
    assert sev_fn.iloc[0]["sample_index"] == 2

    sev_fp = analyzer.get_severe_false_positives()
    assert len(sev_fp) == 1
    assert sev_fp.iloc[0]["sample_index"] == 3

    # Export report
    report_file = analyzer.export_error_report(tmp_path)
    assert report_file.exists()
    assert (tmp_path / "errors.csv").exists()
