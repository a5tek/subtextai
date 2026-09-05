"""
Unit Tests for Model Architectures (Classical Baseline)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.models.baseline import ClassicalBaseline


@pytest.fixture
def dummy_train_data():
    texts = [
        "this test is killing me haha so funny",
        "hilarious video of cute cats dying of laughter",
        "traffic makes me angry today",
        "overwhelmed with work deadlines and projects",
        "super stressed about exams next monday morning",
        "studying all night for finals feeling exhausted",
        "feeling empty and numb inside for months",
        "constant sadness and crying every single day",
        "isolated and disconnected from my friends",
        "ending my life tonight goodbye everyone",
        "cannot take the pain anymore this is my final farewell",
        "goodbye world I have the means ready tonight",
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    return texts, labels


def test_classical_baseline_fit_and_predict(dummy_train_data):
    """Verify classical baseline fits and produces valid predictions."""
    texts, labels = dummy_train_data
    model = ClassicalBaseline(min_df=1)

    assert not model.is_fitted
    model.fit(texts, labels)
    assert model.is_fitted

    preds = model.predict(texts)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(texts)
    assert set(preds).issubset({0, 1, 2, 3})


def test_classical_baseline_predict_proba_shape_and_sums(dummy_train_data):
    """Verify predict_proba outputs (N, num_classes) with rows summing to 1."""
    texts, labels = dummy_train_data
    model = ClassicalBaseline(min_df=1).fit(texts, labels)

    probs = model.predict_proba(texts)
    assert probs.shape == (len(texts), 4)

    # Check each row sums to 1.0 within numerical tolerance
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(texts)), atol=1e-5)


def test_classical_baseline_unfitted_raises_error():
    """Verify calling predict on unfitted model raises RuntimeError."""
    model = ClassicalBaseline()
    with pytest.raises(RuntimeError):
        model.predict(["some random text"])
    with pytest.raises(RuntimeError):
        model.predict_proba(["some random text"])


def test_classical_baseline_save_and_load(dummy_train_data):
    """Verify serialization and deserialization produces identical predictions."""
    texts, labels = dummy_train_data
    model = ClassicalBaseline(min_df=1).fit(texts, labels)
    orig_preds = model.predict(texts)
    orig_probs = model.predict_proba(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "baseline.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = ClassicalBaseline.load(model_path)
        assert loaded_model.is_fitted
        loaded_preds = loaded_model.predict(texts)
        loaded_probs = loaded_model.predict_proba(texts)

        np.testing.assert_array_equal(orig_preds, loaded_preds)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-6)
