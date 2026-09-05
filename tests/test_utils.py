"""
Unit Tests for Utility Modules
Tests reproducibility, logging, and configuration loading.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.utils.reproducibility import set_seed
from src.utils.logging import setup_logger
from src.utils.config import load_config


def test_set_seed_reproducibility():
    """Verify that set_seed ensures reproducible numpy random generation."""
    set_seed(42)
    val1 = np.random.rand(5)
    
    set_seed(42)
    val2 = np.random.rand(5)
    
    np.testing.assert_allclose(val1, val2)


def test_setup_logger():
    """Verify logger creation and handler assignment."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1


def test_load_config_valid():
    """Verify loading of an existing YAML configuration."""
    config_path = Path("configs/baseline.yaml")
    assert config_path.exists(), "configs/baseline.yaml must exist"
    
    cfg = load_config(config_path)
    assert isinstance(cfg, dict)
    assert cfg.get("experiment_name") == "baseline_tfidf_logreg"
    assert "model" in cfg
    assert "data" in cfg


def test_load_config_missing():
    """Verify FileNotFoundError on nonexistent configuration path."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_path_123.yaml")
