"""
Unit Tests for Tokenization and Dataset Pipeline
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.tokenization.tokenizer import DistressDataset, SubtextTokenizer, create_data_loaders


@pytest.fixture
def dummy_tokenizer():
    # Use roberta-base or a lightweight bert tokenizer for testing
    return AutoTokenizer.from_pretrained("roberta-base")


def test_distress_dataset_tensor_shapes(dummy_tokenizer):
    """Verify DistressDataset produces correct tensor dimensions and types."""
    texts = ["I feel very overwhelmed today", "Just watching funny videos online lol"]
    labels = [1, 0]
    max_len = 64

    dataset = DistressDataset(texts, labels, dummy_tokenizer, max_length=max_len)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert "label" in item0

    assert item0["input_ids"].shape == (max_len,)
    assert item0["attention_mask"].shape == (max_len,)
    assert item0["input_ids"].dtype == torch.long
    assert item0["attention_mask"].dtype == torch.long
    assert item0["label"].item() == 1


def test_subtext_tokenizer_batch_encode():
    """Verify SubtextTokenizer encodes single strings and batches cleanly."""
    sub_tok = SubtextTokenizer("roberta-base", max_length=32)
    assert sub_tok.vocab_size > 1000

    encoding = sub_tok.encode(["Hello world", "Testing tokenization pipeline"])
    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert encoding["input_ids"].shape == (2, 32)
    assert encoding["attention_mask"].shape == (2, 32)


def test_create_data_loaders(dummy_tokenizer):
    """Verify DataLoader generation across train/val/test partitions."""
    train_texts = ["train post 1", "train post 2", "train post 3", "train post 4"]
    train_labels = [0, 1, 2, 3]
    val_texts = ["val post 1", "val post 2"]
    val_labels = [0, 1]
    test_texts = ["test post 1", "test post 2"]
    test_labels = [2, 3]

    train_l, val_l, test_l = create_data_loaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        dummy_tokenizer,
        batch_size=2,
        eval_batch_size=2,
        max_length=16,
    )

    assert len(train_l) == 2
    assert len(val_l) == 1
    assert len(test_l) == 1

    batch = next(iter(train_l))
    assert batch["input_ids"].shape == (2, 16)
    assert batch["label"].shape == (2,)
