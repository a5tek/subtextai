"""
Tokenization and PyTorch Dataset Module

Wraps Hugging Face Tokenizers for RoBERTa and domain-adapted transformers.
Provides PyTorch Dataset and DataLoader pipelines with deterministic padding and masking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class DistressDataset(Dataset):
    """PyTorch Dataset yielding tokenized sequences, attention masks, and severity labels."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        """
        Args:
            texts: List of sanitized text posts.
            labels: List of integer labels {0, 1, 2, 3}.
            tokenizer: Hugging Face PreTrainedTokenizer instance.
            max_length: Maximum sequence length.
        """
        self.texts = [str(t) for t in texts]
        self.labels = [int(lbl) for lbl in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # Shape: (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0), # Shape: (max_length,)
            "label": torch.tensor(label, dtype=torch.long),
        }


class SubtextTokenizer:
    """Wrapper managing pretrained tokenizer initialization, special tokens, and batch encodings."""

    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        max_length: int = 256,
    ):
        self.model_name = pretrained_model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a single text or batch of texts into tensor inputs.
        
        Args:
            texts: String or list of strings.
            padding: Padding strategy ('max_length' or 'longest').
            truncation: Whether to truncate to self.max_length.
            
        Returns:
            Dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    max_length: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    
    Args:
        train_texts, train_labels: Training samples.
        val_texts, val_labels: Validation samples.
        test_texts, test_labels: Held-out test samples.
        tokenizer: Hugging Face tokenizer instance.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        max_length: Sequence token length.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = DistressDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = DistressDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = DistressDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
