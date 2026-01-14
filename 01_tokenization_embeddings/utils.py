"""
Utility functions for tokenization and embeddings.

This module contains reusable helper functions that can be shared
across different tokenizer implementations.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


# =============================================================================
# Special Tokens Configuration
# =============================================================================

DEFAULT_SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}


# =============================================================================
# Text Processing Utilities
# =============================================================================

def split_into_words(text: str, pattern: str = r'\S+|\s+') -> list[str]:
    """
    Split text into words/whitespace chunks.
    
    Args:
        text: Input text to split
        pattern: Regex pattern for splitting (default keeps whitespace separate)
    
    Returns:
        List of word/whitespace chunks
    """
    return re.findall(pattern, text)


def get_word_frequencies(text: str) -> dict[tuple[str, ...], int]:
    """
    Convert text to word frequency dictionary.
    
    Each word is represented as a tuple of characters (for BPE processing).
    
    Args:
        text: Input text corpus
    
    Returns:
        Dictionary mapping character tuples to their frequencies
    """
    words = split_into_words(text)
    word_freqs: Counter[tuple[str, ...]] = Counter()
    for word in words:
        word_freqs[tuple(word)] += 1
    return dict(word_freqs)


def extract_unique_chars(word_freqs: dict[tuple[str, ...], int]) -> set[str]:
    """
    Extract all unique characters from word frequency dictionary.
    
    Args:
        word_freqs: Dictionary of word tuples to frequencies
    
    Returns:
        Set of unique characters
    """
    chars = set()
    for word in word_freqs.keys():
        chars.update(word)
    return chars


# =============================================================================
# BPE Algorithm Utilities
# =============================================================================

def count_pair_frequencies(word_freqs: dict[tuple[str, ...], int]) -> Counter:
    """
    Count frequency of adjacent token pairs across all words.
    
    Args:
        word_freqs: Dictionary of word tuples to frequencies
    
    Returns:
        Counter of (token1, token2) pairs to their frequencies
    """
    pair_freqs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i + 1])] += freq
    return pair_freqs


def merge_pair_in_words(
    word_freqs: dict[tuple[str, ...], int],
    pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    """
    Merge all occurrences of a token pair in the vocabulary.
    
    Args:
        word_freqs: Dictionary of word tuples to frequencies
        pair: The (token1, token2) pair to merge
    
    Returns:
        New word frequency dictionary with merged tokens
    """
    new_word_freqs = {}
    merged = pair[0] + pair[1]
    
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    
    return new_word_freqs


def apply_merges_to_word(word: str, merges: dict[tuple[str, str], str]) -> list[str]:
    """
    Apply learned BPE merges to tokenize a single word.
    
    Args:
        word: Input word to tokenize
        merges: Dictionary of (token1, token2) -> merged_token
    
    Returns:
        List of subword tokens
    """
    if not word:
        return []
    
    tokens = list(word)
    for (p1, p2), merged in merges.items():
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == p1 and tokens[i + 1] == p2:
                tokens = tokens[:i] + [merged] + tokens[i + 2:]
            else:
                i += 1
    return tokens


# =============================================================================
# Vocabulary Management
# =============================================================================

def build_initial_vocab(
    chars: set[str],
    special_tokens: dict[str, int] | None = None
) -> dict[str, int]:
    """
    Build initial vocabulary from characters and special tokens.
    
    Args:
        chars: Set of unique characters
        special_tokens: Optional special tokens to include first
    
    Returns:
        Vocabulary dictionary mapping tokens to IDs
    """
    vocab = dict(special_tokens) if special_tokens else {}
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """
    Create inverse vocabulary (ID -> token).
    
    Args:
        vocab: Vocabulary dictionary (token -> ID)
    
    Returns:
        Inverse vocabulary dictionary (ID -> token)
    """
    return {v: k for k, v in vocab.items()}


# =============================================================================
# Serialization Utilities
# =============================================================================

def save_tokenizer_data(
    path: str,
    vocab_size: int,
    vocab: dict[str, int],
    merges: dict[tuple[str, str], str]
) -> None:
    """
    Save tokenizer data to JSON file.
    
    Args:
        path: File path to save to
        vocab_size: Target vocabulary size
        vocab: Vocabulary dictionary
        merges: BPE merges dictionary
    """
    data = {
        "vocab_size": vocab_size,
        "vocab": vocab,
        "merges": {f"{k[0]}|||{k[1]}": v for k, v in merges.items()},
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_tokenizer_data(path: str) -> dict[str, Any]:
    """
    Load tokenizer data from JSON file.
    
    Args:
        path: File path to load from
    
    Returns:
        Dictionary with vocab_size, vocab, and merges
    """
    data = json.loads(Path(path).read_text())
    return {
        "vocab_size": data["vocab_size"],
        "vocab": data["vocab"],
        "merges": {tuple(k.split("|||")): v for k, v in data["merges"].items()},
    }


# =============================================================================
# Padding & Batching Utilities
# =============================================================================

def pad_sequence(
    token_ids: list[int],
    max_length: int,
    pad_token_id: int = 0,
    truncate: bool = True
) -> list[int]:
    """
    Pad or truncate a sequence to a fixed length.
    
    Args:
        token_ids: List of token IDs
        max_length: Target length
        pad_token_id: ID to use for padding
        truncate: Whether to truncate if longer than max_length
    
    Returns:
        Padded/truncated list of token IDs
    """
    if truncate and len(token_ids) > max_length:
        return token_ids[:max_length]
    
    pad_len = max_length - len(token_ids)
    return token_ids + [pad_token_id] * pad_len


def create_attention_mask(
    token_ids: list[int],
    pad_token_id: int = 0
) -> list[int]:
    """
    Create attention mask (1 for real tokens, 0 for padding).
    
    Args:
        token_ids: List of token IDs
        pad_token_id: ID used for padding
    
    Returns:
        List of 0s and 1s indicating valid positions
    """
    return [0 if tid == pad_token_id else 1 for tid in token_ids]


def batch_encode_with_padding(
    encoded_sequences: list[list[int]],
    max_length: int | None = None,
    pad_token_id: int = 0
) -> dict[str, torch.Tensor]:
    """
    Batch encode sequences with padding and attention masks.
    
    Args:
        encoded_sequences: List of encoded token ID lists
        max_length: Optional max length (defaults to longest sequence)
        pad_token_id: ID to use for padding
    
    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors
    """
    batch_max_len = max(len(seq) for seq in encoded_sequences)
    if max_length:
        batch_max_len = min(batch_max_len, max_length)
    
    input_ids = []
    attention_masks = []
    
    for ids in encoded_sequences:
        ids = ids[:batch_max_len]  # Truncate
        mask = [1] * len(ids)
        
        pad_len = batch_max_len - len(ids)
        ids = ids + [pad_token_id] * pad_len
        mask = mask + [0] * pad_len
        
        input_ids.append(ids)
        attention_masks.append(mask)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================

def to_tensor(token_ids: list[int], dtype: torch.dtype = torch.long) -> torch.Tensor:
    """
    Convert token ID list to PyTorch tensor.
    
    Args:
        token_ids: List of token IDs
        dtype: Tensor data type
    
    Returns:
        PyTorch tensor of token IDs
    """
    return torch.tensor(token_ids, dtype=dtype)


def from_tensor(tensor: torch.Tensor) -> list[int]:
    """
    Convert PyTorch tensor to list of token IDs.
    
    Args:
        tensor: PyTorch tensor of token IDs
    
    Returns:
        List of token IDs
    """
    return tensor.tolist()
