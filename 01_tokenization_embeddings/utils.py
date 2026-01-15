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


# =============================================================================
# WordPiece Algorithm Utilities
# =============================================================================

WORDPIECE_PREFIX = "##"


def split_word_for_wordpiece(word: str, prefix: str = WORDPIECE_PREFIX) -> tuple[str, ...]:
    """
    Split a word into WordPiece format (first char + ##prefixed rest).
    
    Args:
        word: Input word
        prefix: Continuation prefix (default: ##)
    
    Returns:
        Tuple of tokens in WordPiece format
    """
    word = word.strip()
    if not word:
        return ()
    if len(word) == 1:
        return (word,)
    return (word[0],) + tuple(prefix + c for c in word[1:])


def get_wordpiece_word_splits(text: str, prefix: str = WORDPIECE_PREFIX) -> dict[tuple[str, ...], int]:
    """
    Split text into words in WordPiece format with frequencies.
    
    Args:
        text: Input text corpus
        prefix: Continuation prefix (default: ##)
    
    Returns:
        Dictionary mapping WordPiece token tuples to frequencies
    """
    words = split_into_words(text)
    word_splits: Counter[tuple[str, ...]] = Counter()
    
    for word in words:
        split = split_word_for_wordpiece(word, prefix)
        if split:
            word_splits[split] += 1
    
    return dict(word_splits)


def get_token_frequencies(word_splits: dict[tuple[str, ...], int]) -> Counter:
    """
    Count frequency of each individual token in word splits.
    
    Args:
        word_splits: Dictionary of word tuples to frequencies
    
    Returns:
        Counter of token frequencies
    """
    token_freqs: Counter[str] = Counter()
    for word, freq in word_splits.items():
        for token in word:
            token_freqs[token] += freq
    return token_freqs


def compute_wordpiece_pair_scores(
    word_splits: dict[tuple[str, ...], int],
    token_freqs: Counter
) -> dict[tuple[str, str], float]:
    """
    Compute WordPiece likelihood scores for adjacent pairs.
    
    Score = freq(pair) / (freq(token1) * freq(token2))
    
    Args:
        word_splits: Dictionary of word tuples to frequencies
        token_freqs: Counter of individual token frequencies
    
    Returns:
        Dictionary of pair -> score
    """
    pair_freqs: Counter[tuple[str, str]] = Counter()
    
    for word, freq in word_splits.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i + 1])] += freq
    
    pair_scores = {}
    for pair, freq in pair_freqs.items():
        denom = token_freqs[pair[0]] * token_freqs[pair[1]]
        if denom > 0:
            pair_scores[pair] = freq / denom
    
    return pair_scores


def merge_wordpiece_pair(
    word_splits: dict[tuple[str, ...], int],
    pair: tuple[str, str],
    prefix: str = WORDPIECE_PREFIX
) -> dict[tuple[str, ...], int]:
    """
    Merge all occurrences of a pair in WordPiece word splits.
    
    Args:
        word_splits: Dictionary of word tuples to frequencies
        pair: The (token1, token2) pair to merge
        prefix: Continuation prefix to remove when merging
    
    Returns:
        New word splits dictionary with merged tokens
    """
    new_word_splits = {}
    # Remove prefix from second token when merging
    merged = pair[0] + pair[1].replace(prefix, "")
    
    for word, freq in word_splits.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_splits[tuple(new_word)] = freq
    
    return new_word_splits


def wordpiece_tokenize_word(
    word: str,
    vocab: dict[str, int],
    prefix: str = WORDPIECE_PREFIX,
    unk_token: str = "<UNK>"
) -> list[str]:
    """
    Tokenize a word using greedy longest-match-first (WordPiece inference).
    
    Args:
        word: Input word to tokenize
        vocab: Vocabulary dictionary
        prefix: Continuation prefix
        unk_token: Unknown token string
    
    Returns:
        List of WordPiece tokens
    """
    if not word:
        return []
    
    tokens = []
    start = 0
    
    while start < len(word):
        end = len(word)
        found = False
        
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = prefix + substr
            
            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            end -= 1
        
        if not found:
            tokens.append(unk_token)
            start += 1
        else:
            start = end
    
    return tokens


def decode_wordpiece_tokens(
    tokens: list[str],
    prefix: str = WORDPIECE_PREFIX
) -> str:
    """
    Reconstruct text from WordPiece tokens.
    
    Args:
        tokens: List of WordPiece tokens
        prefix: Continuation prefix to remove
    
    Returns:
        Reconstructed text string
    """
    text_parts = []
    for token in tokens:
        if token.startswith(prefix):
            text_parts.append(token[len(prefix):])
        else:
            if text_parts:
                text_parts.append(" ")
            text_parts.append(token)
    return "".join(text_parts)


# =============================================================================
# Shared Token Embedding Layer
# =============================================================================

class TokenEmbeddingLayer(torch.nn.Module):
    """
    Reusable token embedding layer for any tokenizer.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Dimension of embeddings
        pad_token_id: ID of padding token (zeroed out)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


# =============================================================================
# Unigram Language Model Utilities
# =============================================================================

import math

SENTENCEPIECE_SPACE = "▁"  # Unicode U+2581


def get_all_substrings(word: str, max_len: int = 16) -> list[str]:
    """
    Get all substrings of a word up to max_len.
    
    Args:
        word: Input word
        max_len: Maximum substring length
    
    Returns:
        List of all substrings
    """
    substrings = []
    for i in range(len(word)):
        for j in range(i + 1, min(i + max_len + 1, len(word) + 1)):
            substrings.append(word[i:j])
    return substrings


def build_initial_unigram_vocab(
    text: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None
) -> dict[str, float]:
    """
    Build initial large vocabulary with substring frequencies.
    
    Args:
        text: Training corpus
        vocab_size: Target initial vocab size (will be pruned later)
        special_tokens: Special tokens to include
    
    Returns:
        Dictionary mapping tokens to log probabilities
    """
    # Count all character and substring frequencies
    substr_counts: Counter[str] = Counter()
    words = split_into_words(text)
    
    for word in words:
        word = word.strip()
        if word:
            # Add all substrings
            for substr in get_all_substrings(word):
                substr_counts[substr] += 1
    
    # Keep top vocab_size substrings by frequency
    total_count = sum(substr_counts.values())
    vocab_probs = {}
    
    # Add special tokens with small probability
    if special_tokens:
        for token in special_tokens:
            vocab_probs[token] = math.log(1e-10)
    
    # Add most frequent substrings
    for token, count in substr_counts.most_common(vocab_size):
        if token not in vocab_probs:
            vocab_probs[token] = math.log(count / total_count)
    
    return vocab_probs


def viterbi_tokenize(
    word: str,
    vocab_probs: dict[str, float],
    unk_token: str = "<UNK>"
) -> list[str]:
    """
    Tokenize word using Viterbi algorithm to find most likely segmentation.
    
    Args:
        word: Input word to tokenize
        vocab_probs: Dictionary of token -> log probability
        unk_token: Unknown token string
    
    Returns:
        List of tokens (most likely segmentation)
    """
    if not word:
        return []
    
    n = len(word)
    
    # best_score[i] = best log probability to reach position i
    best_score = [-float('inf')] * (n + 1)
    best_score[0] = 0.0
    
    # best_edge[i] = (start_position, token) for best path to position i
    best_edge: list[tuple[int, str] | None] = [None] * (n + 1)
    
    for end in range(1, n + 1):
        for start in range(max(0, end - 16), end):  # Max token length = 16
            substr = word[start:end]
            if substr in vocab_probs:
                score = best_score[start] + vocab_probs[substr]
                if score > best_score[end]:
                    best_score[end] = score
                    best_edge[end] = (start, substr)
    
    # Backtrack to find best segmentation
    if best_edge[n] is None:
        # Fallback: character-level tokenization
        tokens = []
        for char in word:
            if char in vocab_probs:
                tokens.append(char)
            else:
                tokens.append(unk_token)
        return tokens
    
    tokens = []
    pos = n
    while pos > 0:
        edge = best_edge[pos]
        if edge is None:
            break
        start, token = edge
        tokens.append(token)
        pos = start
    
    return list(reversed(tokens))


def compute_unigram_loss(
    word_freqs: dict[str, int],
    vocab_probs: dict[str, float]
) -> float:
    """
    Compute total negative log likelihood of corpus under current vocab.
    
    Args:
        word_freqs: Word frequency dictionary
        vocab_probs: Token log probabilities
    
    Returns:
        Total loss (negative log likelihood)
    """
    total_loss = 0.0
    
    for word, freq in word_freqs.items():
        tokens = viterbi_tokenize(word, vocab_probs)
        word_loss = 0.0
        for token in tokens:
            if token in vocab_probs:
                word_loss -= vocab_probs[token]
            else:
                word_loss -= math.log(1e-10)  # UNK penalty
        total_loss += word_loss * freq
    
    return total_loss


def compute_token_importance(
    word_freqs: dict[str, int],
    vocab_probs: dict[str, float],
    token: str
) -> float:
    """
    Compute how much loss increases if token is removed.
    
    Args:
        word_freqs: Word frequency dictionary
        vocab_probs: Token log probabilities
        token: Token to evaluate
    
    Returns:
        Increase in loss if token is removed
    """
    if token not in vocab_probs:
        return 0.0
    
    # Current loss
    current_loss = compute_unigram_loss(word_freqs, vocab_probs)
    
    # Loss without this token
    vocab_without = {k: v for k, v in vocab_probs.items() if k != token}
    loss_without = compute_unigram_loss(word_freqs, vocab_without)
    
    return loss_without - current_loss


def prune_unigram_vocab(
    word_freqs: dict[str, int],
    vocab_probs: dict[str, float],
    target_size: int,
    special_tokens: set[str],
    prune_ratio: float = 0.1
) -> dict[str, float]:
    """
    Prune vocabulary by removing least important tokens.
    
    Args:
        word_freqs: Word frequency dictionary
        vocab_probs: Current token log probabilities
        target_size: Target vocabulary size
        special_tokens: Tokens that should not be pruned
        prune_ratio: Fraction to prune each iteration
    
    Returns:
        Pruned vocabulary
    """
    vocab = dict(vocab_probs)
    
    while len(vocab) > target_size:
        # Compute importance for each token
        importance = {}
        for token in vocab:
            if token not in special_tokens and len(token) > 1:
                importance[token] = compute_token_importance(word_freqs, vocab, token)
        
        if not importance:
            break
        
        # Remove tokens with lowest importance
        num_to_remove = max(1, int(len(importance) * prune_ratio))
        num_to_remove = min(num_to_remove, len(vocab) - target_size)
        
        sorted_tokens = sorted(importance.items(), key=lambda x: x[1])
        for token, _ in sorted_tokens[:num_to_remove]:
            del vocab[token]
    
    return vocab


# =============================================================================
# SentencePiece Utilities (treats space as regular character)
# =============================================================================

def normalize_for_sentencepiece(text: str, add_prefix_space: bool = True) -> str:
    """
    Normalize text for SentencePiece: replace spaces with special marker.
    
    Args:
        text: Input text
        add_prefix_space: Whether to add space marker at start of words
    
    Returns:
        Normalized text with space markers
    """
    # Replace spaces with the special space character
    # Add space marker at word boundaries
    normalized = text.replace(" ", SENTENCEPIECE_SPACE)
    if add_prefix_space and not normalized.startswith(SENTENCEPIECE_SPACE):
        normalized = SENTENCEPIECE_SPACE + normalized
    return normalized


def denormalize_from_sentencepiece(text: str) -> str:
    """
    Convert SentencePiece output back to normal text.
    
    Args:
        text: Text with space markers
    
    Returns:
        Normal text with spaces
    """
    return text.replace(SENTENCEPIECE_SPACE, " ").strip()


def get_sentencepiece_word_freqs(text: str) -> dict[str, int]:
    """
    Get word frequencies with SentencePiece normalization.
    
    Args:
        text: Input corpus
    
    Returns:
        Dictionary of normalized words to frequencies
    """
    word_freqs: Counter[str] = Counter()
    
    # Split on whitespace but treat each word as having leading space marker
    for word in text.split():
        if word:
            normalized = SENTENCEPIECE_SPACE + word
            word_freqs[normalized] += 1
    
    return dict(word_freqs)


# =============================================================================
# Word2Vec Utilities (CBOW & Skip-gram)
# =============================================================================

def create_context_target_pairs(
    word_ids: list[int],
    window_size: int = 2,
    mode: str = "cbow"
) -> list[tuple]:
    """
    Create context-target pairs for Word2Vec training.
    
    Args:
        word_ids: List of word indices
        window_size: Context window size on each side
        mode: "cbow" for (context, target) or "skipgram" for (target, context)
    
    Returns:
        List of training pairs
    """
    pairs = []
    
    for i in range(window_size, len(word_ids) - window_size):
        target = word_ids[i]
        context = [
            word_ids[j]
            for j in range(i - window_size, i + window_size + 1)
            if j != i
        ]
        
        if mode == "cbow":
            pairs.append((context, target))
        else:  # skipgram
            for ctx in context:
                pairs.append((target, ctx))
    
    return pairs


def compute_embedding_similarity(
    embed1: torch.Tensor,
    embed2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    
    Args:
        embed1: First embedding vector
        embed2: Second embedding vector
    
    Returns:
        Cosine similarity score
    """
    import torch.nn.functional as F
    return F.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0)).item()


def find_nearest_embeddings(
    query_embed: torch.Tensor,
    all_embeddings: torch.Tensor,
    top_k: int = 5,
    exclude_indices: set[int] | None = None
) -> list[tuple[int, float]]:
    """
    Find nearest embeddings to a query vector.
    
    Args:
        query_embed: Query embedding vector
        all_embeddings: Matrix of all embeddings (vocab_size, embed_dim)
        top_k: Number of nearest neighbors to return
        exclude_indices: Indices to exclude from results
    
    Returns:
        List of (index, similarity) tuples
    """
    import torch.nn.functional as F
    
    similarities = F.cosine_similarity(
        query_embed.unsqueeze(0),
        all_embeddings,
        dim=1
    )
    
    exclude_indices = exclude_indices or set()
    
    sorted_indices = similarities.argsort(descending=True)
    
    results = []
    for idx in sorted_indices:
        if idx.item() not in exclude_indices:
            results.append((idx.item(), similarities[idx].item()))
            if len(results) >= top_k:
                break
    
    return results
