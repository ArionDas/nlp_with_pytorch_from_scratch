"""
SentencePiece Tokenizer with PyTorch Integration

SentencePiece (used by T5, LLaMA, Mistral) - language-agnostic subword tokenizer.

Key differences from other tokenizers:
- Treats text as raw character stream (no pre-tokenization)
- Uses ▁ (U+2581) to mark word boundaries instead of spaces
- Can use either BPE or Unigram algorithm internally
- Works directly on unicode, language-agnostic

This implementation supports both BPE and Unigram modes.

Usage:
    python sentencepiece_tokenizer_pytorch.py                           # Default (BPE mode)
    python sentencepiece_tokenizer_pytorch.py --mode unigram            # Unigram mode
    python sentencepiece_tokenizer_pytorch.py --text "Your text here"   # Custom text
"""

import argparse
import math
from collections import Counter

import torch

from utils import (
    # Constants
    DEFAULT_SPECIAL_TOKENS,
    SENTENCEPIECE_SPACE,
    # Text processing
    split_into_words,
    # BPE algorithm
    count_pair_frequencies,
    merge_pair_in_words,
    apply_merges_to_word,
    # Unigram algorithm
    build_initial_unigram_vocab,
    viterbi_tokenize,
    prune_unigram_vocab,
    compute_unigram_loss,
    # SentencePiece utilities
    normalize_for_sentencepiece,
    denormalize_from_sentencepiece,
    get_sentencepiece_word_freqs,
    # Vocabulary
    build_initial_vocab,
    invert_vocab,
    extract_unique_chars,
    # Serialization & batching
    batch_encode_with_padding,
    to_tensor,
    from_tensor,
    TokenEmbeddingLayer,
)


class SentencePieceTokenizerPyTorch:
    """
    SentencePiece tokenizer with PyTorch tensor support.
    
    Key characteristics:
    - Language-agnostic: works on raw unicode
    - Uses ▁ to mark word boundaries (beginning of word)
    - Supports both BPE and Unigram algorithms
    - No pre-tokenization needed
    """
    
    def __init__(self, vocab_size: int = 1000, mode: str = "bpe"):
        """
        Args:
            vocab_size: Target vocabulary size
            mode: Algorithm to use - "bpe" or "unigram"
        """
        self.vocab_size = vocab_size
        self.mode = mode.lower()
        
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.merges: dict[tuple[str, str], str] = {}  # For BPE mode
        self.vocab_probs: dict[str, float] = {}  # For Unigram mode
        
        self.special_tokens = DEFAULT_SPECIAL_TOKENS.copy()
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.unk_token_id = self.special_tokens["<UNK>"]
        self.bos_token_id = self.special_tokens["<BOS>"]
        self.eos_token_id = self.special_tokens["<EOS>"]
    
    def _get_sentencepiece_word_freqs(self, text: str) -> dict[tuple[str, ...], int]:
        """Convert text to SentencePiece format with word frequencies."""
        word_freqs: Counter[tuple[str, ...]] = Counter()
        
        for word in text.split():
            if word:
                # Add space marker at beginning of word
                normalized = SENTENCEPIECE_SPACE + word
                word_freqs[tuple(normalized)] += 1
        
        return dict(word_freqs)
    
    def _train_bpe(self, text: str, verbose: bool = True):
        """Train using BPE algorithm with SentencePiece normalization."""
        word_freqs = self._get_sentencepiece_word_freqs(text)
        
        # Build initial vocab from characters
        chars = set()
        for word in word_freqs.keys():
            chars.update(word)
        
        self.vocab = build_initial_vocab(chars, self.special_tokens)
        
        num_merges = self.vocab_size - len(self.vocab)
        
        if verbose:
            print(f"Training SentencePiece (BPE): {len(self.vocab)} -> {self.vocab_size} tokens")
        
        for i in range(num_merges):
            pair_freqs = count_pair_frequencies(word_freqs)
            if not pair_freqs:
                break
            
            best_pair = pair_freqs.most_common(1)[0][0]
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)
            word_freqs = merge_pair_in_words(word_freqs, best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{num_merges}")
        
        self.inverse_vocab = invert_vocab(self.vocab)
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
    
    def _train_unigram(self, text: str, verbose: bool = True):
        """Train using Unigram algorithm with SentencePiece normalization."""
        # Normalize text for SentencePiece
        normalized_text = normalize_for_sentencepiece(text)
        
        # Get word frequencies
        word_freqs = get_sentencepiece_word_freqs(text)
        
        # Build initial large vocabulary
        initial_size = self.vocab_size * 10
        self.vocab_probs = build_initial_unigram_vocab(
            normalized_text, initial_size, self.special_tokens
        )
        
        if verbose:
            print(f"Training SentencePiece (Unigram): {len(self.vocab_probs)} -> {self.vocab_size} tokens")
        
        # Prune to target size
        if len(self.vocab_probs) > self.vocab_size:
            self.vocab_probs = prune_unigram_vocab(
                word_freqs,
                self.vocab_probs,
                self.vocab_size,
                set(self.special_tokens.keys()),
                prune_ratio=0.2
            )
        
        # Build vocab mapping
        self.vocab = dict(self.special_tokens)
        for token in sorted(self.vocab_probs.keys()):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.inverse_vocab = invert_vocab(self.vocab)
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
    
    def train(self, text: str, verbose: bool = True):
        """
        Train tokenizer on corpus.
        
        Args:
            text: Training corpus
            verbose: Whether to print progress
        """
        if self.mode == "bpe":
            self._train_bpe(text, verbose)
        elif self.mode == "unigram":
            self._train_unigram(text, verbose)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'bpe' or 'unigram'.")
    
    def _tokenize_bpe(self, word: str) -> list[str]:
        """Tokenize using BPE merges."""
        if not word:
            return []
        return apply_merges_to_word(word, self.merges)
    
    def _tokenize_unigram(self, word: str) -> list[str]:
        """Tokenize using Viterbi algorithm."""
        if not word:
            return []
        return viterbi_tokenize(word, self.vocab_probs, "<UNK>")
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into SentencePiece tokens.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens with ▁ marking word boundaries
        """
        tokens = []
        
        for word in text.split():
            if word:
                # Add space marker at beginning
                normalized = SENTENCEPIECE_SPACE + word
                
                if self.mode == "bpe":
                    word_tokens = self._tokenize_bpe(normalized)
                else:
                    word_tokens = self._tokenize_unigram(normalized)
                
                tokens.extend(word_tokens)
        
        return tokens
    
    def encode(
        self,
        text: str,
        max_length: int | None = None,
        padding: bool = False,
        return_tensors: bool = True,
        add_special_tokens: bool = False
    ) -> torch.Tensor | list[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        if padding and max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        if return_tensors:
            return to_tensor(token_ids)
        return token_ids
    
    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
        add_special_tokens: bool = False
    ) -> dict[str, torch.Tensor]:
        """Encode batch of texts with padding and attention mask."""
        encoded = [
            self.encode(t, max_length, False, False, add_special_tokens)
            for t in texts
        ]
        return batch_encode_with_padding(encoded, max_length, self.pad_token_id)
    
    def decode(self, token_ids: torch.Tensor | list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = from_tensor(token_ids)
        
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, "<UNK>")
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # Join and denormalize (replace ▁ with space)
        text = "".join(tokens)
        return denormalize_from_sentencepiece(text)
    
    def decode_batch(self, batch_ids: torch.Tensor, skip_special: bool = True) -> list[str]:
        """Decode batch of token IDs."""
        return [self.decode(ids, skip_special) for ids in batch_ids]
    
    def save(self, path: str):
        """Save tokenizer to JSON file."""
        import json
        from pathlib import Path
        
        data = {
            "vocab_size": self.vocab_size,
            "mode": self.mode,
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "vocab_probs": self.vocab_probs,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def load(self, path: str):
        """Load tokenizer from JSON file."""
        import json
        from pathlib import Path
        
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.vocab_size = data["vocab_size"]
        self.mode = data.get("mode", "bpe")
        self.vocab = data["vocab"]
        self.merges = {tuple(k.split("|||")): v for k, v in data.get("merges", {}).items()}
        self.vocab_probs = data.get("vocab_probs", {})
        self.inverse_vocab = invert_vocab(self.vocab)
    
    def get_embedding_layer(self, embed_dim: int = 128) -> TokenEmbeddingLayer:
        """Create an embedding layer for this tokenizer."""
        return TokenEmbeddingLayer(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            pad_token_id=self.pad_token_id
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SentencePiece Tokenizer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Input text to tokenize")
    parser.add_argument("--corpus", "-c", type=str, default=None,
                        help="Path to corpus file for training")
    parser.add_argument("--vocab-size", "-v", type=int, default=256,
                        help="Vocabulary size (default: 256)")
    parser.add_argument("--mode", "-m", type=str, default="bpe",
                        choices=["bpe", "unigram"],
                        help="Algorithm mode: bpe or unigram (default: bpe)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load pre-trained tokenizer")
    return parser.parse_args()


def get_default_corpus() -> str:
    return """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Natural language processing enables computers to understand human language.
    Deep learning models learn representations from data.
    Transformers have revolutionized natural language processing.
    Attention is all you need for sequence modeling.
    SentencePiece treats text as raw character sequences.
    The special character marks word boundaries.
    """ * 20


def safe_print(text: str):
    """Print with fallback for Unicode characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters for Windows console
        safe_text = text.replace(SENTENCEPIECE_SPACE, "_")
        print(safe_text)


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"SentencePiece Tokenizer with PyTorch (mode: {args.mode})")
    print("=" * 60)
    
    tokenizer = SentencePieceTokenizerPyTorch(vocab_size=args.vocab_size, mode=args.mode)
    
    if args.load:
        print(f"Loading tokenizer from: {args.load}")
        tokenizer.load(args.load)
        print(f"Loaded vocab size: {len(tokenizer.vocab)}, mode: {tokenizer.mode}")
    else:
        if args.corpus:
            print(f"Loading corpus from: {args.corpus}")
            with open(args.corpus, 'r', encoding='utf-8') as f:
                corpus = f.read()
        else:
            corpus = get_default_corpus()
        
        tokenizer.train(corpus)
    
    if args.text:
        print("\n" + "=" * 60)
        print("Tokenizing Input Text")
        print("=" * 60)
        
        text = args.text
        tokens = tokenizer.tokenize(text)
        tensor = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(tensor)
        
        print(f"Text:      '{text}'")
        safe_print(f"Tokens:    {tokens}")
        print(f"Token IDs: {tensor.tolist()}")
        print(f"Decoded:   '{decoded}'")
    else:
        print("\n" + "=" * 60)
        print("Demo: Single Text Encoding")
        print("=" * 60)
        
        text = "Machine learning is amazing!"
        tokens = tokenizer.tokenize(text)
        tensor = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(tensor)
        
        print(f"Text:      '{text}'")
        safe_print(f"Tokens:    {tokens}")
        print(f"Tensor:    {tensor}")
        print(f"Shape:     {tensor.shape}")
        print(f"Decoded:   '{decoded}'")
        
        print("\n" + "=" * 60)
        print("Demo: Word Boundary Markers")
        print("=" * 60)
        
        test_text = "Hello world"
        test_tokens = tokenizer.tokenize(test_text)
        print(f"Text: '{test_text}'")
        safe_print(f"Tokens: {test_tokens}")
        print(f"Note: '_' (Unicode U+2581) marks the beginning of each word")
        
        print("\n" + "=" * 60)
        print("Demo: Batch Encoding")
        print("=" * 60)
        
        texts = [
            "Hello world!",
            "Deep learning works.",
            "Transformers are great.",
        ]
        
        batch = tokenizer.encode_batch(texts, max_length=15, add_special_tokens=True)
        
        print(f"Texts: {texts}")
        print(f"\nInput IDs shape: {batch['input_ids'].shape}")
        print(f"Input IDs:\n{batch['input_ids']}")
        
        decoded_batch = tokenizer.decode_batch(batch['input_ids'])
        print(f"\nDecoded batch: {decoded_batch}")
        
        print("\n" + "=" * 60)
        print("Demo: Embeddings")
        print("=" * 60)
        
        embed_layer = tokenizer.get_embedding_layer(embed_dim=64)
        embeddings = embed_layer(batch['input_ids'])
        
        print(f"Input shape:     {batch['input_ids'].shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Vocab size:      {len(tokenizer.vocab)}")
        
        # Show some vocabulary tokens with space marker
        print("\n" + "=" * 60)
        print("Sample Vocabulary (with word boundary markers)")
        print("=" * 60)
        
        sample_tokens = [t for t in list(tokenizer.vocab.keys())[:20] 
                        if t not in tokenizer.special_tokens]
        safe_print(f"Sample tokens: {sample_tokens}")
    
    tokenizer.save(f"sentencepiece_{args.mode}_vocab_pytorch.json")
    print(f"\nTokenizer saved to sentencepiece_{args.mode}_vocab_pytorch.json")


if __name__ == "__main__":
    main()
