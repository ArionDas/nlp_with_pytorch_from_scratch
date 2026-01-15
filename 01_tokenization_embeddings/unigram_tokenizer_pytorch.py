"""
Unigram Language Model Tokenizer with PyTorch Integration

Unigram LM (used by SentencePiece, XLNet, ALBERT) - probabilistic tokenization.

Unlike BPE/WordPiece which build vocab bottom-up by merging, Unigram starts with
a large vocab and prunes it down by removing tokens that least impact the corpus
likelihood. Uses Viterbi algorithm to find the most likely segmentation.

Usage:
    python unigram_tokenizer_pytorch.py                           # Use default corpus
    python unigram_tokenizer_pytorch.py --text "Your text here"   # Custom text
    python unigram_tokenizer_pytorch.py --corpus path/to/file.txt # Train on file
"""

import argparse
import math

import torch

from utils import (
    # Constants
    DEFAULT_SPECIAL_TOKENS,
    # Text processing
    split_into_words,
    # Unigram algorithm
    build_initial_unigram_vocab,
    viterbi_tokenize,
    compute_unigram_loss,
    prune_unigram_vocab,
    # Vocabulary
    invert_vocab,
    # Serialization
    save_tokenizer_data,
    load_tokenizer_data,
    # Batching
    batch_encode_with_padding,
    to_tensor,
    from_tensor,
    # Embedding
    TokenEmbeddingLayer,
)


class UnigramTokenizerPyTorch:
    """
    Unigram Language Model tokenizer with PyTorch tensor support.
    
    Key characteristics:
    - Probabilistic: each token has a probability, find most likely segmentation
    - Top-down: starts with large vocab, prunes to target size
    - Uses Viterbi algorithm for tokenization (dynamic programming)
    - Can produce multiple valid segmentations (we use most likely)
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}  # token -> id
        self.vocab_probs: dict[str, float] = {}  # token -> log probability
        self.inverse_vocab: dict[int, str] = {}
        
        self.special_tokens = DEFAULT_SPECIAL_TOKENS.copy()
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.unk_token_id = self.special_tokens["<UNK>"]
        self.bos_token_id = self.special_tokens["<BOS>"]
        self.eos_token_id = self.special_tokens["<EOS>"]
    
    def train(self, text: str, verbose: bool = True, initial_vocab_multiplier: int = 10):
        """
        Train Unigram tokenizer by building large vocab then pruning.
        
        Args:
            text: Training corpus
            verbose: Whether to print progress
            initial_vocab_multiplier: Initial vocab is target * this multiplier
        """
        if verbose:
            print(f"Training Unigram LM tokenizer...")
        
        # Build word frequency dictionary
        word_freqs = {}
        for word in split_into_words(text):
            word = word.strip()
            if word:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Build initial large vocabulary
        initial_size = self.vocab_size * initial_vocab_multiplier
        self.vocab_probs = build_initial_unigram_vocab(
            text, initial_size, self.special_tokens
        )
        
        if verbose:
            print(f"Initial vocab size: {len(self.vocab_probs)}")
            initial_loss = compute_unigram_loss(word_freqs, self.vocab_probs)
            print(f"Initial loss: {initial_loss:.2f}")
        
        # Prune vocabulary to target size
        if len(self.vocab_probs) > self.vocab_size:
            if verbose:
                print(f"Pruning to target size: {self.vocab_size}")
            
            self.vocab_probs = prune_unigram_vocab(
                word_freqs,
                self.vocab_probs,
                self.vocab_size,
                set(self.special_tokens.keys()),
                prune_ratio=0.2
            )
        
        # Build token -> id mapping
        self.vocab = dict(self.special_tokens)
        for token in sorted(self.vocab_probs.keys()):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.inverse_vocab = invert_vocab(self.vocab)
        
        if verbose:
            final_loss = compute_unigram_loss(word_freqs, self.vocab_probs)
            print(f"Final vocab size: {len(self.vocab)}")
            print(f"Final loss: {final_loss:.2f}")
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using Viterbi algorithm.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens (most likely segmentation)
        """
        words = split_into_words(text)
        tokens = []
        
        for word in words:
            word = word.strip()
            if word:
                word_tokens = viterbi_tokenize(word, self.vocab_probs, "<UNK>")
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
        """
        Encode text to token IDs.
        """
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
        
        return "".join(tokens)
    
    def decode_batch(self, batch_ids: torch.Tensor, skip_special: bool = True) -> list[str]:
        """Decode batch of token IDs."""
        return [self.decode(ids, skip_special) for ids in batch_ids]
    
    def get_token_probability(self, token: str) -> float:
        """Get the probability of a token."""
        if token in self.vocab_probs:
            return math.exp(self.vocab_probs[token])
        return 0.0
    
    def save(self, path: str):
        """Save tokenizer to JSON file."""
        import json
        from pathlib import Path
        
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "vocab_probs": self.vocab_probs,
            "merges": {},
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def load(self, path: str):
        """Load tokenizer from JSON file."""
        import json
        from pathlib import Path
        
        data = json.loads(Path(path).read_text())
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
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
        description="Unigram Language Model Tokenizer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Input text to tokenize")
    parser.add_argument("--corpus", "-c", type=str, default=None,
                        help="Path to corpus file for training")
    parser.add_argument("--vocab-size", "-v", type=int, default=256,
                        help="Vocabulary size (default: 256)")
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
    Unigram models use probabilistic tokenization.
    The Viterbi algorithm finds the most likely segmentation.
    """ * 20


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Unigram Language Model Tokenizer with PyTorch")
    print("=" * 60)
    
    tokenizer = UnigramTokenizerPyTorch(vocab_size=args.vocab_size)
    
    if args.load:
        print(f"Loading tokenizer from: {args.load}")
        tokenizer.load(args.load)
        print(f"Loaded vocab size: {len(tokenizer.vocab)}")
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
        print(f"Tokens:    {tokens}")
        print(f"Token IDs: {tensor.tolist()}")
        print(f"Decoded:   '{decoded}'")
        
        # Show token probabilities
        print("\nToken probabilities:")
        for token in tokens[:5]:
            prob = tokenizer.get_token_probability(token)
            print(f"  '{token}': {prob:.6f}")
    else:
        print("\n" + "=" * 60)
        print("Demo: Single Text Encoding")
        print("=" * 60)
        
        text = "Machine learning is amazing!"
        tokens = tokenizer.tokenize(text)
        tensor = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(tensor)
        
        print(f"Text:      '{text}'")
        print(f"Tokens:    {tokens}")
        print(f"Tensor:    {tensor}")
        print(f"Shape:     {tensor.shape}")
        print(f"Decoded:   '{decoded}'")
        
        print("\n" + "=" * 60)
        print("Demo: Token Probabilities")
        print("=" * 60)
        
        print("Top tokens by probability:")
        sorted_probs = sorted(tokenizer.vocab_probs.items(), key=lambda x: x[1], reverse=True)
        for token, log_prob in sorted_probs[:10]:
            if token not in tokenizer.special_tokens:
                print(f"  '{token}': {math.exp(log_prob):.6f}")
        
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
    
    tokenizer.save("unigram_vocab_pytorch.json")
    print(f"\nTokenizer saved to unigram_vocab_pytorch.json")


if __name__ == "__main__":
    main()
