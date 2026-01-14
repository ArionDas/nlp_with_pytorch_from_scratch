"""
Byte-Pair Encoding (BPE) Tokenizer with PyTorch Integration

BPE algorithm + PyTorch tensors for neural network compatibility.

Usage:
    python bpe_tokenizer_pytorch.py                           # Use default corpus
    python bpe_tokenizer_pytorch.py --text "Your text here"   # Custom text
    python bpe_tokenizer_pytorch.py --corpus path/to/file.txt # Train on file
"""

import argparse

import torch
import torch.nn as nn

from utils import *

class BPETokenizerPyTorch:
    """
    Byte-Pair Encoding tokenizer with PyTorch tensor support.
    
    Implements the BPE algorithm used by GPT-2/GPT-3 for subword tokenization,
    with built-in support for PyTorch tensors, batching, and padding.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        
        self.special_tokens = DEFAULT_SPECIAL_TOKENS.copy()
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.unk_token_id = self.special_tokens["<UNK>"]
        self.bos_token_id = self.special_tokens["<BOS>"]
        self.eos_token_id = self.special_tokens["<EOS>"]
    
    def train(self, text: str, verbose: bool = True):
        """
        Train BPE tokenizer on a text corpus.
        
        Args:
            text: Training corpus
            verbose: Whether to print progress
        """
        word_freqs = get_word_frequencies(text)
        chars = extract_unique_chars(word_freqs)
        self.vocab = build_initial_vocab(chars, self.special_tokens)
        
        num_merges = self.vocab_size - len(self.vocab)
        
        if verbose:
            print(f"Training BPE: {len(self.vocab)} -> {self.vocab_size} tokens")
        
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
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text
        
        Returns:
            List of subword tokens
        """
        words = split_into_words(text)
        tokens = []
        for word in words:
            tokens.extend(apply_merges_to_word(word, self.merges))
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
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            return_tensors: Whether to return PyTorch tensor
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            Token IDs as tensor or list
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
        """
        Encode batch of texts with padding and attention mask.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        encoded = [
            self.encode(t, max_length, False, False, add_special_tokens)
            for t in texts
        ]
        return batch_encode_with_padding(encoded, max_length, self.pad_token_id)
    
    def decode(self, token_ids: torch.Tensor | list[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs as tensor or list
            skip_special: Whether to skip special tokens
        
        Returns:
            Decoded text string
        """
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
    
    def save(self, path: str):
        """Save tokenizer to JSON file."""
        save_tokenizer_data(path, self.vocab_size, self.vocab, self.merges)
    
    def load(self, path: str):
        """Load tokenizer from JSON file."""
        data = load_tokenizer_data(path)
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        self.inverse_vocab = invert_vocab(self.vocab)


class TokenEmbedding(nn.Module):
    """Learnable token embeddings from BPE tokenizer."""
    
    def __init__(self, tokenizer: BPETokenizerPyTorch, embed_dim: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer.vocab),
            embedding_dim=embed_dim,
            padding_idx=tokenizer.pad_token_id
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BPE Tokenizer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Input text to tokenize (if not provided, runs demo)"
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default=None,
        help="Path to corpus file for training (uses default if not provided)"
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=256,
        help="Vocabulary size (default: 256)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load pre-trained tokenizer"
    )
    return parser.parse_args()


def get_default_corpus() -> str:
    return """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Natural language processing enables computers to understand human language.
    Deep learning models learn representations from data.
    Transformers have revolutionized natural language processing.
    Attention is all you need for sequence modeling.
    """ * 20


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BPE Tokenizer with PyTorch")
    print("=" * 60)
    
    tokenizer = BPETokenizerPyTorch(vocab_size=args.vocab_size)
    
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
        print("Demo: Batch Encoding with Padding")
        print("=" * 60)
        
        texts = [
            "Hello world!",
            "Machine learning is great.",
            "NLP with PyTorch.",
        ]
        
        batch = tokenizer.encode_batch(texts, max_length=20, add_special_tokens=True)
        
        print(f"Texts: {texts}")
        print(f"\nInput IDs shape: {batch['input_ids'].shape}")
        print(f"Input IDs:\n{batch['input_ids']}")
        print(f"\nAttention Mask:\n{batch['attention_mask']}")
        
        decoded_batch = tokenizer.decode_batch(batch['input_ids'])
        print(f"\nDecoded batch: {decoded_batch}")
        
        print("\n" + "=" * 60)
        print("Demo: Token Embeddings (nn.Embedding)")
        print("=" * 60)
        
        embed_layer = TokenEmbedding(tokenizer, embed_dim=64)
        embeddings = embed_layer(batch['input_ids'])
        
        print(f"Input shape:     {batch['input_ids'].shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Vocab size:      {len(tokenizer.vocab)}")
    
    tokenizer.save("bpe_vocab_pytorch.json")
    print(f"\nTokenizer saved to bpe_vocab_pytorch.json")


if __name__ == "__main__":
    main()
