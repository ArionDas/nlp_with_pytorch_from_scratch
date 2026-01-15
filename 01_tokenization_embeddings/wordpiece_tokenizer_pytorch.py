"""
WordPiece Tokenizer with PyTorch Integration

WordPiece algorithm (used by BERT) + PyTorch tensors for neural network compatibility.

Unlike BPE which merges the most frequent pair, WordPiece merges the pair that
maximizes the likelihood of the training data (score = freq(ab) / (freq(a) * freq(b))).

Usage:
    python wordpiece_tokenizer_pytorch.py                           # Use default corpus
    python wordpiece_tokenizer_pytorch.py --text "Your text here"   # Custom text
    python wordpiece_tokenizer_pytorch.py --corpus path/to/file.txt # Train on file
"""

import argparse
import torch
from utils import *

class WordPieceTokenizerPyTorch:
    """
    WordPiece tokenizer with PyTorch tensor support.
    
    Implements the WordPiece algorithm used by BERT for subword tokenization.
    Key difference from BPE: uses likelihood-based scoring instead of frequency.
    
    WordPiece score = freq(ab) / (freq(a) * freq(b))
    This favors merging rare pairs that always appear together.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        
        # BERT-style special tokens
        self.special_tokens = DEFAULT_SPECIAL_TOKENS.copy()
        self.special_tokens["[CLS]"] = len(self.special_tokens)
        self.special_tokens["[SEP]"] = len(self.special_tokens)
        self.special_tokens["[MASK]"] = len(self.special_tokens)
        
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.unk_token_id = self.special_tokens["<UNK>"]
        self.cls_token_id = self.special_tokens["[CLS]"]
        self.sep_token_id = self.special_tokens["[SEP]"]
        self.mask_token_id = self.special_tokens["[MASK]"]
    
    def train(self, text: str, verbose: bool = True):
        """
        Train WordPiece tokenizer on a text corpus.
        
        Args:
            text: Training corpus
            verbose: Whether to print progress
        """
        # Get word splits in WordPiece format
        word_splits = get_wordpiece_word_splits(text)
        
        # Extract all unique tokens and build initial vocab
        all_tokens = extract_unique_chars(word_splits)
        self.vocab = build_initial_vocab(all_tokens, self.special_tokens)
        
        initial_vocab_size = len(self.vocab)
        num_merges = self.vocab_size - initial_vocab_size
        
        if verbose:
            print(f"Training WordPiece: {initial_vocab_size} -> {self.vocab_size} tokens")
        
        for i in range(num_merges):
            # Compute token frequencies and pair scores
            token_freqs = get_token_frequencies(word_splits)
            pair_scores = compute_wordpiece_pair_scores(word_splits, token_freqs)
            
            if not pair_scores:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break
            
            # Find pair with highest likelihood score
            best_pair = max(pair_scores, key=pair_scores.get)
            best_score = pair_scores[best_pair]
            
            # Create merged token and add to vocab
            merged_token = best_pair[0] + best_pair[1].replace(WORDPIECE_PREFIX, "")
            self.vocab[merged_token] = len(self.vocab)
            
            # Update word splits with merged pair
            word_splits = merge_wordpiece_pair(word_splits, best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}' (score: {best_score:.4f})")
        
        self.inverse_vocab = invert_vocab(self.vocab)
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into WordPiece tokens.
        
        Args:
            text: Input text
        
        Returns:
            List of WordPiece tokens
        """
        words = split_into_words(text)
        tokens = []
        
        for word in words:
            word = word.strip()
            if word:
                tokens.extend(wordpiece_tokenize_word(word, self.vocab))
        
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
            add_special_tokens: Whether to add [CLS]/[SEP] tokens (BERT-style)
        
        Returns:
            Token IDs as tensor or list
        """
        tokens = self.tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.cls_token_id)
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            token_ids.append(self.sep_token_id)
        
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
        """
        encoded = [
            self.encode(t, max_length, False, False, add_special_tokens)
            for t in texts
        ]
        return batch_encode_with_padding(encoded, max_length, self.pad_token_id)
    
    def decode(self, token_ids: torch.Tensor | list[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = from_tensor(token_ids)
        
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, "<UNK>")
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)
        
        return decode_wordpiece_tokens(tokens)
    
    def decode_batch(self, batch_ids: torch.Tensor, skip_special: bool = True) -> list[str]:
        """Decode batch of token IDs."""
        return [self.decode(ids, skip_special) for ids in batch_ids]
    
    def save(self, path: str):
        """Save tokenizer to JSON file."""
        save_tokenizer_data(path, self.vocab_size, self.vocab, {})
    
    def load(self, path: str):
        """Load tokenizer from JSON file."""
        data = load_tokenizer_data(path)
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
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
        description="WordPiece Tokenizer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Input text to tokenize (if not provided, runs demo)")
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
    BERT uses WordPiece tokenization for subword units.
    The masked language model predicts hidden tokens.
    """ * 20


def main():
    args = parse_args()
    
    print("=" * 60)
    print("WordPiece Tokenizer with PyTorch")
    print("=" * 60)
    
    tokenizer = WordPieceTokenizerPyTorch(vocab_size=args.vocab_size)
    
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
            "BERT uses WordPiece.",
            "Transformers are great.",
        ]
        
        batch = tokenizer.encode_batch(texts, max_length=20, add_special_tokens=True)
        
        print(f"Texts: {texts}")
        print(f"\nInput IDs shape: {batch['input_ids'].shape}")
        print(f"Input IDs:\n{batch['input_ids']}")
        print(f"\nAttention Mask:\n{batch['attention_mask']}")
        
        decoded_batch = tokenizer.decode_batch(batch['input_ids'])
        print(f"\nDecoded batch: {decoded_batch}")
        
        print("\n" + "=" * 60)
        print("Demo: Token Embeddings (using shared TokenEmbeddingLayer)")
        print("=" * 60)
        
        embed_layer = tokenizer.get_embedding_layer(embed_dim=64)
        embeddings = embed_layer(batch['input_ids'])
        
        print(f"Input shape:     {batch['input_ids'].shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Vocab size:      {len(tokenizer.vocab)}")
        
        print("\n" + "=" * 60)
        print("WordPiece Tokenization Example")
        print("=" * 60)
        
        test_word = "tokenization"
        wp_tokens = tokenizer.tokenize(test_word)
        print(f"Word: '{test_word}'")
        print(f"WordPiece tokens: {wp_tokens}")
        print(f"Note: ## prefix indicates continuation of previous token")
    
    tokenizer.save("wordpiece_vocab_pytorch.json")
    print(f"\nTokenizer saved to wordpiece_vocab_pytorch.json")


if __name__ == "__main__":
    main()
