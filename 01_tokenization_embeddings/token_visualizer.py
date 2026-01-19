"""
Token Visualizer - Map words/chunks to IDs with visual output.

Provides interactive visualization of how different tokenizers break down text
into tokens and their corresponding IDs.

Usage:
    python token_visualizer.py                                    # Demo
    python token_visualizer.py --text "Your text here"            # Custom text
    python token_visualizer.py --tokenizer bpe --text "Hello"     # Specific tokenizer
"""

import argparse
from typing import Protocol

from utils import (
    DEFAULT_SPECIAL_TOKENS,
    split_into_words,
    build_initial_vocab,
    invert_vocab,
)


# =============================================================================
# Tokenizer Protocol (for type hints)
# =============================================================================

class TokenizerProtocol(Protocol):
    """Protocol for tokenizers to ensure consistent interface."""
    vocab: dict[str, int]
    inverse_vocab: dict[int, str]
    
    def tokenize(self, text: str) -> list[str]: ...
    def encode(self, text: str, **kwargs) -> list[int]: ...
    def decode(self, token_ids: list[int], **kwargs) -> str: ...


# =============================================================================
# Simple Character Tokenizer (for demonstration)
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer for visualization."""
    
    def __init__(self):
        self.vocab: dict[str, int] = dict(DEFAULT_SPECIAL_TOKENS)
        self.inverse_vocab: dict[int, str] = {}
    
    def fit(self, text: str):
        """Build vocabulary from text."""
        for char in sorted(set(text)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = invert_vocab(self.vocab)
    
    def tokenize(self, text: str) -> list[str]:
        return list(text)
    
    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(c, self.vocab["<UNK>"]) for c in text]
    
    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.inverse_vocab.get(i, "?") for i in token_ids)


class WordTokenizer:
    """Simple word-level tokenizer for visualization."""
    
    def __init__(self):
        self.vocab: dict[str, int] = dict(DEFAULT_SPECIAL_TOKENS)
        self.inverse_vocab: dict[int, str] = {}
    
    def fit(self, text: str):
        """Build vocabulary from text."""
        words = split_into_words(text)
        for word in sorted(set(words)):
            word = word.strip()
            if word and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.inverse_vocab = invert_vocab(self.vocab)
    
    def tokenize(self, text: str) -> list[str]:
        return [w for w in split_into_words(text) if w.strip()]
    
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
    
    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.inverse_vocab.get(i, "<UNK>") for i in token_ids)


# =============================================================================
# Visualization Functions
# =============================================================================

def colorize_token(token: str, token_id: int) -> str:
    """
    Create colored representation of a token.
    Uses ANSI codes for terminal colors.
    """
    # Color palette based on token ID
    colors = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
        "\033[97m",  # White
    ]
    reset = "\033[0m"
    color = colors[token_id % len(colors)]
    return f"{color}{token}{reset}"


def visualize_tokenization(
    text: str,
    tokens: list[str],
    token_ids: list[int],
    tokenizer_name: str = "Tokenizer"
) -> str:
    """
    Create a visual representation of tokenization.
    
    Returns a formatted string showing:
    - Original text
    - Token breakdown
    - Token to ID mapping
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Tokenizer: {tokenizer_name}")
    lines.append(f"{'='*60}")
    
    # Original text
    lines.append(f"\nOriginal Text:")
    lines.append(f"  \"{text}\"")
    
    # Token count
    lines.append(f"\nToken Count: {len(tokens)}")
    
    # Token breakdown with boxes
    lines.append(f"\nToken Breakdown:")
    token_display = "  "
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        # Escape special characters for display
        display_token = repr(token)[1:-1] if token in [' ', '\n', '\t'] else token
        token_display += f"[{display_token}]"
    lines.append(token_display)
    
    # Token to ID mapping table
    lines.append(f"\nToken -> ID Mapping:")
    lines.append(f"  {'Token':<20} {'ID':>6} {'Length':>8}")
    lines.append(f"  {'-'*20} {'-'*6} {'-'*8}")
    
    for token, tid in zip(tokens, token_ids):
        display_token = repr(token) if len(token) == 1 and not token.isalnum() else f"'{token}'"
        lines.append(f"  {display_token:<20} {tid:>6} {len(token):>8}")
    
    # ID sequence
    lines.append(f"\nID Sequence:")
    lines.append(f"  {token_ids}")
    
    return "\n".join(lines)


def visualize_vocab_sample(
    vocab: dict[str, int],
    max_items: int = 30
) -> str:
    """Show a sample of the vocabulary."""
    lines = []
    lines.append(f"\nVocabulary Sample (showing {min(max_items, len(vocab))} of {len(vocab)}):")
    lines.append(f"  {'Token':<20} {'ID':>6}")
    lines.append(f"  {'-'*20} {'-'*6}")
    
    for i, (token, tid) in enumerate(vocab.items()):
        if i >= max_items:
            lines.append(f"  ... and {len(vocab) - max_items} more tokens")
            break
        display_token = repr(token) if len(token) == 1 and not token.isalnum() else f"'{token}'"
        lines.append(f"  {display_token:<20} {tid:>6}")
    
    return "\n".join(lines)


def compare_tokenizers(
    text: str,
    tokenizers: dict[str, object]
) -> str:
    """Compare tokenization across multiple tokenizers."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append("TOKENIZER COMPARISON")
    lines.append(f"{'='*60}")
    lines.append(f"\nText: \"{text}\"")
    
    # Header
    lines.append(f"\n{'Tokenizer':<15} {'#Tokens':>8} {'Tokens'}")
    lines.append(f"{'-'*15} {'-'*8} {'-'*40}")
    
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        token_str = str(tokens)
        if len(token_str) > 40:
            token_str = token_str[:37] + "..."
        lines.append(f"{name:<15} {len(tokens):>8} {token_str}")
    
    return "\n".join(lines)


def create_token_heatmap(
    text: str,
    tokens: list[str],
    token_ids: list[int]
) -> str:
    """
    Create ASCII heatmap showing token boundaries.
    """
    lines = []
    lines.append(f"\nToken Heatmap (each token gets unique marker):")
    
    # Create character-level display with token boundaries
    markers = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    char_line = ""
    marker_line = ""
    
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        marker = markers[i % len(markers)]
        char_line += token
        marker_line += marker * len(token)
    
    # Split into chunks for display
    chunk_size = 60
    for start in range(0, len(char_line), chunk_size):
        end = min(start + chunk_size, len(char_line))
        lines.append(f"  Text:   |{char_line[start:end]}|")
        lines.append(f"  Tokens: |{marker_line[start:end]}|")
        lines.append("")
    
    # Legend
    lines.append("  Legend:")
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        if i >= 10:
            lines.append(f"    ... and {len(tokens) - 10} more")
            break
        marker = markers[i % len(markers)]
        lines.append(f"    {marker} = '{token}' (ID: {tid})")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Token Visualizer")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to tokenize")
    parser.add_argument("--tokenizer", type=str, default="all",
                        choices=["char", "word", "bpe", "wordpiece", "all"],
                        help="Tokenizer to use (default: all)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple tokenizers")
    return parser.parse_args()


def get_default_text() -> str:
    return "Machine learning enables computers to learn from data automatically."


def main():
    args = parse_args()
    text = args.text or get_default_text()
    
    print("=" * 60)
    print("TOKEN VISUALIZER")
    print("=" * 60)
    
    # Initialize tokenizers
    tokenizers = {}
    
    # Character tokenizer
    char_tok = CharTokenizer()
    char_tok.fit(text)
    tokenizers["Character"] = char_tok
    
    # Word tokenizer
    word_tok = WordTokenizer()
    word_tok.fit(text)
    tokenizers["Word"] = word_tok
    
    # Try to import BPE tokenizer
    try:
        from bpe_tokenizer_pytorch import BPETokenizerPyTorch
        bpe_tok = BPETokenizerPyTorch(vocab_size=100)
        bpe_tok.train(text * 10, verbose=False)
        tokenizers["BPE"] = bpe_tok
    except ImportError:
        print("(BPE tokenizer not available)")
    
    # Try to import WordPiece tokenizer
    try:
        from wordpiece_tokenizer_pytorch import WordPieceTokenizerPyTorch
        wp_tok = WordPieceTokenizerPyTorch(vocab_size=100)
        wp_tok.train(text * 10, verbose=False)
        tokenizers["WordPiece"] = wp_tok
    except ImportError:
        print("(WordPiece tokenizer not available)")
    
    if args.compare or args.tokenizer == "all":
        # Compare all tokenizers
        print(compare_tokenizers(text, tokenizers))
        
        # Detailed view for each
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else list(range(len(tokens)))
            
            # Handle tensor return types
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            
            print(visualize_tokenization(text, tokens, token_ids, name))
    else:
        # Single tokenizer
        tok_map = {
            "char": ("Character", char_tok),
            "word": ("Word", word_tok),
        }
        
        if "bpe" in [args.tokenizer] and "BPE" in tokenizers:
            tok_map["bpe"] = ("BPE", tokenizers["BPE"])
        if "wordpiece" in [args.tokenizer] and "WordPiece" in tokenizers:
            tok_map["wordpiece"] = ("WordPiece", tokenizers["WordPiece"])
        
        if args.tokenizer in tok_map:
            name, tokenizer = tok_map[args.tokenizer]
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            
            print(visualize_tokenization(text, tokens, token_ids, name))
            print(create_token_heatmap(text, tokens, token_ids))
            print(visualize_vocab_sample(tokenizer.vocab))
    
    # Interactive demo
    print("\n" + "=" * 60)
    print("DETAILED TOKEN ANALYSIS")
    print("=" * 60)
    
    # Use BPE if available, else word tokenizer
    demo_tok = tokenizers.get("BPE", tokenizers.get("Word"))
    demo_name = "BPE" if "BPE" in tokenizers else "Word"
    
    tokens = demo_tok.tokenize(text)
    token_ids = demo_tok.encode(text)
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    print(f"\nUsing {demo_name} tokenizer:")
    print(create_token_heatmap(text, tokens, token_ids))
    
    # Statistics
    print("\n" + "=" * 60)
    print("TOKENIZATION STATISTICS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print(f"{'-'*30} {'-'*15}")
    print(f"{'Original text length':<30} {len(text):>15}")
    print(f"{'Number of tokens':<30} {len(tokens):>15}")
    print(f"{'Compression ratio':<30} {len(text)/len(tokens):>15.2f}")
    print(f"{'Vocabulary size':<30} {len(demo_tok.vocab):>15}")
    print(f"{'Unique tokens in text':<30} {len(set(tokens)):>15}")
    print(f"{'Avg token length':<30} {sum(len(t) for t in tokens)/len(tokens):>15.2f}")


if __name__ == "__main__":
    main()
