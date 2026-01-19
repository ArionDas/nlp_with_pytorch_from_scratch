"""
One-Hot vs Learned Embeddings Comparison

Compare one-hot encodings with learned embeddings by:
- Computing cosine similarities/distances
- Plotting similarity matrices
- Demonstrating semantic relationships

Usage:
    python embedding_comparison.py                    # Run full comparison
    python embedding_comparison.py --embed-dim 50    # Custom embedding dimension
    python embedding_comparison.py --plot            # Generate plots (requires matplotlib)
"""

import argparse
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    split_into_words,
    invert_vocab,
    compute_embedding_similarity,
    find_nearest_embeddings,
)


# =============================================================================
# Embedding Types
# =============================================================================

def create_one_hot_embeddings(vocab_size: int) -> torch.Tensor:
    """
    Create one-hot embeddings (identity matrix).
    
    Each word is represented by a sparse vector with a single 1.
    
    Args:
        vocab_size: Size of vocabulary
    
    Returns:
        One-hot embedding matrix (vocab_size, vocab_size)
    """
    return torch.eye(vocab_size)


def create_random_embeddings(vocab_size: int, embed_dim: int) -> torch.Tensor:
    """
    Create random dense embeddings.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
    
    Returns:
        Random embedding matrix (vocab_size, embed_dim)
    """
    embeddings = torch.randn(vocab_size, embed_dim)
    # Normalize to unit vectors
    return F.normalize(embeddings, dim=1)


class LearnedEmbeddings(nn.Module):
    """
    Learned embeddings trained on context prediction (simplified Word2Vec).
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        # Average context embeddings
        embeds = self.embeddings(context_ids)
        avg_embed = embeds.mean(dim=1)
        return self.output(avg_embed)
    
    def get_embeddings(self) -> torch.Tensor:
        return self.embeddings.weight.detach()


def train_learned_embeddings(
    word_ids: list[int],
    vocab_size: int,
    embed_dim: int,
    window_size: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    verbose: bool = True
) -> torch.Tensor:
    """
    Train learned embeddings using context prediction.
    
    Args:
        word_ids: List of word indices in corpus
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        window_size: Context window size
        epochs: Training epochs
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        Learned embedding matrix (vocab_size, embed_dim)
    """
    # Create training pairs
    pairs = []
    for i in range(window_size, len(word_ids) - window_size):
        target = word_ids[i]
        context = [word_ids[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        pairs.append((context, target))
    
    if verbose:
        print(f"Training learned embeddings on {len(pairs)} samples...")
    
    model = LearnedEmbeddings(vocab_size, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Mini-batch training
        batch_size = 64
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            contexts = torch.tensor([p[0] for p in batch], dtype=torch.long)
            targets = torch.tensor([p[1] for p in batch], dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (len(pairs) / batch_size):.4f}")
    
    return model.get_embeddings()


# =============================================================================
# Similarity Analysis
# =============================================================================

def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Embedding matrix (vocab_size, embed_dim)
    
    Returns:
        Similarity matrix (vocab_size, vocab_size)
    """
    # Normalize embeddings
    normalized = F.normalize(embeddings, dim=1)
    # Compute cosine similarity
    return torch.mm(normalized, normalized.t())


def compute_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix.
    
    Args:
        embeddings: Embedding matrix (vocab_size, embed_dim)
    
    Returns:
        Distance matrix (vocab_size, vocab_size)
    """
    # Using cdist for efficiency
    return torch.cdist(embeddings, embeddings)


def analyze_similarity_distribution(sim_matrix: torch.Tensor) -> dict:
    """
    Analyze the distribution of similarities.
    
    Args:
        sim_matrix: Similarity matrix
    
    Returns:
        Dictionary of statistics
    """
    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    similarities = sim_matrix[mask]
    
    return {
        "mean": similarities.mean().item(),
        "std": similarities.std().item(),
        "min": similarities.min().item(),
        "max": similarities.max().item(),
        "median": similarities.median().item(),
    }


# =============================================================================
# Visualization (ASCII-based for terminal)
# =============================================================================

def print_similarity_matrix(
    sim_matrix: torch.Tensor,
    words: list[str],
    max_words: int = 10
) -> str:
    """
    Print ASCII representation of similarity matrix.
    """
    n = min(len(words), max_words)
    lines = []
    
    lines.append("\nSimilarity Matrix (subset):")
    
    # Header
    header = "        "
    for word in words[:n]:
        header += f"{word[:6]:>7}"
    lines.append(header)
    
    # Rows
    for i in range(n):
        row = f"{words[i][:6]:<7} "
        for j in range(n):
            sim = sim_matrix[i, j].item()
            # Use symbols for different similarity ranges
            if sim > 0.9:
                sym = "█"
            elif sim > 0.7:
                sym = "▓"
            elif sim > 0.5:
                sym = "▒"
            elif sim > 0.3:
                sym = "░"
            else:
                sym = "·"
            row += f"{sim:>6.2f}{sym}"
        lines.append(row)
    
    return "\n".join(lines)


def print_similarity_heatmap_ascii(
    sim_matrix: torch.Tensor,
    words: list[str],
    max_words: int = 15
) -> str:
    """
    Create ASCII heatmap of similarity matrix.
    """
    n = min(len(words), max_words)
    lines = []
    
    # Gradient characters from low to high similarity (ASCII-safe)
    gradient = " .:-=+*#@"
    
    lines.append("\nASCII Similarity Heatmap:")
    lines.append("(. = low similarity, @ = high similarity)")
    lines.append("")
    
    # Column headers (abbreviated)
    header = "     "
    for word in words[:n]:
        header += word[0].upper()
    lines.append(header)
    
    # Rows
    for i in range(n):
        row = f"{words[i][:4]:<4} "
        for j in range(n):
            sim = sim_matrix[i, j].item()
            # Map similarity to gradient index
            idx = int(sim * (len(gradient) - 1))
            idx = max(0, min(len(gradient) - 1, idx))
            row += gradient[idx]
        lines.append(row)
    
    lines.append("")
    lines.append("Legend: " + "".join(f"{gradient[i]}={i/(len(gradient)-1):.1f} " for i in range(len(gradient))))
    
    return "\n".join(lines)


def plot_similarities_matplotlib(
    one_hot_sim: torch.Tensor,
    learned_sim: torch.Tensor,
    words: list[str],
    output_path: str = "embedding_comparison.png"
):
    """
    Create matplotlib visualization comparing embeddings.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    n = min(len(words), 15)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # One-hot similarities
    ax1 = axes[0]
    im1 = ax1.imshow(one_hot_sim[:n, :n].numpy(), cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax1.set_title("One-Hot Similarities\n(Identity Matrix)")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(words[:n], rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(words[:n], fontsize=8)
    plt.colorbar(im1, ax=ax1)
    
    # Learned similarities
    ax2 = axes[1]
    im2 = ax2.imshow(learned_sim[:n, :n].numpy(), cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax2.set_title("Learned Embedding Similarities\n(Semantic Relationships)")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(words[:n], rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(words[:n], fontsize=8)
    plt.colorbar(im2, ax=ax2)
    
    # Distribution comparison
    ax3 = axes[2]
    
    # Get off-diagonal similarities
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    one_hot_vals = one_hot_sim[:n, :n][mask].numpy()
    learned_vals = learned_sim[:n, :n][mask].numpy()
    
    ax3.hist(one_hot_vals, bins=20, alpha=0.5, label='One-Hot', color='blue')
    ax3.hist(learned_vals, bins=20, alpha=0.5, label='Learned', color='green')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Count')
    ax3.set_title('Similarity Distribution')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="One-Hot vs Learned Embeddings")
    parser.add_argument("--embed-dim", "-d", type=int, default=32,
                        help="Learned embedding dimension (default: 32)")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate matplotlib plots")
    parser.add_argument("--corpus", "-c", type=str, default=None,
                        help="Path to custom corpus")
    return parser.parse_args()


def get_default_corpus() -> str:
    """Load the comprehensive English corpus."""
    import os
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus", "english_corpus.txt")
    
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return f.read() * 5  # Repeat for more training data
    
    # Fallback if corpus file not found
    return """
    The king ruled the kingdom with wisdom and the queen helped him.
    The prince and princess lived in the royal palace together.
    A man and woman walked through the city streets slowly.
    The boy and girl played games in the garden happily.
    Dogs are loyal pets and cats are independent animals.
    The cat chased the small mouse around the old house.
    Birds fly high in the clear blue sky above us.
    Fish swim deep in the cold ocean waters below.
    Machine learning uses data to train intelligent models.
    Deep learning models learn useful representations automatically.
    Natural language processing understands human text well.
    Neural networks contain many layers of neurons inside.
    """ * 50


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ONE-HOT vs LEARNED EMBEDDINGS COMPARISON")
    print("=" * 60)
    
    # Load corpus
    if args.corpus:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            corpus = f.read()
    else:
        corpus = get_default_corpus()
    
    # Build vocabulary
    words_list = [w.lower() for w in split_into_words(corpus) if w.strip() and w.isalpha()]
    word_counts = Counter(words_list)
    
    # Keep words with frequency >= 2
    vocab_words = sorted([w for w, c in word_counts.items() if c >= 2])
    word2idx = {"<UNK>": 0}
    for w in vocab_words:
        word2idx[w] = len(word2idx)
    idx2word = invert_vocab(word2idx)
    
    vocab_size = len(word2idx)
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Corpus size: {len(words_list)} words")
    
    # Prepare word IDs
    word_ids = [word2idx.get(w, 0) for w in words_list if w in word2idx]
    
    # ===========================================
    # Create One-Hot Embeddings
    # ===========================================
    print("\n" + "-" * 40)
    print("ONE-HOT EMBEDDINGS")
    print("-" * 40)
    
    one_hot_embeds = create_one_hot_embeddings(vocab_size)
    one_hot_sim = compute_similarity_matrix(one_hot_embeds)
    
    print(f"Embedding shape: {one_hot_embeds.shape}")
    print(f"Embedding dimension: {vocab_size} (same as vocab size)")
    
    one_hot_stats = analyze_similarity_distribution(one_hot_sim)
    print(f"\nSimilarity Statistics:")
    print(f"  Mean:   {one_hot_stats['mean']:.4f}")
    print(f"  Std:    {one_hot_stats['std']:.4f}")
    print(f"  Min:    {one_hot_stats['min']:.4f}")
    print(f"  Max:    {one_hot_stats['max']:.4f}")
    
    print("\nKey property: All pairs have similarity = 0 (orthogonal)")
    print("One-hot encodings cannot capture semantic relationships!")
    
    # ===========================================
    # Create Learned Embeddings
    # ===========================================
    print("\n" + "-" * 40)
    print("LEARNED EMBEDDINGS")
    print("-" * 40)
    
    learned_embeds = train_learned_embeddings(
        word_ids, vocab_size, args.embed_dim,
        epochs=args.epochs, verbose=True
    )
    learned_sim = compute_similarity_matrix(learned_embeds)
    
    print(f"\nEmbedding shape: {learned_embeds.shape}")
    print(f"Embedding dimension: {args.embed_dim} (dense representation)")
    
    learned_stats = analyze_similarity_distribution(learned_sim)
    print(f"\nSimilarity Statistics:")
    print(f"  Mean:   {learned_stats['mean']:.4f}")
    print(f"  Std:    {learned_stats['std']:.4f}")
    print(f"  Min:    {learned_stats['min']:.4f}")
    print(f"  Max:    {learned_stats['max']:.4f}")
    
    # ===========================================
    # Comparison
    # ===========================================
    print("\n" + "=" * 60)
    print("COMPARISON: ONE-HOT vs LEARNED")
    print("=" * 60)
    
    comparison_table = f"""
    {'Metric':<35} {'One-Hot':>15} {'Learned':>15}
    {'-'*35} {'-'*15} {'-'*15}
    {'Embedding dimension':<35} {vocab_size:>15} {args.embed_dim:>15}
    {'Memory per word (floats)':<35} {vocab_size:>15} {args.embed_dim:>15}
    {'Similarity mean':<35} {one_hot_stats['mean']:>15.4f} {learned_stats['mean']:>15.4f}
    {'Similarity std':<35} {one_hot_stats['std']:>15.4f} {learned_stats['std']:>15.4f}
    {'Captures semantics':<35} {'No':>15} {'Yes':>15}
    {'Sparsity':<35} {'100%':>15} {'0%':>15}
    """
    print(comparison_table)
    
    # ===========================================
    # Semantic Relationships (Learned only)
    # ===========================================
    print("\n" + "=" * 60)
    print("SEMANTIC RELATIONSHIPS (Learned Embeddings)")
    print("=" * 60)
    
    # Find similar words
    test_words = ["king", "man", "cat", "learning"]
    
    for test_word in test_words:
        if test_word in word2idx:
            word_idx = word2idx[test_word]
            query_embed = learned_embeds[word_idx]
            
            similar = find_nearest_embeddings(
                query_embed, learned_embeds, top_k=5,
                exclude_indices={word_idx}
            )
            
            print(f"\nMost similar to '{test_word}':")
            for idx, sim in similar:
                print(f"  {idx2word[idx]}: {sim:.4f}")
    
    # ===========================================
    # ASCII Visualization
    # ===========================================
    
    # Select interesting words for visualization
    viz_words = [idx2word[i] for i in range(min(12, vocab_size)) if i > 0]
    
    print("\n" + "=" * 60)
    print("SIMILARITY HEATMAPS")
    print("=" * 60)
    
    print("\nOne-Hot Embeddings:")
    print(print_similarity_heatmap_ascii(one_hot_sim, viz_words, max_words=12))
    
    print("\nLearned Embeddings:")
    print(print_similarity_heatmap_ascii(learned_sim, viz_words, max_words=12))
    
    # ===========================================
    # Generate Matplotlib Plot
    # ===========================================
    if args.plot:
        plot_similarities_matplotlib(
            one_hot_sim, learned_sim, viz_words,
            output_path="embedding_comparison.png"
        )
    
    # ===========================================
    # Key Takeaways
    # ===========================================
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    
    print("""
    ONE-HOT EMBEDDINGS:
    - Each word is orthogonal to all others (similarity = 0)
    - No semantic information encoded
    - Dimension = vocabulary size (high-dimensional, sparse)
    - Memory inefficient for large vocabularies
    - Cannot generalize to similar words
    
    LEARNED EMBEDDINGS:
    - Similar words have similar vectors (king ~ queen)
    - Capture semantic and syntactic relationships
    - Low-dimensional, dense representations
    - Memory efficient (e.g., 50-300 dimensions vs 50,000+)
    - Enable transfer learning and generalization
    
    WHEN TO USE WHICH:
    - One-hot: Simple baselines, small vocabularies, categorical features
    - Learned: NLP tasks, semantic similarity, downstream models
    """)
    
    # Save embeddings
    torch.save({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'one_hot_embeddings': one_hot_embeds,
        'learned_embeddings': learned_embeds,
        'embed_dim': args.embed_dim,
    }, 'embedding_comparison.pt')
    print("\nEmbeddings saved to embedding_comparison.pt")


if __name__ == "__main__":
    main()
