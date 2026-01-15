"""
Word2Vec: CBOW and Skip-gram Models with PyTorch

Word2Vec learns dense word embeddings by predicting words from context.

- CBOW (Continuous Bag of Words): Predict target word from surrounding context
- Skip-gram: Predict surrounding context words from target word

Usage:
    python word2vec_pytorch.py                              # Train both models
    python word2vec_pytorch.py --model cbow                 # Train CBOW only
    python word2vec_pytorch.py --model skipgram             # Train Skip-gram only
    python word2vec_pytorch.py --corpus path/to/file.txt   # Custom corpus
"""

import argparse
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import (
    DEFAULT_SPECIAL_TOKENS,
    split_into_words,
    build_initial_vocab,
    invert_vocab,
    to_tensor,
)


# =============================================================================
# Data Preparation
# =============================================================================

def build_word_vocab(
    text: str,
    min_freq: int = 1,
    special_tokens: dict[str, int] | None = None
) -> tuple[dict[str, int], dict[int, str], Counter]:
    """
    Build vocabulary from text corpus.
    
    Args:
        text: Input corpus
        min_freq: Minimum word frequency to include
        special_tokens: Special tokens to add
    
    Returns:
        Tuple of (word2idx, idx2word, word_counts)
    """
    words = [w.lower() for w in split_into_words(text) if w.strip() and w.isalpha()]
    word_counts = Counter(words)
    
    # Filter by minimum frequency
    filtered_words = {w for w, c in word_counts.items() if c >= min_freq}
    
    # Build vocab
    word2idx = dict(special_tokens) if special_tokens else {"<UNK>": 0}
    for word in sorted(filtered_words):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    
    idx2word = invert_vocab(word2idx)
    
    return word2idx, idx2word, word_counts


def create_cbow_samples(
    words: list[str],
    word2idx: dict[str, int],
    window_size: int = 2
) -> list[tuple[list[int], int]]:
    """
    Create CBOW training samples: (context_words, target_word).
    
    Args:
        words: List of words in corpus
        word2idx: Word to index mapping
        window_size: Context window size on each side
    
    Returns:
        List of (context_ids, target_id) tuples
    """
    samples = []
    unk_id = word2idx.get("<UNK>", 0)
    
    for i in range(window_size, len(words) - window_size):
        target = words[i]
        if target not in word2idx:
            continue
        
        context = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i:
                ctx_word = words[j]
                context.append(word2idx.get(ctx_word, unk_id))
        
        target_id = word2idx[target]
        samples.append((context, target_id))
    
    return samples


def create_skipgram_samples(
    words: list[str],
    word2idx: dict[str, int],
    window_size: int = 2
) -> list[tuple[int, int]]:
    """
    Create Skip-gram training samples: (target_word, context_word).
    
    Args:
        words: List of words in corpus
        word2idx: Word to index mapping
        window_size: Context window size on each side
    
    Returns:
        List of (target_id, context_id) tuples
    """
    samples = []
    unk_id = word2idx.get("<UNK>", 0)
    
    for i in range(window_size, len(words) - window_size):
        target = words[i]
        if target not in word2idx:
            continue
        
        target_id = word2idx[target]
        
        for j in range(i - window_size, i + window_size + 1):
            if j != i:
                ctx_word = words[j]
                ctx_id = word2idx.get(ctx_word, unk_id)
                samples.append((target_id, ctx_id))
    
    return samples


# =============================================================================
# Datasets
# =============================================================================

class CBOWDataset(Dataset):
    """Dataset for CBOW model."""
    
    def __init__(self, samples: list[tuple[list[int], int]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram model."""
    
    def __init__(self, samples: list[tuple[int, int]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        target, context = self.samples[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class SkipGramNegSamplingDataset(Dataset):
    """Dataset for Skip-gram with Negative Sampling."""
    
    def __init__(
        self,
        samples: list[tuple[int, int]],
        vocab_size: int,
        num_negatives: int = 5,
        word_freqs: Counter | None = None
    ):
        self.samples = samples
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives
        
        # Build negative sampling distribution (word^0.75)
        if word_freqs:
            freqs = [word_freqs.get(i, 1) ** 0.75 for i in range(vocab_size)]
        else:
            freqs = [1.0] * vocab_size
        
        total = sum(freqs)
        self.neg_probs = [f / total for f in freqs]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        target, pos_context = self.samples[idx]
        
        # Sample negative contexts
        neg_contexts = []
        while len(neg_contexts) < self.num_negatives:
            neg = random.choices(range(self.vocab_size), weights=self.neg_probs, k=1)[0]
            if neg != pos_context:
                neg_contexts.append(neg)
        
        return (
            torch.tensor(target, dtype=torch.long),
            torch.tensor(pos_context, dtype=torch.long),
            torch.tensor(neg_contexts, dtype=torch.long)
        )


# =============================================================================
# Models
# =============================================================================

class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) Model.
    
    Predicts target word from average of context word embeddings.
    
    Architecture:
        Context words -> Embeddings -> Average -> Linear -> Softmax -> Target word
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Input embeddings (context words)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Output projection (to vocabulary)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context word indices (batch_size, context_size)
        
        Returns:
            Logits over vocabulary (batch_size, vocab_size)
        """
        # Get embeddings for all context words
        embeds = self.embeddings(context)  # (B, C, E)
        
        # Average context embeddings
        avg_embed = embeds.mean(dim=1)  # (B, E)
        
        # Project to vocabulary
        logits = self.linear(avg_embed)  # (B, V)
        
        return logits
    
    def get_word_embedding(self, word_idx: int) -> torch.Tensor:
        """Get embedding for a single word."""
        return self.embeddings.weight[word_idx].detach()
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all word embeddings."""
        return self.embeddings.weight.detach()


class SkipGramModel(nn.Module):
    """
    Skip-gram Model.
    
    Predicts context words from target word.
    
    Architecture:
        Target word -> Embedding -> Linear -> Softmax -> Context words
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Input embeddings (target words)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Output embeddings (context words)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)
    
    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: Target word indices (batch_size,)
        
        Returns:
            Logits over vocabulary (batch_size, vocab_size)
        """
        # Get target embedding
        embed = self.embeddings(target)  # (B, E)
        
        # Compute similarity with all output embeddings
        logits = torch.matmul(embed, self.out_embeddings.weight.t())  # (B, V)
        
        return logits
    
    def forward_neg_sampling(
        self,
        target: torch.Tensor,
        pos_context: torch.Tensor,
        neg_contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for negative sampling training.
        
        Args:
            target: Target word indices (batch_size,)
            pos_context: Positive context indices (batch_size,)
            neg_contexts: Negative context indices (batch_size, num_neg)
        
        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        # Target embeddings
        target_embed = self.embeddings(target)  # (B, E)
        
        # Positive context embeddings
        pos_embed = self.out_embeddings(pos_context)  # (B, E)
        pos_score = (target_embed * pos_embed).sum(dim=1)  # (B,)
        
        # Negative context embeddings
        neg_embed = self.out_embeddings(neg_contexts)  # (B, N, E)
        neg_score = torch.bmm(neg_embed, target_embed.unsqueeze(2)).squeeze(2)  # (B, N)
        
        return pos_score, neg_score
    
    def get_word_embedding(self, word_idx: int) -> torch.Tensor:
        """Get embedding for a single word."""
        return self.embeddings.weight[word_idx].detach()
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all word embeddings."""
        return self.embeddings.weight.detach()


# =============================================================================
# Training Functions
# =============================================================================

def train_cbow(
    model: CBOWModel,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True
) -> list[float]:
    """Train CBOW model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def train_skipgram(
    model: SkipGramModel,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True
) -> list[float]:
    """Train Skip-gram model with standard cross-entropy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for target, context in dataloader:
            target = target.to(device)
            context = context.to(device)
            
            optimizer.zero_grad()
            logits = model(target)
            loss = criterion(logits, context)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def train_skipgram_neg_sampling(
    model: SkipGramModel,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True
) -> list[float]:
    """Train Skip-gram model with negative sampling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for target, pos_context, neg_contexts in dataloader:
            target = target.to(device)
            pos_context = pos_context.to(device)
            neg_contexts = neg_contexts.to(device)
            
            optimizer.zero_grad()
            
            pos_score, neg_score = model.forward_neg_sampling(
                target, pos_context, neg_contexts
            )
            
            # Negative sampling loss: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


# =============================================================================
# Similarity Functions
# =============================================================================

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def find_similar_words(
    word: str,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    embeddings: torch.Tensor,
    top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Find most similar words to a given word.
    
    Args:
        word: Query word
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        embeddings: Word embeddings matrix (vocab_size, embed_dim)
        top_k: Number of similar words to return
    
    Returns:
        List of (word, similarity) tuples
    """
    if word not in word2idx:
        return []
    
    word_idx = word2idx[word]
    word_embed = embeddings[word_idx]
    
    # Compute similarities with all words
    similarities = F.cosine_similarity(
        word_embed.unsqueeze(0),
        embeddings,
        dim=1
    )
    
    # Get top-k (excluding the word itself)
    top_indices = similarities.argsort(descending=True)[1:top_k + 1]
    
    results = []
    for idx in top_indices:
        similar_word = idx2word[idx.item()]
        sim = similarities[idx].item()
        results.append((similar_word, sim))
    
    return results


def word_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    embeddings: torch.Tensor,
    top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Solve word analogy: A is to B as C is to ?
    
    Uses: vec(B) - vec(A) + vec(C)
    
    Args:
        word_a, word_b, word_c: Analogy words
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        embeddings: Word embeddings matrix
        top_k: Number of results to return
    
    Returns:
        List of (word, similarity) tuples
    """
    if word_a not in word2idx or word_b not in word2idx or word_c not in word2idx:
        return []
    
    vec_a = embeddings[word2idx[word_a]]
    vec_b = embeddings[word2idx[word_b]]
    vec_c = embeddings[word2idx[word_c]]
    
    # B - A + C
    target_vec = vec_b - vec_a + vec_c
    
    # Find most similar
    similarities = F.cosine_similarity(
        target_vec.unsqueeze(0),
        embeddings,
        dim=1
    )
    
    # Exclude input words
    exclude_indices = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    
    top_indices = similarities.argsort(descending=True)
    
    results = []
    for idx in top_indices:
        if idx.item() in exclude_indices:
            continue
        word = idx2word[idx.item()]
        sim = similarities[idx].item()
        results.append((word, sim))
        if len(results) >= top_k:
            break
    
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Word2Vec: CBOW and Skip-gram")
    parser.add_argument("--model", "-m", type=str, default="both",
                        choices=["cbow", "skipgram", "both"],
                        help="Model to train (default: both)")
    parser.add_argument("--corpus", "-c", type=str, default=None,
                        help="Path to corpus file")
    parser.add_argument("--embed-dim", "-d", type=int, default=50,
                        help="Embedding dimension (default: 50)")
    parser.add_argument("--window", "-w", type=int, default=2,
                        help="Context window size (default: 2)")
    parser.add_argument("--epochs", "-e", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--neg-samples", "-n", type=int, default=5,
                        help="Negative samples for Skip-gram (default: 5)")
    return parser.parse_args()


def get_default_corpus() -> str:
    return """
    The king and queen ruled the kingdom with wisdom and grace.
    The prince and princess lived in the royal palace.
    A man and woman walked through the city streets.
    The boy and girl played in the garden together.
    Dogs and cats are popular pets for families.
    The cat chased the mouse around the house.
    Birds fly high in the blue sky above.
    Fish swim in the deep ocean waters below.
    Machine learning uses data to train models.
    Deep learning models learn representations automatically.
    Natural language processing understands human text.
    Neural networks have many layers of neurons.
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    Knowledge is power and wisdom is strength.
    Time flies like an arrow but fruit flies like a banana.
    """ * 50


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Word2Vec: CBOW and Skip-gram")
    print("=" * 60)
    
    # Load corpus
    if args.corpus:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            corpus = f.read()
    else:
        corpus = get_default_corpus()
    
    # Build vocabulary
    word2idx, idx2word, word_counts = build_word_vocab(corpus, min_freq=2)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Prepare words list
    words = [w.lower() for w in split_into_words(corpus) if w.strip() and w.isalpha()]
    words = [w for w in words if w in word2idx]
    print(f"Corpus size: {len(words)} words")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Train CBOW
    if args.model in ["cbow", "both"]:
        print("\n" + "=" * 60)
        print("Training CBOW Model")
        print("=" * 60)
        
        cbow_samples = create_cbow_samples(words, word2idx, args.window)
        print(f"CBOW samples: {len(cbow_samples)}")
        
        cbow_dataset = CBOWDataset(cbow_samples)
        cbow_loader = DataLoader(cbow_dataset, batch_size=args.batch_size, shuffle=True)
        
        cbow_model = CBOWModel(vocab_size, args.embed_dim)
        cbow_losses = train_cbow(cbow_model, cbow_loader, args.epochs, device=device)
        
        print("\nCBOW - Similar words to 'king':")
        cbow_embeddings = cbow_model.get_all_embeddings()
        similar = find_similar_words("king", word2idx, idx2word, cbow_embeddings)
        for word, sim in similar:
            print(f"  {word}: {sim:.4f}")
    
    # Train Skip-gram
    if args.model in ["skipgram", "both"]:
        print("\n" + "=" * 60)
        print("Training Skip-gram Model (with Negative Sampling)")
        print("=" * 60)
        
        sg_samples = create_skipgram_samples(words, word2idx, args.window)
        print(f"Skip-gram samples: {len(sg_samples)}")
        
        # Create dataset with negative sampling
        idx_to_freq = {word2idx[w]: c for w, c in word_counts.items() if w in word2idx}
        sg_dataset = SkipGramNegSamplingDataset(
            sg_samples, vocab_size, args.neg_samples, idx_to_freq
        )
        sg_loader = DataLoader(sg_dataset, batch_size=args.batch_size, shuffle=True)
        
        sg_model = SkipGramModel(vocab_size, args.embed_dim)
        sg_losses = train_skipgram_neg_sampling(sg_model, sg_loader, args.epochs, device=device)
        
        print("\nSkip-gram - Similar words to 'king':")
        sg_embeddings = sg_model.get_all_embeddings()
        similar = find_similar_words("king", word2idx, idx2word, sg_embeddings)
        for word, sim in similar:
            print(f"  {word}: {sim:.4f}")
    
    # Demonstrate word analogies
    print("\n" + "=" * 60)
    print("Word Analogies")
    print("=" * 60)
    
    if args.model in ["skipgram", "both"]:
        embeddings = sg_model.get_all_embeddings()
    else:
        embeddings = cbow_model.get_all_embeddings()
    
    # king - man + woman = queen
    print("\nAnalogy: king - man + woman = ?")
    analogy_results = word_analogy("man", "king", "woman", word2idx, idx2word, embeddings)
    for word, sim in analogy_results[:3]:
        print(f"  {word}: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("Embedding Statistics")
    print("=" * 60)
    print(f"Embedding shape: ({vocab_size}, {args.embed_dim})")
    print(f"Embedding norm (mean): {embeddings.norm(dim=1).mean():.4f}")
    
    # Save embeddings
    torch.save({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'embeddings': embeddings,
        'embed_dim': args.embed_dim,
    }, 'word2vec_embeddings.pt')
    print("\nEmbeddings saved to word2vec_embeddings.pt")


if __name__ == "__main__":
    main()
