"""
Configuration dataclasses for TinyLoRA, LoRA-XS, and GRPO.

These mirror the key hyperparameters discussed in the paper:
  - Frozen SVD rank r (paper uses r=2 by default)
  - Trainable projection dimension u (as few as 1)
  - Weight tying factor n_tie (modules sharing a single v)
  - Target modules in the transformer (Q, K, V, O, up, down, gate)
  - GRPO sampling parameters (group size k, temperature, KL coeff)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TinyLoRAConfig:
    """
    Configuration for TinyLoRA — the core contribution of the paper.

    TinyLoRA replaces the trainable r×r matrix R in LoRA-XS with a tiny
    trainable vector v ∈ R^u projected through fixed random tensors P_i ∈ R^(r×r).

    The weight update becomes:
        W' = W + U @ Σ @ (Σ_i v_i * P_i) @ V^T

    Attributes:
        frozen_rank (int): Rank of the truncated SVD (r). Paper default: 2.
            Higher ranks give more expressive directions but make optimizing
            the small v harder. r=2 was found optimal in ablations.

        trainable_dim (int): Dimension of the trainable vector v (u).
            This is the number of trainable parameters PER module (before tying).
            Can be as small as 1.

        n_tie (int): Weight tying factor — number of modules sharing a single v.
            With n_tie = n_layers * n_modules_per_layer, the entire model shares
            one v, reducing total params to just u.
            Set to 1 for no sharing (each module gets its own v).

        target_modules (List[str]): Which linear layers to adapt.
            Standard transformer has 7 per block: q, k, v, o (attention)
            + up, down, gate (MLP).

        random_seed (int): Seed for generating fixed random projection tensors P.
            These are frozen and never updated — only v is trained.

        alpha (float): Scaling factor for the low-rank update, analogous to
            LoRA's alpha / rank scaling.
    """
    frozen_rank: int = 2
    trainable_dim: int = 1
    n_tie: int = 1  # 1 = no tying; higher = more sharing
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ])
    random_seed: int = 42
    alpha: float = 1.0


@dataclass
class LoRAXSConfig:
    """
    Configuration for LoRA-XS baseline (Bałazy et al., 2025).

    LoRA-XS decomposes the weight update using the truncated SVD of W:
        W' = W + U @ Σ @ R @ V^T
    where only R ∈ R^(r×r) is trainable, and U, Σ, V are frozen.

    This reduces per-module params from O(d*r) [LoRA] to O(r^2) [LoRA-XS].
    At r=1, each module has exactly 1 trainable parameter.

    Attributes:
        rank (int): Rank of the truncated SVD / size of R.
        target_modules (List[str]): Linear layers to adapt.
        alpha (float): Scaling factor for the update.
    """
    rank: int = 1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ])
    alpha: float = 1.0


@dataclass
class GRPOConfig:
    """
    Configuration for Group Relative Policy Optimization (GRPO).

    GRPO (Shao et al., 2024) is the RL algorithm used throughout the paper.
    It generates k completions per prompt, computes group-relative advantages,
    and optimizes via policy gradient — no separate critic model needed.

    The paper's key finding: GRPO enables learning with orders of magnitude
    fewer parameters than SFT because RL provides a sparser, cleaner signal
    (just the reward bit) vs. SFT which must memorize full demonstrations.

    Attributes:
        group_size (int): Number of completions sampled per prompt (k).
            Paper uses k=4 for GSM8K, k=8 for MATH.

        temperature (float): Sampling temperature for generating completions.

        kl_coeff (float): KL penalty coefficient between current and reference
            policy. 0.0 for GSM8K experiments, 0.001 for MATH.

        max_gen_length (int): Maximum generation length in tokens.

        clip_eps (float): PPO-style clipping epsilon for the policy ratio.

        learning_rate (float): Learning rate. Paper sweeps over
            {1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4}.

        num_epochs (int): Number of training epochs (paper uses 3).

        batch_size (int): Batch size (64 for GSM8K, 256 for MATH).

        use_truncated_is (bool): Whether to use truncated importance sampling
            to handle train/inference weight mismatch (from merging LoRA
            weights for vLLM inference).
    """
    group_size: int = 4
    temperature: float = 1.0
    kl_coeff: float = 0.0
    max_gen_length: int = 4096
    clip_eps: float = 0.2
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 64
    use_truncated_is: bool = True
    max_grad_norm: float = 1.0
    reward_type: str = "exact_match"  # binary exact-match reward
