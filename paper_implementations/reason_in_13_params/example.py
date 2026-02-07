"""
Example — End-to-end demonstration of TinyLoRA on a toy transformer.

This script demonstrates the complete pipeline described in the paper:
  1. Build a small transformer model (standing in for Qwen2.5-7B)
  2. Freeze base weights
  3. Apply TinyLoRA with extreme parameter efficiency
  4. Train using GRPO with verifiable rewards
  5. Compare parameter counts across LoRA, LoRA-XS, and TinyLoRA

Since this is a conceptual demo, we use a tiny 2-layer transformer
and a synthetic "math reasoning" task.

Usage:
    python -m paper_implementations.reason_in_13_params.example
"""

import torch
import torch.nn as nn
import math
from typing import Dict

from .config import TinyLoRAConfig, LoRAXSConfig, GRPOConfig
from .tiny_lora import TinyLoRALinear
from .lora_xs import LoRAXSLinear
from .model import (
    apply_tiny_lora, apply_lora_xs, count_trainable_params,
    freeze_base_model, print_adapter_summary,
)
from .grpo import GRPOTrainer


# ─────────────────────────────────────────────────────────────
# Toy Transformer (stands in for Qwen2.5-7B in the real paper)
# ─────────────────────────────────────────────────────────────

class ToyAttention(nn.Module):
    """Minimal multi-head attention with named projections matching real models."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class ToyMLP(nn.Module):
    """Minimal gated MLP matching LLaMA/Qwen architecture."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = ToyAttention(d_model, n_heads)
        self.mlp = ToyMLP(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyTransformer(nn.Module):
    """
    Tiny transformer model for demonstrating TinyLoRA.

    Real paper uses Qwen2.5-7B-Instruct (7.6B params, 28 layers, 7 modules/layer).
    This toy version has configurable layers to show the same concepts.
    """
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, d_ff=512, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ToyTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────────────────────
# Demonstration functions
# ─────────────────────────────────────────────────────────────

def demo_parameter_comparison():
    """
    Compare parameter counts across adaptation methods.

    Reproduces the key insight from Table 1 of the paper:
      Full FT >> LoRA >> LoRA-XS >> TinyLoRA
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON ACROSS ADAPTATION METHODS")
    print("(Conceptual demo with toy transformer)")
    print("=" * 70)

    configs = {
        "TinyLoRA (u=1, full tying)": ("tinylora", TinyLoRAConfig(
            frozen_rank=2, trainable_dim=1,
            n_tie=4 * 7  # 4 layers × 7 modules = full sharing
        )),
        "TinyLoRA (u=1, per-layer tying)": ("tinylora", TinyLoRAConfig(
            frozen_rank=2, trainable_dim=1,
            n_tie=7  # share within each layer's 7 modules
        )),
        "TinyLoRA (u=1, no tying)": ("tinylora", TinyLoRAConfig(
            frozen_rank=2, trainable_dim=1,
            n_tie=1
        )),
        "TinyLoRA (u=4, no tying)": ("tinylora", TinyLoRAConfig(
            frozen_rank=2, trainable_dim=4,
            n_tie=1
        )),
        "LoRA-XS (r=1)": ("loraxs", LoRAXSConfig(rank=1)),
        "LoRA-XS (r=4)": ("loraxs", LoRAXSConfig(rank=4)),
    }

    for name, (method, config) in configs.items():
        model = ToyTransformer(n_layers=4  )
        freeze_base_model(model)

        if method == "tinylora":
            apply_tiny_lora(model, config)
        else:
            apply_lora_xs(model, config)

        stats = count_trainable_params(model)
        print(f"\n  {name}:")
        print(f"    Trainable:  {stats['trainable']:>8,} params  "
              f"({stats['trainable_bytes_bf16']:>6,} bytes in bf16)")
        print(f"    Total:      {stats['total']:>8,} params")
        print(f"    Compression: {stats['compression_ratio']:>8,.0f}x")


def demo_grpo_training():
    """
    Demonstrate a GRPO training step with TinyLoRA.

    Shows the complete pipeline:
      1. Create model and apply TinyLoRA
      2. Generate synthetic "completions" with rewards
      3. Run one GRPO training step
      4. Observe that only the tiny v vectors change
    """
    print("\n" + "=" * 70)
    print("GRPO TRAINING STEP WITH TINYLORA")
    print("=" * 70)

    # Create model with TinyLoRA
    model = ToyTransformer(n_layers=4)
    freeze_base_model(model)

    # 13 params like the paper's headline result!
    # 4 layers × 7 modules = 28 modules, n_tie=2 → 14 groups, u=1 → 14 params
    # (Close to 13 — exact 13 requires specific architecture tuning)
    config = TinyLoRAConfig(
        frozen_rank=2,
        trainable_dim=1,
        n_tie=2,
    )
    apply_tiny_lora(model, config)
    print_adapter_summary(model)

    # Snapshot trainable params before update
    v_before = {
        name: p.clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    # Create GRPO trainer
    grpo_config = GRPOConfig(group_size=4, lr=1e-3)
    trainer = GRPOTrainer(
        model=model,
        group_size=grpo_config.group_size,
        lr=grpo_config.learning_rate,
    )

    # Simulate one training step with synthetic data
    batch_size = 4
    group_size = grpo_config.group_size

    # Synthetic log-probs and rewards (standing in for real generation)
    log_probs = torch.randn(batch_size, group_size, requires_grad=True)
    old_log_probs = log_probs.detach() + 0.01 * torch.randn_like(log_probs)
    rewards = torch.bernoulli(0.5 * torch.ones(batch_size, group_size))

    print(f"Synthetic rewards:\n{rewards}")
    print(f"Mean reward: {rewards.mean():.2f}")

    # Run training step
    metrics = trainer.training_step(log_probs, old_log_probs, rewards)

    print("\n  Training metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    # Show that v vectors changed
    print("\n  Parameter updates (v vectors):")
    for name, p in model.named_parameters():
        if p.requires_grad and name in v_before:
            delta = (p - v_before[name]).abs().sum().item()
            print(f"    {name}: Δ = {delta:.8f}")


def demo_weight_merging():
    """
    Demonstrate weight merging for inference.

    The paper describes merging LoRA weights into the base model
    at each training step for vLLM inference compatibility.
    """
    print("\n" + "=" * 70)
    print("WEIGHT MERGING FOR INFERENCE")
    print("=" * 70)

    model = ToyTransformer(n_layers=2)
    freeze_base_model(model)

    config = TinyLoRAConfig(frozen_rank=2, trainable_dim=1, n_tie=1)
    apply_tiny_lora(model, config)

    # Manually set a non-zero v to show the merge effect
    for module in model.modules():
        if isinstance(module, TinyLoRALinear):
            with torch.no_grad():
                module.v.fill_(0.5)
            break

    # Show that merged weight differs from original
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALinear):
            original_norm = module.weight.data.norm().item()
            merged = module.merged_weight()
            merged_norm = merged.norm().item()
            delta_norm = (merged - module.weight.data).norm().item()
            print(f"  {name}:")
            print(f"    |W_original| = {original_norm:.4f}")
            print(f"    |W_merged|   = {merged_norm:.4f}")
            print(f"    |ΔW|         = {delta_norm:.6f}")
            break


def demo_sft_vs_rl_insight():
    """
    Illustrate the paper's key theoretical insight:
    why RL needs fewer parameters than SFT.

    SFT must memorize entire demonstrations → high information content
    RL only needs the reward bit per completion → sparse, clean signal
    """
    print("\n" + "=" * 70)
    print("KEY INSIGHT: WHY RL NEEDS FEWER PARAMETERS THAN SFT")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Information Content Analysis                   │
    ├────────────────────┬────────────────────────────────────────────┤
    │                    │                                            │
    │   SFT Training     │   RL (GRPO) Training                      │
    │                    │                                            │
    │   Each sample:     │   Each sample:                             │
    │   (prompt, full    │   (prompt, k completions, k reward bits)   │
    │    demonstration)  │                                            │
    │                    │                                            │
    │   Information:     │   Information:                             │
    │   ALL tokens of y  │   Only k × H(R) bits (≤ k bits)           │
    │   are equally      │   when reward is binary                   │
    │   weighted — model │                                            │
    │   must memorize    │   Signal is CLEANLY SEPARATED:             │
    │   EVERYTHING       │   - Task-relevant features correlate       │
    │   (signal + noise) │     with reward                            │
    │                    │   - Noise cancels via resampling            │
    │   Needs: ~1M+      │                                            │
    │   parameters       │   Needs: as few as 13 parameters!          │
    │                    │                                            │
    ├────────────────────┴────────────────────────────────────────────┤
    │                                                                 │
    │   Paper result on GSM8K (Qwen2.5-7B-Instruct):                  │
    │                                                                 │
    │   Method          13 params    120 params    Full FT             │
    │   ─────────────   ─────────    ──────────    ────────            │
    │   RL (GRPO)       91%          95%           95%                 │
    │   SFT             83%          84%           ~95%                │
    │                                                                 │
    │   → RL achieves 95% of full FT with just 120 parameters!        │
    │   → SFT needs 100-1000x MORE parameters for same performance    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 70)
    print("  Learning to Reason in 13 Parameters — Conceptual Demo")
    print("  Paper: Morris et al., 2026 (arXiv:2602.04118)")
    print("█" * 70)

    demo_parameter_comparison()
    demo_grpo_training()
    demo_weight_merging()
    demo_sft_vs_rl_insight()

    print("\n" + "█" * 70)
    print("  Demo complete! See README.md for full paper analysis.")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
