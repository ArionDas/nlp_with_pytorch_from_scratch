# Learning to Reason in 13 Parameters

> **Paper:** [Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118)
> **Authors:** John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar
> **Affiliations:** FAIR at Meta, Cornell University, Carnegie Mellon University
> **Published:** February 4, 2026

---

## Table of Contents

- [TL;DR](#tldr)
- [Motivation](#motivation)
- [Architecture Diagram](#architecture-diagram)
- [The Core Idea: TinyLoRA](#the-core-idea-tinylora)
  - [From LoRA to LoRA-XS to TinyLoRA](#from-lora-to-lora-xs-to-tinylora)
  - [TinyLoRA Parameterization](#tinylora-parameterization)
  - [Parameter Sharing (Weight Tying)](#parameter-sharing-weight-tying)
- [Why RL and Not SFT?](#why-rl-and-not-sft)
  - [Information-Theoretic Argument](#information-theoretic-argument)
  - [Signal Separation](#signal-separation)
- [GRPO: The RL Algorithm](#grpo-the-rl-algorithm)
- [Key Results](#key-results)
- [Ablations & Practical Guidelines](#ablations--practical-guidelines)
- [Implementation Details](#implementation-details)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Broader Implications](#broader-implications)
- [Citation](#citation)

---

## TL;DR

Can a language model learn to solve math problems by training **only 13 parameters** (26 bytes)?

**Yes.** This paper proposes **TinyLoRA**, a method that scales low-rank adapters down to as few as **one trainable parameter**. Combined with **reinforcement learning** (GRPO), TinyLoRA trains Qwen2.5-7B to **91% accuracy on GSM8K** while updating just 13 parameters in bf16. The key insight: RL provides a much sparser, cleaner gradient signal than supervised fine-tuning, enabling effective learning even with extreme parameter constraints.

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Headline Result                          │
│                                                                 │
│  Model:  Qwen2.5-7B-Instruct  (7.6 billion parameters)        │
│  Task:   GSM8K math word problems                               │
│  Method: TinyLoRA + GRPO                                        │
│                                                                 │
│  Trainable Parameters: 13          (out of 7,600,000,000)      │
│  Update Size:          26 bytes    (in bf16)                    │
│  Accuracy:             91%         (vs 95% full fine-tuning)   │
│  Compression Ratio:    ~585,000,000x                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Motivation

Modern reasoning models are trained through **Reinforcement Learning with Verifiable Rewards (RLVR)** — generating multiple candidate solutions, checking answers, and updating the model via policy gradient. While full fine-tuning updates billions of parameters, methods like **LoRA** reduce this to millions. But even LoRA at rank 1 requires ~3M parameters for an 8B model.

This paper asks: **How far can we go?**

The answer is surprisingly far. The authors observe that:

1. **RL makes fundamentally more information-dense updates than SFT** — the gradient signal is cleaner and sparser
2. **Larger models need proportionally fewer parameter updates** — the knowledge is already there, only the "style" needs adjusting
3. **Conventional LoRA can't scale below the model dimension** — new parameterizations are needed

These observations led to TinyLoRA, which breaks through the parameter floor imposed by standard LoRA.

---

## Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        TinyLoRA ARCHITECTURE OVERVIEW                             ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  ┌──────────────────────────── FROZEN PRETRAINED MODEL ─────────────────────────┐ ║
║  │                          (e.g., Qwen2.5-7B-Instruct)                         │ ║
║  │                                                                              │ ║
║  │   Layer 1             Layer 2             ...        Layer N                 │ ║
║  │  ┌─────────────┐   ┌─────────────┐              ┌─────────────┐              │ ║
║  │  │ Attention   │   │ Attention   │              │ Attention   │              │ ║
║  │  │ ┌─────────┐ │   │ ┌─────────┐ │              │ ┌─────────┐ │              │ ║
║  │  │ │ Q_proj  │←┼───┼─┼─ TINY  ─┼─┼──────────────┼─┼─ LORA  ─┼─┤              │ ║
║  │  │ │ K_proj  │←┤   │ │ adapters│ │              │ │ adapters│ │              │ ║
║  │  │ │ V_proj  │←┤   │ │ applied │ │              │ │ applied │ │              │ ║
║  │  │ │ O_proj  │←┤   │ │ to each │ │              │ │ to each │ │              │ ║
║  │  │ └─────────┘ │   │ └─────────┘ │              │ └─────────┘ │              │ ║
║  │  │ MLP         │   │ MLP         │              │ MLP         │              │ ║
║  │  │ ┌─────────┐ │   │ ┌─────────┐ │              │ ┌─────────┐ │              │ ║
║  │  │ │gate_proj│←┤   │ │gate_proj│←┤              │ │gate_proj│←┤              │ ║
║  │  │ │ up_proj │←┤   │ │ up_proj │←┤              │ │ up_proj │←┤              │ ║
║  │  │ │down_proj│←┤   │ │down_proj│←┤              │ │down_proj│←┤              │ ║
║  │  │ └─────────┘ │   │ └─────────┘ │              │ └─────────┘ │              │ ║
║  │  └─────────────┘   └─────────────┘              └─────────────┘              │ ║
║  │        7 adapted         7 adapted                    7 adapted              │ ║
║  │       modules/layer     modules/layer                modules/layer           │ ║
║  └──────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                   ║
║  ┌──────────────────── INSIDE ONE TinyLoRA-ADAPTED LINEAR ──────────────────────┐ ║
║  │                                                                              │ ║
║  │   Input x ──┬──────────────────────────────────────┬──→ Output               │ ║
║  │             │            FROZEN PATH               │                         │ ║
║  │             │         x @ W^T + bias               │                         │ ║
║  │             │                                      │                         │ ║
║  │             │         TinyLoRA PATH (α-scaled)     │                         │ ║
║  │             └── x @ ΔW^T ──────────────────────────┘                         │ ║
║  │                    │                                                         │ ║
║  │                  ΔW = U @ Σ @ R @ V^T                                        │ ║
║  │                          │                                                   │ ║
║  │          ┌───────────────┼───────────────┐                                   │ ║
║  │          │     R = Σᵢ vᵢ · Pᵢ            │                                   │ ║
║  │          │         │       │             │                                   │ ║
║  │          │    TRAINABLE   FROZEN         │                                   │ ║
║  │          │    v ∈ Rᵘ      P ∈ Rᵘˣʳˣʳ     │                                   │ ║
║  │          │  (u params!)  (random,fixed)  │                                   │ ║
║  │          └───────────────────────────────┘                                   │ ║
║  │                                                                              │ ║
║  │   Frozen components from SVD(W):                                             │ ║
║  │     U ∈ R^(d_out × r)   — left singular vectors                              │ ║
║  │     Σ ∈ R^(r × r)       — singular values (diagonal)                         │ ║
║  │     V ∈ R^(d_in × r)    — right singular vectors                             │ ║
║  │                                                                              │ ║
║  └──────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                   ║
║  ┌──────────────────── WEIGHT TYING (KEY TO 13 PARAMS) ─────────────────────────┐ ║
║  │                                                                              │ ║
║  │   Without tying: each module gets its own v → N_layers × 7 × u params        │ ║
║  │                                                                              │ ║
║  │   With n_tie: every n_tie consecutive modules SHARE the same v               │ ║
║  │                                                                              │ ║
║  │   Example (28 layers × 7 modules = 196 modules, u=1):                        │ ║
║  │                                                                              │ ║
║  │    n_tie=1   → 196 unique v's → 196 params                                   │ ║
║  │    n_tie=7   → 28 shared v's  → 28 params   (per-layer sharing)              │ ║
║  │    n_tie=14  → ~14 shared v's → ~14 params                                   │ ║
║  │    n_tie=15  → 13 shared v's  → 13 params   ← THE HEADLINE!                  │ ║
║  │    n_tie=196 → 1 shared v     → 1 param     (full sharing)                   │ ║
║  │                                                                              │ ║
║  │   Each shared v is projected through DIFFERENT random P tensors per module,  │ ║
║  │   so the same v produces DIFFERENT ΔW updates at each adapted layer.         │ ║
║  │                                                                              │ ║
║  └──────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                   ║
║  ┌──────────────────── GRPO TRAINING LOOP ──────────────────────────────────────┐ ║
║  │                                                                              │ ║
║  │   for each epoch:                                                            │ ║
║  │     for each batch of prompts x₁...xₙ:                                        │ ║
║  │       │                                                                      │ ║
║  │       ├─ 1. GENERATE: sample k completions per prompt from π_θ               │ ║
║  │       │      y_{i,1}...y_{i,k} ~ π_θ(·|xᵢ)                                   │ ║
║  │       │                                                                      │ ║
║  │       ├─ 2. REWARD: score each completion (exact match)                      │ ║
║  │       │      rᵢⱼ = verify(xᵢ, yᵢⱼ) ∈ {0, 1}                                  │ ║
║  │       │                                                                      │ ║
║  │       ├─ 3. ADVANTAGE: group-relative normalization                          │ ║
║  │       │      Âᵢⱼ = (rᵢⱼ - mean(rᵢ)) / std(rᵢ)                                │ ║
║  │       │      (No separate critic needed!)                                    │ ║
║  │       │                                                                      │ ║
║  │       ├─ 4. POLICY GRADIENT: clipped surrogate objective                     │ ║
║  │       │      ratio = π_θ(y)/π_old(y)                                         │ ║
║  │       │      L = -min(ratio·Â, clip(ratio,1±ε)·Â)                            │ ║
║  │       │                                                                      │ ║
║  │       ├─ 5. KL PENALTY (optional): β · KL(π_θ || π_ref)                      │ ║
║  │       │                                                                      │ ║
║  │       └─ 6. UPDATE: backprop through only the u trainable params!            │ ║
║  │              ∇_v L → update v (13 scalars get gradients)                     │ ║
║  │                                                                              │ ║
║  └──────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## The Core Idea: TinyLoRA

### From LoRA to LoRA-XS to TinyLoRA

The paper builds on a progression of increasingly parameter-efficient methods:

| Method | Update Formula | Params per Module | For 7B Model (196 modules) |
|--------|---------------|-------------------|---------------------------|
| **Full FT** | W' = W + ΔW | d × k | ~7.6B |
| **LoRA** | W' = W + A·B | d × r + r × k ≈ O(d·r) | ~3M (r=1) |
| **LoRA-XS** | W' = W + U·Σ·R·V^T | r² | 196 (r=1) |
| **TinyLoRA** | W' = W + U·Σ·(Σᵢ vᵢPᵢ)·V^T | u | **1 – 196** (u=1) |

The key bottleneck in LoRA: parameter count scales as O(d·r), where d is the model dimension (e.g., 4096 for 7B models). Even at r=1, you need ~4096 params per module.

**LoRA-XS** breaks this by working in the SVD subspace of W:
```
W' = W + U @ Σ @ R @ V^T
         ─────   ─   ─────
         frozen  train frozen
```
Where U, Σ, V are from the truncated SVD of W, and only R ∈ R^(r×r) is trained. This learns to *recombine the dominant singular directions* of W. At r=1, R is a single scalar.

**TinyLoRA** goes further by replacing R with a *projected* representation:
```
R = Σᵢ vᵢ · Pᵢ     where v ∈ Rᵘ (trainable), Pᵢ ∈ R^(r×r) (frozen random)
```

### TinyLoRA Parameterization

The mathematical formulation:

$$W' = W + \alpha \cdot U \Sigma \left(\sum_{i=1}^{u} v_i P_i\right) V^\top$$

Where:
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$ — frozen pretrained weight
- $U \in \mathbb{R}^{d_{out} \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$, $V \in \mathbb{R}^{d_{in} \times r}$ — from truncated SVD of $W$ (frozen)
- $P_i \in \mathbb{R}^{r \times r}$ — fixed random projection matrices (frozen, unique per module)
- $v \in \mathbb{R}^u$ — **the only trainable parameters** ($u$ scalars)
- $\alpha$ — scaling factor

The frozen rank $r = 2$ was found optimal in ablations. Higher ranks introduce too many degrees of freedom in the frozen components, making the optimization of tiny $v$ harder.

### Parameter Sharing (Weight Tying)

Even with $u = 1$ per module, a model with 28 layers × 7 modules = 196 modules would need 196 parameters. To go lower, **weight tying** shares the same $v$ across groups of modules:

$$\text{Total params} = \frac{n_{layers} \times m_{modules} \times u}{n_{tie}}$$

Crucially, even though tied modules share the same $v$, they produce **different weight updates** because each module has its own unique frozen random projections $P_i$.

**Practical guideline from ablations:** Exhaust the $u$ budget first (down to $u = 1$) before increasing $n_{tie}$. In other words, fewer unique parameters is better than more shared ones.

---

## Why RL and Not SFT?

This is the paper's deepest insight. TinyLoRA only works with RL. The same configurations fail badly with SFT.

### Information-Theoretic Argument

**SFT training data** consists of (prompt, full demonstration) pairs:
- The model must memorize the *entire* output sequence $y$
- It cannot distinguish which features of $y$ are task-relevant (how to solve math) vs. irrelevant (specific phrasing, formatting noise)
- Information content: **all tokens of y** must be absorbed → requires high model capacity

**RL training data** consists of (prompt, k completions, k binary rewards):
- The relevant information is the **reward signal**: just $k \cdot H(\mathcal{R})$ bits per prompt
- When reward is binary: at most **k bits** of useful information per prompt
- This is orders of magnitude less than SFT's requirement

### Signal Separation

RL has a built-in mechanism for separating signal from noise:

- **Task-relevant features** correlate with the reward $r$ → signal accumulates across samples
- **Task-irrelevant features** do not correlate → noise cancels via resampling

SFT lacks this separation: without reward annotation, the model treats all tokens as equally informative, wasting capacity on irrelevant details.

**Empirical evidence:**

| Update Size | RL (GRPO) Accuracy | SFT Accuracy |
|-------------|-------------------|--------------|
| 13 params | **91%** | 83% |
| 120 params | **95%** | 84% |
| 1M params | **95%** | ~93% |
| Full FT | 95% | ~95% |

SFT requires **100–1000x larger updates** to match RL performance.

---

## GRPO: The RL Algorithm

**Group Relative Policy Optimization** (Shao et al., 2024) is the core training algorithm:

1. **Sample** $k$ completions per prompt from current policy $\pi_\theta$
2. **Score** each with verifiable reward (exact-match answer checking)
3. **Normalize** advantages within each group (no critic needed!):

$$\hat{A}_{ij} = \frac{r_{ij} - \text{mean}(r_{i,1:k})}{\text{std}(r_{i,1:k}) + \epsilon}$$

4. **Update** with clipped policy gradient:

$$\mathcal{L} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(y)}{\pi_{old}(y)} \hat{A}, \; \text{clip}\left(\frac{\pi_\theta(y)}{\pi_{old}(y)}, 1 \pm \epsilon\right) \hat{A}\right)\right]$$

5. **Optional KL penalty** against reference policy: $\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$

The crucial point: gradients from GRPO flow back through all frozen components ($U, \Sigma, P, V$) and only update $v$ — the 13 scalars that control the entire model's reasoning behavior.

---

## Key Results

### GSM8K (Math Word Problems)

| Method | Params Trained | Accuracy | Update Size |
|--------|---------------|----------|-------------|
| Base (no training) | 0 | 76% | 0 bytes |
| **TinyLoRA + GRPO** | **1** | **~82%** | **2 bytes** |
| **TinyLoRA + GRPO** | **13** | **91%** | **26 bytes** |
| **TinyLoRA + GRPO** | **120** | **95%** | **240 bytes** |
| LoRA-XS + GRPO | 196 | 95% | 392 bytes |
| LoRA + GRPO (r=1) | ~3M | 95% | ~6 MB |
| Full Fine-tuning | ~7.6B | 95% | ~15 GB |

### MATH & Hard Benchmarks (Qwen2.5-7B-Instruct)

| # Params | GSM8K | MATH500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|----------|-------|---------|---------|---------------|--------|-------|-----|
| 0 (base) | 88.2 | 64.6 | 25.7 | 30.1 | 3.3 | 30.0 | 40.3 |
| **13** | **91.8** | **74.6** | **27.1** | **36.3** | **16.0** | **54.5** | **50.1** |
| 196 | 92.2 | 76.6 | 37.1 | 38.8 | 16.7 | 57.5 | 53.2 |
| 6,272 | 91.9 | 78.0 | 37.5 | 41.0 | 16.7 | 57.5 | 53.8 |
| Full FT | 91.7 | 78.2 | 38.6 | 40.4 | 20.0 | 62.5 | 55.2 |

With 196 params, TinyLoRA retains **87% of the absolute performance improvement** of full fine-tuning across six benchmarks.

### Scaling Across Models

Larger models → fewer parameters needed to reach peak performance:

```
Model Size    Params for 95% of Full FT Performance
─────────     ─────────────────────────────────────
1.5B          ~10,000
3B            ~1,000
7B            ~120
8B            ~100
70B           ~50  (projected)
```

This suggests trillion-scale models may be trainable with **single-digit parameters**.

---

## Ablations & Practical Guidelines

### Frozen Rank $r$

| Frozen Rank | Performance |
|-------------|-------------|
| r = 1 | Good |
| **r = 2** | **Best** (default) |
| r = 4 | Slightly worse |
| r = 8+ | Degrades |

Higher ranks introduce too many degrees of freedom in the frozen SVD components, making optimization of tiny $v$ harder.

### Trading Off $u$ vs $n_{tie}$

For a fixed parameter budget:
- **More $u$, less $n_{tie}$** → better (more expressive per-module updates)
- **Less $u$, more $n_{tie}$** → worse (forced sharing limits expressivity)

**Rule:** Reduce $u$ to 1 before increasing $n_{tie}$.

### Precision in Byte-Constrained Regime

When the constraint is total bytes (not parameters):

| Precision | Bytes/Param | Performance (bit-for-bit) |
|-----------|-------------|--------------------------|
| fp16 | 2 | Good |
| bf16 | 2 | Good |
| **fp32** | **4** | **Best** (surprisingly!) |

fp32 outperforms bf16/fp16 even after accounting for its 2x byte cost.

### Sharing Strategy

| Strategy | Description | Performance |
|----------|-------------|-------------|
| Structured | Same-type modules share (all Q_proj together) | Worse |
| **Tiled** | Nearby modules share regardless of type | **Better** |

---

## Implementation Details

### vLLM Compatibility

Since vLLM only supports standard LoRA with rank ≥ 4, the paper merges TinyLoRA weights into the base model at each training step:

1. Compute $W_{merged} = W + \Delta W_{TinyLoRA}$
2. Use merged weights for vLLM inference (generation)
3. Use true TinyLoRA weights for the final forward pass (gradient computation)
4. Apply **truncated importance sampling** to handle the numerical mismatch

### Hyperparameter Sweep

Learning rates swept: $\{10^{-7}, 5 \times 10^{-7}, 10^{-6}, 5 \times 10^{-6}, 10^{-5}, 10^{-4}, 2 \times 10^{-4}\}$

Best LR is selected per update size, averaged over 3 random seeds.

---

## File Structure

```
reason_in_13_params/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses (TinyLoRA, LoRA-XS, GRPO)
├── tiny_lora.py         # Core TinyLoRA layer + parameter group manager
├── lora_xs.py           # LoRA-XS baseline for comparison
├── model.py             # Model wrapper: inject adapters, count params
├── grpo.py              # GRPO trainer (conceptual)
├── example.py           # End-to-end demo on a toy transformer
└── README.md            # This file
```

---

## Usage

### Quick Demo

```bash
cd nlp_with_pytorch_from_scratch
python -m paper_implementations.reason_in_13_params.example
```

### Applying TinyLoRA to a Model

```python
import torch
from paper_implementations.reason_in_13_params import (
    TinyLoRAConfig, apply_tiny_lora, count_trainable_params
)

# Load your pretrained model
model = load_your_model()  # e.g., Qwen2.5-7B-Instruct

# Freeze base weights
for p in model.parameters():
    p.requires_grad = False

# Apply TinyLoRA — 13 params like the paper!
config = TinyLoRAConfig(
    frozen_rank=2,        # r=2 (optimal from ablations)
    trainable_dim=1,      # u=1 (one scalar per projection basis)
    n_tie=15,             # Share across groups of 15 modules
    target_modules=[      # 7 modules per transformer block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ],
)
model = apply_tiny_lora(model, config)

# Verify
stats = count_trainable_params(model)
print(f"Trainable: {stats['trainable']} params ({stats['trainable_bytes_bf16']} bytes)")
```

### Training with GRPO

```python
from paper_implementations.reason_in_13_params import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    group_size=4,       # k=4 completions per prompt
    kl_coeff=0.0,       # No KL penalty for GSM8K
    clip_eps=0.2,       # PPO-style clipping
    lr=1e-5,
)

# In your training loop:
metrics = trainer.training_step(
    log_probs=current_log_probs,
    old_log_probs=old_log_probs,
    rewards=binary_rewards,
)
```

---

## Broader Implications

### What Does This Tell Us?

1. **Knowledge vs. Style:** The knowledge to solve math problems is already in the pretrained model. TinyLoRA only changes the model's "style" — how it formats and presents reasoning. This ~13-parameter update likely teaches the model to *generate longer, more structured outputs* rather than encoding new mathematical knowledge.

2. **RL is Fundamentally Different from SFT:** RL makes sparse, high-precision updates that target exactly what matters. SFT wastes capacity on irrelevant details. This has profound implications for how we think about fine-tuning.

3. **Scaling Laws for Adaptation:** Larger models need smaller relative updates. As models scale to trillions of parameters, we may need only a handful of parameters to adapt them to new tasks — enabling extreme personalization and multi-tenant serving.

4. **Qwen vs. LLaMA:** Qwen models are ~10x more parameter-efficient than LLaMA at the same update size, suggesting pretraining details significantly affect adaptability.

5. **The Future of LoRA Serving:** If adapters are truly this small (26 bytes!), a single GPU could serve millions of personalized model variants simultaneously.

### Limitations

- Results are limited to **math reasoning tasks** — may not generalize to creative writing, science, or code
- Extreme parameter efficiency may be partially explained by **data contamination** (models may have seen similar problems during pretraining)
- The method relies on the base model already having strong latent capabilities

---

## Citation

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X. and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```
