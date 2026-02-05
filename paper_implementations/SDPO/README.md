# SDPO: Self-Distilled Policy Optimization

Conceptual implementation of **"Reinforcement Learning via Self-Distillation"** (arXiv:2601.20802).

## Overview

SDPO addresses the credit assignment problem in RL for LLMs by using **self-distillation from high-reward trajectories**. Instead of learning only from scalar rewards, SDPO:

1. Generates multiple rollouts per prompt
2. Identifies successful (high-reward) examples
3. **Reprompts** the model with these successes as demonstrations
4. Treats the feedback-conditioned model as a **teacher** for self-distillation
5. Updates the policy using both policy gradients and KL-regularized distillation

## Key Components

### 1. Policy Model (`policy_model.py`)
- Wraps a transformer LM for both student (policy) and teacher roles
- Teacher is maintained as an **EMA** (exponential moving average) of the student
- Supports generation with configurable sampling parameters

### 2. Reward Model (`reward_model.py`)
- Abstract interface for verifiable reward computation
- Includes example implementations:
  - `MathRewardModel`: Extracts and evaluates math answers
  - `RuleBasedCodeRewardModel`: Checks code structure and syntax
  - `VerifiableRewardModel`: Pattern-based verification

### 3. Reprompter (`reprompting.py`)
- Constructs demonstration-augmented prompts
- Selects successful examples from rollout batch
- Formats solutions and feedback using configurable templates
- Supports filtering (e.g., removing `<think>` tags)

### 4. Self-Distillation Loss (`self_distillation.py`)
- Computes **KL divergence** between student and teacher distributions
- Supports multiple variants:
  - Forward KL: `KL(teacher || student)`
  - Reverse KL: `KL(student || teacher)`  
  - JSD (Jensen-Shannon): Interpolated between both
- Optional **top-k distillation** for efficiency
- **Importance sampling** weighting with clipping

### 5. Trainer (`trainer.py`)
- Main training loop implementing the SDPO algorithm:
  ```
  For each batch:
    1. Generate G rollouts per prompt
    2. Compute rewards R(x, y)
    3. Identify successes: {y | R(x, y) >= threshold}
    4. Construct reprompted inputs with demonstrations
    5. Compute losses:
       - L_pg: Policy gradient on original rollouts
       - L_distill: KL(student||teacher) on reprompted inputs
       - L_kl: Reference model KL penalty
    6. Update student: θ ← θ - α∇(L_pg + L_distill + L_kl)
    7. Update teacher EMA: φ ← (1-β)φ + βθ
  ```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SDPO Trainer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Prompts → [Generate G rollouts] → {y₁, y₂, ..., y_G}          │
│                ↓                                                │
│          [Reward Model] → {r₁, r₂, ..., r_G}                    │
│                ↓                                                │
│   Successes: {y_success | r ≥ threshold}                        │
│                ↓                                                │
│   [Reprompter] → Augmented prompt with demonstrations           │
│                ↓                                                │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ Student(θ): forward on original + reprompted        │       │
│   │ Teacher(φ): forward on reprompted (EMA of θ)        │       │
│   └─────────────────────────────────────────────────────┘       │
│                ↓                                                │
│   Loss = PG_loss(θ) + distill(θ, φ) + KL_penalty(θ, φ)          │
│                ↓                                                │
│   Update θ, then φ ← EMA(θ)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from sdpo import SDPOConfig, PolicyModel, MathRewardModel, SDPOTrainer

# Configuration
config = SDPOConfig(
    model_name="gpt2",
    num_samples_per_prompt=4,  # G in GRPO
    self_distillation__success_reward_threshold=0.8,
    self_distillation__alpha=0.5,  # JSD
)

# Initialize components
policy_model = PolicyModel(config.model_name, device="cuda")
reward_model = MathRewardModel()

# Create trainer
trainer = SDPOTrainer(
    config=config,
    policy_model=policy_model,
    reward_model=reward_model,
    train_dataloader=dataloader
)

# Train
trainer.train()
```

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_samples_per_prompt` | Rollouts per prompt (G) | 4 |
| `success_reward_threshold` | Minimum reward for success | 1.0 |
| `alpha` | KL interpolation (0=forward, 1=reverse, 0.5=JSD) | 0.5 |
| `teacher_update_rate` | EMA decay rate (β) | 0.05 |
| `distillation_weight` | Weight for distillation loss | 1.0 |
| `pg_loss_weight` | Weight for policy gradient loss | 1.0 |
| `kl_penalty_weight` | Weight for reference KL penalty | 0.01 |

## Differences from Paper

This is a **conceptual implementation** focused on algorithmic clarity:

1. **Simplified distributed training**: No Ray/FSDP (single-GPU friendly)
2. **No vLLM/SGLang**: Uses HuggingFace generate() 
3. **Basic advantage estimation**: Simplified GRPO without full PPO
4. **Example reward models**: Rule-based rather than learned verifiers

For the full production implementation with distributed training, see the [official repository](https://github.com/lasgroup/SDPO).

## Paper Reference

```bibtex
@article{hubotter2026reinforcement,
  title={Reinforcement Learning via Self-Distillation},
  author={Hübotter, Jonas and Lübeck, Frederike and Behric, Lejs and 
          Baumann, Anton and Bagatella, Marco and Marta, Daniel and 
          Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and 
          Guestrin, Carlos and Krause, Andreas},
  year={2026},
  journal={arXiv preprint arXiv:2601.20802},
}
```

## License

This implementation is for educational purposes. See the original paper repository for the official Apache-2.0 licensed code.
