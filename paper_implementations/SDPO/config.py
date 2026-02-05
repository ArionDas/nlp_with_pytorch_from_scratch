"""
Configuration classes for SDPO (Self-Distilled Policy Optimization).
"""
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class SelfDistillationConfig:
    """Configuration for self-distillation component."""
    # Core settings
    full_logit_distillation: bool = True  # Whether to use full-logit KL distillation
    alpha: float = 0.5  # KL interpolation coefficient: 0.0=forward KL, 1.0=reverse KL, 0.5=JSD
    success_reward_threshold: float = 1.0  # Minimum reward to be considered successful
    
    # Teacher regularization
    teacher_regularization: str = "ema"  # "ema" or "trust-region"
    teacher_update_rate: float = 0.05  # EMA update rate or trust-region mixing coefficient
    
    # Distillation options
    distillation_topk: Optional[int] = None  # Use top-k logits instead of full distribution
    distillation_add_tail: bool = True  # Add tail bucket for top-k distillation
    is_clip: Optional[float] = 2.0  # Importance sampling ratio clip value
    
    # Loss weights
    pg_loss_weight: float = 1.0  # Policy gradient loss weight
    distillation_weight: float = 1.0  # Self-distillation loss weight
    kl_penalty_weight: float = 0.01  # KL divergence penalty weight


@dataclass
class RepromptingConfig:
    """Configuration for reprompting/demonstration construction."""
    max_reprompt_len: int = 10240  # Maximum token length of reprompted prompt
    reprompt_truncation: str = "right"  # "left", "right", or "error"
    
    # Demonstration selection
    dont_reprompt_on_self_success: bool = True  # Don't use own success as demonstration
    remove_thinking_from_demonstration: bool = True  # Remove <think> tags
    
    # Feedback settings
    include_environment_feedback: bool = True
    environment_feedback_only_without_solution: bool = True
    
    # Templates
    solution_template: str = "<|solution|>\n{successful_previous_attempt}\n<|/solution|>\n"
    feedback_template: str = "<|feedback|>\n{feedback_raw}\n<|/feedback|>\n"
    reprompt_template: str = (
        "{prompt}\n\n"
        "Here is a previous attempt and feedback:\n"
        "{solution}"
        "{feedback}"
        "Now, try again with this knowledge.\n"
    )


@dataclass
class SDPOConfig:
    """Main configuration for SDPO training."""
    # Model settings
    model_name: str = "gpt2"  # Base model name
    device: str = "cuda"
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples_per_prompt: int = 4  # Number of rollouts per prompt (G in GRPO)
    
    # Rollout settings
    rollout_batch_size: int = 16
    
    # Sub-configs
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    reprompting: RepromptingConfig = field(default_factory=RepromptingConfig)
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


@dataclass
class RolloutResult:
    """Result of a single rollout."""
    prompt: str
    response: str
    full_sequence: str
    reward: float
    feedback: Optional[str] = None
    is_success: bool = False
