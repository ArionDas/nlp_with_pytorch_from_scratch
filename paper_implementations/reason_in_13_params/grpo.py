"""
Group Relative Policy Optimization (GRPO) — Conceptual Implementation.

GRPO (Shao et al., 2024) is the RL algorithm used throughout the paper.
It is the *critical enabler* of TinyLoRA's success: the paper's central
finding is that RL (GRPO) can learn with 1000x fewer parameters than SFT
because RL provides sparser, cleaner gradient signal.

How GRPO works:
  1. For each prompt x, sample k completions y_1...y_k from the policy π_θ
  2. Score each completion with a verifiable reward r(x, y_i) ∈ {0, 1}
  3. Compute group-relative advantages: Â_i = (r_i - mean(r)) / std(r)
  4. Update policy with clipped policy gradient (similar to PPO)
  5. Optionally add KL penalty against reference policy

Why GRPO works for TinyLoRA:
  - SFT must memorize *entire demonstrations* (high information content)
  - GRPO only needs the *reward bit* per completion (sparse signal)
  - The reward-relevant features correlate with r; noise cancels via resampling
  - This allows learning in extremely low-capacity regimes (<100 params)

Note: This is a conceptual/pedagogical implementation. For actual training,
use frameworks like VERL (Sheng et al., 2024) with vLLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GRPOBatch:
    """A batch of prompts with grouped completions and rewards."""
    prompt_ids: torch.Tensor          # (batch, prompt_len)
    completion_ids: torch.Tensor      # (batch, group_size, completion_len)
    rewards: torch.Tensor             # (batch, group_size) — binary {0, 1}
    attention_mask: torch.Tensor      # (batch, group_size, total_len)


class GRPOTrainer:
    """
    Conceptual GRPO trainer for TinyLoRA fine-tuning.

    This implements the core GRPO algorithm loop:
      1. Generate completions (simulated in this conceptual version)
      2. Score with verifiable reward
      3. Compute group-relative advantages
      4. Policy gradient update with clipping

    The paper demonstrates that this approach, combined with TinyLoRA,
    achieves 91% on GSM8K with only 13 trainable parameters.

    Args:
        model: The language model with TinyLoRA adapters applied.
        ref_model: Frozen reference model for KL penalty (optional).
        reward_fn: Function mapping (prompt, completion) → reward ∈ {0, 1}.
        group_size: Number of completions per prompt (k).
        kl_coeff: KL divergence penalty coefficient (β).
        clip_eps: PPO-style clipping epsilon.
        lr: Learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        reward_fn: Optional[Callable] = None,
        group_size: int = 4,
        kl_coeff: float = 0.0,
        clip_eps: float = 0.2,
        lr: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn or self._default_reward
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm

        # Only optimize trainable params (the tiny v vectors!)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)

        print(f"[GRPO] Optimizing {sum(p.numel() for p in trainable_params)} "
              f"trainable parameters out of "
              f"{sum(p.numel() for p in model.parameters())} total")

    @staticmethod
    def _default_reward(prompt: str, completion: str) -> float:
        """Placeholder binary reward — exact match checking."""
        return 1.0

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages.

        For each prompt's group of k completions:
            Â_i = (r_i - mean(r_1..k)) / (std(r_1..k) + ε)

        This normalization is what makes GRPO not need a separate critic:
        the group itself serves as the baseline.

        Args:
            rewards: (batch, group_size) reward scores

        Returns:
            advantages: (batch, group_size) normalized advantages
        """
        mean_r = rewards.mean(dim=-1, keepdim=True)
        std_r = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean_r) / std_r
        return advantages

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clipped policy gradient loss (PPO-style).

        L_clip = -E[ min( ratio * Â,  clip(ratio, 1±ε) * Â ) ]

        where ratio = exp(log_π_new - log_π_old)

        Args:
            log_probs: (batch, group_size) current policy log-probs
            old_log_probs: (batch, group_size) old policy log-probs
            advantages: (batch, group_size) group-relative advantages

        Returns:
            Scalar loss
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        loss = -torch.min(surr1, surr2).mean()

        return loss

    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence penalty between current and reference policy.

        KL = E[ log(π_θ / π_ref) ] = E[ log_π_θ - log_π_ref ]

        Paper uses β=0 for GSM8K, β=0.001 for MATH.
        """
        kl = (log_probs - ref_log_probs).mean()
        return self.kl_coeff * kl

    def training_step(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Execute one GRPO training step.

        This is the conceptual inner loop:
          1. Compute group-relative advantages from rewards
          2. Compute clipped policy gradient loss
          3. Add KL penalty if reference model exists
          4. Backprop and update the tiny v parameters

        Args:
            log_probs: Current policy log-probabilities (batch, group_size)
            old_log_probs: Old policy log-probabilities (batch, group_size)
            rewards: Binary rewards (batch, group_size)
            ref_log_probs: Reference model log-probabilities (optional)

        Returns:
            Dictionary of training metrics
        """
        # Step 1: Group-relative advantages
        advantages = self.compute_group_advantages(rewards)

        # Step 2: Clipped policy loss
        policy_loss = self.compute_policy_loss(log_probs, old_log_probs, advantages)

        # Step 3: KL penalty
        kl_loss = torch.tensor(0.0)
        if ref_log_probs is not None and self.kl_coeff > 0:
            kl_loss = self.compute_kl_penalty(log_probs, ref_log_probs)

        total_loss = policy_loss + kl_loss

        # Step 4: Optimize — note only TinyLoRA's v vectors have grad!
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }

    def merge_weights_for_inference(self):
        """
        Merge TinyLoRA weights into the base model for fast inference.

        The paper describes merging LoRA weights at each training step
        for vLLM inference, then using truncated importance sampling to
        handle the resulting numerical mismatch.

        In practice: W_merged = W_original + ΔW_tinylora
        """
        from .tiny_lora import TinyLoRALinear
        for module in self.model.modules():
            if isinstance(module, TinyLoRALinear):
                module.weight.data = module.merged_weight()
