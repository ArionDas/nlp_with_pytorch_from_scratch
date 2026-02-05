"""
Self-distillation loss computation for SDPO.

Implements KL divergence-based distillation between student (policy) and 
teacher (feedback-conditioned model) distributions.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfDistillationLoss:
    """
    Computes self-distillation loss using KL divergence.
    
    Supports multiple KL variants:
    - Forward KL: KL(teacher || student) - teacher seeks modes
    - Reverse KL: KL(student || teacher) - student covers modes
    - JSD: Jensen-Shannon Divergence (alpha=0.5) - balanced
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # 0.0=forward KL, 1.0=reverse KL, 0.5=JSD
        topk: Optional[int] = None,
        add_tail: bool = True,
        is_clip: Optional[float] = 2.0
    ):
        """
        Args:
            alpha: Interpolation between forward and reverse KL
            topk: If set, only use top-k logits for distillation
            add_tail: Add tail bucket for top-k distillation
            is_clip: Importance sampling ratio clip value
        """
        self.alpha = alpha
        self.topk = topk
        self.add_tail = add_tail
        self.is_clip = is_clip
    
    def compute_loss(
        self,
        student_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        teacher_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        labels: Optional[torch.Tensor] = None  # [batch, seq_len] for IS weighting
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute self-distillation KL loss.
        
        Returns:
            loss: Scalar loss value
            metrics: Dict with component losses for logging
        """
        # Apply top-k filtering if configured
        if self.topk is not None:
            student_logits, teacher_logits = self._apply_topk(
                student_logits, teacher_logits
            )
        
        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        # Compute probabilities
        student_probs = torch.exp(student_log_probs)
        teacher_probs = torch.exp(teacher_log_probs)
        
        # Compute forward KL: KL(teacher || student)
        forward_kl = torch.sum(
            teacher_probs * (teacher_log_probs - student_log_probs),
            dim=-1
        )  # [batch, seq_len]
        
        # Compute reverse KL: KL(student || teacher)
        reverse_kl = torch.sum(
            student_probs * (student_log_probs - teacher_log_probs),
            dim=-1
        )  # [batch, seq_len]
        
        # Interpolate based on alpha
        if self.alpha == 0.0:
            kl_loss = forward_kl
        elif self.alpha == 1.0:
            kl_loss = reverse_kl
        else:
            kl_loss = (1 - self.alpha) * forward_kl + self.alpha * reverse_kl
        
        # Apply importance sampling weighting if labels provided
        if self.is_clip is not None and labels is not None:
            weights = self._compute_importance_weights(
                student_log_probs, teacher_log_probs, labels
            )
            kl_loss = kl_loss * weights
        
        # Apply attention mask and compute mean
        if attention_mask is not None:
            kl_loss = (kl_loss * attention_mask).sum() / attention_mask.sum()
        else:
            kl_loss = kl_loss.mean()
        
        metrics = {
            'forward_kl': forward_kl.mean().item(),
            'reverse_kl': reverse_kl.mean().item(),
            'distillation_loss': kl_loss.item()
        }
        
        return kl_loss, metrics
    
    def _apply_topk(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply top-k filtering to logits.
        
        Only keep top-k logits from teacher, rest go to tail bucket.
        """
        # Get top-k indices from teacher
        topk_values, topk_indices = torch.topk(
            teacher_logits, 
            k=self.topk, 
            dim=-1
        )
        
        # Create new logits tensor with top-k + optional tail
        vocab_size = teacher_logits.shape[-1]
        new_vocab_size = self.topk + (1 if self.add_tail else 0)
        
        batch_size, seq_len, _ = teacher_logits.shape
        
        # Initialize new logits
        new_teacher_logits = torch.full(
            (batch_size, seq_len, new_vocab_size),
            float('-inf'),
            device=teacher_logits.device,
            dtype=teacher_logits.dtype
        )
        new_student_logits = new_teacher_logits.clone()
        
        # Fill in top-k logits
        new_teacher_logits[..., :self.topk] = torch.gather(
            teacher_logits, -1, topk_indices
        )
        new_student_logits[..., :self.topk] = torch.gather(
            student_logits, -1, topk_indices
        )
        
        # Add tail bucket if configured
        if self.add_tail:
            # Compute tail as logsumexp of remaining tokens
            teacher_tail = torch.logsumexp(
                teacher_logits.scatter(-1, topk_indices, float('-inf')),
                dim=-1,
                keepdim=True
            )
            student_tail = torch.logsumexp(
                student_logits.scatter(-1, topk_indices, float('-inf')),
                dim=-1,
                keepdim=True
            )
            new_teacher_logits[..., -1:] = teacher_tail
            new_student_logits[..., -1:] = student_tail
        
        return new_student_logits, new_teacher_logits
    
    def _compute_importance_weights(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance sampling weights and clip them.
        
        weight = exp(teacher_log_prob - student_log_prob)
        """
        # Gather log probs for labels
        student_token_log_probs = torch.gather(
            student_log_probs, -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        
        teacher_token_log_probs = torch.gather(
            teacher_log_probs, -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute ratio
        log_ratio = teacher_token_log_probs - student_token_log_probs
        ratio = torch.exp(log_ratio)
        
        # Clip
        if self.is_clip is not None:
            ratio = torch.clamp(ratio, 1.0 / self.is_clip, self.is_clip)
        
        return ratio


class PolicyGradientLoss:
    """
    Computes policy gradient loss (typically GRPO-style).
    
    Uses advantage estimates from grouped rewards.
    """
    
    def __init__(self, epsilon: float = 0.2):
        """
        Args:
            epsilon: Clipping parameter for PPO-style objective
        """
        self.epsilon = epsilon
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,  # [batch, seq_len]
        old_log_probs: torch.Tensor,  # [batch, seq_len]
        advantages: torch.Tensor,  # [batch]
        attention_mask: Optional[torch.Tensor] = None,
        use_clipping: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute policy gradient loss.
        
        Args:
            log_probs: Current policy log probs
            old_log_probs: Old policy log probs (for ratio computation)
            advantages: Advantage estimates (broadcast to seq_len)
            attention_mask: Token-level mask
            use_clipping: Whether to use PPO clipping
        
        Returns:
            loss: Policy gradient loss
            metrics: Dict with loss components
        """
        # Expand advantages to match sequence dimension
        advantages = advantages.unsqueeze(-1)  # [batch, 1]
        
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)  # [batch, seq_len]
        
        # Compute unclipped objective
        objective = ratio * advantages
        
        if use_clipping:
            # Clipped objective
            clipped_ratio = torch.clamp(
                ratio, 
                1 - self.epsilon, 
                1 + self.epsilon
            )
            clipped_objective = clipped_ratio * advantages
            
            # Take minimum (pessimistic)
            pg_loss = -torch.min(objective, clipped_objective)
        else:
            # Vanilla policy gradient
            pg_loss = -objective
        
        # Apply mask and compute mean
        if attention_mask is not None:
            pg_loss = (pg_loss * attention_mask).sum() / attention_mask.sum()
        else:
            pg_loss = pg_loss.mean()
        
        metrics = {
            'pg_loss': pg_loss.item(),
            'mean_ratio': ratio.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
        
        return pg_loss, metrics
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,  # [batch]
        group_size: int  # Number of samples per prompt (G in GRPO)
    ) -> torch.Tensor:
        """
        Compute advantages using group-relative normalization (GRPO).
        
        For each group of rollouts from the same prompt:
            advantage = (reward - mean) / std
        
        Args:
            rewards: Flattened rewards for all rollouts
            group_size: Number of rollouts per prompt
        
        Returns:
            advantages: Normalized advantages
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size
        
        advantages = torch.zeros_like(rewards)
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group_rewards = rewards[start_idx:end_idx]
            
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8
            
            advantages[start_idx:end_idx] = (group_rewards - mean) / std
        
        return advantages
