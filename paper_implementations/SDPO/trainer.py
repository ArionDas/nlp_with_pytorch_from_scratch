"""
Main SDPO Trainer - Self-Distilled Policy Optimization.

This trainer implements the core SDPO loop:
1. Generate rollouts from current policy
2. Compute rewards and identify successes
3. Reprompt with successful demonstrations + feedback
4. Compute self-distillation loss (student learns from teacher)
5. Compute policy gradient loss
6. Update student model and teacher EMA
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Iterator
import logging
from tqdm import tqdm

from .config import SDPOConfig, RolloutResult
from .policy_model import PolicyModel
from .reward_model import RewardModel
from .reprompting import Reprompter
from .self_distillation import SelfDistillationLoss, PolicyGradientLoss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDPOTrainer:
    """
    Self-Distilled Policy Optimization Trainer.
    
    Combines:
    - Policy gradient optimization (GRPO-style)
    - Self-distillation from feedback-conditioned teacher
    - EMA-based teacher regularization
    """
    
    def __init__(
        self,
        config: SDPOConfig,
        policy_model: PolicyModel,
        reward_model: RewardModel,
        train_dataloader: DataLoader
    ):
        self.config = config
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.train_dataloader = train_dataloader
        
        # Initialize components
        self.reprompter = Reprompter(config.reprompting)
        self.distillation_loss_fn = SelfDistillationLoss(
            alpha=config.self_distillation.alpha,
            topk=config.self_distillation.distillation_topk,
            add_tail=config.self_distillation.distillation_add_tail,
            is_clip=config.self_distillation.is_clip
        )
        self.pg_loss_fn = PolicyGradientLoss()
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        self.global_step = 0
    
    def train(self):
        """Main training loop."""
        logger.info("Starting SDPO training...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            epoch_metrics = {
                'total_loss': [],
                'pg_loss': [],
                'distillation_loss': [],
                'kl_penalty': [],
                'mean_reward': []
            }
            
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                # Extract prompts from batch
                prompts = batch['prompt'] if isinstance(batch, dict) else batch
                
                # Step 1: Generate rollouts
                rollouts = self.generate_rollouts(prompts)
                
                # Step 2: Compute rewards
                rollouts = self.compute_rewards(rollouts)
                
                # Step 3: Construct reprompted inputs
                reprompted = self.reprompter.construct_demonstrations(
                    rollouts,
                    success_threshold=self.config.self_distillation.success_reward_threshold
                )
                
                # Step 4: Compute losses and update
                metrics = self.train_step(rollouts, reprompted)
                
                # Collect metrics
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self.log_metrics(metrics, self.global_step)
                
                self.global_step += 1
            
            # Epoch summary
            self.log_epoch_summary(epoch_metrics, epoch)
        
        logger.info("Training complete!")
    
    def generate_rollouts(self, prompts: List[str]) -> List[Dict]:
        """
        Generate multiple rollouts per prompt.
        
        Returns:
            List of rollout dictionaries
        """
        all_rollouts = []
        
        for prompt in prompts:
            # Generate G samples per prompt (G = num_samples_per_prompt)
            rollouts = self.policy_model.generate_rollouts(
                [prompt],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_return_sequences=self.config.num_samples_per_prompt
            )
            all_rollouts.extend(rollouts)
        
        return all_rollouts
    
    def compute_rewards(self, rollouts: List[Dict]) -> List[Dict]:
        """
        Compute rewards for all rollouts.
        """
        prompts = [r['prompt'] for r in rollouts]
        responses = [r['response'] for r in rollouts]
        full_sequences = [r['full_sequence'] for r in rollouts]
        
        reward_results = self.reward_model.compute_rewards(
            prompts, responses, full_sequences
        )
        
        # Merge rewards into rollouts
        for rollout, reward_dict in zip(rollouts, reward_results):
            rollout['reward'] = reward_dict['reward']
            rollout['feedback'] = reward_dict.get('feedback', None)
            rollout['is_success'] = (
                reward_dict['reward'] >= self.config.self_distillation.success_reward_threshold
            )
        
        return rollouts
    
    def train_step(
        self,
        rollouts: List[Dict],
        reprompted: List[Dict]
    ) -> Dict[str, float]:
        """
        Single training step: compute losses and update model.
        """
        self.policy_model.train_mode()
        
        # Prepare inputs
        original_texts = [r['full_sequence'] for r in rollouts]
        reprompted_texts = [r['reprompted_prompt'] for r in reprompted]
        
        # Tokenize
        original_inputs = self._tokenize_batch(original_texts)
        reprompted_inputs = self._tokenize_batch(reprompted_texts)
        
        # Forward pass - Student on original prompts
        student_logits = self.policy_model.forward_student(
            original_inputs['input_ids'],
            original_inputs['attention_mask']
        )
        
        # Forward pass - Student on reprompted prompts
        student_rep_logits = self.policy_model.forward_student(
            reprompted_inputs['input_ids'],
            reprompted_inputs['attention_mask']
        )
        
        # Forward pass - Teacher on reprompted prompts (no grad)
        teacher_logits = self.policy_model.forward_teacher(
            reprompted_inputs['input_ids'],
            reprompted_inputs['attention_mask']
        )
        
        # Compute losses
        total_loss = 0.0
        metrics = {}
        
        # 1. Policy Gradient Loss (on original rollouts)
        rewards = torch.tensor(
            [r['reward'] for r in rollouts],
            dtype=torch.float32,
            device=self.policy_model.device
        )
        
        # Compute advantages (GRPO-style grouping)
        advantages = self.pg_loss_fn.compute_advantages(
            rewards,
            group_size=self.config.num_samples_per_prompt
        )
        
        # Get log probs for original rollouts
        labels = original_inputs['input_ids'][:, 1:]  # Shift for next-token prediction
        student_log_probs = self.policy_model.get_log_probs(
            student_logits[:, :-1, :],  # Remove last position
            labels
        )
        
        # Compute PG loss (simplified - would need old policy for full PPO)
        # For SDPO, we use on-policy gradient
        pg_loss, pg_metrics = self.pg_loss_fn.compute_loss(
            log_probs=student_log_probs,
            old_log_probs=student_log_probs.detach(),  # On-policy
            advantages=advantages.unsqueeze(1).expand(-1, student_log_probs.shape[1]),
            attention_mask=original_inputs['attention_mask'][:, 1:],
            use_clipping=False  # Vanilla policy gradient for simplicity
        )
        
        total_loss += pg_loss * self.config.self_distillation.pg_loss_weight
        metrics.update(pg_metrics)
        
        # 2. Self-Distillation Loss (student learns from teacher on reprompted)
        # Align reprompted logits with original sequence length
        rep_labels = reprompted_inputs['input_ids'][:, 1:]
        
        distill_loss, distill_metrics = self.distillation_loss_fn.compute_loss(
            student_logits=student_rep_logits[:, :-1, :],
            teacher_logits=teacher_logits[:, :-1, :],
            attention_mask=reprompted_inputs['attention_mask'][:, 1:],
            labels=rep_labels
        )
        
        total_loss += distill_loss * self.config.self_distillation.distillation_weight
        metrics.update(distill_metrics)
        
        # 3. KL Penalty (student shouldn't diverge too far from initial policy)
        # Simplified: use teacher as reference and compute KL
        ref_kl_loss, _ = self.distillation_loss_fn.compute_loss(
            student_logits=student_logits[:, :-1, :],
            teacher_logits=teacher_logits[:, :student_logits.shape[1]-1, :],
            attention_mask=original_inputs['attention_mask'][:, 1:]
        )
        
        total_loss += ref_kl_loss * self.config.self_distillation.kl_penalty_weight
        metrics['kl_penalty'] = ref_kl_loss.item()
        
        # Backward and update
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Update teacher EMA
        self.policy_model.update_teacher_ema()
        
        # Collect final metrics
        metrics['total_loss'] = total_loss.item()
        metrics['mean_reward'] = rewards.mean().item()
        metrics['success_rate'] = (rewards >= self.config.self_distillation.success_reward_threshold).float().mean().item()
        
        return metrics
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        tokenizer = self.policy_model.tokenizer
        
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.config.reprompting.max_reprompt_len
        )
        
        return {
            'input_ids': encodings['input_ids'].to(self.policy_model.device),
            'attention_mask': encodings['attention_mask'].to(self.policy_model.device)
        }
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console/logger."""
        log_str = f"Step {step}: "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
    
    def log_epoch_summary(self, epoch_metrics: Dict[str, List[float]], epoch: int):
        """Log epoch summary statistics."""
        logger.info(f"\n--- Epoch {epoch + 1} Summary ---")
        for key, values in epoch_metrics.items():
            if values:
                mean_val = sum(values) / len(values)
                logger.info(f"  {key}: {mean_val:.4f}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self.policy_model.save(path)
        logger.info(f"Checkpoint saved to {path}")
