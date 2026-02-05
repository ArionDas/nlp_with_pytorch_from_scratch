"""
Reprompting logic for constructing demonstrations from successful rollouts.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import re


class Reprompter:
    """
    Constructs reprompted inputs by combining original prompts with 
    successful demonstrations and failure feedback.
    """
    
    def __init__(self, config):
        self.config = config
    
    def construct_demonstrations(
        self,
        rollouts: List[Dict],
        success_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Construct reprompted prompts for self-distillation.
        
        For each rollout, finds successful examples from the batch and
        constructs a new prompt that includes them as demonstrations,
        along with feedback from failed attempts.
        
        Args:
            rollouts: List of rollout results with 'prompt', 'response', 'reward', 'feedback'
            success_threshold: Minimum reward to be considered successful
        
        Returns:
            List of reprompted inputs ready for teacher model forward pass
        """
        reprompted = []
        
        # Separate successful and failed rollouts
        successful = [r for r in rollouts if r['reward'] >= success_threshold]
        failed = [r for r in rollouts if r['reward'] < success_threshold]
        
        for rollout in rollouts:
            original_prompt = rollout['prompt']
            
            # Select demonstration
            demonstration = self._select_demonstration(
                rollout, successful, failed
            )
            
            # Select feedback
            feedback = self._select_feedback(
                rollout, successful, failed
            )
            
            # Construct reprompted input
            reprompted_prompt = self._build_reprompt(
                original_prompt, demonstration, feedback
            )
            
            reprompted.append({
                'original_rollout': rollout,
                'reprompted_prompt': reprompted_prompt,
                'demonstration': demonstration,
                'feedback': feedback
            })
        
        return reprompted
    
    def _select_demonstration(
        self,
        current_rollout: Dict,
        successful: List[Dict],
        failed: List[Dict]
    ) -> Optional[str]:
        """
        Select a successful demonstration for the current rollout.
        
        If dont_reprompt_on_self_success is True and current rollout is 
        successful, we still use other successes as examples.
        """
        if not successful:
            return None
        
        # Filter out self if configured
        if self.config.dont_reprompt_on_self_success:
            other_successful = [
                s for s in successful 
                if s['response'] != current_rollout['response']
            ]
            if other_successful:
                successful = other_successful
        
        # Pick the highest reward successful example
        best = max(successful, key=lambda x: x['reward'])
        
        demo_text = best['response']
        
        # Remove thinking tags if configured
        if self.config.remove_thinking_from_demonstration:
            demo_text = self._remove_thinking_tags(demo_text)
        
        return demo_text
    
    def _select_feedback(
        self,
        current_rollout: Dict,
        successful: List[Dict],
        failed: List[Dict]
    ) -> Optional[str]:
        """
        Select feedback to include in the reprompt.
        
        Only include feedback if:
        1. include_environment_feedback is True
        2. environment_feedback_only_without_solution is False OR no successful demonstration
        """
        if not self.config.include_environment_feedback:
            return None
        
        # Check if we should skip feedback when we have a solution
        if (self.config.environment_feedback_only_without_solution 
            and successful):
            return None
        
        # Get feedback from current rollout if it failed (below success threshold)
        if current_rollout['reward'] < self.config.success_threshold and 'feedback' in current_rollout:
            return current_rollout['feedback']
        
        # Or get feedback from a failed example
        if failed:
            sample_failed = failed[0]  # Pick first failed
            return sample_failed.get('feedback', "Previous attempt was incorrect.")
        
        return None
    
    def _build_reprompt(
        self,
        original_prompt: str,
        demonstration: Optional[str],
        feedback: Optional[str]
    ) -> str:
        """Build the final reprompted string."""
        
        # Format solution section
        solution_str = ""
        if demonstration:
            solution_str = self.config.solution_template.format(
                successful_previous_attempt=demonstration
            )
        
        # Format feedback section
        feedback_str = ""
        if feedback:
            feedback_str = self.config.feedback_template.format(
                feedback_raw=feedback
            )
        
        # Build final reprompt
        reprompt = self.config.reprompt_template.format(
            prompt=original_prompt,
            solution=solution_str,
            feedback=feedback_str
        )
        
        # Truncate if necessary
        reprompt = self._truncate_reprompt(reprompt)
        
        return reprompt
    
    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> or similar reasoning tags."""
        # Remove content between thinking tags
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reasoning>.*?</reasoning>',
        ]
        
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL)
        
        return result.strip()
    
    def _truncate_reprompt(self, reprompt: str) -> str:
        """
        Truncate reprompted prompt to max length.
        
        Note: This is a simplified version that truncates by character count.
        In practice, you'd want to truncate by token count using the tokenizer.
        """
        max_len = self.config.max_reprompt_len
        
        if len(reprompt) <= max_len:
            return reprompt
        
        if self.config.reprompt_truncation == "right":
            # Truncate from the end
            return reprompt[:max_len]
        elif self.config.reprompt_truncation == "left":
            # Truncate from the beginning (keep the end with demonstrations)
            return reprompt[-max_len:]
        else:  # error
            raise ValueError(
                f"Reprompt exceeds max length ({len(reprompt)} > {max_len})"
            )
