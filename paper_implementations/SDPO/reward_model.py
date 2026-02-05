"""
Reward model interface and implementations for SDPO.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import re


class RewardModel(ABC):
    """Abstract base class for reward models."""
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        full_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute rewards and optional feedback for rollouts.
        
        Returns:
            List of dicts with 'reward' (float) and optional 'feedback' (str)
        """
        pass


class VerifiableRewardModel(RewardModel):
    """
    Example reward model for verifiable domains (like math or code).
    
    Checks if response contains correct answer pattern or passes tests.
    """
    
    def __init__(self, correct_answer_pattern: Optional[str] = None):
        self.correct_answer_pattern = correct_answer_pattern
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        full_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        results = []
        
        for prompt, response, full_seq in zip(prompts, responses, full_sequences):
            reward = 0.0
            feedback = None
            
            # Example: Check for correct answer pattern
            if self.correct_answer_pattern:
                if re.search(self.correct_answer_pattern, response):
                    reward = 1.0
                    feedback = "Correct answer found!"
                else:
                    reward = 0.0
                    feedback = "Answer pattern not found. Try again."
            else:
                # Simple length-based reward as placeholder
                reward = min(len(response) / 100.0, 1.0)
                feedback = f"Response length: {len(response)} chars"
            
            results.append({
                'reward': reward,
                'feedback': feedback
            })
        
        return results


class RuleBasedCodeRewardModel(RewardModel):
    """
    Example reward model for code generation tasks.
    
    Checks for syntactic correctness and presence of key elements.
    """
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        full_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        results = []
        
        for prompt, response, full_seq in zip(prompts, responses, full_sequences):
            reward = 0.0
            feedback_parts = []
            
            # Check for code block
            if '```' in response or response.strip().startswith('def ') or response.strip().startswith('class '):
                reward += 0.3
                feedback_parts.append("Code block detected")
            else:
                feedback_parts.append("No code block found")
            
            # Check for function definition
            if 'def ' in response:
                reward += 0.3
                feedback_parts.append("Function definition found")
            else:
                feedback_parts.append("Missing function definition")
            
            # Check for return statement
            if 'return ' in response:
                reward += 0.2
                feedback_parts.append("Return statement found")
            else:
                feedback_parts.append("Missing return statement")
            
            # Check indentation (basic Python check)
            lines = response.split('\n')
            has_indentation = any(line.startswith('    ') or line.startswith('\t') for line in lines)
            if has_indentation:
                reward += 0.2
                feedback_parts.append("Proper indentation detected")
            else:
                feedback_parts.append("Check indentation")
            
            results.append({
                'reward': reward,
                'feedback': "; ".join(feedback_parts)
            })
        
        return results


class MathRewardModel(RewardModel):
    """
    Example reward model for math problems.
    
    Extracts final answer and checks against expected format.
    """
    
    def __init__(self, extract_boxed_answer: bool = True):
        self.extract_boxed_answer = extract_boxed_answer
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from text, looking for boxed notation or final number."""
        # Try to extract boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Try to find answer after common markers
        markers = ['answer is', 'final answer', 'result is', 'equals']
        for marker in markers:
            if marker in text.lower():
                # Look for number or expression after marker
                pattern = rf'{marker}\s*[:=]?\s*([^\n\.]+)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        full_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        results = []
        
        for prompt, response, full_seq in zip(prompts, responses, full_sequences):
            reward = 0.0
            feedback = None
            
            # Extract answer
            answer = self._extract_answer(response)
            
            if answer:
                reward = 0.5  # Partial credit for having an answer format
                feedback = f"Answer extracted: {answer}"
                
                # Check for reasoning steps
                if any(marker in response.lower() for marker in ['step', 'first', 'then', 'therefore']):
                    reward += 0.3
                    feedback += "; Shows reasoning steps"
                
                # Check for calculation
                if any(marker in response for marker in ['=', '+', '-', '*', '/']):
                    reward += 0.2
                    feedback += "; Contains calculations"
            else:
                reward = 0.0
                feedback = "Could not extract final answer. Use \\boxed{} or clearly state your answer."
            
            results.append({
                'reward': reward,
                'feedback': feedback
            })
        
        return results
