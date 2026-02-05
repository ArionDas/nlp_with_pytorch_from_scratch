"""
Policy model wrapper for SDPO with generation capabilities and teacher model support.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


class PolicyModel(nn.Module):
    """
    Policy model wrapper that supports both student (policy) and teacher models.
    
    The teacher model is maintained as an EMA (exponential moving average) of the
    student model weights, used for self-distillation.
    """
    
    def __init__(self, model_name: str, device: str = "cuda", teacher_update_rate: float = 0.05):
        super().__init__()
        self.device = device
        self.teacher_update_rate = teacher_update_rate
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.student_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.student_model.to(device)
        
        # Create teacher model as EMA copy
        self.teacher_model = copy.deepcopy(self.student_model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def generate_rollouts(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate rollouts (completions) for given prompts.
        
        Returns:
            List of dicts with 'prompt', 'response', and 'full_sequence' keys
        """
        self.student_model.eval()
        rollouts = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.student_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode and separate prompt from response
            prompt_len = inputs['input_ids'].shape[1]
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                response_ids = output[prompt_len:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                rollouts.append({
                    'prompt': prompt,
                    'response': response,
                    'full_sequence': full_text
                })
        
        return rollouts
    
    def forward_student(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through student model, returning logits."""
        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def forward_teacher(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through teacher model, returning logits."""
        with torch.no_grad():
            outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def update_teacher_ema(self):
        """
        Update teacher model using exponential moving average of student weights.
        Teacher <- (1 - beta) * Teacher + beta * Student
        """
        beta = self.teacher_update_rate
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(),
                self.student_model.parameters()
            ):
                teacher_param.data.mul_(1 - beta).add_(student_param.data, alpha=beta)
    
    def get_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for given labels.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
        
        Returns:
            Log probabilities: [batch_size, seq_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs for the actual labels
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs
    
    def parameters(self, recurse: bool = True):
        """Return student model parameters (for optimizer)."""
        return self.student_model.parameters(recurse=recurse)
    
    def train_mode(self):
        """Set student model to training mode."""
        self.student_model.train()
    
    def eval_mode(self):
        """Set student model to eval mode."""
        self.student_model.eval()
    
    def save(self, path: str):
        """Save student model."""
        self.student_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load student model."""
        self.student_model = AutoModelForCausalLM.from_pretrained(path)
        self.student_model.to(self.device)
        # Re-initialize teacher
        self.teacher_model = copy.deepcopy(self.student_model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
