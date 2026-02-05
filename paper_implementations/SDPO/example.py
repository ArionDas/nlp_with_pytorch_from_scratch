"""
Example usage of SDPO trainer.

This demonstrates how to set up and use the SDPO trainer with a toy example.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import SDPOConfig
from policy_model import PolicyModel
from reward_model import MathRewardModel
from trainer import SDPOTrainer


class SimplePromptDataset(Dataset):
    """Simple dataset of prompts for demonstration."""
    
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}


def main():
    """Example setup and training loop."""
    
    # Example math prompts
    prompts = [
        "Solve: What is 2 + 3?",
        "Solve: What is 5 * 4?",
        "Solve: What is 10 - 7?",
        "Solve: What is 15 / 3?",
        "Solve: What is 2^3?",
    ] * 4  # Replicate for larger dataset
    
    # Configuration
    from config import SelfDistillationConfig, RepromptingConfig
    
    config = SDPOConfig(
        model_name="gpt2",  # Small model for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-5,
        max_new_tokens=128,
        temperature=0.7,
        num_samples_per_prompt=4,  # G in GRPO - generate 4 samples per prompt
        
        # Self-distillation settings
        self_distillation=SelfDistillationConfig(
            success_reward_threshold=0.8,
            alpha=0.5,  # JSD
            distillation_weight=1.0,
            pg_loss_weight=1.0,
            kl_penalty_weight=0.01,
        ),
        
        # Reprompting settings
        reprompting=RepromptingConfig(
            include_environment_feedback=True,
            dont_reprompt_on_self_success=True,
        ),
    )
    
    print("Initializing SDPO training...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Samples per prompt: {config.num_samples_per_prompt}")
    
    # Create dataset and dataloader
    dataset = SimplePromptDataset(prompts)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Initialize policy model
    print("\nLoading policy model...")
    policy_model = PolicyModel(
        model_name=config.model_name,
        device=config.device,
        teacher_update_rate=config.self_distillation.teacher_update_rate
    )
    
    # Initialize reward model
    print("Loading reward model...")
    reward_model = MathRewardModel(extract_boxed_answer=True)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SDPOTrainer(
        config=config,
        policy_model=policy_model,
        reward_model=reward_model,
        train_dataloader=dataloader
    )
    
    # Run training
    print("\n" + "="*50)
    print("Starting SDPO Training Loop")
    print("="*50 + "\n")
    
    # In a real scenario, this would run the full training
    # For demo, we'll just show the structure
    print("Training loop structure:")
    print("1. Generate rollouts (G samples per prompt)")
    print("2. Compute rewards using verifiable reward model")
    print("3. Identify successful demonstrations")
    print("4. Reprompt with demonstrations + feedback")
    print("5. Compute self-distillation loss (student learns from teacher)")
    print("6. Compute policy gradient loss")
    print("7. Update student model")
    print("8. Update teacher EMA")
    print("\nNote: This is a conceptual implementation.")
    print("Actual training would require significant compute resources.")
    
    # Uncomment to actually run training (requires GPU and time)
    # trainer.train()
    
    print("\n" + "="*50)
    print("Example Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
