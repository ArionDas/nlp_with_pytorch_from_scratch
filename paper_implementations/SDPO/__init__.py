"""
SDPO: Self-Distilled Policy Optimization

A conceptual implementation of the paper:
"Reinforcement Learning via Self-Distillation" (arXiv:2601.20802)

This implementation provides the core algorithmic components of SDPO:
- Self-distillation from feedback-conditioned teacher model
- EMA-based teacher regularization  
- Reprompting with successful demonstrations
- Combined policy gradient and distillation losses
"""

from .config import SDPOConfig, SelfDistillationConfig, RepromptingConfig, RolloutResult
from .policy_model import PolicyModel
from .reward_model import RewardModel, VerifiableRewardModel, RuleBasedCodeRewardModel, MathRewardModel
from .trainer import SDPOTrainer
from .reprompting import Reprompter
from .self_distillation import SelfDistillationLoss, PolicyGradientLoss

__all__ = [
    'SDPOConfig',
    'SelfDistillationConfig', 
    'RepromptingConfig',
    'RolloutResult',
    'PolicyModel',
    'RewardModel',
    'VerifiableRewardModel',
    'RuleBasedCodeRewardModel',
    'MathRewardModel',
    'SDPOTrainer',
    'Reprompter',
    'SelfDistillationLoss',
    'PolicyGradientLoss',
]
