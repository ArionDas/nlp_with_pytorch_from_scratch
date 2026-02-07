"""
Learning to Reason in 13 Parameters — Conceptual Implementation
================================================================

Paper: "Learning to Reason in 13 Parameters"
Authors: John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar
Affiliation: FAIR at Meta, Cornell University, Carnegie Mellon University
Link: https://arxiv.org/abs/2602.04118

This package provides a lightweight conceptual implementation of TinyLoRA,
a method for scaling low-rank adapters to sizes as small as one parameter.
The key insight is that RL (specifically GRPO) makes fundamentally more
information-dense updates than SFT, enabling learning with orders of
magnitude fewer parameters.

Modules:
    - config: Configuration dataclasses for TinyLoRA, LoRA-XS, and GRPO
    - tiny_lora: Core TinyLoRA layer — projects a tiny trainable vector
                 through fixed random tensors into the SVD subspace
    - lora_xs: LoRA-XS baseline — learns a small r×r matrix in SVD space
    - model: Wrapper that injects TinyLoRA/LoRA-XS into a transformer model
    - grpo: Group Relative Policy Optimization trainer
    - example: End-to-end demonstration script
"""

from .config import TinyLoRAConfig, LoRAXSConfig, GRPOConfig
from .tiny_lora import TinyLoRALinear
from .lora_xs import LoRAXSLinear
from .model import apply_tiny_lora, apply_lora_xs, count_trainable_params
from .grpo import GRPOTrainer
