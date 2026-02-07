"""
Model wrapper — injects TinyLoRA or LoRA-XS adapters into a transformer model.

This module provides utility functions to:
  1. Identify target linear layers in a transformer (Q, K, V, O, MLP projections)
  2. Replace them with TinyLoRA or LoRA-XS wrapped versions
  3. Manage weight tying across modules
  4. Count and report trainable parameters

The paper adapts 7 linear layers per transformer block:
  Attention: q_proj, k_proj, v_proj, o_proj
  MLP:       up_proj, down_proj, gate_proj

Example parameter counts for Qwen2.5-7B (28 layers, 7 modules/layer = 196 modules):
  - TinyLoRA u=1, n_tie=196 (full sharing): 1 param
  - TinyLoRA u=1, n_tie=1 (no sharing):     196 params
  - TinyLoRA u=1, n_tie=28 (share per layer): 7 params
  - TinyLoRA u=1, n_tie=14 (13 groups):     ~13 params  ← the paper's headline!
  - LoRA-XS r=1:                              196 params
  - LoRA r=1:                                 ~3M params
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

from .tiny_lora import TinyLoRALinear, TinyLoRAParameterGroup
from .lora_xs import LoRAXSLinear
from .config import TinyLoRAConfig, LoRAXSConfig


def find_target_modules(
    model: nn.Module,
    target_names: List[str],
) -> List[Tuple[str, nn.Module, str]]:
    """
    Find all linear layers matching target names in the model.

    Returns list of (parent_name, parent_module, attr_name) tuples
    for each matching linear layer.
    """
    targets = []
    for name, module in model.named_modules():
        for attr_name in dir(module):
            if attr_name in target_names:
                layer = getattr(module, attr_name, None)
                if isinstance(layer, nn.Linear):
                    targets.append((name, module, attr_name))
    return targets


def _find_linear_layers_recursive(
    model: nn.Module,
    target_names: List[str],
    prefix: str = "",
) -> List[Tuple[nn.Module, str, str]]:
    """
    Recursively find linear layers whose name matches one of target_names.

    Returns list of (parent_module, attribute_name, full_path) tuples.
    """
    results = []
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name in target_names and isinstance(child, nn.Linear):
            results.append((model, name, full_name))
        else:
            results.extend(
                _find_linear_layers_recursive(child, target_names, full_name)
            )
    return results


def apply_tiny_lora(
    model: nn.Module,
    config: TinyLoRAConfig,
) -> nn.Module:
    """
    Apply TinyLoRA adapters to a model in-place.

    This replaces target linear layers with TinyLoRALinear wrappers.
    Original weights are frozen; only tiny v vectors are trainable.

    Weight tying is managed via TinyLoRAParameterGroup:
      - Modules are assigned shared v parameters in order
      - Every n_tie consecutive modules share the same v

    Args:
        model: The pretrained model (e.g., Qwen2.5-7B-Instruct)
        config: TinyLoRA configuration

    Returns:
        The modified model (same object, modified in-place)
    """
    # Find all target layers
    targets = _find_linear_layers_recursive(model, config.target_modules)
    n_modules = len(targets)

    if n_modules == 0:
        print("[TinyLoRA] Warning: No target modules found! "
              f"Searched for: {config.target_modules}")
        return model

    # Create parameter group for weight tying
    dtype = next(model.parameters()).dtype
    param_group = TinyLoRAParameterGroup(
        trainable_dim=config.trainable_dim,
        n_tie=config.n_tie,
        dtype=dtype,
    )

    # Replace each target layer with TinyLoRALinear
    for module_id, (parent, attr_name, full_path) in enumerate(targets):
        original_layer = getattr(parent, attr_name)

        # Get shared (or unique) v parameter
        shared_v = param_group.get_shared_v()

        # Create TinyLoRA-wrapped layer
        tiny_lora_layer = TinyLoRALinear(
            original_linear=original_layer,
            frozen_rank=config.frozen_rank,
            trainable_dim=config.trainable_dim,
            alpha=config.alpha,
            shared_v=shared_v,
            random_seed=config.random_seed,
            module_id=module_id,
        )

        # Replace in parent
        setattr(parent, attr_name, tiny_lora_layer)

    total_params = param_group.total_trainable_params
    total_bytes = total_params * (2 if dtype == torch.bfloat16 else 4)

    print(f"[TinyLoRA] Applied to {n_modules} modules")
    print(f"[TinyLoRA] Weight tying: n_tie={config.n_tie} "
          f"({len(param_group.all_params)} unique v vectors)")
    print(f"[TinyLoRA] Total trainable: {total_params} params "
          f"({total_bytes} bytes)")

    return model


def apply_lora_xs(
    model: nn.Module,
    config: LoRAXSConfig,
) -> nn.Module:
    """
    Apply LoRA-XS adapters to a model in-place.

    Replaces target linear layers with LoRAXSLinear wrappers.
    Each adapted module trains a small r×r matrix R.

    Args:
        model: The pretrained model
        config: LoRA-XS configuration

    Returns:
        The modified model
    """
    targets = _find_linear_layers_recursive(model, config.target_modules)

    for parent, attr_name, full_path in targets:
        original_layer = getattr(parent, attr_name)
        lora_xs_layer = LoRAXSLinear(
            original_linear=original_layer,
            rank=config.rank,
            alpha=config.alpha,
        )
        setattr(parent, attr_name, lora_xs_layer)

    n_modules = len(targets)
    params_per_module = config.rank ** 2
    total = n_modules * params_per_module
    print(f"[LoRA-XS] Applied to {n_modules} modules "
          f"({params_per_module} params each, {total} total)")

    return model


def count_trainable_params(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable vs frozen parameters in a model.

    Returns a dict with:
      - trainable: Number of trainable parameters
      - frozen: Number of frozen parameters
      - total: Total parameters
      - trainable_bytes: Size of trainable params in bytes (bf16)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    # Count unique parameters (handles weight tying)
    seen_ids = set()
    unique_trainable = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_trainable += p.numel()

    return {
        "trainable": unique_trainable,
        "frozen": frozen,
        "total": total,
        "trainable_bytes_bf16": unique_trainable * 2,
        "trainable_bytes_fp32": unique_trainable * 4,
        "compression_ratio": total / max(unique_trainable, 1),
    }


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters, typically called before applying adapters."""
    for param in model.parameters():
        param.requires_grad = False


def print_adapter_summary(model: nn.Module) -> None:
    """Print a summary of adapter modules in the model."""
    tiny_lora_count = 0
    lora_xs_count = 0

    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALinear):
            tiny_lora_count += 1
        elif isinstance(module, LoRAXSLinear):
            lora_xs_count += 1

    stats = count_trainable_params(model)

    print("\n" + "=" * 60)
    print("Adapter Summary")
    print("=" * 60)
    if tiny_lora_count:
        print(f"  TinyLoRA modules:  {tiny_lora_count}")
    if lora_xs_count:
        print(f"  LoRA-XS modules:   {lora_xs_count}")
    print(f"  Trainable params:  {stats['trainable']:,}")
    print(f"  Frozen params:     {stats['frozen']:,}")
    print(f"  Total params:      {stats['total']:,}")
    print(f"  Update size (bf16):{stats['trainable_bytes_bf16']:,} bytes")
    print(f"  Compression ratio: {stats['compression_ratio']:,.0f}x")
    print("=" * 60 + "\n")
