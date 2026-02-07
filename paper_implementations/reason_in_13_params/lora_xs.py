"""
LoRA-XS — Baseline comparison from "LoRA-XS: Low-Rank Adaptation with
Extremely Small Number of Parameters" (Bałazy et al., 2025).

LoRA-XS is the stepping stone between standard LoRA and TinyLoRA.
It uses the truncated SVD of the frozen weight to define a structured
low-rank update where only a small r×r matrix R is trained.

Hierarchy of methods (from most to fewest parameters):
  LoRA:       W' = W + A @ B                  → O(d * r) params/module
  LoRA-XS:    W' = W + U @ Σ @ R @ V^T        → O(r^2)   params/module
  TinyLoRA:   W' = W + U @ Σ @ (Σ v_i P_i) @ V^T → O(u) params/module  (u ≤ r^2)

For a 7B model with 7 adapted modules per layer and 28 layers:
  LoRA r=1:   ~3M params
  LoRA-XS r=1: 196 params (one per module)
  TinyLoRA u=1, full tying: 1 param (!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAXSLinear(nn.Module):
    """
    A frozen linear layer with a LoRA-XS adapter.

    The update uses the truncated SVD of the frozen weight:
        W' = W + α * U @ Σ @ R @ V^T

    where U, Σ, V come from SVD(W) and are frozen, and only R ∈ R^(r×r)
    is trainable [r^2 parameters per module].

    This can be viewed as learning to recombine the dominant singular
    directions of W, and outperforms randomly-initialized LoRA.

    Args:
        original_linear (nn.Linear): Pretrained layer to adapt.
        rank (int): Rank r of the truncated SVD (and size of R).
        alpha (float): Scaling factor for the update.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 1,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.d_out, self.d_in = original_linear.weight.shape
        self.rank = rank
        self.alpha = alpha

        # Freeze original
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Truncated SVD
        U_full, s_full, Vh_full = torch.linalg.svd(
            self.weight.data.float(), full_matrices=False
        )
        r = min(rank, min(self.d_out, self.d_in))

        self.register_buffer("U", U_full[:, :r].to(self.weight.dtype))
        self.register_buffer("sigma", torch.diag(s_full[:r]).to(self.weight.dtype))
        self.register_buffer("Vt", Vh_full[:r, :].to(self.weight.dtype))

        # Trainable R ∈ R^(r×r) — initialized to zero (no initial change)
        self.R = nn.Parameter(torch.zeros(r, r, dtype=self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        delta_W = self.alpha * (self.U @ self.sigma @ self.R @ self.Vt)
        lora_out = F.linear(x, delta_W)
        return base_out + lora_out

    def merged_weight(self) -> torch.Tensor:
        """Return W + ΔW for inference."""
        delta_W = self.alpha * (self.U @ self.sigma @ self.R @ self.Vt)
        return self.weight.data + delta_W.detach()

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"trainable_params={self.rank ** 2}"
        )
