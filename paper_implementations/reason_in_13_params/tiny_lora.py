"""
TinyLoRA — The core contribution of "Learning to Reason in 13 Parameters".

TinyLoRA extends LoRA-XS to enable adapting a frozen linear layer with an
arbitrarily small number of trainable parameters (down to 1).

Architecture:
  Given a frozen weight matrix W ∈ R^(d_out × d_in):

  1. Compute truncated SVD:  W ≈ U @ Σ @ V^T
     where U ∈ R^(d_out × r), Σ ∈ R^(r × r), V ∈ R^(d_in × r)

  2. Generate u fixed random projection matrices P_i ∈ R^(r × r), i=1..u
     These are frozen (never updated).

  3. Learn a tiny vector v ∈ R^u (the ONLY trainable parameters!)

  4. Reconstruct the r×r mixing matrix:   R = Σ_i  v_i * P_i

  5. Apply the update:   W' = W + α * U @ Σ @ R @ V^T

With weight tying (n_tie > 1), multiple modules share the same v,
pushing total trainable params even lower.

At u=1 with full weight tying across all modules, the ENTIRE model
is adapted by a SINGLE scalar parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TinyLoRALinear(nn.Module):
    """
    A frozen linear layer augmented with a TinyLoRA adapter.

    The forward pass computes:
        output = x @ W^T + bias + α * x @ (V @ R^T @ Σ @ U^T)^T

    where R = Σ_i v_i * P_i  is the reconstructed mixing matrix,
    and only v ∈ R^u is trainable.

    Args:
        original_linear (nn.Linear): The pretrained linear layer to adapt.
        frozen_rank (int): Rank r of the truncated SVD.
        trainable_dim (int): Dimension u of the trainable vector v.
        alpha (float): Scaling factor for the LoRA update.
        shared_v (Optional[nn.Parameter]): If provided, use this shared
            parameter vector instead of creating a new one (for weight tying).
        random_seed (int): Seed for generating the fixed projection tensors P.
        module_id (int): Unique identifier for this module, used to ensure
            different modules get different random projections even when
            sharing the same v.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        frozen_rank: int = 2,
        trainable_dim: int = 1,
        alpha: float = 1.0,
        shared_v: Optional[nn.Parameter] = None,
        random_seed: int = 42,
        module_id: int = 0,
    ):
        super().__init__()

        self.d_out, self.d_in = original_linear.weight.shape
        self.frozen_rank = frozen_rank
        self.trainable_dim = trainable_dim
        self.alpha = alpha

        # ── Step 1: Freeze original weights ──
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # ── Step 2: Truncated SVD of the frozen weight ──
        # W ≈ U @ diag(s) @ Vh  →  U ∈ R^(d_out×r), s ∈ R^r, Vh ∈ R^(r×d_in)
        U_full, s_full, Vh_full = torch.linalg.svd(self.weight.data.float(), full_matrices=False)

        r = min(frozen_rank, min(self.d_out, self.d_in))
        self.register_buffer("U", U_full[:, :r].to(self.weight.dtype))        # (d_out, r)
        self.register_buffer("sigma", torch.diag(s_full[:r]).to(self.weight.dtype))  # (r, r)
        self.register_buffer("Vt", Vh_full[:r, :].to(self.weight.dtype))       # (r, d_in)

        # ── Step 3: Fixed random projection tensors P ∈ R^(u, r, r) ──
        # Each P_i is a random r×r matrix, frozen. Different per module.
        gen = torch.Generator()
        gen.manual_seed(random_seed + module_id)
        P = torch.randn(trainable_dim, r, r, generator=gen, dtype=self.weight.dtype)
        # Normalize each P_i so the scale of the update is controlled by v
        P = P / (P.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        self.register_buffer("P", P)  # (u, r, r) — frozen

        # ── Step 4: Trainable vector v ∈ R^u ──
        if shared_v is not None:
            # Weight tying: reuse a shared parameter
            self.v = shared_v
            self._owns_v = False
        else:
            self.v = nn.Parameter(torch.zeros(trainable_dim))
            self._owns_v = True

    def _reconstruct_R(self) -> torch.Tensor:
        """
        Reconstruct the r×r mixing matrix:  R = Σ_i v_i * P_i

        Returns:
            R: Tensor of shape (r, r)
        """
        # v: (u,)  P: (u, r, r)  →  einsum → (r, r)
        R = torch.einsum("u, u r s -> r s", self.v, self.P)
        return R

    def _compute_delta_W(self) -> torch.Tensor:
        """
        Compute the low-rank weight update:
            ΔW = α * U @ Σ @ R @ V^T

        Returns:
            delta_W: Tensor of shape (d_out, d_in)
        """
        R = self._reconstruct_R()                     # (r, r)
        # U @ Σ @ R @ Vt
        delta = self.U @ self.sigma @ R @ self.Vt      # (d_out, d_in)
        return self.alpha * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:  output = x @ (W + ΔW)^T + bias

        For efficiency, we compute the LoRA branch separately:
            output = F.linear(x, W, bias) + x @ ΔW^T
        """
        # Frozen base forward
        base_out = F.linear(x, self.weight, self.bias)

        # TinyLoRA update branch
        delta_W = self._compute_delta_W()              # (d_out, d_in)
        lora_out = F.linear(x, delta_W)                # no bias for ΔW

        return base_out + lora_out

    def merged_weight(self) -> torch.Tensor:
        """Return the merged weight W + ΔW (useful for inference with vLLM)."""
        return self.weight.data + self._compute_delta_W().detach()

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, "
            f"frozen_rank={self.frozen_rank}, trainable_dim={self.trainable_dim}, "
            f"alpha={self.alpha}, owns_v={self._owns_v}"
        )


class TinyLoRAParameterGroup:
    """
    Manages weight tying across multiple TinyLoRA modules.

    When n_tie > 1, multiple modules share the same trainable vector v.
    This class creates the shared parameter(s) and assigns them to modules
    in groups of size n_tie.

    Example — 28 layers × 7 modules = 196 total modules:
      - n_tie=1:    196 independent v vectors → 196*u params
      - n_tie=7:    28 shared v vectors (one per layer) → 28*u params
      - n_tie=196:  1 global shared v → u params (as few as 1!)

    The paper found: "exhaust the u budget (down to u=1) before increasing n_tie"
    """

    def __init__(self, trainable_dim: int, n_tie: int, dtype: torch.dtype = torch.float32):
        self.trainable_dim = trainable_dim
        self.n_tie = n_tie
        self.dtype = dtype
        self._params: list[nn.Parameter] = []
        self._counter = 0

    def get_shared_v(self) -> nn.Parameter:
        """
        Get (or create) the shared v parameter for the current module.

        A new parameter is created every n_tie calls; in between, the
        same parameter is reused.
        """
        if self._counter % self.n_tie == 0:
            v = nn.Parameter(torch.zeros(self.trainable_dim, dtype=self.dtype))
            self._params.append(v)
        self._counter += 1
        return self._params[-1]

    @property
    def all_params(self) -> list[nn.Parameter]:
        """Return all unique shared v parameters."""
        return self._params

    @property
    def total_trainable_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return len(self._params) * self.trainable_dim
