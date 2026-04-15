"""
HebbianLastLayer — ABCD Hebbian plasticity rule on the frozen last layer.
=========================================================================

This module implements the generalised ABCD Hebbian update rule for the
actor's last ``Linear(hidden_dim -> num_actions)`` weight matrix.  The
layer itself is NOT mutated — the controller owns its own ``W`` tensor
of shape ``(num_envs, out, in)`` which is updated at each simulation step.

Biases are never modified.
"""

from __future__ import annotations

from typing import Dict, Union

import torch
from torch import Tensor


class HebbianLastLayer:
    """Per-environment Hebbian plasticity controller.

    Owns ``num_envs`` independent last-layer weight matrices (one per
    environment slot) and applies per-environment ABCD updates each step.
    Nothing is shared between environments except the frozen checkpoint.

    Parameters
    ----------
    W_checkpoint : Tensor
        Frozen reference weights ``(out, in)``.  Never modified.
    hebbian_rules : dict of Tensor
        Per-weight ABCD + decay tensors.  Each value may have shape
        ``(out, in)`` (same rules for every environment — will be expanded
        to ``(num_envs, out, in)``) or ``(num_envs, out, in)`` (already
        pre-expanded per-env).
    eta : float or Tensor
        Global learning rate.  May be scalar or broadcastable to
        ``(num_envs, out, in)``.  Ignored if ``hebbian_rules`` contains
        an ``"eta"`` key.
    w_max : float
        Symmetric weight clipping bound.
    use_oja_coefficient : bool
        If True, modulate ABCD updates by Oja-like coefficient that pulls
        drift back toward the checkpoint.
    device : str or torch.device
        Target device.
    num_envs : int
        Number of environment slots.  Each gets its own weight matrix.
    """

    def __init__(
        self,
        W_checkpoint: Tensor,
        hebbian_rules: Dict[str, Tensor],
        eta: Union[float, Tensor] = 0.01,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
        num_envs: int = 1,
    ) -> None:
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.device = torch.device(device)
        self.num_envs = num_envs

        # Frozen checkpoint: (out, in) — never mutated
        self.W_checkpoint = W_checkpoint.detach().clone().to(self.device)

        # Per-environment weights: (num_envs, out, in)
        self.W = self.W_checkpoint.unsqueeze(0).expand(num_envs, -1, -1).contiguous().clone()

        # ABCD + lam — expand (out, in) → (num_envs, out, in) if needed
        self.A = self._expand_rule(hebbian_rules["A"])
        self.B = self._expand_rule(hebbian_rules["B"])
        self.C = self._expand_rule(hebbian_rules["C"])
        self.D = self._expand_rule(hebbian_rules["D"])
        self.lam = self._expand_rule(hebbian_rules["lam"])

        # eta: from rules dict, or scalar, or provided tensor
        if "eta" in hebbian_rules:
            self.eta = self._expand_rule(hebbian_rules["eta"])
        elif isinstance(eta, Tensor):
            self.eta = eta.to(self.device)
        else:
            self.eta = eta  # scalar

    def _expand_rule(self, tensor: Tensor) -> Tensor:
        """Bring a rule tensor to ``(num_envs, out, in)`` on the target device."""
        t = tensor.to(self.device)
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(self.num_envs, -1, -1).contiguous().clone()
        elif t.dim() == 3:
            if t.shape[0] != self.num_envs:
                raise ValueError(
                    f"Rule tensor has {t.shape[0]} envs but expected {self.num_envs}"
                )
            t = t.contiguous().clone()
        else:
            raise ValueError(f"Unexpected rule tensor shape: {t.shape}")
        return t

    def reset_weights(self) -> None:
        """Reset all per-environment weights to the frozen checkpoint."""
        self.W.copy_(self.W_checkpoint.unsqueeze(0).expand_as(self.W))

    def reset_weights_individual(self, k: int, slice_size: int) -> None:
        """Reset the weights for one individual's environment slice.

        Rows ``[k * slice_size, (k + 1) * slice_size)`` are reset to the
        frozen checkpoint; all other rows are left untouched.
        """
        start, end = k * slice_size, (k + 1) * slice_size
        self.W[start:end].copy_(self.W_checkpoint.unsqueeze(0).expand(end - start, -1, -1))

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply per-environment ABCD update (no averaging).

        Parameters
        ----------
        x : Tensor
            Presynaptic activations ``(num_envs, in)``.
        y : Tensor
            Postsynaptic activations ``(num_envs, out)``.
        eps : float
            Stability epsilon for the Oja coefficient.
        """
        N = self.num_envs
        if x.shape[0] != N or y.shape[0] != N:
            raise ValueError(
                f"Expected batch={N}, got x={tuple(x.shape)}, y={tuple(y.shape)}"
            )

        # Classical Hebb outer product per env: (N, out, in)
        xy = torch.einsum("ei,ej->eij", y, x)

        if self.use_oja_coefficient:
            y2 = y.unsqueeze(2) ** 2  # (N, out, 1)
            w_diff = self.W - self.W_checkpoint.unsqueeze(0)  # (N, out, in)
            k = 1.0 - (y2 * w_diff) / (xy + eps)
        else:
            k = 1.0

        dW = self.eta * k * (
            self.A * xy
            + self.B * x.unsqueeze(1)  # (N, 1, in)
            + self.C * y.unsqueeze(2)  # (N, out, 1)
            + self.D
        )

        # W = W * (1 - lam) + lam * W_checkpoint + dW, then clip
        self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint.unsqueeze(0))
        self.W.clamp_(-self.w_max, self.w_max)

    def get_weight_snapshot(self) -> Tensor:
        """Return a detached copy of current per-env weights ``(num_envs, out, in)``."""
        return self.W.detach().clone()
