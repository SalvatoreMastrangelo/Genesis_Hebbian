"""
HebbianLastLayer — ABCD Hebbian plasticity rule on existing last layer.
=======================================================================

This module implements the generalised ABCD Hebbian update rule that
modifies the **existing** last linear layer of the frozen actor in-place.
No new layer is created — the frozen ``Linear(hidden_dim -> num_actions)``
weight matrix (typically ``Linear(64 -> 7)``) is directly mutated at each
simulation step.

Biases are **never** modified.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn


class HebbianLastLayer:
    """Wraps the actor's existing last Linear layer with Hebbian plasticity.

    Modifies ONLY ``.weight`` in-place each step.  Bias is never modified.

    This is NOT an ``nn.Module`` — it is a controller that mutates the
    existing layer's weight buffer during rollout.

    Now supports **per-environment weight matrices** with independent updates.
    Each environment has its own weight matrix and receives its own Hebbian update
    based on its individual activations (no averaging).

    Parameters
    ----------
    linear_layer : nn.Linear
        The actor's last linear layer (hidden_dim -> num_actions).
    hebbian_rules : dict
        Per-weight ABCD + lambda tensors from ``decode_hebbian_genes``.
        Keys: ``A, B, C, D, lam`` and optionally ``eta``.
    eta : float or Tensor
        Global learning rate (scalar) or per-weight (num_actions, hidden_dim) tensor.
    w_max : float
        Symmetric weight clipping bound.
    device : str or torch.device
        Target device for all tensors.
    num_envs : int
        Number of environments. If > 1, maintains per-environment weight matrices.
        Default: 1 (backward compatible with single-individual case).
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        hebbian_rules: Dict[str, Tensor],
        eta: Union[float, Tensor] = 0.01,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
        num_envs: int = 1,
    ) -> None:
        self.layer = linear_layer
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.device = torch.device(device)
        self.num_envs = num_envs

        # Infer dimensions
        num_actions, hidden_dim = linear_layer.weight.data.shape

        # Store checkpoint weights (bias untouched throughout)
        W_checkpoint_single = linear_layer.weight.data.clone().to(self.device)

        # Per-environment weight matrices: (num_envs, num_actions, hidden_dim)
        if num_envs > 1:
            self.W_checkpoint = W_checkpoint_single.unsqueeze(0).expand(num_envs, -1, -1).clone()
            self.W = W_checkpoint_single.unsqueeze(0).expand(num_envs, -1, -1).clone()
        else:
            # Single environment: store as (num_actions, hidden_dim) for backward compat
            self.W_checkpoint = W_checkpoint_single
            self.W = W_checkpoint_single.clone()

        # Per-weight ABCD + decay
        self.A = hebbian_rules["A"].to(self.device)
        self.B = hebbian_rules["B"].to(self.device)
        self.C = hebbian_rules["C"].to(self.device)
        self.D = hebbian_rules["D"].to(self.device)
        self.lam = hebbian_rules["lam"].to(self.device)

        # eta: either scalar (global) or (out, in) tensor (per-weight)
        if "eta" in hebbian_rules:
            self.eta = hebbian_rules["eta"].to(self.device)
        elif isinstance(eta, Tensor):
            self.eta = eta.to(self.device)
        else:
            self.eta = eta

    def reset_weights(self) -> None:
        """Reset layer weights to checkpoint values.

        Called at: episode start AND generation start.
        Bias is never modified or reset.
        """
        if self.num_envs > 1:
            self.W.copy_(self.W_checkpoint)
        else:
            self.layer.weight.data.copy_(self.W_checkpoint)
            self.W.copy_(self.W_checkpoint)

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply ABCD rule with per-environment updates (no averaging).

        Called AFTER each forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations ``(batch, hidden_dim)`` — input to last layer.
            If num_envs > 1, batch dimension must equal num_envs.
        y : Tensor
            Postsynaptic activations ``(batch, num_actions)`` — output of last
            layer (before tanh scaling).
            If num_envs > 1, batch dimension must equal num_envs.
        eps : float
            Small constant for numerical stability in k computation.
        """
        if self.num_envs == 1:
            # Single environment: use original logic but operate on self.W
            x_mean = x.mean(dim=0)  # (hidden_dim,)
            y_mean = y.mean(dim=0)  # (num_actions,)

            # Classical Hebb term: (num_actions, hidden_dim)
            xy = torch.outer(y_mean, x_mean)

            # Compute coefficient k if enabled
            if self.use_oja_coefficient:
                y2 = y_mean.unsqueeze(1) ** 2  # (num_actions, 1)
                w_diff = self.W - self.W_checkpoint  # (num_actions, hidden_dim)
                k = 1.0 - (y2 * w_diff) / (xy + eps)  # (num_actions, hidden_dim)
            else:
                k = 1.0

            # ABCD update modulated by k
            dW = self.eta * k * (
                self.A * xy
                + self.B * x_mean.unsqueeze(0)
                + self.C * y_mean.unsqueeze(1)
                + self.D
            )

            # Update weights
            self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint)
            self.W.clamp_(-self.w_max, self.w_max)

            # Write back to layer
            self.layer.weight.data.copy_(self.W)
        else:
            # Multiple environments: per-environment updates (no averaging)
            batch_size = x.shape[0]
            if batch_size != self.num_envs:
                raise ValueError(
                    f"Expected batch_size={self.num_envs}, got {batch_size}"
                )

            # Each environment's activations: (num_envs, hidden_dim/num_actions)
            # Classical Hebb term per-environment: (num_envs, num_actions, hidden_dim)
            xy = torch.einsum("ei,ej->eij", y, x)

            # Compute coefficient k if enabled (per-environment)
            if self.use_oja_coefficient:
                y2 = y.unsqueeze(2) ** 2  # (num_envs, num_actions, 1)
                w_diff = self.W - self.W_checkpoint  # (num_envs, num_actions, hidden_dim)
                k = 1.0 - (y2 * w_diff) / (xy + eps)  # (num_envs, num_actions, hidden_dim)
            else:
                k = 1.0

            # ABCD update modulated by k (per-environment)
            dW = self.eta * k * (
                self.A * xy
                + self.B * x.unsqueeze(1)  # (num_envs, 1, hidden_dim)
                + self.C * y.unsqueeze(2)  # (num_envs, num_actions, 1)
                + self.D
            )

            # Update per-environment weights
            self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint)
            self.W.clamp_(-self.w_max, self.w_max)

            # Write first environment's weights back to layer (for compatibility)
            self.layer.weight.data.copy_(self.W[0])

    def get_weight_snapshot(self) -> Tensor:
        """Return a detached copy of current layer weights (for logging).

        If num_envs > 1, returns all per-environment weights (num_envs, out, in).
        """
        return self.W.detach().clone()


class BatchedHebbianLastLayer:
    """Batched Hebbian plasticity with **per-environment** weight matrices.

    Now supports true per-environment updates where each environment (out of P*S total)
    has its own weight matrix and receives its own Hebbian update based on its
    individual activations (no averaging across environments).

    Can be used in two modes:
    1. Population evaluation: P individuals, S envs each, updates per-environment
    2. Direct per-environment: N environments, updates per-environment

    Parameters
    ----------
    linear_layer : nn.Linear
        The frozen actor's last layer (reference to weight buffer).
    hebbian_rules_list : list[dict]
        List of N dicts, each with keys A, B, C, D, lam (and optionally eta).
        hebbian_rules_list[i] contains the rules for environment i.
        If using population mode, hebbian_rules_list should have length P (one per individual),
        which will be replicated S times.
    eta : float or Tensor
        Global scalar eta or (N, out, in) per-environment eta.
    w_max : float
        Weight clipping bound.
    device : str or torch.device
        Target device.
    num_envs : int
        Total number of environments (P*S). Each gets its own weight matrix.
    pop_size : int (optional)
        Number of individuals P. If provided, rules will be replicated S times.
    slice_size : int (optional)
        Number of envs per individual S. Only used if pop_size is provided.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        hebbian_rules_list: list,
        eta: Union[float, Tensor] = 0.01,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
        num_envs: int = None,
        pop_size: int = None,
        slice_size: int = None,
    ) -> None:
        self.linear_layer = linear_layer
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.device = torch.device(device)

        # Infer dimensions from the layer
        out_features, in_features = linear_layer.weight.shape

        # Frozen weights: (out, in)
        self.W_checkpoint = linear_layer.weight.data.clone().to(self.device)

        # Handle num_envs from pop_size and slice_size if provided
        if num_envs is None:
            if pop_size is not None and slice_size is not None:
                num_envs = pop_size * slice_size
                self.pop_size = pop_size
                self.slice_size = slice_size
            else:
                num_envs = len(hebbian_rules_list)
                self.pop_size = 1
                self.slice_size = num_envs
        else:
            self.pop_size = pop_size if pop_size is not None else 1
            self.slice_size = slice_size if slice_size is not None else num_envs

        self.num_envs = num_envs

        # Expand rules if population mode (replicate each individual's rules S times)
        if pop_size is not None and len(hebbian_rules_list) == pop_size:
            expanded_rules = []
            for i in range(pop_size):
                for _ in range(slice_size):
                    expanded_rules.append(hebbian_rules_list[i])
            hebbian_rules_list = expanded_rules

        # Per-environment weight matrices: (N, out, in)
        self.W = self.W_checkpoint.unsqueeze(0).expand(num_envs, -1, -1).clone().to(self.device)

        # Stack ABCD+decay from each environment's rules into (N, out, in) tensors
        self.A = torch.stack([r["A"] for r in hebbian_rules_list]).to(self.device)
        self.B = torch.stack([r["B"] for r in hebbian_rules_list]).to(self.device)
        self.C = torch.stack([r["C"] for r in hebbian_rules_list]).to(self.device)
        self.D = torch.stack([r["D"] for r in hebbian_rules_list]).to(self.device)
        self.lam = torch.stack([r["lam"] for r in hebbian_rules_list]).to(self.device)

        # eta: scalar, (N, out, in), or extracted from rules
        if "eta" in hebbian_rules_list[0]:
            self.eta = torch.stack([r["eta"] for r in hebbian_rules_list]).to(self.device)
        elif isinstance(eta, Tensor):
            self.eta = eta.to(self.device)
        else:
            self.eta = eta  # scalar

    def reset_weights(self) -> None:
        """Reset all per-environment weights to the frozen checkpoint."""
        self.W.copy_(self.W_checkpoint.unsqueeze(0).expand(self.num_envs, -1, -1))

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply per-environment ABCD Hebbian updates (NO averaging).

        Each environment receives its own update based on its individual activations.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations (N, in) where N = num_envs.
            Each row is one environment's activations.
        y : Tensor
            Postsynaptic activations (N, out) where N = num_envs.
            Each row is one environment's outputs.
        eps : float
            Small constant for numerical stability in k computation.
        """
        N = self.num_envs
        out, inp = self.W.shape[1], self.W.shape[2]

        if x.shape[0] != N or y.shape[0] != N:
            raise ValueError(
                f"Expected batch_size={N}, got x.shape={x.shape}, y.shape={y.shape}"
            )

        # Classical Hebbian term per-environment: (N, out, in)
        # outer[e,i,j] = y[e,i] * x[e,j]
        outer = torch.einsum("ei,ej->eij", y, x)

        # Compute coefficient k if enabled: per-environment, element-wise normalized by Oja's rule
        # k_{e,i,j} = 1 - (y_{e,j}² · (w_{e,i,j} - w_checkpoint_{i,j})) / (outer_{e,i,j} + eps)
        if self.use_oja_coefficient:
            y2 = y.unsqueeze(2) ** 2  # (N, out, 1) for broadcasting
            w_diff = self.W - self.W_checkpoint.unsqueeze(0)  # (N, out, in)
            k = 1.0 - (y2 * w_diff) / (outer + eps)  # (N, out, in)
        else:
            k = 1.0  # Fall back to standard ABCD rule

        # Compute delta W modulated by k: (N, out, in)
        dW = self.eta * k * (
            self.A * outer
            + self.B * x.unsqueeze(1)   # (N, 1, in) → broadcast to (N, out, in)
            + self.C * y.unsqueeze(2)   # (N, out, 1) → broadcast to (N, out, in)
            + self.D
        )

        # Update: W = W * (1 - lambda) + lambda * W_checkpoint + dW, then clamp
        self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint.unsqueeze(0))
        self.W.clamp_(-self.w_max, self.w_max)

        # Write back the **first** environment's weights to the frozen linear_layer
        # (this is only for compatibility with old code paths that expect a single layer.weight)
        # In the batched actor wrapper, we use self.W directly, not this buffer.
        self.linear_layer.weight.data.copy_(self.W[0])

    def get_weight_snapshots(self) -> Tensor:
        """Return per-environment weight snapshots for logging.

        Returns
        -------
        Tensor
            (N, out, in) detached copy of all per-environment weights.
        """
        return self.W.detach().clone()
