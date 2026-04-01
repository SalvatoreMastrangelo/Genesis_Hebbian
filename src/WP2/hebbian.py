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
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        hebbian_rules: Dict[str, Tensor],
        eta: Union[float, Tensor] = 0.01,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.layer = linear_layer
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.device = torch.device(device)

        # Store checkpoint weights (bias untouched throughout)
        self.W_checkpoint = linear_layer.weight.data.clone().to(self.device)

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
        self.layer.weight.data.copy_(self.W_checkpoint)

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply ABCD rule modulated by Oja-like coefficient k.

        Called AFTER each forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations ``(batch, hidden_dim)`` — input to last layer.
        y : Tensor
            Postsynaptic activations ``(batch, num_actions)`` — output of last
            layer (before tanh scaling).
        eps : float
            Small constant for numerical stability in k computation.
        """
        # For batched envs, average activations across the batch dimension
        x_mean = x.mean(dim=0)  # (hidden_dim,)
        y_mean = y.mean(dim=0)  # (num_actions,)

        # Classical Hebb term: (num_actions, hidden_dim)
        xy = torch.outer(y_mean, x_mean)

        # Compute coefficient k if enabled (element-wise normalized by Oja's rule)
        # k_{i,j} = 1 - (y_j² · (w_{i,j} - w_checkpoint_{i,j})) / (x_i · y_j)
        if self.use_oja_coefficient:
            y2 = y_mean.unsqueeze(1) ** 2  # (num_actions, 1) for broadcasting
            w_diff = self.layer.weight.data - self.W_checkpoint  # (num_actions, hidden_dim)
            k = 1.0 - (y2 * w_diff) / (xy + eps)  # (num_actions, hidden_dim)
        else:
            k = 1.0  # Fall back to standard ABCD rule

        # ABCD update modulated by k: dW shape (num_actions, hidden_dim)
        dW = self.eta * k * (
            self.A * xy                         # classical Hebb  (num_actions, hidden_dim)
            + self.B * x_mean.unsqueeze(0)      # presynaptic     (num_actions, hidden_dim)
            + self.C * y_mean.unsqueeze(1)      # postsynaptic    (num_actions, hidden_dim)
            + self.D                             # bias/drift      (num_actions, hidden_dim)
        )

        # Decay (towards checkpoint) + Hebbian update (bias untouched)
        # W = W * (1 - lambda) + lambda * W_checkpoint + dW
        self.layer.weight.data.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint)
        self.layer.weight.data.clamp_(-self.w_max, self.w_max)

    def get_weight_snapshot(self) -> Tensor:
        """Return a detached copy of current layer weights (for logging)."""
        return self.layer.weight.data.detach().clone()


class BatchedHebbianLastLayer:
    """Batched Hebbian plasticity for population-level vectorized evaluation.

    Manages per-individual weight matrices (P, out, in) where P is population size.
    Each of P individuals has its own Hebbian rules (A, B, C, D, lam, eta).

    During rollout with P*S total envs (P individuals, S envs each), the forward
    pass groups envs by individual, computes per-individual Hebbian updates via
    per-individual mean activations, and modifies per-individual weight matrices
    in-place.

    Parameters
    ----------
    linear_layer : nn.Linear
        The frozen actor's last layer (reference to weight buffer).
    hebbian_rules_list : list[dict]
        List of P dicts, each with keys A, B, C, D, lam (and optionally eta).
        hebbian_rules_list[i] contains the rules for individual i.
    eta : float or Tensor
        Global scalar eta or (P, out, in) per-individual eta.
    w_max : float
        Weight clipping bound.
    device : str or torch.device
        Target device.
    pop_size : int
        Number of individuals P.
    slice_size : int
        Number of envs per individual S.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        hebbian_rules_list: list,
        eta: Union[float, Tensor] = 0.01,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
        pop_size: int = 1,
        slice_size: int = 8192,
    ) -> None:
        self.linear_layer = linear_layer
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.device = torch.device(device)
        self.pop_size = pop_size
        self.slice_size = slice_size

        # Infer dimensions from the layer
        out_features, in_features = linear_layer.weight.shape

        # Frozen weights: (out, in)
        self.W_checkpoint = linear_layer.weight.data.clone().to(self.device)

        # Per-individual weight matrices: (P, out, in)
        self.W = self.W_checkpoint.unsqueeze(0).expand(pop_size, -1, -1).clone().to(self.device)

        # Stack ABCD+decay from each individual's rules into (P, out, in) tensors
        self.A = torch.stack([r["A"] for r in hebbian_rules_list]).to(self.device)
        self.B = torch.stack([r["B"] for r in hebbian_rules_list]).to(self.device)
        self.C = torch.stack([r["C"] for r in hebbian_rules_list]).to(self.device)
        self.D = torch.stack([r["D"] for r in hebbian_rules_list]).to(self.device)
        self.lam = torch.stack([r["lam"] for r in hebbian_rules_list]).to(self.device)

        # eta: scalar, (P, out, in), or extracted from rules
        if "eta" in hebbian_rules_list[0]:
            self.eta = torch.stack([r["eta"] for r in hebbian_rules_list]).to(self.device)
        elif isinstance(eta, Tensor):
            self.eta = eta.to(self.device)
        else:
            self.eta = eta  # scalar

    def reset_weights(self) -> None:
        """Reset all per-individual weights to the frozen checkpoint."""
        self.W.copy_(self.W_checkpoint.unsqueeze(0).expand(self.pop_size, -1, -1))

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply batched ABCD Hebbian update modulated by Oja-like coefficient k.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations (P*S, in).
        y : Tensor
            Postsynaptic activations (P*S, out).
        eps : float
            Small constant for numerical stability in k computation.
        """
        P = self.pop_size
        S = self.slice_size
        out, inp = self.W.shape[1], self.W.shape[2]

        # Reshape into per-individual batches
        x_batched = x.view(P, S, inp)      # (P, S, in)
        y_batched = y.view(P, S, out)      # (P, S, out)

        # Per-individual mean activations
        x_mean = x_batched.mean(dim=1)     # (P, in)
        y_mean = y_batched.mean(dim=1)     # (P, out)

        # Classical Hebbian term via einsum: (P, out, in)
        outer = torch.einsum("pi,pj->pij", y_mean, x_mean)

        # Compute coefficient k if enabled: per-individual, element-wise normalized by Oja's rule
        # k_{p,i,j} = 1 - (y_{p,j}² · (w_{p,i,j} - w_checkpoint_{i,j})) / (x_{p,i} · y_{p,j})
        if self.use_oja_coefficient:
            y2 = y_mean.unsqueeze(2) ** 2  # (P, out, 1) for broadcasting
            w_diff = self.W - self.W_checkpoint.unsqueeze(0)  # (P, out, in)
            k = 1.0 - (y2 * w_diff) / (outer + eps)  # (P, out, in)
        else:
            k = 1.0  # Fall back to standard ABCD rule

        # Compute delta W modulated by k: (P, out, in)
        dW = self.eta * k * (
            self.A * outer
            + self.B * x_mean.unsqueeze(1)   # (P, 1, in) → broadcast to (P, out, in)
            + self.C * y_mean.unsqueeze(2)   # (P, out, 1) → broadcast to (P, out, in)
            + self.D
        )

        # Update: W = W * (1 - lambda) + lambda * W_checkpoint + dW, then clamp
        self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint.unsqueeze(0))
        self.W.clamp_(-self.w_max, self.w_max)

        # Write back the **first** individual's weights to the frozen linear_layer
        # (this is only for compatibility with old code paths that expect a single layer.weight)
        # In the batched actor wrapper, we use self.W directly, not this buffer.
        self.linear_layer.weight.data.copy_(self.W[0])

    def get_weight_snapshots(self) -> Tensor:
        """Return per-individual weight snapshots for logging.

        Returns
        -------
        Tensor
            (P, out, in) detached copy of all per-individual weights.
        """
        return self.W.detach().clone()
