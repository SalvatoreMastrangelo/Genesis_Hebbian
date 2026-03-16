"""
HebbianLastLayer — ABCD Hebbian plasticity rule on existing last layer.
=======================================================================

This module implements the generalised ABCD Hebbian update rule that
modifies the **existing** last linear layer of the frozen actor in-place.
No new layer is created — the pretrained ``Linear(64 -> 5)`` weight matrix
is directly mutated at each simulation step.

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
        The actor's last linear layer (64 -> 5).
    hebbian_rules : dict
        Per-weight ABCD + lambda tensors from ``decode_hebbian_genes``.
        Keys: ``A, B, C, D, lam`` and optionally ``eta``.
    eta : float or Tensor
        Global learning rate (scalar) or per-weight (5, 64) tensor.
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
        device: str | torch.device = "cpu",
    ) -> None:
        self.layer = linear_layer
        self.w_max = w_max
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

    def hebbian_update(self, x: Tensor, y: Tensor) -> None:
        """Apply ABCD rule to modify last layer weights in-place.

        Called AFTER each forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations ``(batch, 64)`` — input to last layer.
        y : Tensor
            Postsynaptic activations ``(batch, 5)`` — output of last layer
            (before tanh scaling).
        """
        # For batched envs, average activations across batch
        x_mean = x.mean(dim=0)  # (64,)
        y_mean = y.mean(dim=0)  # (5,)

        # ABCD update: dW shape (5, 64) matches layer.weight
        dW = self.eta * (
            self.A * torch.outer(y_mean, x_mean)       # classical Hebb  (5, 64)
            + self.B * x_mean.unsqueeze(0)              # presynaptic     (5, 64)
            + self.C * y_mean.unsqueeze(1)              # postsynaptic    (5, 64)
            + self.D                                     # bias/drift      (5, 64)
        )

        # Decay + update (bias untouched)
        self.layer.weight.data.mul_(1.0 - self.lam).add_(dW)
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
        device: str | torch.device = "cpu",
        pop_size: int = 1,
        slice_size: int = 8192,
    ) -> None:
        self.linear_layer = linear_layer
        self.w_max = w_max
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

    def hebbian_update(self, x: Tensor, y: Tensor) -> None:
        """Apply batched ABCD Hebbian update.

        Parameters
        ----------
        x : Tensor
            Presynaptic activations (P*S, in).
        y : Tensor
            Postsynaptic activations (P*S, out).
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

        # Compute delta W: (P, out, in)
        dW = self.eta * (
            self.A * outer
            + self.B * x_mean.unsqueeze(1)   # (P, 1, in) → broadcast to (P, out, in)
            + self.C * y_mean.unsqueeze(2)   # (P, out, 1) → broadcast to (P, out, in)
            + self.D
        )

        # Update: W = W * (1 - lambda) + dW, then clamp
        self.W.mul_(1.0 - self.lam).add_(dW)
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
