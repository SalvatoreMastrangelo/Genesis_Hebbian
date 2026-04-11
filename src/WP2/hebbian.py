"""
Hebbian Rules Creator — Generate ABCD plasticity rules for a loaded controller.
===============================================================================

This module provides utilities to create Hebbian ABCD (and optionally decay/eta)
tensors for the last layer of a loaded actor checkpoint, with flexible initialization
strategies.

Typical workflow:
    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")
    rules = create_hebbian_rules(
        actor,
        init_method="uniform",
        init_range=(-0.05, 0.05),
        add_decay=True,
        device="cuda"
    )
    # rules is a dict with keys: A, B, C, D, lam (+ optionally eta)
    # ready to pass to HebbianLastLayer or evolution
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from .checkpoint_loader import get_actor_dimensions


def create_hebbian_rules(
    actor: nn.Module,
    init_method: Literal["uniform", "zero"] = "zero",
    init_range: Tuple[float, float] = (-0.1, 0.1),
    add_decay: bool = True,
    decay_init: Literal["uniform", "zero"] = "zero",
    decay_range: Tuple[float, float] = (0.0, 0.1),
    add_eta: bool = False,
    eta_value: float = 0.01,
    device: str | torch.device = "cpu",
) -> Dict[str, Tensor]:
    """Create Hebbian ABCD rules (and optionally decay/eta) for the actor's last layer.

    Parameters
    ----------
    actor : nn.Module
        The loaded actor network (typically from load_wp1_actor).
    init_method : {"uniform", "zero"}
        Initialization method for A, B, C, D tensors.
        - "uniform": uniformly sample from [init_range[0], init_range[1]]
        - "zero": initialize to 0
    init_range : tuple[float, float]
        Range for uniform initialization (low, high). Default: (-0.1, 0.1).
    add_decay : bool
        Whether to add decay (lam) tensor. Default: True.
    decay_init : {"uniform", "zero"}
        Initialization method for decay tensor.
        - "uniform": uniformly sample from [decay_range[0], decay_range[1]]
        - "zero": initialize to 0
    decay_range : tuple[float, float]
        Range for decay uniform initialization (low, high). Default: (0.0, 0.1).
    add_eta : bool
        Whether to add per-weight learning rate (eta) tensor. Default: False.
        If False, eta is assumed to be a global scalar parameter elsewhere.
    eta_value : float
        Default learning rate value if add_eta=True. Default: 0.01.
    device : str or torch.device
        Device to place tensors on. Default: "cpu".

    Returns
    -------
    rules : dict[str, Tensor]
        Dictionary with keys:
        - "A": (num_actions, hidden_dim) Hebbian coefficient
        - "B": (num_actions, hidden_dim) presynaptic bias coefficient
        - "C": (num_actions, hidden_dim) postsynaptic bias coefficient
        - "D": (num_actions, hidden_dim) constant drift coefficient
        - "lam": (num_actions, hidden_dim) decay/reset coefficient [if add_decay=True]
        - "eta": (num_actions, hidden_dim) learning rate [if add_eta=True]

    Raises
    ------
    RuntimeError
        If actor dimensions cannot be inferred or last layer not found.
    ValueError
        If init_method or decay_init not in {"uniform", "zero"}.
    """
    device = torch.device(device)

    # Validate initialization methods
    if init_method not in ("uniform", "zero"):
        raise ValueError(f"init_method must be 'uniform' or 'zero', got {init_method}")
    if decay_init not in ("uniform", "zero"):
        raise ValueError(f"decay_init must be 'uniform' or 'zero', got {decay_init}")

    # Get last layer dimensions
    hidden_dim, num_actions = get_actor_dimensions(actor)
    shape = (num_actions, hidden_dim)

    # Initialize A, B, C, D
    rules = {}
    for coeff in ["A", "B", "C", "D"]:
        if init_method == "uniform":
            tensor = torch.empty(shape, device=device).uniform_(
                init_range[0], init_range[1]
            )
        else:  # zero
            tensor = torch.zeros(shape, device=device)
        rules[coeff] = tensor

    # Initialize decay (lam)
    if add_decay:
        if decay_init == "uniform":
            lam = torch.empty(shape, device=device).uniform_(
                decay_range[0], decay_range[1]
            )
        else:  # zero
            lam = torch.zeros(shape, device=device)
        rules["lam"] = lam

    # Initialize eta
    if add_eta:
        eta = torch.full(shape, eta_value, device=device)
        rules["eta"] = eta

    return rules


def attach_hebbian_rules_to_actor(
    actor: nn.Module,
    rules: Dict[str, Tensor] | None = None,
    **create_kwargs,
) -> nn.Module:
    """Attach Hebbian rules as buffers to the actor module.

    If rules are not provided, they are created using create_hebbian_rules
    with the provided kwargs.

    Parameters
    ----------
    actor : nn.Module
        The loaded actor network.
    rules : dict[str, Tensor], optional
        Pre-computed Hebbian rules. If None, created via create_hebbian_rules
        with kwargs. Default: None.
    **create_kwargs
        Additional kwargs passed to create_hebbian_rules if rules is None.
        Examples: init_method, init_range, add_decay, device.

    Returns
    -------
    actor : nn.Module
        The same actor module, now with hebbian_rules attribute containing
        a dict of registered buffers A, B, C, D, lam, etc.

    Examples
    --------
    >>> actor = attach_hebbian_rules_to_actor(
    ...     actor,
    ...     init_method="uniform",
    ...     init_range=(-0.05, 0.05),
    ...     device="cuda"
    ... )
    >>> actor.hebbian_rules["A"]  # access rules
    """
    # Create rules if not provided
    if rules is None:
        rules = create_hebbian_rules(actor, **create_kwargs)

    # Register each rule as a buffer (persistent, non-trainable tensor)
    hebbian_rules_dict = {}
    for name, tensor in rules.items():
        buffer_name = f"hebbian_{name}"
        actor.register_buffer(buffer_name, tensor)
        hebbian_rules_dict[name] = tensor

    # Store reference for easy access
    actor.hebbian_rules = hebbian_rules_dict

    return actor


def extract_hebbian_rules_from_actor(actor: nn.Module) -> Dict[str, Tensor]:
    """Extract Hebbian rules from an actor that has them attached.

    Parameters
    ----------
    actor : nn.Module
        Actor with hebbian rules attached via attach_hebbian_rules_to_actor.

    Returns
    -------
    rules : dict[str, Tensor]
        Dictionary with keys A, B, C, D, lam, etc.

    Raises
    ------
    RuntimeError
        If hebbian_rules attribute not found on actor.
    """
    if not hasattr(actor, "hebbian_rules"):
        raise RuntimeError(
            "Actor does not have hebbian_rules attribute. "
            "Use attach_hebbian_rules_to_actor first."
        )
    return actor.hebbian_rules




def get_hebbian_genome_dim(
    actor: nn.Module,
    add_decay: bool = True,
    add_eta: bool = False,
) -> int:
    """Calculate genome dimension per individual for evolving Hebbian rules.

    Each individual in the population needs to evolve its own set of ABCD rules
    (and optionally lam and eta). This function returns how many values each
    individual's genome needs for these rules.

    For a last layer with `n_weights = num_actions * hidden_dim`:
    - Base: 4 × n_weights (A, B, C, D)
    - With decay: + n_weights (lam)
    - With eta: + n_weights (eta)

    Parameters
    ----------
    actor : nn.Module
        The actor network (to infer hidden_dim and num_actions).
    add_decay : bool
        Whether decay (lam) is evolved per individual. Default: True.
    add_eta : bool
        Whether eta is evolved per individual. Default: False.

    Returns
    -------
    genome_dim : int
        Total Hebbian genome dimension for one individual's controller.

    Examples
    --------
    >>> hidden_dim, num_actions = 64, 7
    >>> n_weights = hidden_dim * num_actions  # 448
    >>> dim_base = 4 * 448  # 1792
    >>> dim_with_decay = 5 * 448  # 2240
    >>> dim_with_both = 6 * 448  # 2688
    """
    hidden_dim, num_actions = get_actor_dimensions(actor)
    n_weights = hidden_dim * num_actions

    # Base: A, B, C, D
    dim = 4 * n_weights

    if add_decay:
        dim += n_weights
    if add_eta:
        dim += n_weights

    return dim


class HebbianController:
    """Self-contained controller with frozen actor + per-controller Hebbian rules and weights.

    Each controller has its own independent:
    - Hebbian rules (A, B, C, D, lam, eta)
    - Weight matrix (evolved separately)
    - Forward pass logic with Hebbian updates

    Easy to assign one controller per drone.

    Usage
    -----
    ```python
    # Create one controller per drone
    controllers = []
    for ind_idx in range(population_size):
        rules = {
            "A": evolved_rules_A[ind_idx],
            "B": evolved_rules_B[ind_idx],
            "C": evolved_rules_C[ind_idx],
            "D": evolved_rules_D[ind_idx],
            "lam": evolved_lam[ind_idx],
            "eta": 0.01,  # scalar or per-weight
        }
        ctrl = HebbianController(
            actor=frozen_actor,
            rules=rules,
            w_checkpoint=w_checkpoint,  # frozen baseline weights
            device="cuda"
        )
        controllers.append(ctrl)
        drone_list[ind_idx].controller = ctrl

    # During simulation
    for step in range(episode_length):
        for i, ctrl in enumerate(controllers):
            obs = env.get_obs(i)
            x, y = get_activations(obs)

            # Forward with current plasticity
            actions = ctrl(obs)

            # Update weights based on activations
            ctrl.hebbian_update(x, y)

            env.step(i, actions)
    ```

    Parameters
    ----------
    actor : nn.Module
        The frozen actor (from load_wp1_actor). Used only for forward pass.
    rules : dict[str, Tensor]
        Per-controller Hebbian rules with keys:
        - "A": (num_actions, hidden_dim) — Hebbian coefficient
        - "B": (num_actions, hidden_dim) — presynaptic bias
        - "C": (num_actions, hidden_dim) — postsynaptic bias
        - "D": (num_actions, hidden_dim) — constant drift
        - "lam": (num_actions, hidden_dim) — decay/reset to checkpoint
        - "eta": float or (num_actions, hidden_dim) — learning rate
    w_checkpoint : Tensor
        Frozen baseline weights (num_actions, hidden_dim) to reset/decay toward.
        Usually from the pretrained actor's last layer.
    w_max : float
        Weight clipping bound. Default: 3.0
    use_oja_coefficient : bool
        Whether to apply Oja normalization during updates. Default: True
    device : str or torch.device
        Device for computation. Default: "cpu"
    """

    def __init__(
        self,
        actor: nn.Module,
        rules: Dict[str, Tensor],
        w_checkpoint: Tensor,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = actor
        self.device = torch.device(device)
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient

        # Store per-controller rules
        self.A = rules["A"].to(self.device)
        self.B = rules["B"].to(self.device)
        self.C = rules["C"].to(self.device)
        self.D = rules["D"].to(self.device)
        self.lam = rules["lam"].to(self.device)

        # eta: scalar or per-weight
        if isinstance(rules.get("eta"), Tensor):
            self.eta = rules["eta"].to(self.device)
        else:
            self.eta = rules.get("eta", 0.01)  # default 0.01

        # Per-controller weight matrix (starts as copy of checkpoint)
        self.W_checkpoint = w_checkpoint.clone().to(self.device)
        self.W = self.W_checkpoint.clone()

        # Cache the last layer reference for forward pass
        self.last_layer = None
        for module in reversed(list(self.actor.actor.modules())):
            if isinstance(module, nn.Linear):
                self.last_layer = module
                break

        if self.last_layer is None:
            raise RuntimeError("Could not find last Linear layer in actor network")

    def reset_weights(self) -> None:
        """Reset this controller's weights to the frozen checkpoint."""
        self.W.copy_(self.W_checkpoint)

    def __call__(self, obs: Tensor, hidden_states=None) -> Tensor:
        """Forward pass using this controller's Hebbian-updated weights.

        Parameters
        ----------
        obs : Tensor
            Observation(s) to process.
            - Shape: (obs_dim,) for single observation
            - Shape: (batch, obs_dim) for batched observations
        hidden_states : Tensor, optional
            LSTM hidden states (h, c) tuple. If None, initialized internally.
            Each element shape: (num_layers, batch, rnn_hidden_size)

        Returns
        -------
        actions : Tensor
            Scaled actions matching the actor's action format.
            - Shape: (num_actions,) for single obs
            - Shape: (batch, num_actions) for batched obs
        """
        # Ensure obs is on the correct device
        if obs.device != self.device:
            obs = obs.to(self.device)

        # Ensure batch dimension for LSTM processing
        obs_expanded = obs.unsqueeze(0) if obs.dim() == 1 else obs.unsqueeze(0)

        # Get hidden activations from LSTM + MLP backbone
        with torch.no_grad():
            # LSTM step
            hidden_in = self.actor.memory_a(obs_expanded, hidden_states=hidden_states)

            # MLP layers before last
            x = hidden_in
            for layer in self.actor.actor.children():
                if layer is self.last_layer:
                    break
                x = layer(x)

            # Remove batch dimension
            if x.dim() == 3:
                x = x.squeeze(0)

        # Apply last layer with this controller's weights
        z = torch.nn.functional.linear(x, self.W, bias=self.last_layer.bias)

        # Apply tanh and scaling
        a = torch.tanh(z)
        actions = self.actor._scale(a)

        return actions

    def hebbian_update(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> None:
        """Apply Hebbian plasticity update to this controller's weights.

        Parameters
        ----------
        x : Tensor
            Presynaptic activation (hidden_dim,) for this controller's environment.
        y : Tensor
            Postsynaptic activation (num_actions,) for this controller's environment.
        eps : float
            Small constant for numerical stability. Default: 1e-8
        """
        # Ensure tensors are on correct device
        x = x.to(self.device)
        y = y.to(self.device)

        # Outer product: (num_actions, hidden_dim)
        outer = torch.outer(y, x)

        # Oja coefficient if enabled: modulate by weight drift from checkpoint
        if self.use_oja_coefficient:
            y2 = y.unsqueeze(1) ** 2  # (num_actions, 1)
            w_diff = self.W - self.W_checkpoint  # (num_actions, hidden_dim)
            k = 1.0 - (y2 * w_diff) / (outer + eps)  # (num_actions, hidden_dim)
        else:
            k = 1.0

        # Compute delta W: (num_actions, hidden_dim)
        dW = self.eta * k * (
            self.A * outer
            + self.B * x.unsqueeze(0)      # (1, hidden_dim) → broadcast
            + self.C * y.unsqueeze(1)      # (num_actions, 1) → broadcast
            + self.D
        )

        # Update weights: W = W * (1 - lam) + lam * W_checkpoint + dW
        self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint)
        self.W.clamp_(-self.w_max, self.w_max)


class HebbianControllerBatch:
    """Batch forward & Hebbian updates for multiple controllers with temporal continuity.

    Runs all controllers in parallel with **persistent LSTM hidden states** across timesteps.
    This maintains temporal continuity within an episode while allowing batch processing.

    Each controller still maintains its own independent rules and weights.
    Hidden states accumulate across calls within an episode and reset at episode boundaries.

    Usage
    -----
    ```python
    # Create controllers
    controllers = [HebbianController(...) for _ in range(num_envs)]
    batch = HebbianControllerBatch(controllers)

    # Simulate episode
    batch.reset_hidden_states()  # Start fresh at episode begin
    for step in range(episode_length):
        obs_batch = torch.stack([obs for each env])  # (num_envs, obs_dim)
        x_batch = torch.stack([x for each env])      # (num_envs, hidden_dim)
        y_batch = torch.stack([y for each env])      # (num_envs, num_actions)

        # Batched forward (LSTM state carries over from previous step)
        actions_batch = batch.forward_batch(obs_batch)

        # Batched Hebbian update
        batch.hebbian_update(x_batch, y_batch)

        env.step(actions_batch)

        # LSTM hidden states persist to next iteration - temporal continuity!
    ```

    Parameters
    ----------
    controllers : list[HebbianController]
        List of HebbianController instances to update in parallel.
        All must have same architecture (num_actions, hidden_dim).
    device : str or torch.device, optional
        Device for computation. If None, uses device of first controller.
    """

    def __init__(
        self,
        controllers: list,
        device: str | torch.device | None = None,
    ) -> None:
        if not controllers:
            raise ValueError("Must provide at least one controller")

        self.controllers = controllers
        self.num_controllers = len(controllers)

        # Infer device from first controller if not specified
        if device is None:
            device = controllers[0].device
        self.device = torch.device(device)

        # Verify all controllers have same architecture
        num_actions, hidden_dim = controllers[0].W.shape
        for i, ctrl in enumerate(controllers[1:], 1):
            out, inp = ctrl.W.shape
            if (out, inp) != (num_actions, hidden_dim):
                raise ValueError(
                    f"Controller {i} has shape ({out}, {inp}), "
                    f"expected ({num_actions}, {hidden_dim})"
                )

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Per-controller hidden states for batch processing
        # Shape: (num_controllers, num_layers, hidden_dim) for both h and c
        # This allows independent reset per controller while maintaining batch efficiency
        self._h_batch = None  # Per-controller h states
        self._c_batch = None  # Per-controller c states
        self._hidden_states_initialized = False

    def hebbian_update_batch(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        eps: float = 1e-8,
    ) -> None:
        """Update all controllers' weights in parallel using batched activations.

        Parameters
        ----------
        x_batch : Tensor
            Presynaptic activations (num_controllers, hidden_dim).
            Each row is activations for one environment's controller.
        y_batch : Tensor
            Postsynaptic activations (num_controllers, num_actions).
            Each row is outputs for one environment's controller.
        eps : float
            Small constant for numerical stability. Default: 1e-8
        """
        # Validate shapes
        if x_batch.shape != (self.num_controllers, self.hidden_dim):
            raise ValueError(
                f"x_batch shape {x_batch.shape} doesn't match "
                f"({self.num_controllers}, {self.hidden_dim})"
            )
        if y_batch.shape != (self.num_controllers, self.num_actions):
            raise ValueError(
                f"y_batch shape {y_batch.shape} doesn't match "
                f"({self.num_controllers}, {self.num_actions})"
            )

        # Ensure tensors on correct device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Stack all weight matrices: (num_controllers, num_actions, hidden_dim)
        W_batch = torch.stack([ctrl.W for ctrl in self.controllers])
        W_checkpoint_batch = torch.stack(
            [ctrl.W_checkpoint for ctrl in self.controllers]
        )

        # Batch outer product: (num_controllers, num_actions, hidden_dim)
        # outer[e,i,j] = y_batch[e,i] * x_batch[e,j]
        outer = torch.einsum("ei,ej->eij", y_batch, x_batch)

        # Compute Oja coefficient if enabled
        if self.controllers[0].use_oja_coefficient:
            y2 = y_batch.unsqueeze(2) ** 2  # (num_controllers, num_actions, 1)
            w_diff = W_batch - W_checkpoint_batch  # (num_controllers, num_actions, hidden_dim)
            k = 1.0 - (y2 * w_diff) / (outer + eps)
        else:
            k = 1.0

        # Compute delta W for all controllers: (num_controllers, num_actions, hidden_dim)
        # Each controller uses its own A, B, C, D, lam rules
        dW_batch = []
        for i, ctrl in enumerate(self.controllers):
            dW = ctrl.eta * k[i] * (
                ctrl.A * outer[i]
                + ctrl.B * x_batch[i].unsqueeze(0)      # broadcast
                + ctrl.C * y_batch[i].unsqueeze(1)      # broadcast
                + ctrl.D
            )
            dW_batch.append(dW)

        dW_batch = torch.stack(dW_batch)

        # Update all weights: W = W * (1 - lam) + lam * W_checkpoint + dW
        # Again, each controller uses its own lam
        for i, ctrl in enumerate(self.controllers):
            W_batch[i].mul_(1.0 - ctrl.lam).add_(dW_batch[i]).add_(
                ctrl.lam * W_checkpoint_batch[i]
            )
            W_batch[i].clamp_(-ctrl.w_max, ctrl.w_max)

        # Write updated weights back to controllers
        for i, ctrl in enumerate(self.controllers):
            ctrl.W.copy_(W_batch[i])

    def hebbian_update(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        eps: float = 1e-8,
    ) -> None:
        """Alias for hebbian_update_batch (same behavior)."""
        self.hebbian_update_batch(x_batch, y_batch, eps)

    def reset_hidden_states(self, controller_indices: list | None = None) -> None:
        """Reset LSTM hidden states for specified controllers or all.

        Allows independent reset of hidden states per controller, useful when
        individual environments reset without affecting others.

        Parameters
        ----------
        controller_indices : list, optional
            Indices of controllers to reset (e.g., [0, 2]). If None, resets all.
            Default: None (reset all)

        Usage
        -----
        ```python
        batch = HebbianControllerBatch(controllers)
        batch.reset_hidden_states()  # Reset all at episode start

        for step in range(episode_length):
            actions = batch.forward_batch(obs_batch)
            batch.hebbian_update(x_batch, y_batch)

            # Individual env resets mid-episode
            done_indices = [0, 3]  # Envs 0 and 3 finished
            batch.reset_hidden_states(done_indices)  # Only reset those
        ```
        """
        if not self._hidden_states_initialized:
            # Not yet initialized, nothing to reset
            return

        if controller_indices is None:
            # Reset all controllers
            self._h_batch.zero_()
            self._c_batch.zero_()
        else:
            # Reset specific controllers
            for idx in controller_indices:
                if not (0 <= idx < self.num_controllers):
                    raise IndexError(f"Controller index {idx} out of range [0, {self.num_controllers})")
                self._h_batch[idx].zero_()
                self._c_batch[idx].zero_()

    def reset_controller_state_and_weights(
        self,
        controller_indices: int | list | None = None,
    ) -> None:
        """Reset both hidden state and weights for specified controllers.

        Convenience method that resets LSTM hidden states and Hebbian weights
        back to checkpoint for one or more controllers. Useful for mid-episode
        environment resets.

        Parameters
        ----------
        controller_indices : int, list, or None
            Index or list of indices to reset.
            - int: Reset single controller at that index
            - list: Reset all controllers in list
            - None: Reset all controllers
            Default: None (reset all)

        Raises
        ------
        IndexError
            If any index is out of range [0, num_controllers)

        Usage
        -----
        ```python
        batch = HebbianControllerBatch(controllers)
        batch.reset_hidden_states()  # Episode start

        for step in range(episode_length):
            actions = batch.forward_batch(obs_batch)
            batch.hebbian_update(x_batch, y_batch)

            # Env 3 finished early, reset it completely
            batch.reset_controller_state_and_weights(3)

            # Or reset multiple at once
            batch.reset_controller_state_and_weights([1, 5, 7])
        ```
        """
        # Normalize input to list
        if controller_indices is None:
            indices = list(range(self.num_controllers))
        elif isinstance(controller_indices, int):
            indices = [controller_indices]
        else:
            indices = list(controller_indices)

        # Validate indices
        for idx in indices:
            if not (0 <= idx < self.num_controllers):
                raise IndexError(
                    f"Controller index {idx} out of range [0, {self.num_controllers})"
                )

        # Reset hidden states for these controllers
        self.reset_hidden_states(indices)

        # Reset weights for these controllers
        for idx in indices:
            self.controllers[idx].reset_weights()

    def forward_batch(
        self,
        obs_batch: Tensor,
        hidden_states_batch: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for all controllers in parallel.

        Since all controllers share the same frozen actor backbone, we run the
        LSTM + MLP once for all observations, then apply each controller's
        unique last-layer weights in batch.

        Parameters
        ----------
        obs_batch : Tensor
            Batched observations (num_controllers, obs_dim).
        hidden_states_batch : Tensor, optional
            Batched LSTM hidden states. If None, initialized internally.
            Should be tuple of (h, c) where each is (num_layers, num_controllers, rnn_hidden_size).

        Returns
        -------
        actions_batch : Tensor
            Batched actions (num_controllers, num_actions).
        """
        # Validate shape
        if obs_batch.shape[0] != self.num_controllers:
            raise ValueError(
                f"obs_batch has {obs_batch.shape[0]} observations, "
                f"expected {self.num_controllers}"
            )

        # Ensure obs on correct device
        obs_batch = obs_batch.to(self.device)

        actor = self.controllers[0].actor

        with torch.no_grad():
            # LSTM step: (num_controllers, obs_dim) → (num_controllers, lstm_hidden_dim)
            # Memory.forward internally does unsqueeze(0), so pass 2D input
            #
            # Maintain per-controller hidden states for independent reset capability
            # When first initialized, _h_batch and _c_batch store (num_controllers, num_layers, hidden_dim)

            # Initialize per-controller hidden states on first call
            if not self._hidden_states_initialized:
                # Let Memory initialize hidden states for the batch
                actor.memory_a.hidden_states = None
                hidden_in_init = actor.memory_a(obs_batch)
                if hidden_in_init.dim() == 3:
                    hidden_in_init = hidden_in_init.squeeze(0)

                # Extract initialized hidden states and store per-controller
                h, c = actor.memory_a.hidden_states
                # h, c shape: (num_layers, num_controllers, hidden_dim)
                # Transpose to (num_controllers, num_layers, hidden_dim) for per-controller access
                self._h_batch = h.transpose(0, 1).clone()
                self._c_batch = c.transpose(0, 1).clone()
                self._hidden_states_initialized = True
                hidden_in = hidden_in_init
            else:
                # Use stored per-controller hidden states
                # Transpose back to (num_layers, num_controllers, hidden_dim) for LSTM
                h_batch_fmt = self._h_batch.transpose(0, 1)
                c_batch_fmt = self._c_batch.transpose(0, 1)
                actor.memory_a.hidden_states = (h_batch_fmt, c_batch_fmt)

                hidden_in = actor.memory_a(obs_batch)
                if hidden_in.dim() == 3:
                    hidden_in = hidden_in.squeeze(0)

                # Update stored hidden states after forward
                h_new, c_new = actor.memory_a.hidden_states
                self._h_batch = h_new.transpose(0, 1).clone()
                self._c_batch = c_new.transpose(0, 1).clone()

            # hidden_in shape: (num_controllers, lstm_hidden_dim)

            # Squeeze sequence dimension: (1, num_controllers, lstm_hidden_dim) → (num_controllers, lstm_hidden_dim)
            if hidden_in.dim() == 3:
                hidden_in = hidden_in.squeeze(0)

            # MLP layers before last
            x = hidden_in
            for layer in actor.actor.children():
                # Stop before the last linear layer (which has output_features = num_actions)
                if isinstance(layer, nn.Linear) and layer.out_features == self.num_actions:
                    break
                x = layer(x)

            # x should now be (num_controllers, hidden_dim)

        # Apply last layer with each controller's weights in parallel
        # x shape: (num_controllers, hidden_dim)
        # W shape: (num_controllers, num_actions, hidden_dim)
        W_batch = torch.stack([ctrl.W for ctrl in self.controllers])
        b_batch = torch.stack([ctrl.last_layer.bias for ctrl in self.controllers])

        # Batched linear: z[i] = x[i] @ W[i]^T + b[i]
        # x: (num_controllers, hidden_dim)
        # W_batch: (num_controllers, num_actions, hidden_dim)
        # Using einsum: "eh,enh->en" sums over h (hidden_dim)
        # Result: (num_controllers, num_actions)
        z = torch.einsum("eh,enh->en", x, W_batch) + b_batch

        # Apply tanh and scaling
        a = torch.tanh(z)
        actions_batch = actor._scale(a)

        return actions_batch
