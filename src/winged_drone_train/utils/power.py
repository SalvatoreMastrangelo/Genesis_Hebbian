"""Actuation and power utilities for the drone environment.

This module is intentionally independent from Genesis internals.  It provides:

- scale_and_clamp_actions:
    Utility to map normalized actions to physical actuator ranges.

- ActuatorDynamics:
    Helper that scales normalized actions, optionally simulates latency, and
    adds Gaussian noise to the outputs.

- compute_power_consumption:
    Lightweight power consumption model for propellers and servos, based on
    thrust and servo torque/velocity.  Designed to be called directly from
    the environment using quantities already available there.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple
import csv

import torch


# -----------------------------------------------------------------------------
# Action scaling utilities
# -----------------------------------------------------------------------------


def scale_and_clamp_actions(
    raw_actions: torch.Tensor,
    throttle_limit: Tuple[float, float],
    joint_limits: Tuple[Sequence[float], Sequence[float]],
) -> torch.Tensor:
    """Scale and clamp normalized actions to physical actuator ranges.

    Convention:
        * raw_actions[:, 0]  → throttle command in approximately [0, 1].
        * raw_actions[:, 1:] → servo commands in approximately [-1, 1].

    Throttle is scaled to [min_thr, max_thr].
    Servo commands are scaled elementwise by their absolute joint limits and
    clamped to the provided per–joint ranges.

    Args:
        raw_actions: (B, A) tensor with normalized actions.
        throttle_limit: (min_thr, max_thr) tuple.
        joint_limits: (mins, maxs) where each is a sequence of length A-1
            describing the physical min/max angle for each servo.

    Returns:
        (B, A) tensor with scaled and clamped actions.
    """
    if raw_actions.ndim != 2:
        raise ValueError("raw_actions is expected to be 2D (batch, num_actions).")

    B, A = raw_actions.shape
    if A < 1:
        raise ValueError("raw_actions must contain at least one action (throttle).")

    min_thr, max_thr = float(throttle_limit[0]), float(throttle_limit[1])

    mins, maxs = joint_limits
    joint_min = torch.as_tensor(mins, dtype=torch.float32, device=raw_actions.device)
    joint_max = torch.as_tensor(maxs, dtype=torch.float32, device=raw_actions.device)
    if joint_min.numel() != A - 1 or joint_max.numel() != A - 1:
        raise ValueError(
            f"joint_limits must have length A-1={A-1}, "
            f"got {joint_min.numel()} and {joint_max.numel()}."
        )

    scaled = torch.empty_like(raw_actions, device=raw_actions.device)

    # Throttle: scale [0,1] → [min_thr, max_thr]
    scaled[:, 0] = torch.clamp(raw_actions[:, 0] * max_thr, min=min_thr, max=max_thr)

    # Servos: scale by max magnitude and clamp to [min, max] for each DOF
    if A > 1:
        servo_raw = raw_actions[:, 1:] * joint_max  # approximate symmetric range
        servo_clamped = torch.max(torch.min(servo_raw, joint_max), joint_min)
        scaled[:, 1:] = servo_clamped

    return scaled


# -----------------------------------------------------------------------------
# Latency configuration
# -----------------------------------------------------------------------------


@dataclass
class LatencyConfig:
    """Configuration for the action latency model.

    Attributes:
        simulate_latency: If True, enable simple FIFO latency.
        latency_min: Minimum latency (in control steps).
        latency_max: Maximum latency (in control steps).
        random_latency_per_step:
            - If False: sample one latency per env (per episode) and keep it.
            - If True: sample a new latency at each control step.
    """

    simulate_latency: bool = False
    latency_min: int = 0
    latency_max: int = 0
    random_latency_per_step: bool = False


# -----------------------------------------------------------------------------
# Actuator dynamics
# -----------------------------------------------------------------------------


class ActuatorDynamics:
    """Scale, delay and optionally noisify actuator commands.

    Typical usage::

        dyn = ActuatorDynamics(
            num_envs=64,
            num_actions=7,
            throttle_limit=(0.0, 1.0),
            joint_limits=(joint_mins, joint_maxs),
            latency_cfg=LatencyConfig(simulate_latency=True, latency_min=1, latency_max=3),
            throttle_noise_std=0.02,
            servo_noise_std=0.01,
            device=device,
        )

        servo_pos, throttle = dyn.process_actions(raw_actions)

    The class keeps its internal buffers on the chosen device and is therefore
    cheap to call at every control step.
    """

    def __init__(
        self,
        num_envs: int,
        num_actions: int,
        *,
        throttle_limit: Tuple[float, float] = (0.0, 1.0),
        joint_limits: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        latency_cfg: Optional[LatencyConfig] = None,
        throttle_noise_std: float = 0.0,
        servo_noise_std: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)

        # Store limits
        self.throttle_min, self.throttle_max = float(throttle_limit[0]), float(throttle_limit[1])

        if joint_limits is not None:
            mins, maxs = joint_limits
            self.joint_limits_min = torch.as_tensor(mins, dtype=torch.float32, device=self.device)
            self.joint_limits_max = torch.as_tensor(maxs, dtype=torch.float32, device=self.device)
            if self.joint_limits_min.numel() != self.num_actions - 1:
                raise ValueError(
                    "joint_limits must specify one min/max per servo action "
                    f"(expected {self.num_actions - 1}, got {self.joint_limits_min.numel()})."
                )
        else:
            # Default: symmetric [-1,1] range for all servos
            n_servos = self.num_actions - 1
            self.joint_limits_min = -torch.ones(n_servos, device=self.device)
            self.joint_limits_max = torch.ones(n_servos, device=self.device)

        # Latency configuration
        self.latency_cfg = latency_cfg or LatencyConfig()
        self.simulate_latency = bool(self.latency_cfg.simulate_latency)
        self.latency_min = int(self.latency_cfg.latency_min)
        self.latency_max = int(self.latency_cfg.latency_max)
        self.random_latency_per_step = bool(self.latency_cfg.random_latency_per_step)

        if self.latency_max < self.latency_min:
            self.latency_max = self.latency_min

        if self.simulate_latency and self.latency_max > 0:
            K = self.latency_max + 1  # length of FIFO history
            self._action_buffer = torch.zeros(
                (self.num_envs, K, self.num_actions), device=self.device, dtype=torch.float32
            )
            self._current_latency = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.long
            )
        else:
            self._action_buffer = None
            self._current_latency = None

        # Noise parameters
        self.throttle_noise_std = float(throttle_noise_std)
        self.servo_noise_std = float(servo_noise_std)

    # ------------------------------------------------------------------
    # Latency control
    # ------------------------------------------------------------------
    def reset_envs(self, env_indices: torch.Tensor) -> None:
        """Reset latency buffers for a subset of environments.

        This should typically be called during environment reset to avoid
        cross–episode leakage of past actions.
        """
        if not self.simulate_latency or self._action_buffer is None:
            return

        env_indices = env_indices.to(self.device, dtype=torch.long)
        self._action_buffer[env_indices] = 0.0

        if not self.random_latency_per_step and self.latency_max > 0:
            # Fixed latency sampled once per episode
            rand_delays = torch.randint(
                low=self.latency_min,
                high=self.latency_max + 1,
                size=(env_indices.numel(),),
                device=self.device,
            )
            self._current_latency[env_indices] = rand_delays
        else:
            # Will be overwritten at the next step if random per step.
            self._current_latency[env_indices] = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def process_actions(self, raw_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale, delay and perturb the input actions.

        Args:
            raw_actions: Normalized actions of shape (B, num_actions) where
                B should equal num_envs in typical usage.

        Returns:
            (servo_positions, throttle_output):
                - servo_positions has shape (B, num_actions-1);
                - throttle_output has shape (B,).
        """
        if raw_actions.ndim != 2:
            raise ValueError("raw_actions must be 2D (batch, num_actions).")

        B, A = raw_actions.shape
        if A != self.num_actions:
            raise ValueError(
                f"Second dimension of raw_actions must be num_actions={self.num_actions}, got {A}."
            )

        raw_actions = raw_actions.to(self.device)

        # 1) Scale and clamp to physical limits
        scaled = scale_and_clamp_actions(
            raw_actions,
            throttle_limit=(self.throttle_min, self.throttle_max),
            joint_limits=(self.joint_limits_min.tolist(), self.joint_limits_max.tolist()),
        )

        # 2) Apply latency model if enabled
        if self.simulate_latency and self._action_buffer is not None:
            # Shift FIFO buffer and insert current commands at the front
            self._action_buffer[:, 1:] = self._action_buffer[:, :-1].clone()
            self._action_buffer[:, 0, :] = scaled

            if self.random_latency_per_step and self.latency_max > 0:
                self._current_latency = torch.randint(
                    low=self.latency_min,
                    high=self.latency_max + 1,
                    size=(self.num_envs,),
                    device=self.device,
                    dtype=torch.long,
                )

            idx = self._current_latency.view(-1, 1, 1).expand(-1, 1, self.num_actions)
            applied_actions = torch.gather(self._action_buffer, dim=1, index=idx).squeeze(1)
        else:
            applied_actions = scaled

        # 3) Split into throttle and servos
        throttle_out = applied_actions[:, 0]  # (B,)
        if self.num_actions > 1:
            servo_out = applied_actions[:, 1:]
        else:
            servo_out = torch.empty((B, 0), device=self.device)

        # 4) Add optional actuator noise
        if self.servo_noise_std > 0.0 and servo_out.numel() > 0:
            servo_noise = torch.randn_like(servo_out) * self.servo_noise_std
            servo_out = servo_out + servo_noise
            servo_out = torch.max(
                torch.min(servo_out, self.joint_limits_max.unsqueeze(0)),
                self.joint_limits_min.unsqueeze(0),
            )

        if self.throttle_noise_std > 0.0:
            thr_noise = torch.randn_like(throttle_out) * self.throttle_noise_std
            throttle_out = (throttle_out + thr_noise).clamp(
                min=self.throttle_min,
                max=self.throttle_max,
            )

        return servo_out, throttle_out

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def get_latency_buffer_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the internal latency buffer and current delays.

        This is mainly useful for debugging or logging.
        """
        return self._action_buffer, self._current_latency


# -----------------------------------------------------------------------------
# Power consumption model
# -----------------------------------------------------------------------------
def _default_catalog_path() -> Path:
    base = Path(__file__).resolve().parents[3]
    urdf_dir = base / "genesis" / "assets" / "urdf" / "mydrone"
    return urdf_dir / "actuators.csv"


def _clean_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if not isinstance(name, str):
        return None
    value = name.strip()
    if not value or value.lower() == "none":
        return None
    return value


@lru_cache(maxsize=8)
def _read_actuator_catalog(path_str: str) -> dict[str, dict]:
    path = Path(path_str)
    if not path.exists():
        return {}
    catalog: dict[str, dict] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _clean_name(row.get("name"))
            if not name:
                continue
            kind = (row.get("type") or "").strip().lower()

            def fkey(k: str) -> Optional[float]:
                v = (row.get(k) or "").strip()
                if not v or v.lower() == "none":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            catalog[name] = {
                "type": kind,
                "c0": fkey("c0"),
                "c1": fkey("c1"),
                "c2": fkey("c2"),
                "max_thrust": fkey("max_thrust"),
                "radius": fkey("radius"),
            }
    return catalog


def _propeller_actuator_names_from_config(aero_config: Optional[dict]) -> tuple[str, ...]:
    if not aero_config:
        return ()
    links = aero_config.get("links", {}) or {}
    names = []
    for _, info in links.items():
        if not isinstance(info, dict):
            continue
        if (info.get("type") or "").strip().lower() != "propeller":
            continue
        act_name = _clean_name(info.get("actuator"))
        if act_name:
            names.append(act_name)
    return tuple(names)


def _default_prop_coeffs(
    drone_name: str,
    device: torch.device,
    n_propellers: int,
    *,
    actuator_catalog_path: Optional[Path] = None,
    aero_config: Optional[dict] = None,
    propeller_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Return polynomial coefficients for propeller power as (N, 3) tensor.

    The polynomial is: P_prop = c0 + c1 * T + c2 * T^2  (per propeller),
    where T is the propeller thrust.
    """

    # Generic morphing drone / default model
    base = torch.tensor([[0.0, 15.053, 2.431]], device=device, dtype=torch.float32)
    coeffs = base if n_propellers == 1 else base.expand(n_propellers, 3).clone()

    if propeller_names:
        names = [_clean_name(n) for n in propeller_names]
        names = [n for n in names if n]
    else:
        csv_path = _default_catalog_path()
        if actuator_catalog_path is not None:
            csv_path = Path(actuator_catalog_path)
        names = list(_propeller_actuator_names_from_config(aero_config))

    if not names:
        return coeffs

    if len(names) < n_propellers:
        names = names + [names[0]] * (n_propellers - len(names))
    if len(names) > n_propellers:
        names = names[:n_propellers]

    csv_path = actuator_catalog_path or _default_catalog_path()
    catalog = _read_actuator_catalog(str(csv_path))
    for i, name in enumerate(names):
        row = catalog.get(name)
        if not row or row.get("type") != "propeller":
            continue
        c0 = row.get("c0")
        c1 = row.get("c1")
        c2 = row.get("c2")
        if c0 is None or c1 is None or c2 is None:
            continue
        coeffs[i] = torch.tensor([c0, c1, c2], device=device, dtype=torch.float32)

    return coeffs


def _default_servo_power_constants(
    drone_name: str, device: torch.device, n_servos: int
) -> torch.Tensor:
    """Return servo power constants (R, kV, kI) as (n_servos, 3) tensor.

    For historical reasons this mimics the constants used in the original
    environment code, but in a more flexible way.
    """
    if n_servos == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)

    # Default (morphing drone) configuration:
    #   2 × KST X10 V8.0  +  4 × KST X08 Plus V2.0  (then repeat X08 if needed)
    constants_x10 = torch.tensor([2.80, 1.25, 0.35], device=device, dtype=torch.float32)
    constants_x08 = torch.tensor([8.84, 1.39, 0.55], device=device, dtype=torch.float32)

    if n_servos <= 2:
        return constants_x10.unsqueeze(0).expand(n_servos, 3)

    # First two servos = X10, rest = X08
    first_two = constants_x10.unsqueeze(0).expand(2, 3)
    remaining = constants_x08.unsqueeze(0).expand(n_servos - 2, 3)
    return torch.cat([first_two, remaining], dim=0)


def _default_torque_multipliers(
    drone_name: str,
    device: torch.device,
    n_servos: int,
    sweep_multiplier: float,
    twist_multiplier: float,
    tail_multiplier: float,
) -> torch.Tensor:
    """Return transmission ratios for the servo power model (output/actuator).
    """
    if n_servos == 0:
        return torch.empty((0,), device=device, dtype=torch.float32)

    base = [
        float(sweep_multiplier),
        float(sweep_multiplier),
        float(twist_multiplier),
        float(twist_multiplier),
        float(tail_multiplier),
        float(tail_multiplier),
    ]
    if n_servos <= len(base):
        vals = base[:n_servos]
    else:
        # Repeat "tail" multiplier or use 1.0 for extra servos
        extra = [1.0] * (n_servos - len(base))
        vals = base + extra

    return torch.tensor(vals, device=device, dtype=torch.float32)


@torch.no_grad()
def compute_power_consumption(
    thrust: torch.Tensor,
    servo_torque: torch.Tensor,
    servo_velocity: torch.Tensor,
    *,
    drone_name: str = "morphing_drone",
    sweep_multiplier: float = 2.0,
    twist_multiplier: float = 2.5,
    tail_multiplier: float = 2.0,
    prop_coefficients: Optional[torch.Tensor] = None,
    actuator_catalog_path: Optional[str | Path] = None,
    aero_config: Optional[dict] = None,
    propeller_names: Optional[Sequence[str]] = None,
    servo_power_constants: Optional[torch.Tensor] = None,
    torque_multipliers: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate power consumption of propellers and servos.

    This function is intentionally stateless and cheap enough to be called
    at every simulation step.  It reproduces the same structure as the original
    environment code, but with a cleaner and more configurable interface.

    Args:
        thrust:
            (B, n_prop) tensor with propeller thrust (e.g. Newtons).
        servo_torque:
            (B, n_servos) tensor with joint/output-side servo torques.
        servo_velocity:
            (B, n_servos) tensor with joint/output-side servo angular velocities.
        drone_name:
            Name of the drone model. Used to select default coefficients
            when prop_coefficients / servo_power_constants are not provided.
        sweep_multiplier, twist_multiplier, tail_multiplier:
            Multipliers used to scale servo torque/velocity for the morphing
            drone model..
        prop_coefficients:
            Optional (n_prop, 3) or (1, 3) tensor with [c0, c1, c2] for each
            propeller.  If None, a default set is chosen based on `drone_name`.
        actuator_catalog_path:
            Optional path to actuators.csv used to resolve propeller c0/c1/c2.
        aero_config:
            Optional aero configuration dict used to map propeller actuator names.
        propeller_names:
            Optional list of propeller actuator names to resolve c0/c1/c2.
        servo_power_constants:
            Optional (n_servos, 3) tensor with [R, kV, kI] per servo.  If None,
            default values are selected based on `drone_name`.
        torque_multipliers:
            Optional (n_servos,) tensor. If None, defaults are chosen based on
            `drone_name` and the sweep/twist/tail multipliers. Interpreted as
            transmission ratios M (output/actuator).
        device:
            Torch device. If None, inferred from `thrust`.
        return_components:
            If True, return a tuple (total_power, prop_power, servo_power).
            Otherwise, return only `total_power`.

    Returns:
        If return_components is False:
            total_power: (B,) tensor with total power consumption per env.
        If True:
            (total_power, prop_power, servo_power), each (B,).
    """
    if thrust.ndim != 2:
        raise ValueError("thrust must have shape (B, n_prop).")
    if servo_torque.ndim != 2 or servo_velocity.ndim != 2:
        raise ValueError("servo_torque and servo_velocity must have shape (B, n_servos).")
    if servo_torque.shape != servo_velocity.shape:
        raise ValueError("servo_torque and servo_velocity must have the same shape.")

    device = torch.device(device) if device is not None else thrust.device
    thrust = thrust.to(device)
    servo_torque = servo_torque.to(device)
    servo_velocity = servo_velocity.to(device)

    B, n_prop = thrust.shape
    _, n_servos = servo_torque.shape

    # ------------------------------------------------------------------
    # Propeller power: polynomial in thrust
    # ------------------------------------------------------------------
    if prop_coefficients is None:
        csv_path = Path(actuator_catalog_path) if actuator_catalog_path is not None else None
        if aero_config is None:
            from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroParameters

            aero_config = SimpleDroneAeroParameters.as_dict()
        prop_coefficients = _default_prop_coeffs(
            drone_name,
            device,
            n_prop,
            actuator_catalog_path=csv_path,
            aero_config=aero_config,
            propeller_names=propeller_names,
        )
    else:
        prop_coefficients = prop_coefficients.to(device)
        if prop_coefficients.ndim != 2 or prop_coefficients.shape[1] != 3:
            raise ValueError("prop_coefficients must have shape (N, 3).")

        if prop_coefficients.shape[0] == 1 and n_prop > 1:
            prop_coefficients = prop_coefficients.expand(n_prop, 3)
        elif prop_coefficients.shape[0] != n_prop:
            raise ValueError(
                f"prop_coefficients first dim must be 1 or n_prop={n_prop}, "
                f"got {prop_coefficients.shape[0]}."
            )

    c0 = prop_coefficients[:, 0].view(1, n_prop)
    c1 = prop_coefficients[:, 1].view(1, n_prop)
    c2 = prop_coefficients[:, 2].view(1, n_prop)

    cp_each = c0 + c1 * thrust + c2 * thrust ** 2  # (B, n_prop)
    cp_each = torch.clamp(cp_each, min=0.0)
    prop_power = cp_each.sum(dim=1)  # (B,)

    # ------------------------------------------------------------------
    # Servo power: simple electric motor model
    # ------------------------------------------------------------------
    if n_servos == 0:
        servo_power = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        if servo_power_constants is None:
            servo_power_constants = _default_servo_power_constants(
                drone_name, device, n_servos
            )
        else:
            servo_power_constants = servo_power_constants.to(device)
            if servo_power_constants.shape != (n_servos, 3):
                raise ValueError(
                    f"servo_power_constants must have shape (n_servos, 3), got {servo_power_constants.shape}."
                )

        if torque_multipliers is None:
            torque_multipliers = _default_torque_multipliers(
                drone_name,
                device,
                n_servos,
                sweep_multiplier=sweep_multiplier,
                twist_multiplier=twist_multiplier,
                tail_multiplier=tail_multiplier,
            )
        else:
            torque_multipliers = torque_multipliers.to(device)
            if torque_multipliers.shape != (n_servos,):
                raise ValueError(
                    f"torque_multipliers must have shape (n_servos,), got {torque_multipliers.shape}."
                )

        R_const = servo_power_constants[:, 0].view(1, n_servos)  # (1, n_servos)
        kV = servo_power_constants[:, 1].view(1, n_servos)
        kI = servo_power_constants[:, 2].view(1, n_servos)

        multipliers = torque_multipliers.view(1, n_servos)

        # Convert joint/output-side values to actuator-side using ratio M:
        # tau_act = tau_out / M, omega_act = omega_out * M.
        min_ratio = 1e-3
        sign = torch.where(
            multipliers >= 0.0,
            torch.ones_like(multipliers),
            -torch.ones_like(multipliers),
        )
        safe_multipliers = sign * multipliers.abs().clamp(min=min_ratio)

        T = servo_torque / safe_multipliers   # (B, n_servos)
        V = servo_velocity * safe_multipliers # (B, n_servos)

        # P = (V * T) / (kV * kI) + (R * kV / kI) * T^2
        denom = (kV * kI).clamp(min=1e-6)
        P = (V * T) / denom + (R_const * kV / kI.clamp(min=1e-6)) * T ** 2
        P = torch.clamp(P, min=0.0)  # no negative power

        servo_power = P.sum(dim=1)  # (B,)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_power = prop_power + servo_power

    if return_components:
        return total_power, prop_power, servo_power
    return total_power
