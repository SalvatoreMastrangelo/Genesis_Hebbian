"""Actuation and control-side helpers for the winged drone."""

from winged_drone_train.control.power import (
    ActuatorDynamics,
    LatencyConfig,
    compute_power_consumption,
    scale_and_clamp_actions,
)

__all__ = [
    "ActuatorDynamics",
    "LatencyConfig",
    "compute_power_consumption",
    "scale_and_clamp_actions",
]
