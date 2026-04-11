"""
Fitness objective functions for WP2 multi-objective evolution.
==============================================================

Each function takes per-episode metrics and returns a scalar fitness value.
All objectives are oriented for **maximisation** (NSGA-II convention).
Costs are negated so that maximising them minimises the underlying cost.

The objective set is extensible: add a new function and register it in
``OBJECTIVE_REGISTRY``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from WP2_old.config import HebbianEvolutionConfig


# ============================================================================
#  Individual objective functions
# ============================================================================

def velocity_objective(metrics: Dict[str, np.ndarray]) -> float:
    """Mean forward velocity across evaluation episodes (maximise)."""
    v = metrics.get("velocities", np.array([]))
    if v.size == 0:
        return 0.0
    return float(np.mean(v))


def energy_objective(metrics: Dict[str, np.ndarray]) -> float:
    """Negative total energy consumption (maximise = lower energy).

    Stored as -E_tot so that NSGA-II maximisation minimises energy.
    """
    e = metrics.get("energies", np.array([]))
    if e.size == 0:
        return -10.0  # sentinel for invalid
    return float(-np.mean(e))


def progress_objective(metrics: Dict[str, np.ndarray]) -> float:
    """Mean forward progress / distance covered (maximise)."""
    p = metrics.get("progresses", np.array([]))
    if p.size == 0:
        return 0.0
    return float(np.mean(p))


def smoothness_objective(metrics: Dict[str, np.ndarray]) -> float:
    """Negative action jerk — penalises erratic control (maximise = smoother).

    Jerk = mean absolute second derivative of actions across timesteps.
    """
    jerks = metrics.get("action_jerks", np.array([]))
    if jerks.size == 0:
        return 0.0
    return float(-np.mean(jerks))


def crash_rate_objective(metrics: Dict[str, np.ndarray]) -> float:
    """Negative crash rate (maximise = fewer crashes).

    crash_rate in [0, 1]; we negate so maximisation minimises crashes.
    """
    crashes = metrics.get("crash_flags", np.array([]))
    if crashes.size == 0:
        return 0.0
    return float(-np.mean(crashes))


# ============================================================================
#  Registry
# ============================================================================

OBJECTIVE_REGISTRY: Dict[str, Callable[[Dict[str, np.ndarray]], float]] = {
    "velocity": velocity_objective,
    "energy": energy_objective,
    "progress": progress_objective,
    "smoothness": smoothness_objective,
    "crash_rate": crash_rate_objective,
}


def compute_fitness(
    metrics: Dict[str, np.ndarray],
    cfg: HebbianEvolutionConfig,
) -> List[float]:
    """Compute the fitness tuple for an individual given episode metrics.

    Returns a list of floats matching the active objectives in config order.
    """
    obj_names = cfg.active_objective_names()
    fitness = []
    for name in obj_names:
        fn = OBJECTIVE_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown objective: {name}")
        fitness.append(fn(metrics))
    return fitness


def default_fitness(cfg: HebbianEvolutionConfig) -> List[float]:
    """Return sentinel fitness values for invalid individuals."""
    n = cfg.num_active_objectives()
    # Use zeros/negatives as sentinels
    sentinels = {
        "velocity": 0.0,
        "energy": -10.0,
        "progress": 0.0,
        "smoothness": 0.0,
        "crash_rate": -1.0,
    }
    return [sentinels.get(name, 0.0) for name in cfg.active_objective_names()]
