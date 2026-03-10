"""
WP1 — Infrastructure: Config System, Logging & Plotting
========================================================

This package implements the first work-package of the morphology-agnostic
flight-controller project.  Its purpose is to make every hyperparameter
configurable from a single file and every training run fully logged and
reproducible.

Modules
-------
config
    Unified ``RunConfig`` dataclass hierarchy.  Covers PPO hyperparameters,
    reward weights, observation flags, controller architecture, training
    settings, and URDF catalog options — all in one serialisable object with
    YAML round-trip and CLI override support.

run_manager
    Creates and enforces a timestamped run-directory layout
    (``config.yaml``, ``checkpoints/``, ``tb/``, ``eval/``, ``plots/``).

csv_logger
    Writes a per-iteration CSV with reward, forward-progress, energy, and
    an episode-termination breakdown (wall crash / angle crash / collision /
    success fractions).

plotting
    Reads the CSV log and auto-generates four diagnostic figures: reward
    curve, v_mean + E_tot, termination breakdown (stacked area), and
    forward progress.

train
    Drop-in training entry point that wires ``RunConfig`` → legacy config
    dicts → ``WingedDroneEnv`` / ``Gen_Env`` → RSL-RL ``OnPolicyRunner``,
    with integrated CSV logging and auto-plotting.

Quick start
-----------
.. code-block:: bash

    # Default single-morphology run
    python -m WP1.train

    # From a YAML config with CLI overrides
    python -m WP1.train --cfg configs/default.yaml --cfg.ppo.learning_rate 3e-4

    # Multi-morphology foundation training
    python -m WP1.train --cfg configs/foundation.yaml

    # Regenerate plots for an existing run
    python -m WP1.plotting logs/runs/2025-06-01_12-00-00_drone-forest
"""

from WP1.config import RunConfig
from WP1.run_manager import RunManager
from WP1.csv_logger import CSVLogger
from WP1.plotting import plot_run

__all__ = ["RunConfig", "RunManager", "CSVLogger", "plot_run"]
