"""
Run-folder management for WP1.

Every training run lands in a self-contained, timestamped folder under
``logs/runs/`` so the artefacts uniquely identify the run::

    logs/runs/<YYYY-MM-DD_HH-MM-SS>_<exp_name>/
    ├── config.yaml      # YAML payload (defaults + overrides), for reproducibility
    ├── cfgs.pkl         # legacy 5-tuple pickle (loaded by winged_drone_train.eval)
    ├── tb/              # TensorBoard event files + model_*.pt checkpoints
    ├── eval/            # post-training evaluation artefacts
    └── plots/           # optional diagnostic figures
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from WP1.config_loader import dump_yaml


class RunManager:
    """Create and own the directory layout for a WP1 training run.

    Parameters
    ----------
    exp_name : str
        Experiment tag used in the folder name.
    root : str | Path
        Parent directory of all runs. Defaults to ``logs/runs``.
    resume : bool
        If true, look for the latest matching folder and append ``_resumed``
        to the new run's name rather than starting from a fresh timestamp.
    """

    def __init__(
        self,
        exp_name: str,
        root: str | Path = "logs/runs",
        resume: bool = False,
    ) -> None:
        self.exp_name = exp_name
        self.root = Path(root)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{stamp}_{exp_name}"

        if resume:
            previous = self._find_latest(exp_name)
            if previous is not None:
                folder_name = previous.name + "_resumed"

        self.run_dir = self.root / folder_name
        self.tb_dir = self.run_dir / "tb"
        self.eval_dir = self.run_dir / "eval"
        self.plots_dir = self.run_dir / "plots"

        for d in (self.run_dir, self.tb_dir, self.eval_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"[RunManager] Run directory: {self.run_dir}")

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    @property
    def log_dir(self) -> Path:
        """Directory passed to ``OnPolicyRunner`` (TensorBoard + checkpoints)."""
        return self.tb_dir

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #

    def save_yaml_snapshot(self, payload: Dict[str, Any]) -> None:
        """Write the resolved YAML payload to ``config.yaml`` inside the run."""
        dump_yaml(payload, self.run_dir / "config.yaml")

    def save_legacy_cfg_pickle(
        self,
        env_cfg: Dict[str, Any],
        obs_cfg: Dict[str, Any],
        reward_cfg: Dict[str, Any],
        command_cfg: Dict[str, Any],
        train_cfg: Dict[str, Any],
    ) -> Path:
        """Pickle the 5-tuple of cfgs into ``cfgs.pkl`` so downstream tools
        (``winged_drone_train.eval.evaluation``, custom plotters, etc.) can
        load this run the same way they load ``logs/ea/<exp_name>`` runs.
        """
        path = self.run_dir / "cfgs.pkl"
        with path.open("wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _find_latest(self, exp_name: str) -> Optional[Path]:
        if not self.root.exists():
            return None
        candidates = sorted(
            [d for d in self.root.iterdir() if d.is_dir() and exp_name in d.name],
            key=lambda p: p.name,
        )
        return candidates[-1] if candidates else None
