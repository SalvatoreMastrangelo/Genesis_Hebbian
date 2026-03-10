"""
Per-iteration CSV logger with episode-termination breakdown.
=============================================================

Captures key training metrics after every PPO update and appends them to
a CSV file inside the run's ``eval/`` directory.  The resulting file is
the primary data source for ``WP1.plotting.plot_run()``.

Data sources
------------
The logger reads from two sources each iteration:

1. **``env.extras["episode"]``** — populated by ``WingedDroneEnv.reset()``
   whenever environments are reset.  Contains per-reset-batch averages of
   reward components (``rew_progress``, ``rew_energy``, ...) and raw
   termination counts (``num_wall_crashed``, ``num_angle_crashed``,
   ``num_collision``, ``num_success``).

2. **RSL-RL runner buffers** — ``runner.rewbuffer`` and
   ``runner.lenbuffer`` provide running-mean episode reward and length.

Columns
-------
===================  ===========================================================
Column               Description
===================  ===========================================================
``iter``             PPO iteration index (0-based)
``mean_reward``      Mean total episodic reward (from runner buffer)
``v_mean``           Mean progress-reward component (speed tracking)
``E_tot``            Mean energy-penalty component
``progress``         Mean final X position of completed episodes (metres)
``crash_rate``       Fraction of terminations due to any crash type
``wall_crash_frac``  Fraction due to lateral-wall or ground crashes
``angle_crash_frac`` Fraction due to exceeding safe roll/pitch angles
``collision_frac``   Fraction due to obstacle collisions
``success_frac``     Fraction of episodes that reached the corridor end
``timeout_frac``     Fraction of episodes that timed out (placeholder)
``mean_episode_length``  Mean episode length in env steps
===================  ===========================================================

Usage
-----
.. code-block:: python

    from WP1.csv_logger import CSVLogger

    csv_log = CSVLogger("logs/runs/.../eval/training_log.csv")

    for iteration in range(max_iters):
        # ... PPO update ...
        csv_log.log(iteration, env.extras, mean_reward=..., mean_episode_length=...)

    csv_log.close()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# Column names for the CSV output
CSV_COLUMNS: List[str] = [
    "iter",
    "mean_reward",
    "v_mean",
    "E_tot",
    "progress",
    "crash_rate",
    "wall_crash_frac",
    "angle_crash_frac",
    "collision_frac",
    "success_frac",
    "timeout_frac",
    "mean_episode_length",
]


class CSVLogger:
    """Append one row of training metrics per PPO iteration to a CSV file.

    The file is opened in write mode on construction (overwriting any
    existing file at the same path), and a header row is written
    immediately.  Subsequent calls to :meth:`log` append one data row
    per iteration.  The file is flushed after every write so that
    partial results are available even if training is interrupted.

    Parameters
    ----------
    path : str or Path
        Output CSV file path.  Parent directories are created if needed.

    Attributes
    ----------
    path : Path
        Resolved output file path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def log(
        self,
        iteration: int,
        extras: Dict[str, Any],
        mean_reward: Optional[float] = None,
        mean_episode_length: Optional[float] = None,
    ) -> None:
        """Write one CSV row for the current PPO iteration.

        Termination counts from ``extras["episode"]`` are normalised into
        fractions (each in [0, 1]) so that they sum to 1.  If no
        terminations occurred in this iteration the fractions are all 0.

        Parameters
        ----------
        iteration : int
            Zero-based PPO iteration index.
        extras : Dict[str, Any]
            The ``env.extras`` dictionary from ``WingedDroneEnv`` (or
            ``Gen_Env``).  The method looks for the ``"episode"`` sub-dict
            with keys:

            - ``num_wall_crashed`` — count of wall / ground crashes
            - ``num_angle_crashed`` — count of angle-limit crashes
            - ``num_collision`` — count of obstacle collisions
            - ``num_success`` — count of successful corridor traversals
            - ``rew_progress`` — mean progress-reward component
            - ``rew_energy`` — mean energy-penalty component
            - ``final_x`` — mean final X position (metres)

        mean_reward : float, optional
            Mean total episodic reward for this iteration.
        mean_episode_length : float, optional
            Mean episode length in environment steps.
        """
        ep = extras.get("episode", {})

        # Termination counts (these are sums over reset envs, not fractions)
        # We normalise by the sum of all termination types to get fractions.
        wall = float(ep.get("num_wall_crashed", 0))
        angle = float(ep.get("num_angle_crashed", 0))
        collision = float(ep.get("num_collision", 0))
        success = float(ep.get("num_success", 0))

        total_terms = wall + angle + collision + success
        # timeout is implicit: envs that reset but aren't in any crash/success category
        # We approximate timeout_frac from time_outs if available
        timeout_frac = 0.0

        if total_terms > 0:
            wall_frac = wall / total_terms
            angle_frac = angle / total_terms
            collision_frac = collision / total_terms
            success_frac = success / total_terms
            crash_rate = (wall + angle + collision) / total_terms
        else:
            wall_frac = angle_frac = collision_frac = success_frac = crash_rate = 0.0

        # Reward components
        v_mean = float(ep.get("rew_progress", 0.0))
        e_tot = float(ep.get("rew_energy", 0.0))
        progress = float(ep.get("final_x", 0.0))

        row = {
            "iter": iteration,
            "mean_reward": f"{mean_reward:.6g}" if mean_reward is not None else "",
            "v_mean": f"{v_mean:.6g}",
            "E_tot": f"{e_tot:.6g}",
            "progress": f"{progress:.4f}",
            "crash_rate": f"{crash_rate:.4f}",
            "wall_crash_frac": f"{wall_frac:.4f}",
            "angle_crash_frac": f"{angle_frac:.4f}",
            "collision_frac": f"{collision_frac:.4f}",
            "success_frac": f"{success_frac:.4f}",
            "timeout_frac": f"{timeout_frac:.4f}",
            "mean_episode_length": f"{mean_episode_length:.2f}" if mean_episode_length is not None else "",
        }

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush pending writes and close the underlying file handle.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        try:
            self._file.close()
        except Exception:
            pass
