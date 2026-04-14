"""
Per-iteration CSV logger with dynamic field capture.
====================================================

Captures ALL training metrics after every PPO update and appends them to
a CSV file inside the run's ``eval/`` directory.  The resulting file is
the primary data source for ``WP1.plotting.plot_run()``.

Unlike the legacy hardcoded logger, this version:
- Dynamically discovers and saves all fields from ``env.extras["episode"]``
- Captures all PPO metrics (mean_reward, mean_episode_length, etc.)
- Adds derived metrics (crash_rate, termination fractions)
- Never loses information — new metrics are added to the CSV on first occurrence
- Maintains backward compatibility with existing plotting code

Data sources
------------
1. **``env.extras["episode"]``** — environment-reported per-episode statistics
2. **PPO runner buffers** — mean_reward, mean_episode_length
3. **Derived metrics** — crash rates and termination breakdowns

Usage
-----
.. code-block:: python

    from WP1.csv_logger import CSVLogger

    csv_log = CSVLogger("logs/runs/.../eval/training_log.csv")

    for iteration in range(max_iters):
        # ... PPO update ...
        csv_log.log(
            iteration,
            env.extras,
            ppo_metrics={
                "mean_reward": ...,
                "mean_episode_length": ...,
            }
        )

    csv_log.close()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class CSVLogger:
    """Dynamically log all training metrics per PPO iteration to a CSV file.

    The file is opened in write mode on construction (overwriting any
    existing file at the same path), and a header row is written based on
    the first data row. Subsequent calls to :meth:`log` append one data row
    per iteration. The file is flushed after every write so that partial
    results are available even if training is interrupted.

    The logger dynamically discovers columns from the data itself:
    - Iteration number is always first
    - PPO metrics (mean_reward, mean_episode_length) are second
    - Environment episode statistics come next
    - Derived metrics (crash_rate, termination fractions) are appended

    Parameters
    ----------
    path : str or Path
        Output CSV file path.  Parent directories are created if needed.

    Attributes
    ----------
    path : Path
        Resolved output file path.
    _fieldnames : List[str]
        Dynamically determined column names (written to header on first call).
    _header_written : bool
        Whether the CSV header has been written yet.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = None
        self._fieldnames: List[str] = []
        self._header_written = False

    def log(
        self,
        iteration: int,
        extras: Dict[str, Any],
        ppo_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write one CSV row for the current PPO iteration, capturing all available data.

        Automatically discovers and saves all fields from extras["episode"] and
        ppo_metrics without losing information. Derived metrics (crash_rate,
        termination fractions) are computed and added.

        Parameters
        ----------
        iteration : int
            Zero-based PPO iteration index.
        extras : Dict[str, Any]
            The ``env.extras`` dictionary from ``WingedDroneEnv`` (or ``Gen_Env``).
            The method looks for the ``"episode"`` sub-dict with arbitrary keys
            (e.g., ``num_wall_crashed``, ``rew_progress``, ``final_x``, etc.).
        ppo_metrics : Dict[str, Any], optional
            PPO-related metrics like ``mean_reward``, ``mean_episode_length``.
            Keys are used as-is in the CSV.
        """
        if ppo_metrics is None:
            ppo_metrics = {}

        ep = extras.get("episode", {})

        # Build the complete row data
        row: Dict[str, str] = {"iter": str(iteration)}

        # Add PPO metrics
        for key, val in ppo_metrics.items():
            if val is not None:
                if isinstance(val, float):
                    row[key] = f"{val:.6g}"
                else:
                    row[key] = str(val)

        # Add all episode statistics from env.extras["episode"]
        for key, val in ep.items():
            if val is not None:
                if isinstance(val, float):
                    row[key] = f"{val:.6g}"
                elif isinstance(val, int):
                    row[key] = str(val)
                else:
                    row[key] = str(val)

        # Compute derived termination metrics
        wall = float(ep.get("num_wall_crashed", 0))
        angle = float(ep.get("num_angle_crashed", 0))
        collision = float(ep.get("num_collision", 0))
        success = float(ep.get("num_success", 0))
        total_terms = wall + angle + collision + success

        if total_terms > 0:
            row["crash_rate"] = f"{(wall + angle + collision) / total_terms:.4f}"
            row["wall_crash_frac"] = f"{wall / total_terms:.4f}"
            row["angle_crash_frac"] = f"{angle / total_terms:.4f}"
            row["collision_frac"] = f"{collision / total_terms:.4f}"
            row["success_frac"] = f"{success / total_terms:.4f}"
        else:
            row["crash_rate"] = "0.0000"
            row["wall_crash_frac"] = "0.0000"
            row["angle_crash_frac"] = "0.0000"
            row["collision_frac"] = "0.0000"
            row["success_frac"] = "0.0000"

        # On first call, discover all column names and write header
        if not self._header_written:
            # Build ordered field names: iter, ppo metrics, episode data, derived metrics
            self._fieldnames = ["iter"]
            self._fieldnames.extend(sorted(ppo_metrics.keys()))
            self._fieldnames.extend(sorted(ep.keys()))
            self._fieldnames.extend([
                "crash_rate", "wall_crash_frac", "angle_crash_frac",
                "collision_frac", "success_frac"
            ])
            # Remove duplicates while preserving order
            seen = set()
            self._fieldnames = [f for f in self._fieldnames if not (f in seen or seen.add(f))]

            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._header_written = True

        # Write row, using empty string for missing fields
        row_with_defaults = {f: row.get(f, "") for f in self._fieldnames}
        self._writer.writerow(row_with_defaults)
        self._file.flush()

    def close(self) -> None:
        """Flush pending writes and close the underlying file handle.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        try:
            self._file.close()
        except Exception:
            pass
