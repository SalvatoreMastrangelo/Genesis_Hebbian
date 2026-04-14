"""
Auto-generate training diagnostic plots from a run folder's CSV log.
=====================================================================

Reads the per-iteration CSV produced by :class:`WP1.csv_logger.CSVLogger`
and generates four publication-ready diagnostic figures:

1. **Reward curve** — raw per-iteration mean reward (transparent) overlaid
   with an exponential moving average (EMA, window = 25 iterations).

2. **v_mean & E_tot** — side-by-side plots of the progress-reward and
   energy-penalty components over training.  Useful for diagnosing
   whether the policy is learning to fly efficiently.

3. **Termination breakdown** — stacked-area chart showing the fraction
   of episode terminations due to wall crash, angle crash, obstacle
   collision, and successful corridor traversal.  A healthy training run
   should show crash fractions declining and success fraction rising.

4. **Forward progress** — EMA-smoothed mean final X position of
   completed episodes, indicating how far into the forest corridor the
   drone reaches on average.

All figures are saved as 150-dpi PNGs in the run folder's ``plots/``
sub-directory.

Usage
-----
**Standalone** (after a completed or interrupted run):

.. code-block:: bash

    python -m WP1.plotting logs/runs/2025-06-01_12-00-00_my-experiment

**Programmatic** (called automatically at the end of ``WP1.train``):

.. code-block:: python

    from WP1.plotting import plot_run
    plot_run("logs/runs/2025-06-01_12-00-00_my-experiment")

Dependencies
------------
Requires ``matplotlib`` and ``numpy``.  If ``matplotlib`` is not
installed the function prints a warning and returns without error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _smooth(y: np.ndarray, window: int = 25) -> np.ndarray:
    """Apply exponential moving average (EMA) smoothing.

    Parameters
    ----------
    y : np.ndarray
        1-D array of raw values.
    window : int
        EMA span.  The smoothing factor is ``alpha = 2 / (window + 1)``.

    Returns
    -------
    np.ndarray
        Smoothed array of the same length as *y*.
    """
    if len(y) < 2:
        return y
    alpha = 2.0 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def plot_run(run_dir: str | Path, csv_name: str = "training_log.csv") -> None:
    """Generate all standard diagnostic plots for a training run.

    Reads ``<run_dir>/eval/<csv_name>`` and saves figures to
    ``<run_dir>/plots/``.

    Parameters
    ----------
    run_dir : str or Path
        Path to the top-level run directory (the one containing
        ``config.yaml``, ``eval/``, ``plots/``, etc.).
    csv_name : str
        Name of the CSV file inside ``eval/``.  Defaults to
        ``"training_log.csv"``.

    Notes
    -----
    - NaN values in the CSV are handled gracefully: they are masked out
      of line plots and replaced with 0 in stacked-area charts.
    - If matplotlib is not installed, a warning is printed and the
      function returns without error.
    """
    if not HAS_MPL:
        print("[plot_run] matplotlib not available — skipping plots.")
        return

    run_dir = Path(run_dir)
    csv_path = run_dir / "eval" / csv_name
    if not csv_path.is_file():
        print(f"[plot_run] CSV not found: {csv_path}")
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    import csv
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[plot_run] CSV is empty.")
        return

    def col(name: str) -> np.ndarray:
        """Extract a named column as a float array (missing → NaN)."""
        vals = []
        for r in rows:
            v = r.get(name, "")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        return np.array(vals)

    iters = col("iter")

    # --- 1. Reward curve ---
    # Try various column name formats
    reward = col("Train_mean_reward")
    if np.all(np.isnan(reward)):
        reward = col("mean_reward")
    if np.all(np.isnan(reward)):
        reward = col("reward")

    # If no explicit reward column, try to derive from reward components
    if np.all(np.isnan(reward)):
        reward_components = [
            "rew_progress", "rew_energy", "rew_crash", "rew_height",
            "rew_angular", "rew_smooth", "rew_obstacle", "rew_success",
            "rew_cosmetic", "rew_stability"
        ]
        component_cols = [col(c) for c in reward_components]
        # Only use components that have data
        valid_cols = [c for c in component_cols if not np.all(np.isnan(c))]
        if valid_cols:
            reward = np.nanmean(np.column_stack(valid_cols), axis=1)

    if not np.all(np.isnan(reward)):
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = ~np.isnan(reward)
        ax.plot(iters[mask], reward[mask], alpha=0.3, color="steelblue", label="raw")
        ax.plot(iters[mask], _smooth(reward[mask]), color="steelblue", linewidth=2, label="smoothed")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Training Reward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "reward_curve.png", dpi=150)
        plt.close(fig)
        print(f"[plot_run] Saved reward_curve.png")

    # --- 2. Progress and Energy components ---
    v_mean = col("Episode_rew_progress")  # New: progress reward component
    e_tot = col("Episode_rew_energy")      # New: energy reward component
    if np.all(np.isnan(v_mean)):
        v_mean = col("v_mean")  # Fall back to old name
    if np.all(np.isnan(v_mean)):
        v_mean = col("rew_progress")  # Fall back to unprefixed name
    if np.all(np.isnan(e_tot)):
        e_tot = col("E_tot")    # Fall back to old name
    if np.all(np.isnan(e_tot)):
        e_tot = col("rew_energy")  # Fall back to unprefixed name

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not np.all(np.isnan(v_mean)):
        mask = ~np.isnan(v_mean)
        axes[0].plot(iters[mask], _smooth(v_mean[mask]), color="forestgreen", linewidth=2)
        axes[0].set_title("Progress Reward")
        axes[0].set_xlabel("Iteration")
        axes[0].grid(True, alpha=0.3)
    if not np.all(np.isnan(e_tot)):
        mask = ~np.isnan(e_tot)
        axes[1].plot(iters[mask], _smooth(e_tot[mask]), color="firebrick", linewidth=2)
        axes[1].set_title("Energy Penalty")
        axes[1].set_xlabel("Iteration")
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "v_mean_energy.png", dpi=150)
    plt.close(fig)
    print(f"[plot_run] Saved v_mean_energy.png")

    # --- 3. Termination breakdown (stacked area) ---
    # New CSV has Episode_num_* (counts), old had *_frac (fractions)
    wall_crashes = col("Episode_num_wall_crashed")
    angle_crashes = col("Episode_num_angle_crashed")
    collisions = col("Episode_num_collision")
    successes = col("Episode_num_success")

    # If counts are all NaN, try old fraction columns
    if np.all(np.isnan(wall_crashes)):
        data = np.column_stack([col(c) for c in ["wall_crash_frac", "angle_crash_frac", "collision_frac", "success_frac"]])
    else:
        # Convert counts to fractions (assume they sum to the episode count per iteration)
        counts = np.column_stack([wall_crashes, angle_crashes, collisions, successes])
        counts = np.nan_to_num(counts, nan=0.0)
        totals = np.sum(counts, axis=1, keepdims=True)
        totals[totals == 0] = 1  # Avoid division by zero
        data = counts / totals

    data = np.nan_to_num(data, nan=0.0)

    labels = ["Wall crash", "Angle crash", "Collision", "Success"]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(iters, data.T, labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.set_title("Episode Termination Breakdown")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "termination_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"[plot_run] Saved termination_breakdown.png")

    # --- 4. Progress (final_x) ---
    progress = col("Episode_final_x")
    if np.all(np.isnan(progress)):
        progress = col("progress")  # Fall back to old name
    if np.all(np.isnan(progress)):
        progress = col("final_x")  # Fall back to unprefixed name
    if not np.all(np.isnan(progress)):
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = ~np.isnan(progress)
        ax.plot(iters[mask], _smooth(progress[mask]), color="darkorchid", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean final X position")
        ax.set_title("Forward Progress Over Training")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "progress.png", dpi=150)
        plt.close(fig)
        print(f"[plot_run] Saved progress.png")


# Allow standalone execution: python -m WP1.plotting <run_folder>
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m WP1.plotting <run_folder>")
        sys.exit(1)
    plot_run(sys.argv[1])
