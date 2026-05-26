"""
WP2 CMA-ES run analysis plots.
================================

Four targeted plots for a CMA-ES Hebbian-rules run:

1. plot_rules_distribution      — A/B/C/D histograms (bins) for the best individual.
2. plot_fitness_progress_std    — std of fitness and progress over generations.
3. plot_rule_weight_correlation — |rule| vs |checkpoint weight| scatter per rule.
4. plot_cma_state               — CMA-ES state variables (sigma, axis ratio,
                                  condition number) across generations.

``analyze_run(run_dir)`` generates all four.

Usage::

    python -m WP2.plot_cma <run_folder>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  Helpers
# ============================================================================

def _save_fig(fig, plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cma] Saved {name}.png")


def _load_config(run_dir: Path):
    """Load HebbianEvolutionConfig from the run's reproducibility directory."""
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return None
    from WP2.config import HebbianEvolutionConfig
    return HebbianEvolutionConfig.from_yaml(cfg_path)


def _find_last_gen_dir(run_dir: Path) -> Optional[Path]:
    """Return the highest-numbered generation directory."""
    gen_root = run_dir / "generations"
    if not gen_root.is_dir():
        return None
    dirs = sorted(gen_root.glob("gen_*"))
    return dirs[-1] if dirs else None


def _load_best_genome(run_dir: Path) -> Optional[np.ndarray]:
    """Load the genome of the best individual from the last generation."""
    gen_dir = _find_last_gen_dir(run_dir)
    if gen_dir is None:
        return None
    sols_path = gen_dir / "solutions.npy"
    fits_path = gen_dir / "fitnesses.npy"
    if not sols_path.is_file() or not fits_path.is_file():
        return None
    solutions = np.load(sols_path)   # (P, n_genes)
    fitnesses = np.load(fits_path)   # (P,)
    return solutions[int(np.argmax(fitnesses))]


def _decode_block(genome: np.ndarray, block_idx: int, n_weights: int,
                  lo: float, hi: float) -> np.ndarray:
    """Decode one ABCD block from [0, 1] genome space to [lo, hi]."""
    start = block_idx * n_weights
    return genome[start:start + n_weights] * (hi - lo) + lo


def _load_csv_as_arrays(csv_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a CSV with a header row into a dict of float numpy arrays."""
    if not csv_path.is_file():
        return None
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    result: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        result[key] = np.array(vals)
    return result


def _load_population_csv(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load cma_population.csv into a dict of numpy arrays."""
    return _load_csv_as_arrays(run_dir / "results" / "cma_population.csv")


def _load_summary_csv(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load cma_summary.csv into a dict of numpy arrays."""
    return _load_csv_as_arrays(run_dir / "results" / "cma_summary.csv")


def _load_checkpoint_last_layer(run_dir: Path, expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Return the last Linear layer's weights as a flat numpy array, matching expected_shape."""
    ckpt_path = run_dir / "reproducibility" / "wp1_actor.pt"
    if not ckpt_path.is_file():
        return None
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        last_weight = None
        for v in state_dict.values():
            if hasattr(v, "shape") and tuple(v.shape) == expected_shape:
                last_weight = v
        if last_weight is None:
            return None
        return last_weight.cpu().numpy().flatten()
    except Exception as exc:
        print(f"[plot_cma] Could not load checkpoint weights: {exc}")
        return None


# ============================================================================
#  1. Rules distribution (histograms, best individual)
# ============================================================================

def plot_rules_distribution(run_dir: str | Path) -> None:
    """Histogram of A/B/C/D values for the best individual in the last generation.

    All four rules are shown side-by-side in a single PNG.
    """
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)

    cfg = _load_config(run_dir)
    if cfg is None or not cfg.hebbian.enabled:
        print("[plot_cma] No config or Hebbian disabled — skipping rules distribution.")
        return

    genome = _load_best_genome(run_dir)
    if genome is None:
        print("[plot_cma] No genome data found — skipping rules distribution.")
        return

    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim  # 7 × 64 = 448
    rules = {
        "A": _decode_block(genome, 0, n_weights, *cfg.hebbian.A_range),
        "B": _decode_block(genome, 1, n_weights, *cfg.hebbian.B_range),
        "C": _decode_block(genome, 2, n_weights, *cfg.hebbian.C_range),
        "D": _decode_block(genome, 3, n_weights, *cfg.hebbian.D_range),
    }

    colors = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (name, vals), color in zip(axes, rules.items(), colors):
        ax.hist(vals, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"{name} value")
        ax.set_ylabel("Count")
        ax.set_title(f"Rule {name}  (μ={vals.mean():.3f}, σ={vals.std():.3f})")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Hebbian Rule Distributions — Best Individual (Last Generation)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, run_dir / "plots", "rules_distribution")


# ============================================================================
#  2. Std of fitness and progress over generations
# ============================================================================

def plot_fitness_progress_std(run_dir: str | Path) -> None:
    """Plot per-generation std of fitness and std of progress on shared x-axis.

    Uses a twin y-axis so both quantities can be compared despite different scales.
    """
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)

    data = _load_population_csv(run_dir)
    if data is None:
        print("[plot_cma] No cma_population.csv found — skipping std plot.")
        return

    gens = data["generation"]
    unique_gens = np.unique(gens)
    std_fitness  = np.array([data["fitness"][gens == g].std()  for g in unique_gens])
    std_progress = np.array([data["progress"][gens == g].std() for g in unique_gens])

    color_fit  = "#1f77b4"
    color_prog = "#d62728"

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(unique_gens, std_fitness, color=color_fit, linewidth=2, label="Fitness std")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness std", color=color_fit)
    ax1.tick_params(axis="y", labelcolor=color_fit)

    ax2 = ax1.twinx()
    ax2.plot(unique_gens, std_progress, color=color_prog, linewidth=2,
             linestyle="--", label="Progress std")
    ax2.set_ylabel("Progress std [m]", color=color_prog)
    ax2.tick_params(axis="y", labelcolor=color_prog)

    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    ax1.set_title("Population Diversity — Std of Fitness and Progress per Generation")
    ax1.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, run_dir / "plots", "fitness_progress_std")


# ============================================================================
#  3. Rule magnitude vs checkpoint weight magnitude
# ============================================================================

def plot_rule_weight_correlation(run_dir: str | Path) -> None:
    """Scatter |rule_i| vs |w_checkpoint_i| / max|w| for each of A, B, C, D.

    Pearson r is shown in each subplot title.  All four rules are in one PNG.
    """
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)

    cfg = _load_config(run_dir)
    if cfg is None or not cfg.hebbian.enabled:
        print("[plot_cma] No config or Hebbian disabled — skipping correlation plot.")
        return

    genome = _load_best_genome(run_dir)
    if genome is None:
        print("[plot_cma] No genome data found — skipping correlation plot.")
        return

    expected_shape = (cfg.hebbian.num_actions, cfg.hebbian.hidden_dim)
    w0 = _load_checkpoint_last_layer(run_dir, expected_shape)
    if w0 is None:
        print("[plot_cma] Could not load checkpoint weights — skipping correlation plot.")
        return

    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    rules = {
        "A": _decode_block(genome, 0, n_weights, *cfg.hebbian.A_range),
        "B": _decode_block(genome, 1, n_weights, *cfg.hebbian.B_range),
        "C": _decode_block(genome, 2, n_weights, *cfg.hebbian.C_range),
        "D": _decode_block(genome, 3, n_weights, *cfg.hebbian.D_range),
    }

    w_abs = np.abs(w0)
    w_rel = w_abs / (w_abs.max() + 1e-8)   # relative weight magnitude in [0, 1]

    colors = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (name, vals), color in zip(axes, rules.items(), colors):
        rule_abs = np.abs(vals)
        r = float(np.corrcoef(rule_abs, w_rel)[0, 1])
        ax.scatter(w_rel, rule_abs, alpha=0.35, s=12, color=color, edgecolors="none")
        ax.set_xlabel("|w| / max|w|")
        ax.set_ylabel(f"|{name}|")
        ax.set_title(f"Rule {name}  (r = {r:+.3f})")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Rule Magnitude vs Checkpoint Weight Magnitude — Best Individual", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, run_dir / "plots", "rule_weight_correlation")


# ============================================================================
#  4. CMA-ES state variables (sigma, axis ratio, condition number) per gen
# ============================================================================

def _column_is_present(arr: Optional[np.ndarray]) -> bool:
    """True if `arr` exists and has at least one finite value."""
    return arr is not None and np.isfinite(arr).any()


def plot_cma_state(run_dir: str | Path) -> None:
    """Plot CMA-ES internal state variables over generations.

    Always plots sigma. If `axis_ratio` and `cond_number` columns are present
    in cma_summary.csv (newer runs), they are added as extra panels. For older
    runs without those columns, a single-panel sigma-only figure is produced.
    """
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)

    data = _load_summary_csv(run_dir)
    if data is None or "generation" not in data or "sigma" not in data:
        print("[plot_cma] No cma_summary.csv (or missing sigma) — skipping CMA state plot.")
        return

    gens = data["generation"]
    sigma = data["sigma"]
    axis_ratio = data.get("axis_ratio")
    cond_number = data.get("cond_number")

    has_ar = _column_is_present(axis_ratio)
    has_cn = _column_is_present(cond_number)
    n_panels = 1 + int(has_ar) + int(has_cn)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    axes = axes[0]

    axes[0].plot(gens, sigma, color="#1f77b4", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("σ (step size)")
    axes[0].set_title("CMA-ES sigma")
    axes[0].grid(True, which="both", alpha=0.3, linestyle="--")

    idx = 1
    if has_ar:
        axes[idx].plot(gens, axis_ratio, color="#2ca02c", linewidth=2)
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel("Generation")
        axes[idx].set_ylabel("axis ratio  max(D)/min(D)")
        axes[idx].set_title("Axis ratio of C")
        axes[idx].grid(True, which="both", alpha=0.3, linestyle="--")
        idx += 1
    if has_cn:
        axes[idx].plot(gens, cond_number, color="#d62728", linewidth=2)
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel("Generation")
        axes[idx].set_ylabel("cond(C)")
        axes[idx].set_title("Condition number of C")
        axes[idx].grid(True, which="both", alpha=0.3, linestyle="--")

    if not (has_ar or has_cn):
        fig.suptitle("CMA-ES State — sigma only (axis ratio / cond unavailable for this run)",
                     fontsize=12)
    else:
        fig.suptitle("CMA-ES State Variables across Generations", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, run_dir / "plots", "cma_state")


# ============================================================================
#  Master analysis function
# ============================================================================

def analyze_run(run_dir: str | Path) -> None:
    """Generate all CMA-ES plots for a WP2 run."""
    if not HAS_MPL:
        print("[analyze_run] matplotlib not available — skipping plots.")
        return
    run_dir = Path(run_dir)
    print(f"[analyze_run] Generating plots for {run_dir}")
    plot_rules_distribution(run_dir)
    plot_fitness_progress_std(run_dir)
    plot_rule_weight_correlation(run_dir)
    plot_cma_state(run_dir)
    print(f"[analyze_run] All plots saved to {run_dir / 'plots'}")


# ============================================================================
#  CLI entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m WP2.plot_cma <run_folder>")
        sys.exit(1)
    analyze_run(sys.argv[1])
