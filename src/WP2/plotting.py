"""
WP2 analysis and visualisation functions.
==========================================

Each plot is a standalone function accepting a run directory path.
``analyze_run(run_dir)`` generates all plots at once.

Plots are saved to ``{run_dir}/plots/`` as PNG + PDF.

Required plots from plan:
1. Pareto front (2D/3D scatter)
2. Pareto front evolution (generation as color)
3. Hypervolume convergence
4. Per-objective convergence
5. Hebbian parameter distributions (violin)
6. Hebbian parameter heatmap
7. Weight dynamics (timestep within episode)
8. Morphology diversity
9. Ablation comparison (bar chart)
10. Objective correlation (pairwise scatter)
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """EMA smoothing."""
    if len(y) < 2:
        return y
    alpha = 2.0 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def _load_gen_summary(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load generation_summary.csv into arrays."""
    csv_path = run_dir / "results" / "generation_summary.csv"
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        result[key] = np.array(vals)
    return result


def _load_pareto_history(run_dir: Path) -> List[Dict]:
    """Load pareto_history.csv."""
    csv_path = run_dir / "results" / "pareto_history.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_pop_history(run_dir: Path) -> List[Dict]:
    """Load population_history.csv."""
    csv_path = run_dir / "results" / "population_history.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_objective_names(run_dir: Path) -> List[str]:
    """Infer objective names from CSV headers."""
    csv_path = run_dir / "results" / "generation_summary.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    return [h.replace("best_", "") for h in headers if h.startswith("best_")]


def _save_fig(fig, plots_dir: Path, name: str) -> None:
    """Save a figure as PNG and PDF."""
    fig.savefig(plots_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(plots_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {name}.png + .pdf")


# ============================================================================
#  1. Pareto front (2D / 3D scatter)
# ============================================================================

def plot_pareto_front(run_dir: str | Path) -> None:
    """Plot the final Pareto front."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    obj_names = _get_objective_names(run_dir)
    rows = _load_pareto_history(run_dir)
    if not rows or len(obj_names) < 2:
        return

    # Get last generation
    gens = [int(float(r["generation"])) for r in rows]
    last_gen = max(gens)
    last_rows = [r for r in rows if int(float(r["generation"])) == last_gen]

    # Extract fitness values
    fitness_cols = [f"fitness_{name}" for name in obj_names]
    data = []
    for r in last_rows:
        vals = []
        for col in fitness_cols:
            try:
                vals.append(float(r.get(col, "nan")))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        data.append(vals)
    data = np.array(data)

    n_obj = len(obj_names)
    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data[:, 0], data[:, 1], c="steelblue", edgecolors="navy", s=60, alpha=0.8)
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_title(f"Pareto Front (Gen {last_gen})")
        ax.grid(True, alpha=0.3)
        _save_fig(fig, plots_dir, "pareto_front")

    elif n_obj >= 3:
        # 3D scatter with first 3 objectives
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c="steelblue", edgecolors="navy", s=60, alpha=0.8,
        )
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_zlabel(obj_names[2])
        ax.set_title(f"Pareto Front (Gen {last_gen})")
        _save_fig(fig, plots_dir, "pareto_front")

        # Also make pairwise 2D plots
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(data[:, i], data[:, j], c="steelblue", edgecolors="navy", s=50, alpha=0.7)
                ax.set_xlabel(obj_names[i])
                ax.set_ylabel(obj_names[j])
                ax.set_title(f"Pareto Front: {obj_names[i]} vs {obj_names[j]}")
                ax.grid(True, alpha=0.3)
                _save_fig(fig, plots_dir, f"pareto_{obj_names[i]}_vs_{obj_names[j]}")


# ============================================================================
#  2. Pareto front evolution
# ============================================================================

def plot_pareto_evolution(run_dir: str | Path) -> None:
    """Plot Pareto front evolution across generations (color = generation)."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    obj_names = _get_objective_names(run_dir)
    rows = _load_pareto_history(run_dir)
    if not rows or len(obj_names) < 2:
        return

    fitness_cols = [f"fitness_{name}" for name in obj_names]
    gens = np.array([int(float(r["generation"])) for r in rows])
    data = np.array([
        [float(r.get(col, "nan")) for col in fitness_cols]
        for r in rows
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        data[:, 0], data[:, 1], c=gens, cmap="viridis",
        edgecolors="grey", s=40, alpha=0.7, linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Generation")
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Pareto Front Evolution")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "pareto_evolution")


# ============================================================================
#  3. Hypervolume convergence
# ============================================================================

def plot_hypervolume(run_dir: str | Path) -> None:
    """Plot hypervolume indicator across generations."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    obj_names = _get_objective_names(run_dir)
    rows = _load_pareto_history(run_dir)
    if not rows or len(obj_names) < 2:
        return

    fitness_cols = [f"fitness_{name}" for name in obj_names]

    # Group by generation
    gen_data: Dict[int, List[List[float]]] = {}
    for r in rows:
        g = int(float(r["generation"]))
        vals = [float(r.get(col, "nan")) for col in fitness_cols]
        gen_data.setdefault(g, []).append(vals)

    # Compute hypervolume for each generation (2D approximation)
    # Reference point: slightly worse than worst observed values
    all_vals = np.array([v for vlist in gen_data.values() for v in vlist])
    if all_vals.size == 0:
        return
    ref_point = np.min(all_vals, axis=0) - 0.1 * np.abs(np.min(all_vals, axis=0))

    def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
        """Simple 2D hypervolume computation."""
        # Filter dominated points
        pts = points[np.all(points > ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        # Sort by first objective descending
        pts = pts[pts[:, 0].argsort()[::-1]]
        hv = 0.0
        prev_y = ref[1]
        for p in pts:
            if p[1] > prev_y:
                hv += (p[0] - ref[0]) * (p[1] - prev_y)
                prev_y = p[1]
        return hv

    gens = sorted(gen_data.keys())
    hvs = []
    for g in gens:
        pts = np.array(gen_data[g])
        if len(obj_names) == 2:
            hvs.append(_hv_2d(pts[:, :2], ref_point[:2]))
        else:
            # For >2 objectives, just track Pareto front size as proxy
            hvs.append(float(len(pts)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, hvs, "o-", color="darkorange", linewidth=2, markersize=4)
    ylabel = "Hypervolume" if len(obj_names) == 2 else "Pareto Front Size"
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title("Hypervolume Convergence")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "hypervolume_convergence")


# ============================================================================
#  4. Per-objective convergence
# ============================================================================

def plot_objective_convergence(run_dir: str | Path) -> None:
    """Plot mean / best / worst for each objective across generations."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _load_gen_summary(run_dir)
    obj_names = _get_objective_names(run_dir)
    if not data or not obj_names:
        return

    gens = data.get("generation", np.array([]))
    n_obj = len(obj_names)

    fig, axes = plt.subplots(1, n_obj, figsize=(6 * n_obj, 5), squeeze=False)
    colors = ["steelblue", "firebrick", "forestgreen", "darkorange", "purple"]

    for i, name in enumerate(obj_names):
        ax = axes[0, i]
        best_key = f"best_{name}"
        mean_key = f"mean_{name}"

        if best_key in data:
            ax.plot(gens, data[best_key], "-", color=colors[i % len(colors)],
                    linewidth=2, label="best")
        if mean_key in data:
            ax.plot(gens, data[mean_key], "--", color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.7, label="mean")

        ax.set_xlabel("Generation")
        ax.set_ylabel(name)
        ax.set_title(f"{name} Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "fitness_convergence")


# ============================================================================
#  5. Hebbian parameter distributions (violin)
# ============================================================================

def plot_hebbian_distributions(run_dir: str | Path) -> None:
    """Plot Hebbian parameter distributions across generations (violin)."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load config to get genome structure
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return

    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig.from_yaml(cfg_path)
    if not cfg.hebbian.enabled:
        return

    hebb_dim = cfg.hebbian_genome_dim()
    n_weights = 448  # 64 * 7

    rows = _load_pop_history(run_dir)
    if not rows:
        return

    # Group genomes by generation
    gen_genomes: Dict[int, List[List[float]]] = {}
    for r in rows:
        g = int(float(r["generation"]))
        try:
            genome = ast.literal_eval(r["genome"])
        except (ValueError, SyntaxError):
            continue
        gen_genomes.setdefault(g, []).append(genome[:hebb_dim])

    if not gen_genomes:
        return

    # Sample a few generations for the violin plot
    all_gens = sorted(gen_genomes.keys())
    if len(all_gens) <= 6:
        sample_gens = all_gens
    else:
        indices = np.linspace(0, len(all_gens) - 1, 6, dtype=int)
        sample_gens = [all_gens[i] for i in indices]

    param_names = ["A", "B", "C", "D", "lambda"]
    if cfg.hebbian.evolve_eta:
        param_names.append("eta")

    fig, axes = plt.subplots(
        len(param_names), 1,
        figsize=(max(8, 2 * len(sample_gens)), 3 * len(param_names)),
        squeeze=False,
    )

    for p_idx, param_name in enumerate(param_names):
        ax = axes[p_idx, 0]
        start = p_idx * n_weights
        end = start + n_weights

        gen_vals = []
        gen_labels = []
        for g in sample_gens:
            genomes = np.array(gen_genomes[g])
            if genomes.shape[1] < end:
                continue
            block = genomes[:, start:end].flatten()
            gen_vals.append(block)
            gen_labels.append(str(g))

        if gen_vals:
            parts = ax.violinplot(gen_vals, showmeans=True, showextrema=True)
            ax.set_xticks(range(1, len(gen_labels) + 1))
            ax.set_xticklabels(gen_labels)
            ax.set_xlabel("Generation")
        ax.set_ylabel(f"{param_name} (normalised)")
        ax.set_title(f"{param_name} Distribution Across Generations")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "hebbian_param_distributions")


# ============================================================================
#  6. Hebbian parameter heatmap
# ============================================================================

def plot_hebbian_heatmap(run_dir: str | Path) -> None:
    """Plot ABCD heatmap for the best Pareto solution."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load best Pareto individual's Hebbian rules
    rules_path = run_dir / "pareto_solutions" / "individual_000" / "hebbian_rules.yaml"
    if not rules_path.is_file():
        return

    import yaml
    with open(rules_path, "r") as f:
        rules = yaml.safe_load(f)

    param_names = ["A", "B", "C", "D", "lam"]
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))

    for i, name in enumerate(param_names):
        if name not in rules:
            continue
        matrix = np.array(rules[name])  # (5, 64)
        im = axes[i].imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        axes[i].set_title(name)
        axes[i].set_xlabel("Input neuron (64)")
        axes[i].set_ylabel("Output neuron (5)")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle("Hebbian ABCD Parameters — Best Pareto Solution", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "hebbian_param_heatmap")


# ============================================================================
#  7. Weight dynamics (placeholder — requires rollout trace data)
# ============================================================================

def plot_weight_dynamics(run_dir: str | Path) -> None:
    """Placeholder: plot weight evolution within an episode.

    This requires running a single-episode rollout with weight snapshots,
    which is not collected during standard evolution.
    Generates an info message instead.
    """
    print("[plot] Weight dynamics plot requires a dedicated trace rollout (not generated during evolution).")


# ============================================================================
#  8. Morphology diversity
# ============================================================================

def plot_morphology_diversity(run_dir: str | Path) -> None:
    """Plot morphology genome diversity across generations."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return

    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig.from_yaml(cfg_path)
    if not cfg.morphology.evolve:
        return

    hebb_dim = cfg.hebbian_genome_dim()
    morph_dim = cfg.morphology_genome_dim()

    rows = _load_pop_history(run_dir)
    if not rows:
        return

    gen_std: Dict[int, float] = {}
    for r in rows:
        g = int(float(r["generation"]))
        try:
            genome = ast.literal_eval(r["genome"])
        except (ValueError, SyntaxError):
            continue
        morph = genome[hebb_dim:hebb_dim + morph_dim]
        gen_std.setdefault(g, []).append(morph)

    gens = sorted(gen_std.keys())
    stds = []
    for g in gens:
        arr = np.array(gen_std[g])
        stds.append(np.mean(np.std(arr, axis=0)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, stds, "o-", color="teal", linewidth=2, markersize=4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Genome Std Dev")
    ax.set_title("Morphology Diversity")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "morphology_diversity")


# ============================================================================
#  10. Objective correlation (pairwise scatter)
# ============================================================================

def plot_objective_correlation(run_dir: str | Path) -> None:
    """Pairwise objective scatter from the final population."""
    if not HAS_MPL:
        return
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    obj_names = _get_objective_names(run_dir)
    rows = _load_pop_history(run_dir)
    if not rows or len(obj_names) < 2:
        return

    # Last generation
    gens = [int(float(r["generation"])) for r in rows]
    last_gen = max(gens)
    last_rows = [r for r in rows if int(float(r["generation"])) == last_gen]

    fitness_cols = [f"fitness_{name}" for name in obj_names]
    data = []
    for r in last_rows:
        vals = [float(r.get(col, "nan")) for col in fitness_cols]
        data.append(vals)
    data = np.array(data)

    n = len(obj_names)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            if i == j:
                ax.hist(data[:, i], bins=20, color="steelblue", alpha=0.7)
                ax.set_xlabel(obj_names[i])
            else:
                ax.scatter(data[:, j], data[:, i], s=15, alpha=0.5, c="steelblue")
                ax.set_xlabel(obj_names[j])
                ax.set_ylabel(obj_names[i])
            ax.grid(True, alpha=0.2)

    fig.suptitle("Objective Correlations (Final Generation)", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "objective_correlation")


# ============================================================================
#  Master analysis function
# ============================================================================

def analyze_run(run_dir: str | Path) -> None:
    """Generate all standard plots for a WP2 run."""
    if not HAS_MPL:
        print("[analyze_run] matplotlib not available — skipping plots.")
        return

    run_dir = Path(run_dir)
    print(f"[analyze_run] Generating plots for {run_dir}")

    plot_pareto_front(run_dir)
    plot_pareto_evolution(run_dir)
    plot_hypervolume(run_dir)
    plot_objective_convergence(run_dir)
    plot_hebbian_distributions(run_dir)
    plot_hebbian_heatmap(run_dir)
    plot_weight_dynamics(run_dir)
    plot_morphology_diversity(run_dir)
    plot_objective_correlation(run_dir)

    print(f"[analyze_run] All plots saved to {run_dir / 'plots'}")


# ============================================================================
#  CLI entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m WP2.plotting <run_folder>")
        sys.exit(1)
    analyze_run(sys.argv[1])
