"""
HebbianCMAES — CMA-ES loop for Hebbian rules optimisation.
==========================================================

Uses the pycma ``CMAEvolutionStrategy`` to search for Hebbian rules that
maximise the WP1 scalar reward (sum of per-step ``env.last_reward_total``
over the episode, averaged across evaluation environments).

- **Single objective**: WP1 reward sum.
- **Rules only**: morphology is always fixed.
- **Catalog support**: optionally evaluates each candidate against a set of
  pre-built URDFs for robustness across morphologies.
- **CMA-ES mechanics**: pycma handles step-size adaptation and covariance
  matrix update; no crossover/mutation operators needed.

Usage
-----
>>> cfg = HebbianEvolutionConfig.from_yaml("configs/cma_es_rules_only.yaml")
>>> runner = HebbianCMAES(cfg)
>>> runner.run()
"""

from __future__ import annotations

import csv
import datetime
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cma
import numpy as np
import torch
from tabulate import tabulate

from WP2.config import HebbianEvolutionConfig
from WP2.evaluate import (
    evaluate_population_cma_batched,
    evaluate_population_multi_urdf,
    _load_catalog,
)
from WP2.utils import (
    save_git_info,
    save_environment_info,
    save_rng_state,
    load_rng_state,
    decode_hebbian_genes,
)


# ============================================================================
#  Helpers
# ============================================================================

def minmax_normalize_fitness(fitnesses: np.ndarray) -> np.ndarray:
    """Min-max normalize a generation's fitness vector to [0, 1].

    The best individual maps to 1.0 and the worst to 0.0. A constant vector
    (max == min, e.g. the all-zero fallback after an eval failure) is returned
    unchanged — there is no rank information to rescale.
    """
    fitnesses = np.asarray(fitnesses, dtype=float)
    span = fitnesses.max() - fitnesses.min()
    if span == 0.0:
        return fitnesses
    return (fitnesses - fitnesses.min()) / span


def _format_time(seconds: float) -> Tuple[int, int, int]:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return h, m, s


def _load_catalog_paths(catalog_path: str) -> List[str]:
    """Load a ``catalog.txt`` into absolute URDF paths (one per line)."""
    p = Path(catalog_path)
    base = p.parent
    entries: List[str] = []
    with open(p, "r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            entries.append(str((base / name).resolve()))
    if not entries:
        raise ValueError(f"Catalog {catalog_path} is empty")
    return entries


def _generate_random_urdfs(
    out_dir: Path,
    n: int,
    seed: int,
    include_standard_mydrone: bool = True,
) -> List[str]:
    """Sample ``n`` random URDFs using the same sampler as WP1 training.

    When ``include_standard_mydrone`` is True (default), the first URDF is the
    standard-mydrone baseline and the remaining ``n - 1`` are drawn uniformly
    in the normalized drone genome space via ``Chromosome_Drone``. When False,
    all ``n`` URDFs are randomly sampled. URDFs and a ``catalog.txt`` are
    written to ``out_dir`` for reproducibility.
    """
    from general_policy.catalog import build_catalog

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = build_catalog(
        catalog_dir=out_dir,
        n=n,
        seed=seed,
        include_standard_mydrone=include_standard_mydrone,
    )
    return [str(p) for p in paths]


def _read_norm_genomes_from_catalog(catalog_dir: Path) -> Optional[List[List[float]]]:
    """Read ``genomes.txt`` from a catalog dir and return normalized genomes.

    ``genomes.txt`` stores the *physical* genome of each URDF, one per row as
    ``<urdf_filename>: g0, g1, ..., g14`` (row order matches ``catalog.txt``).
    Each is mapped back into normalized ``[0, 1]^D`` space via
    ``Chromosome_Drone.from_physical`` so it can be perturbed and re-emitted.
    Returns ``None`` if the file is missing or any row is unparsable / has the
    wrong arity (e.g. an ``<unknown>`` placeholder).
    """
    from morph_evolution.chromosome_drone import Chromosome_Drone

    genomes_file = Path(catalog_dir) / "genomes.txt"
    if not genomes_file.exists():
        return None

    norm: List[List[float]] = []
    for line in genomes_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, _, rhs = line.partition(":")
        parts = [p.strip() for p in rhs.split(",") if p.strip()]
        try:
            phys = [float(p) for p in parts]
        except ValueError:
            return None
        if len(phys) != Chromosome_Drone.num_genes():
            return None
        norm.append([float(x) for x in Chromosome_Drone.from_physical(phys)])
    return norm or None


# ============================================================================
#  CSV initialisation / append
# ============================================================================

def _init_csvs(pop_path: Path, summary_path: Path) -> None:
    for p in (pop_path, summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with open(pop_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "individual_idx",
            "fitness", "velocity", "progress", "crash_rate", "cot", "v_deviation",
        ])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "pop_size",
            "best_fitness", "mean_fitness", "worst_fitness", "std_fitness",
            "sigma", "axis_ratio", "cond_number",
            "mean_velocity", "mean_progress", "mean_crash_rate", "mean_cot", "mean_v_deviation",
            "uh_noiseS", "uh_sigma_factor",
        ])


def _init_baseline_csv(path: Path) -> None:
    """Initialise a single-controller (baseline or specialist) summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "fitness", "velocity", "progress", "crash_rate", "cot", "v_deviation",
        ])


def _append_baseline_csv(
    path: Path,
    gen: int,
    baseline: Dict[str, float],
) -> None:
    """Append a baseline/specialist per-generation row."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            gen,
            f"{baseline['fitness']:.6g}",
            f"{baseline['velocity']:.6g}",
            f"{baseline['progress']:.6g}",
            f"{baseline['crash_rate']:.6g}",
            f"{baseline.get('cot', float('nan')):.6g}",
            f"{baseline.get('v_deviation', float('nan')):.6g}",
        ])


# Metrics written for both the best individual and the baseline in the
# held-out validation summary (in column order).
_VAL_METRIC_KEYS = ("fitness", "velocity", "progress", "crash_rate", "cot", "v_deviation")


def _init_validation_csv(path: Path, include_specialist: bool = False) -> None:
    """Initialise the held-out validation summary CSV.

    One row per validation pass with the best-fitness individual's metrics and
    the zero-rules baseline's metrics on the SAME ``n_val_envs`` held-out
    forests, so a generation-by-generation best-vs-baseline curve can be drawn.
    When ``include_specialist`` is set, a third ``specialist_*`` block is added
    (same forests under CRN, so all three controllers are directly comparable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["generation", "best_idx"]
    header += [f"best_{k}" for k in _VAL_METRIC_KEYS]
    header += [f"baseline_{k}" for k in _VAL_METRIC_KEYS]
    if include_specialist:
        header += [f"specialist_{k}" for k in _VAL_METRIC_KEYS]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def _append_validation_csv(
    path: Path,
    gen: int,
    best_idx: int,
    best: Dict[str, float],
    baseline: Dict[str, float],
    specialist: Optional[Dict[str, float]] = None,
) -> None:
    """Append one held-out validation row (best individual + baseline [+ specialist]).

    Pass ``specialist`` (possibly an empty dict on eval failure) only when the
    CSV was initialised with ``include_specialist=True``, so the column count
    stays consistent with the header.
    """
    row: List = [gen, best_idx]
    row += [f"{best.get(k, float('nan')):.6g}" for k in _VAL_METRIC_KEYS]
    row += [f"{baseline.get(k, float('nan')):.6g}" for k in _VAL_METRIC_KEYS]
    if specialist is not None:
        row += [f"{specialist.get(k, float('nan')):.6g}" for k in _VAL_METRIC_KEYS]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _append_population_csv(
    path: Path,
    gen: int,
    fitnesses: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        P = len(fitnesses)
        for i in range(P):
            writer.writerow([
                gen, i,
                f"{fitnesses[i]:.6g}",
                f"{metrics['velocities'][i]:.6g}",
                f"{metrics['progresses'][i]:.6g}",
                f"{metrics['crash_flags'][i]:.6g}",
                f"{metrics['cots'][i]:.6g}",
                f"{metrics['v_deviations'][i]:.6g}",
            ])


def _append_summary_csv(
    path: Path,
    gen: int,
    fitnesses: np.ndarray,
    metrics: Dict[str, np.ndarray],
    sigma: float,
    axis_ratio: float,
    cond_number: float,
    uh: Optional[Dict[str, float]] = None,
) -> None:
    P = len(fitnesses)
    nan = float("nan")
    uh_noise = uh["noiseS"] if uh is not None else nan
    uh_factor = uh["factor"] if uh is not None else nan
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            gen, P,
            f"{fitnesses.max():.6g}",
            f"{fitnesses.mean():.6g}",
            f"{fitnesses.min():.6g}",
            f"{fitnesses.std():.6g}",
            f"{sigma:.6g}",
            f"{axis_ratio:.6g}",
            f"{cond_number:.6g}",
            f"{metrics['velocities'].mean():.6g}",
            f"{metrics['progresses'].mean():.6g}",
            f"{metrics['crash_flags'].mean():.6g}",
            f"{metrics['cots'].mean():.6g}",
            f"{metrics['v_deviations'].mean():.6g}",
            f"{uh_noise:.6g}",
            f"{uh_factor:.6g}",
        ])


def _extract_cma_state(es) -> tuple[float, float]:
    """Return (axis_ratio, cond_number) from a pycma strategy, NaN on failure.

    axis_ratio = max(D)/min(D) where D are singular values of B*D (sqrt eigenvalues of C).
    cond_number = axis_ratio**2 = cond(C).
    """
    try:
        D = getattr(es, "D", None)
        if D is None:
            sm = getattr(es, "sm", None)
            D = getattr(sm, "D", None) if sm is not None else None
        if D is None:
            C = np.asarray(es.C)
            eigs = np.linalg.eigvalsh(C)
            eigs = eigs[eigs > 0]
            if eigs.size == 0:
                return float("nan"), float("nan")
            D = np.sqrt(eigs)
        D = np.asarray(D, dtype=float)
        D = D[D > 0]
        if D.size == 0:
            return float("nan"), float("nan")
        ar = float(D.max() / D.min())
        return ar, ar * ar
    except Exception:
        return float("nan"), float("nan")


def _print_generation_table(
    gen: int,
    fitnesses: np.ndarray,
    metrics: Dict[str, np.ndarray],
    sigma: float,
    total_elapsed: Optional[float] = None,
    iter_elapsed: Optional[float] = None,
    eta_seconds: Optional[float] = None,
    baseline: Optional[Dict[str, float]] = None,
    specialist: Optional[Dict[str, float]] = None,
    total_generations: Optional[int] = None,
    uh: Optional[Dict[str, float]] = None,
) -> None:
    gen_str = f"{gen}/{total_generations}" if total_generations is not None else str(gen)
    timing_parts = [f"CMA-ES Generation {gen_str}", f"Population={len(fitnesses)}"]

    if total_elapsed is not None:
        h, m, s = _format_time(total_elapsed)
        timing_parts.append(f"Elapsed: {h}:{m:02d}:{s:02d}")

    if iter_elapsed is not None:
        im, is_ = divmod(int(iter_elapsed), 60)
        timing_parts.append(f"Iter: {im}:{is_:02d}")

    if eta_seconds is not None:
        h, m, s = _format_time(eta_seconds)
        timing_parts.append(f"ETA: {h}:{m:02d}:{s:02d}")

    timing_parts.append(f"σ={sigma:.4g}")

    if uh is not None:
        timing_parts.append(
            f"UH: noiseS={uh['noiseS']:+.3f} ×σ={uh['factor']:.4f}"
        )

    header_str = " | ".join(timing_parts)
    print(f"\n{'═' * 90}")
    print(f"  {header_str}")
    print(f"{'═' * 90}")

    # Metric rows: (display name, array, baseline_key)
    rows = [
        ("Fitness (reward)", fitnesses,                "fitness"),
        ("Velocity [m/s]",   metrics["velocities"],    "velocity"),
        ("Vel. Dev. [m/s]",  metrics["v_deviations"],  "v_deviation"),
        ("Progress [m]",     metrics["progresses"],    "progress"),
        ("COT",              metrics["cots"],          "cot"),
        ("Crash Rate",       metrics["crash_flags"],   "crash_rate"),
    ]

    # "Best/Worst" always refer to the fitness-best / fitness-worst individual,
    # so every row in the summary (and the breakdown below) describes the
    # *same* drone in the Best column and the *same* drone in the Worst column.
    best_idx = int(np.argmax(fitnesses))
    worst_idx = int(np.argmin(fitnesses))

    extra_cols: List[Tuple[str, Dict[str, float]]] = []
    if baseline is not None:
        extra_cols.append(("Baseline", baseline))
    if specialist is not None:
        extra_cols.append(("Specialist", specialist))

    base_headers = ["Metric", "Best (Hebb)", "Median (Hebb)", "Worst (Hebb)", "Std"]
    if extra_cols:
        headers = base_headers + [name for name, _ in extra_cols]
    else:
        headers = ["Metric", "Best", "Median", "Worst", "Std"]

    table_rows = []
    for name, arr, key in rows:
        row = [
            name,
            f"{arr[best_idx]:.4g}",
            f"{np.median(arr):.4g}",
            f"{arr[worst_idx]:.4g}",
            f"{arr.std():.4g}",
        ]
        for _, ref in extra_cols:
            row.append(f"{ref.get(key, float('nan')):.4g}")
        table_rows.append(row)

    if extra_cols:
        tag = " | ".join(["Hebb"] + [name for name, _ in extra_cols])
        print(f"\n  Metrics Summary ({tag}):")
    else:
        print("\n  Metrics Summary:")
    print(tabulate(
        table_rows,
        headers=headers,
        tablefmt="grid",
        numalign="center",
        stralign="left",
    ))

    # Reward breakdown (per-component episode-sum values)
    comp_arr = metrics.get("reward_components")
    comp_names = metrics.get("reward_names") or []
    if comp_arr is not None and len(comp_names) and comp_arr.size:
        extra_comps = [(label, ref.get("reward_components", {}) or {})
                       for label, ref in extra_cols]

        breakdown_rows = []
        for i, name in enumerate(comp_names):
            col = comp_arr[:, i]
            row = [
                name,
                f"{col[best_idx]:.4g}",
                f"{np.median(col):.4g}",
                f"{col[worst_idx]:.4g}",
                f"{col.std():.4g}",
            ]
            for _, comps in extra_comps:
                row.append(f"{comps.get(name, float('nan')):.4g}")
            breakdown_rows.append(row)

        if extra_cols:
            breakdown_headers = [
                "Reward Component",
                "Best (Hebb)", "Median (Hebb)", "Worst (Hebb)", "Std",
            ] + [name for name, _ in extra_cols]
        else:
            breakdown_headers = [
                "Reward Component", "Best (Hebb)", "Median (Hebb)", "Worst (Hebb)", "Std",
            ]

        print("\n  Reward Breakdown (per-component episode sum):")
        print(tabulate(
            breakdown_rows,
            headers=breakdown_headers,
            tablefmt="grid",
            numalign="center",
            stralign="left",
        ))
    print()


# ============================================================================
#  Main class
# ============================================================================

class HebbianCMAES:
    """CMA-ES loop for Hebbian plasticity rules optimisation.

    Maximises the WP1 scalar reward sum (``env.last_reward_total`` accumulated
    over an episode) using the pycma ``CMAEvolutionStrategy``.

    Parameters
    ----------
    cfg : HebbianEvolutionConfig
    """

    def __init__(self, cfg: HebbianEvolutionConfig) -> None:
        self.cfg = cfg

        self.n_genes = cfg.hebbian_genome_dim()
        if self.n_genes == 0:
            raise ValueError("Hebbian genome dimension is 0. Enable hebbian in config.")

        # Pre-load frozen actor
        from WP2.frozen_actor import load_frozen_actor
        from WP1.config import RunConfig
        self._model, self._last_layer, _, _ = load_frozen_actor(
            cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device
        )
        self._wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
        self._model_and_layer = (
            self._model, self._last_layer,
            cfg.hebbian.num_actions, cfg.hebbian.hidden_dim,
        )

        # Run directory (created early so URDF auto-generation can write here)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{stamp}_{cfg.exp_name}"
        self.run_dir = Path(cfg.base_dir) / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.repro_dir = self.run_dir / "reproducibility"
        self.gen_dir = self.run_dir / "generations"
        self.results_dir = self.run_dir / "results"
        for d in (self.repro_dir, self.gen_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        #  Resolve URDFs and pick evaluation path
        # ------------------------------------------------------------------
        #
        # The multi-URDF path runs ALL URDFs in a single Genesis scene via
        # ``MultiSceneEvalEnv`` — one scene per URDF, reused across generations.
        # The single-URDF legacy path is preserved byte-for-byte for N=1 runs
        # without an explicit catalog (default/fixed morphology).
        self._catalog: Optional[List[Tuple[str, str]]] = None
        self._urdf_paths: Optional[List[str]] = None
        self._use_multi_urdf: bool = False

        if cfg.catalog.path:
            self._urdf_paths = _load_catalog_paths(cfg.catalog.path)
            self._use_multi_urdf = True
            print(f"[HebbianCMAES] Catalog: {len(self._urdf_paths)} URDFs "
                  f"from {cfg.catalog.path}  → multi-URDF path")
        elif cfg.catalog.num_urdfs > 1 or cfg.catalog.force_multi_urdf:
            n = max(1, cfg.catalog.num_urdfs)
            gen_dir = self.run_dir / "urdfs"
            print(f"[HebbianCMAES] No catalog — generating {n} random URDFs "
                  f"into {gen_dir}")
            self._urdf_paths = _generate_random_urdfs(
                gen_dir,
                n,
                cfg.seed,
                include_standard_mydrone=cfg.catalog.include_standard_mydrone,
            )
            self._use_multi_urdf = True
            force_note = " (force_multi_urdf=True)" if (n == 1 and cfg.catalog.force_multi_urdf) else ""
            print(f"[HebbianCMAES] Generated {len(self._urdf_paths)} URDFs "
                  f"→ multi-URDF path{force_note}")
        else:
            print("[HebbianCMAES] No catalog — legacy single-URDF path "
                  "(default/fixed morphology)")
            from general_policy.catalog import write_single_urdf_catalog
            from winged_drone_train.defaults import default_mydrone_urdf_path
            write_single_urdf_catalog(
                self.run_dir / "urdfs",
                default_mydrone_urdf_path(),
            )

        # When ``catalog.mutate`` drives the periodic refresh, the current
        # catalog's normalized genomes are the source of truth for the next
        # mutation (perturb → rebuild → repeat). Capture them once up front
        # from the starting catalog's ``genomes.txt`` so subsequent mutations
        # accumulate from in-memory state (preserving sub-bin drift on the
        # discrete genes). Fail fast if they can't be read.
        self._urdf_norm_genomes: Optional[List[List[float]]] = None
        if (
            self._use_multi_urdf
            and getattr(cfg.catalog, "mutate", False)
            and int(getattr(cfg.catalog, "refresh_urdfs_every", 0) or 0) > 0
        ):
            cat_dir = Path(self._urdf_paths[0]).parent
            self._urdf_norm_genomes = _read_norm_genomes_from_catalog(cat_dir)
            if self._urdf_norm_genomes is None:
                raise ValueError(
                    "catalog.mutate=true requires a parsable genomes.txt in the "
                    f"starting catalog dir, but none was found at {cat_dir}. "
                    "Auto-generated catalogs always include one."
                )

        # For rules-only evolution: pre-build environment once (will be reused)
        self._env = None
        self._env_urdf_path = None

        # ------------------------------------------------------------------
        #  Held-out validation env (optional)
        # ------------------------------------------------------------------
        # A SEPARATE pool of `n_val_envs` Genesis envs, split equally across the
        # `validation_catalog` URDFs (default: standard mydrone). Every `period`
        # generations the best individual and the zero-rules baseline are both
        # evaluated here on freshly-generated, held-out forests — see
        # `_build_validation_env_once` / the validation block in `run()`.
        self._val_env = None
        self._val_urdf_paths: Optional[List[str]] = None
        if self.cfg.validation.enable:
            vc = (self.cfg.validation.validation_catalog or "").strip()
            if vc and vc.lower() != "none":
                self._val_urdf_paths = _load_catalog_paths(vc)
                print(f"[HebbianCMAES] Validation catalog: "
                      f"{len(self._val_urdf_paths)} URDFs from {vc}")
            else:
                from winged_drone_train.defaults import default_mydrone_urdf_path
                self._val_urdf_paths = [str(default_mydrone_urdf_path())]
                print("[HebbianCMAES] Validation catalog empty → standard mydrone drone")

        # CSV paths
        self.pop_csv_path = self.results_dir / "cma_population.csv"
        self.summary_csv_path = self.results_dir / "cma_summary.csv"
        self.baseline_csv_path = self.results_dir / "baseline_summary.csv"
        self.specialist_csv_path = self.results_dir / "specialist_summary.csv"
        self.validation_csv_path = self.results_dir / "validation_summary.csv"
        _init_csvs(self.pop_csv_path, self.summary_csv_path)
        _init_baseline_csv(self.baseline_csv_path)
        if self.cfg.evaluation.run_specialist:
            _init_baseline_csv(self.specialist_csv_path)
        if self.cfg.validation.enable:
            _init_validation_csv(
                self.validation_csv_path,
                include_specialist=self.cfg.evaluation.run_specialist,
            )

        # Timing
        self._run_start_time: Optional[float] = None
        self._gen_start_time: Optional[float] = None

        # All-time best trackers: one entry per saved metric.
        # Each value: {"genome": np.ndarray, "gen": int, "idx": int,
        #              "fitness": float, <metric_key>: float}
        self._best_trackers: Dict[str, Optional[dict]] = {
            "fitness":            None,  # higher is better
            "progress":           None,  # higher is better
            "velocity_deviation": None,  # lower is better
            "cost_of_transport":  None,  # lower is better
            "crash_rate":         None,  # lower is better
        }

    # ------------------------------------------------------------------
    #  Environment setup (rules-only only)
    # ------------------------------------------------------------------

    def _build_env_once(self) -> None:
        """Build the environment once and keep it alive across generations.

        - **Multi-URDF path** (``self._use_multi_urdf``): build a
          ``MultiSceneEvalEnv`` with N independent Genesis scenes (one
          ``WingedDroneEnv`` per URDF, same pattern as WP1's ``Gen_Env``).
          Sized so each drone gets ``(num_eval_envs // N) // H * H`` parallel
          envs, where H is the CMA-ES population size (so every
          (urdf, individual) pair gets the same integer number of forests).
        - **Single-URDF legacy path**: build the WP1 default ``WingedDroneEnv``
          with ``num_eval_envs`` envs.  Identical to the original behavior.
        """
        import genesis as gs

        if self._use_multi_urdf:
            from WP2.evaluate import _build_multi_urdf_env

            N = len(self._urdf_paths)
            total = self.cfg.evaluation.num_eval_envs
            # Population size for sizing: same formula the CMA-ES loop uses.
            n_pop_cfg = self.cfg.cmaes.population_size
            H = n_pop_cfg if n_pop_cfg > 0 else int(4 + 3 * np.log(self.n_genes))
            # Round envs/drone so it's divisible by H (so F = envs/drone / H is integer).
            F = max(1, (total // N) // H)
            envs_per_drone = H * F
            total_slots = N * envs_per_drone

            print(f"[HebbianCMAES] Building MultiSceneEvalEnv: N={N} URDFs × "
                  f"{envs_per_drone} envs/drone = {total_slots} slots "
                  f"(H={H}, F={F} forests per (urdf, individual))...", flush=True)
            try:
                n_workers = int(getattr(self.cfg.evaluation, "num_eval_workers", 1))
                if not gs._initialized:
                    if n_workers > 1:
                        print("[HebbianCMAES] Skipping main-process gs.init() "
                              "(parallel workers each init Genesis in their own process)",
                              flush=True)
                    else:
                        print("[HebbianCMAES] Initializing Genesis (main process)...",
                              flush=True)
                        gs.init(logging_level="error", backend=gs.gpu)
                        print("[HebbianCMAES] Genesis initialized", flush=True)
                n_gpus = int(getattr(self.cfg.evaluation, "num_gpus", 0)) or None
                self._env = _build_multi_urdf_env(
                    urdf_paths=self._urdf_paths,
                    cfg=self.cfg,
                    wp1_cfg=self._wp1_cfg,
                    device=self.cfg.device,
                    num_envs_per_drone=envs_per_drone,
                    num_workers=n_workers,
                    num_gpus=n_gpus,
                )
                self._env_urdf_path = list(self._urdf_paths)
                print(f"[HebbianCMAES] MultiSceneEvalEnv ready", flush=True)
            except Exception as exc:
                # Do NOT swallow: silently continuing makes the CMA-ES loop
                # rebuild the env from within evaluate.py, spawning ANOTHER
                # set of workers while the just-failed ones may still hold
                # GPU memory. That cascades into all-zero-fitness runs that
                # waste hours. Surface the failure and stop the run.
                self._env = None
                self._env_urdf_path = None
                print(f"[HebbianCMAES] Failed to build MultiSceneEvalEnv: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                raise
            return

        # Legacy single-URDF path — unchanged.
        from WP2.evaluate import _build_env

        print("[HebbianCMAES] Building environment (rules-only, single URDF)...")
        try:
            if not gs._initialized:
                gs.init(logging_level="error", backend=gs.gpu)

            num_envs = self.cfg.evaluation.num_eval_envs
            self._env, self._env_urdf_path = _build_env(
                self.cfg, self._wp1_cfg, self.cfg.device,
                num_envs_override=num_envs,
            )
            print(f"[HebbianCMAES] Environment ready: {num_envs} envs")
        except Exception as exc:
            print(f"[HebbianCMAES] Failed to build environment: {exc}")
            self._env = None
            self._env_urdf_path = None

    def _cleanup_env(self) -> None:
        """Destroy the pre-built environment (called at run end or before refresh)."""
        if self._env is None:
            return
        import genesis as gs
        print("[HebbianCMAES] Destroying environment...")
        # ParallelMultiSceneEvalEnv runs its Genesis scenes inside subprocesses,
        # so gs.destroy() on the main process can't reach them — explicitly
        # ask the wrapper to shutdown its workers first.
        shutdown = getattr(self._env, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[HebbianCMAES] Warning: parallel env shutdown failed: {exc}")
        try:
            if gs._initialized:
                gs.destroy()
        except Exception as exc:
            print(f"[HebbianCMAES] Warning: gs.destroy() failed: {exc}")
        self._env = None
        self._env_urdf_path = None

    # ------------------------------------------------------------------
    #  Held-out validation environment
    # ------------------------------------------------------------------

    def _build_validation_env_once(self) -> None:
        """Reserve the held-out validation env: ``n_val_envs`` slots split
        equally across the validation URDFs, each URDF in its own Genesis scene.

        Built the same way as the population eval env (``_build_multi_urdf_env``),
        so it lives in the same Genesis runtime and inherits the CRN setting
        (``evaluation.crn``: on only when true, legacy otherwise). Sized for a
        single genome per pass (P=1), so every slot becomes one forest:
        ``F = n_val_envs // num_validation_urdfs``.
        """
        import genesis as gs
        from WP2.evaluate import _build_multi_urdf_env

        N = len(self._val_urdf_paths)
        total = int(self.cfg.validation.n_val_envs)
        envs_per_drone = max(1, total // N)
        actual = envs_per_drone * N
        if actual != total:
            print(f"[HebbianCMAES] validation: n_val_envs={total} not divisible by "
                  f"{N} URDFs → using {envs_per_drone} envs/URDF ({actual} total)")

        n_workers = int(getattr(self.cfg.evaluation, "num_eval_workers", 1))
        print(f"[HebbianCMAES] Building validation env: N={N} URDFs × "
              f"{envs_per_drone} envs = {actual} held-out forests "
              f"(period={self.cfg.validation.period})...", flush=True)
        try:
            if not gs._initialized and n_workers <= 1:
                gs.init(logging_level="error", backend=gs.gpu)
            n_gpus = int(getattr(self.cfg.evaluation, "num_gpus", 0)) or None
            self._val_env = _build_multi_urdf_env(
                urdf_paths=self._val_urdf_paths,
                cfg=self.cfg,
                wp1_cfg=self._wp1_cfg,
                device=self.cfg.device,
                num_envs_per_drone=envs_per_drone,
                num_workers=n_workers,
                num_gpus=n_gpus,
            )
            print("[HebbianCMAES] Validation env ready", flush=True)
        except Exception as exc:
            self._val_env = None
            print(f"[HebbianCMAES] Failed to build validation env: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            raise

    def _cleanup_validation_env(self) -> None:
        """Release the validation env. Shuts down parallel workers if any; the
        sequential scenes are torn down by ``_cleanup_env``'s ``gs.destroy()``."""
        if self._val_env is None:
            return
        shutdown = getattr(self._val_env, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[HebbianCMAES] Warning: validation env shutdown failed: {exc}")
        self._val_env = None

    # ------------------------------------------------------------------
    #  Periodic URDF refresh
    # ------------------------------------------------------------------

    def _refresh_urdfs(self, gen: int) -> None:
        """Refresh the morphology pool and rebuild the eval env.

        Triggered every ``catalog.refresh_urdfs_every`` inner generations
        (multi-URDF path only). With ``catalog.mutate=False`` (default) this
        resamples a brand-new *random* URDF population; with ``mutate=True`` it
        instead perturbs the current catalog's genomes (see ``_mutate_urdfs``).
        New URDFs are written under ``urdfs_gen_XXX/`` for reproducibility.
        Seed = ``cfg.seed + gen`` so each refresh is deterministic given the
        run seed.
        """
        if not self._use_multi_urdf:
            return
        if getattr(self.cfg.catalog, "mutate", False):
            self._mutate_urdfs(gen)
            return
        n = max(1, self.cfg.catalog.num_urdfs)
        new_dir = self.run_dir / f"urdfs_gen_{gen:03d}"
        print(
            f"\n[HebbianCMAES] === Refreshing URDFs at gen {gen} → {new_dir} ==="
        )
        print(
            f"[HebbianCMAES] Sampling {n} new random URDFs "
            f"(include_standard_mydrone="
            f"{self.cfg.catalog.include_standard_mydrone}, "
            f"seed={self.cfg.seed + gen})"
        )
        new_paths = _generate_random_urdfs(
            new_dir,
            n,
            self.cfg.seed + gen,
            include_standard_mydrone=self.cfg.catalog.include_standard_mydrone,
        )
        # The held-out validation env shares the Genesis runtime, so the
        # `gs.destroy()` inside `_cleanup_env()` tears down its scenes too —
        # release and rebuild it alongside the population env.
        self._cleanup_validation_env()
        self._cleanup_env()
        self._urdf_paths = new_paths
        self._build_env_once()
        if self.cfg.validation.enable:
            self._build_validation_env_once()

    def _mutate_urdfs(self, gen: int) -> None:
        """Mutate the current catalog's genomes and rebuild the eval env.

        Used when ``catalog.mutate=True``. Each URDF's normalized genome (held
        in ``self._urdf_norm_genomes``) is perturbed by per-gene Gaussian noise
        of std ``catalog.mutation_std`` and clamped back to ``[0, 1]``; the noisy
        genomes become the new in-memory state, so successive refreshes drift
        cumulatively from where the last one left off. Every URDF is perturbed,
        including the standard-mydrone baseline. The mutated genomes are
        materialised into ``urdfs_gen_XXX/`` (discrete NACA genes snapped at
        emit time) and a ``catalog.txt`` + ``genomes.txt`` are written there.
        Seed = ``cfg.seed + gen`` so each mutation is deterministic.
        """
        from WP2_Outer_Loop.urdf_population import materialize_urdfs, write_catalog_txt
        from general_policy.catalog import _write_genomes_txt
        from morph_evolution.chromosome_drone import Chromosome_Drone

        current = self._urdf_norm_genomes
        if current is None:
            # Lazy fallback (should have been captured in __init__): recover
            # from the current catalog dir's genomes.txt.
            current = _read_norm_genomes_from_catalog(
                Path(self._urdf_paths[0]).parent
            )
            if current is None:
                raise ValueError(
                    "catalog.mutate=true could not recover the current catalog's "
                    "normalized genomes (no parsable genomes.txt found)."
                )

        std = float(self.cfg.catalog.mutation_std)
        new_dir = self.run_dir / f"urdfs_gen_{gen:03d}"
        print(
            f"\n[HebbianCMAES] === Mutating URDFs at gen {gen} → {new_dir} ==="
        )
        print(
            f"[HebbianCMAES] Perturbing {len(current)} current genomes "
            f"(Gaussian std={std}, clamp [0,1], seed={self.cfg.seed + gen})"
        )

        rng = np.random.default_rng(self.cfg.seed + gen)
        mutated: List[List[float]] = []
        for g in current:
            g_arr = np.asarray(g, dtype=float)
            noisy = g_arr + rng.normal(0.0, std, size=g_arr.shape)
            mutated.append(np.clip(noisy, 0.0, 1.0).tolist())

        # Build URDFs first (UrdfMaker needs the live Genesis runtime, which
        # _cleanup_env() tears down), mirroring the random-refresh ordering.
        new_paths = materialize_urdfs(mutated, new_dir)
        write_catalog_txt(new_paths, new_dir)
        # Persist the physical (snapped) genomes for inspection / reproducibility,
        # matching what every other catalog dir carries.
        phys = [
            Chromosome_Drone.to_physical(Chromosome_Drone.snap_genome_norm(g))
            for g in mutated
        ]
        _write_genomes_txt(new_dir, [Path(p) for p in new_paths], phys)

        # The held-out validation env shares the Genesis runtime, so the
        # `gs.destroy()` inside `_cleanup_env()` tears down its scenes too —
        # release and rebuild it alongside the population env.
        self._cleanup_validation_env()
        self._cleanup_env()
        self._urdf_paths = new_paths
        self._urdf_norm_genomes = mutated
        self._build_env_once()
        if self.cfg.validation.enable:
            self._build_validation_env_once()

    # ------------------------------------------------------------------
    #  Reference controller evaluation (zero Hebbian rules)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Uncertainty handling (UH-CMA-ES, σ-only arm)
    # ------------------------------------------------------------------

    def _uh_reevaluate(self, solutions: List[np.ndarray]) -> np.ndarray:
        """Re-evaluate the full population on the CURRENT forests (no refresh).

        Reuses the pre-built env (size matches the full population), so the
        only differences from the just-completed evaluation are the irreducible
        per-rollout stochastic sources — stochastic actor sampling and per-step
        aero force noise — which is exactly the rank-noise CRN cannot remove.
        Forests/DR draws are NOT refreshed here on purpose: refreshing would
        fold forest-draw variance (which CRN makes common across individuals)
        back into the measurement.
        """
        existing_env = (
            (self._env, self._env_urdf_path) if self._env is not None else None
        )
        if self._use_multi_urdf:
            fitnesses, _ = evaluate_population_multi_urdf(
                solutions,
                self.cfg,
                self._model_and_layer,
                self._wp1_cfg,
                urdf_paths=self._urdf_paths,
                existing_env=existing_env,
                verbose=False,
            )
        else:
            fitnesses, _ = evaluate_population_cma_batched(
                solutions,
                self.cfg,
                self._model_and_layer,
                self._wp1_cfg,
                catalog=self._catalog,
                existing_env=existing_env,
                verbose=False,
            )
        return np.asarray(fitnesses, dtype=float)

    def _apply_uncertainty_handling(
        self,
        es,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
        gen: int,
    ) -> float:
        """Measure rank-noise via a full-population re-eval and bump sigma.

        Reuses pycma's exact Hansen statistic (``NoiseHandler.update_measure``)
        and its step-size treatment (``treat``); only the re-evaluation is
        driven through our batched evaluator instead of pycma's per-individual
        ``func``. Returns the applied sigma multiplier.
        """
        nh = self._nh
        try:
            re_fit = self._uh_reevaluate(solutions)
        except Exception as exc:
            print(f"[UH] gen {gen}: re-evaluation failed ({exc}); skipping")
            return 1.0

        # pycma operates in minimisation space (the optimiser sees -reward).
        # The measure is purely rank-based (|rankDelta|, symmetric limits) so
        # the sign does not change noiseS, but negate for faithfulness.
        fit_min = list((-np.asarray(fitnesses, dtype=float)))
        refit_min = list((-re_fit))
        nh.fit = fit_min
        nh.fitre = refit_min
        # Full re-eval is already paid for, so default to the whole population
        # for a robust measure; honour a positive uh_reevals subset if set.
        if nh.lam_reeval:
            nh.idx = nh.indices(fit_min)
        else:
            nh.idx = np.arange(len(fit_min))

        nh.update_measure()
        factor = nh.treat()  # in {1.0, alphasigma}; evaluations stays 1 (maxevals=1)

        old_sigma = float(es.sigma)
        es.sigma *= factor
        # Stash for this gen's CSV row + table (consumed in _after_generation).
        self._last_uh = {
            "noiseS": float(nh.noiseS),
            "factor": float(factor),
            "sigma_pre": old_sigma,
            "sigma_post": float(es.sigma),
            "n_reeval": int(len(nh.idx)),
        }
        print(
            f"[UH] gen {gen}: noiseS={nh.noiseS:+.4f} factor={factor:.4f} "
            f"sigma {old_sigma:.4g} → {float(es.sigma):.4g} "
            f"(reeval {len(nh.idx)}/{len(solutions)} indiv, no forest refresh)"
        )
        return factor

    @staticmethod
    def _pack_eval_result(
        fitnesses: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Reduce a single-genome eval (P=1) to a flat per-metric dict."""
        result = {
            "fitness":     float(fitnesses[0]),
            "velocity":    float(metrics["velocities"][0]),
            "progress":    float(metrics["progresses"][0]),
            "crash_rate":  float(metrics["crash_flags"][0]),
            "cot":         float(metrics["cots"][0]),
            "v_deviation": float(metrics["v_deviations"][0]),
        }
        comp_arr = metrics.get("reward_components")
        names = metrics.get("reward_names", []) or []
        if comp_arr is not None and len(names) and comp_arr.size:
            result["reward_components"] = {
                name: float(comp_arr[0, i]) for i, name in enumerate(names)
            }
        return result

    def _evaluate_genome_on_validation(
        self,
        genome: np.ndarray,
        verbose: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Evaluate ONE genome (with its Hebbian rules) on the held-out
        validation env. Used for the generation's best-fitness individual.

        Assumes the caller has already refreshed the validation forests for
        this pass (so the baseline that follows flies the same layouts).
        """
        if self._val_env is None:
            return None
        try:
            fitnesses, metrics = evaluate_population_multi_urdf(
                [np.asarray(genome, dtype=float)],
                self.cfg,
                self._model_and_layer,
                self._wp1_cfg,
                urdf_paths=self._val_urdf_paths,
                existing_env=(self._val_env, self._val_urdf_paths),
                verbose=verbose,
            )
            return self._pack_eval_result(fitnesses, metrics)
        except Exception as exc:
            print(f"[HebbianCMAES] Validation (best individual) eval failed: {exc}")
            return None

    def _evaluate_reference_actor(
        self,
        label: str,
        ckpt_path: Optional[str],
        ckpt_cfg_path: Optional[str],
        verbose: bool = False,
        env_override=None,
        urdf_paths_override: Optional[List[str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Evaluate a frozen reference actor with zero Hebbian rules.

        Shared implementation for the *baseline*, the *specialist*, and the
        held-out *validation baseline* curves. By default the reference actor
        runs on the SAME env (same forests, same speed grid, same URDFs) as the
        population, which is the whole point — every generation's comparison is
        fair by construction. Passing ``env_override`` / ``urdf_paths_override``
        instead runs it on that env (used for the held-out validation pass).

        ABCD=0 (genome=0.5 for symmetric [-1,1] ranges) means no plasticity
        update.  Decay is also zeroed (config copy) so weights stay exactly
        at the checkpoint values throughout the episode.

        Parameters
        ----------
        label : str
            "Baseline" or "Specialist" — only used in log strings.
        ckpt_path, ckpt_cfg_path : str or None
            Reference actor checkpoint + matching WP1 config. If both are
            None/empty the main frozen actor is reused (used by baseline's
            legacy "same checkpoint" mode; not valid for specialist).
        """
        import copy

        ref_cfg = copy.deepcopy(self.cfg)
        ref_cfg.hebbian.decay = 0.0

        if ckpt_path:
            ref_cfg.checkpoint_path = ckpt_path
            ref_cfg.checkpoint_config_path = ckpt_cfg_path

            # Re-infer last-layer dims from this checkpoint so the genome
            # length matches a (possibly different) architecture.
            import torch as _torch
            from WP2.frozen_actor import last_actor_linear_key
            _ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
            _last_key = last_actor_linear_key(_sd)
            if _last_key is not None:
                ref_cfg.hebbian.num_actions = _sd[_last_key].shape[0]
                ref_cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
            del _ckpt, _sd

        n_weights = ref_cfg.hebbian.num_actions * ref_cfg.hebbian.hidden_dim
        n_genes_ref = ref_cfg.hebbian_genome_dim()

        ref_genome = np.full(n_genes_ref, 0.5)
        if ref_cfg.hebbian.evolve_decay:
            # ABCD occupy 4 × abcd_block_size genes; the per-weight decay block
            # starts right after them.
            decay_start = 4 * ref_cfg.hebbian.abcd_block_size()
            ref_genome[decay_start: decay_start + n_weights] = 0.0

        # Target the override env (held-out validation) when given, else the
        # population eval env. An override always routes through the multi-URDF
        # path (the validation env is a MultiSceneEvalEnv even for one URDF).
        if env_override is not None:
            existing_env = (env_override, urdf_paths_override)
            use_multi = True
            urdf_paths = urdf_paths_override
        else:
            existing_env = (
                (self._env, self._env_urdf_path)
                if self._env is not None
                else None
            )
            use_multi = self._use_multi_urdf
            urdf_paths = self._urdf_paths
        try:
            if verbose:
                print(f"[HebbianCMAES] Evaluating {label.lower()} "
                      f"(zero-Hebbian, zero-decay)...")
            if use_multi:
                fitnesses, metrics = evaluate_population_multi_urdf(
                    [ref_genome],
                    ref_cfg,
                    self._model_and_layer,
                    self._wp1_cfg,
                    urdf_paths=urdf_paths,
                    existing_env=existing_env,
                    verbose=verbose,
                )
            else:
                fitnesses, metrics = evaluate_population_cma_batched(
                    [ref_genome],
                    ref_cfg,
                    self._model_and_layer,
                    self._wp1_cfg,
                    catalog=self._catalog,
                    existing_env=existing_env,
                    verbose=verbose,
                )
            result = self._pack_eval_result(fitnesses, metrics)
            if verbose:
                print(
                    f"[HebbianCMAES] {label}: fitness={result['fitness']:.4g}  "
                    f"vel={result['velocity']:.4g}  prog={result['progress']:.4g}  "
                    f"crash={result['crash_rate']:.4g}  cot={result['cot']:.4g}  "
                    f"v_dev={result['v_deviation']:.4g} m/s"
                )
            return result
        except Exception as exc:
            print(f"[HebbianCMAES] {label} evaluation failed: {exc}")
            return None

    def _evaluate_baseline(self, verbose: bool = False) -> Optional[Dict[str, float]]:
        """Evaluate the (optionally separate) baseline actor with zero rules."""
        return self._evaluate_reference_actor(
            "Baseline",
            ckpt_path=self.cfg.baseline_checkpoint_path or None,
            ckpt_cfg_path=self.cfg.baseline_checkpoint_config_path or None,
            verbose=verbose,
        )

    def _evaluate_specialist(self, verbose: bool = False) -> Optional[Dict[str, float]]:
        """Evaluate the specialist actor with zero Hebbian rules.

        Unlike the baseline, the specialist requires its own checkpoint pair
        (validated up-front in ``run.py``); there is no "reuse the frozen
        actor" fallback.
        """
        return self._evaluate_reference_actor(
            "Specialist",
            ckpt_path=self.cfg.specialist_checkpoint_path or None,
            ckpt_cfg_path=self.cfg.specialist_checkpoint_config_path or None,
            verbose=verbose,
        )

    def _run_validation_pass(
        self,
        gen: int,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
        verbose: bool = False,
    ) -> None:
        """Evaluate the generation's best individual AND the zero-rules baseline
        (AND the specialist, when ``evaluation.run_specialist`` is set) on the
        held-out validation env, then log + print the comparison.

        Forests are regenerated once at the start of the pass so they are new
        each period; all controllers then fly the SAME layouts (no refresh
        between them) for a fair paired comparison under CRN.
        """
        if self._val_env is None:
            return

        # Fresh held-out forests for this pass (new trees, distinct from the
        # population's eval pool). Both controllers below see these same layouts.
        try:
            self._val_env.refresh_forests()
        except Exception as exc:
            print(f"[HebbianCMAES] Validation forest refresh failed: {exc}")
            return

        best_idx = int(np.argmax(fitnesses))
        best_genome = solutions[best_idx]

        # All controllers fly the SAME freshly-refreshed forests (no refresh
        # between them) so the comparison is paired under CRN.
        val_best = self._evaluate_genome_on_validation(best_genome, verbose=verbose)
        val_baseline = self._evaluate_reference_actor(
            "Validation-Baseline",
            ckpt_path=self.cfg.baseline_checkpoint_path or None,
            ckpt_cfg_path=self.cfg.baseline_checkpoint_config_path or None,
            verbose=verbose,
            env_override=self._val_env,
            urdf_paths_override=self._val_urdf_paths,
        )

        # Optional third controller: the specialist actor, evaluated on the very
        # same held-out forests as best + baseline.
        val_specialist = None
        if self.cfg.evaluation.run_specialist:
            val_specialist = self._evaluate_reference_actor(
                "Validation-Specialist",
                ckpt_path=self.cfg.specialist_checkpoint_path or None,
                ckpt_cfg_path=self.cfg.specialist_checkpoint_config_path or None,
                verbose=verbose,
                env_override=self._val_env,
                urdf_paths_override=self._val_urdf_paths,
            )

        if val_best is None or val_baseline is None:
            print(f"[HebbianCMAES] Validation pass at gen {gen} produced no result.")
            return

        # When the specialist is enabled, always write its block (NaN-filled on
        # eval failure) so the CSV column count matches the header.
        spec_for_csv = None
        if self.cfg.evaluation.run_specialist:
            spec_for_csv = val_specialist if val_specialist is not None else {}

        _append_validation_csv(
            self.validation_csv_path, gen, best_idx, val_best, val_baseline,
            specialist=spec_for_csv,
        )
        self._print_validation(
            gen, best_idx, val_best, val_baseline, specialist=val_specialist
        )

    def _print_validation(
        self,
        gen: int,
        best_idx: int,
        best: Dict[str, float],
        baseline: Dict[str, float],
        specialist: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print the held-out best-vs-baseline[-vs-specialist] comparison table."""
        n_val = len(self._val_urdf_paths) if self._val_urdf_paths else 0
        rows = [
            ("Fitness (reward)", "fitness"),
            ("Velocity [m/s]",   "velocity"),
            ("Vel. Dev. [m/s]",  "v_deviation"),
            ("Progress [m]",     "progress"),
            ("COT",              "cot"),
            ("Crash Rate",       "crash_rate"),
        ]

        def _pct(num: float, den: float) -> str:
            return (f"{(num - den) / abs(den) * 100.0:+.1f}%"
                    if abs(den) > 1e-12 else "n/a")

        table_rows = []
        for name, key in rows:
            b = best.get(key, float("nan"))
            r = baseline.get(key, float("nan"))
            row = [name, f"{b:.4g}", f"{r:.4g}"]
            if specialist is not None:
                s = specialist.get(key, float("nan"))
                row += [f"{s:.4g}", _pct(b, r), _pct(b, s)]
            else:
                row.append(_pct(b, r))
            table_rows.append(row)

        if specialist is not None:
            headers = ["Metric", "Best (Hebb)", "Baseline", "Specialist",
                       "Δ vs base", "Δ vs spec"]
        else:
            headers = ["Metric", "Best (Hebb)", "Baseline", "Δ vs baseline"]

        print(f"\n  Held-out Validation (gen {gen}, best individual #{best_idx}, "
              f"{n_val} URDF(s) × fresh forests):")
        print(tabulate(
            table_rows,
            headers=headers,
            tablefmt="grid",
            numalign="center",
            stralign="left",
        ))
        print()

    # ------------------------------------------------------------------
    #  Reproducibility
    # ------------------------------------------------------------------

    def _save_reproducibility(self) -> None:
        import shutil

        self.cfg.to_yaml(self.repro_dir / "config.yaml")

        if self.cfg.checkpoint_config_path and Path(self.cfg.checkpoint_config_path).is_file():
            shutil.copy2(self.cfg.checkpoint_config_path, self.repro_dir / "wp1_config.yaml")

        if self.cfg.checkpoint_path and Path(self.cfg.checkpoint_path).is_file():
            shutil.copy2(self.cfg.checkpoint_path, self.repro_dir / "wp1_actor.pt")

        save_git_info(self.repro_dir / "git_info.txt")
        save_environment_info(self.repro_dir / "environment.txt")

    # ------------------------------------------------------------------
    #  Per-generation bookkeeping
    # ------------------------------------------------------------------

    def _save_generation(
        self,
        gen: int,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
        es,
    ) -> None:
        gen_path = self.gen_dir / f"gen_{gen:03d}"
        gen_path.mkdir(parents=True, exist_ok=True)

        # Population + fitnesses as numpy
        np.save(gen_path / "solutions.npy", np.array(solutions))
        np.save(gen_path / "fitnesses.npy", fitnesses)

        # CMA-ES internal state (for resume)
        import pickle
        with open(gen_path / "cmaes_state.pkl", "wb") as f:
            pickle.dump(es, f)

        # RNG state
        save_rng_state(gen_path / "rng_state.pkl")

    def _after_generation(
        self,
        gen: int,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
        metrics: Dict[str, np.ndarray],
        es,
        baseline: Optional[Dict[str, float]] = None,
        specialist: Optional[Dict[str, float]] = None,
    ) -> None:
        import time

        sigma = float(es.sigma)
        axis_ratio, cond_number = _extract_cma_state(es)

        # UH measurement for this gen (set by _apply_uncertainty_handling, None
        # on non-measurement gens / when UH is disabled). Consume + clear so it
        # is logged exactly once.
        uh = getattr(self, "_last_uh", None)
        self._last_uh = None

        # CSV logging
        _append_population_csv(self.pop_csv_path, gen, fitnesses, metrics)
        _append_summary_csv(
            self.summary_csv_path, gen, fitnesses, metrics,
            sigma, axis_ratio, cond_number, uh=uh,
        )
        if baseline is not None:
            _append_baseline_csv(self.baseline_csv_path, gen, baseline)
        if specialist is not None:
            _append_baseline_csv(self.specialist_csv_path, gen, specialist)

        # Generation checkpoint
        self._save_generation(gen, solutions, fitnesses, es)

        # Timing
        now = time.perf_counter()
        iter_elapsed = now - self._gen_start_time if self._gen_start_time else None
        total_elapsed = now - self._run_start_time if self._run_start_time else None
        eta_secs = None
        if gen > 0 and total_elapsed:
            avg = total_elapsed / gen
            eta_secs = avg * (self.cfg.evolution.num_generations - gen)

        _print_generation_table(
            gen, fitnesses, metrics, sigma,
            total_elapsed=total_elapsed,
            iter_elapsed=iter_elapsed,
            eta_seconds=eta_secs,
            baseline=baseline,
            specialist=specialist,
            total_generations=self.cfg.evolution.num_generations,
            uh=uh,
        )

    # ------------------------------------------------------------------
    #  All-time best tracking
    # ------------------------------------------------------------------

    _METRIC_SPECS = {
        # key in _best_trackers → (metrics_array_key, higher_is_better)
        "fitness":            ("reward_sums",  True),
        "progress":           ("progresses",   True),
        "velocity_deviation": ("v_deviations", False),
        "cost_of_transport":  ("cots",         False),
        "crash_rate":         ("crash_flags",  False),
    }

    def _update_best_trackers(
        self,
        gen: int,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ) -> None:
        """Update all-time best for each tracked metric after a generation."""
        metric_arrays = dict(metrics)
        metric_arrays["reward_sums"] = fitnesses

        for tracker_key, (arr_key, higher_is_better) in self._METRIC_SPECS.items():
            arr = metric_arrays[arr_key]
            idx = int(np.argmax(arr) if higher_is_better else np.argmin(arr))
            value = float(arr[idx])

            prev = self._best_trackers[tracker_key]
            is_better = (
                prev is None
                or (higher_is_better and value > prev["value"])
                or (not higher_is_better and value < prev["value"])
            )
            if is_better:
                self._best_trackers[tracker_key] = {
                    "genome":             solutions[idx].copy(),
                    "gen":                gen,
                    "idx":                idx,
                    "value":              value,
                    # all metrics for this individual
                    "fitness":            float(fitnesses[idx]),
                    "progress":           float(metric_arrays["progresses"][idx]),
                    "velocity_deviation": float(metric_arrays["v_deviations"][idx]),
                    "cost_of_transport":  float(metric_arrays["cots"][idx]),
                    "crash_rate":         float(metric_arrays["crash_flags"][idx]),
                }

    def _save_best_individual(
        self,
        subdir: Path,
        genome: np.ndarray,
        gen: int,
        idx: int,
        extra_fields: Dict,
    ) -> None:
        """Save genome.npy, hebbian_rules.yaml, and fitness.yaml to *subdir*."""
        import yaml

        subdir.mkdir(parents=True, exist_ok=True)
        np.save(subdir / "genome.npy", genome)

        rules = decode_hebbian_genes(
            list(np.clip(genome, 0.0, 1.0)),
            self.cfg.hebbian,
            out_features=self.cfg.hebbian.num_actions,
            in_features=self.cfg.hebbian.hidden_dim,
        )
        rules_dict = {k: v.cpu().numpy().tolist() for k, v in rules.items()}
        with open(subdir / "hebbian_rules.yaml", "w") as f:
            yaml.dump(rules_dict, f, sort_keys=False)

        record = {"generation": gen, "individual_idx": idx, **extra_fields}
        with open(subdir / "fitness.yaml", "w") as f:
            yaml.dump(record, f, sort_keys=False)

    # ------------------------------------------------------------------
    #  Finalisation
    # ------------------------------------------------------------------

    def _finalize(
        self,
        es,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
    ) -> None:
        best_idx = int(np.argmax(fitnesses)) if len(fitnesses) else 0
        best_fitness = float(fitnesses[best_idx]) if len(fitnesses) else 0.0

        print(f"\n[HebbianCMAES] Done. Best fitness: {best_fitness:.6g} "
              f"(individual {best_idx})")
        print(f"[HebbianCMAES] Results saved to: {self.run_dir}")

        # Save all-time best for each tracked metric
        best_dir = self.run_dir / "best_individual"

        folder_names = {
            "fitness":            "fitness",
            "progress":           "progress",
            "velocity_deviation": "velocity_deviation",
            "cost_of_transport":  "cost_of_transport",
            "crash_rate":         "crash_rate",
        }

        for tracker_key, folder_name in folder_names.items():
            entry = self._best_trackers[tracker_key]
            if entry is None:
                continue
            subdir = best_dir / folder_name
            self._save_best_individual(
                subdir,
                entry["genome"],
                entry["gen"],
                entry["idx"],
                {
                    "fitness":            entry["fitness"],
                    "progress":           entry["progress"],
                    "velocity_deviation": entry["velocity_deviation"],
                    "cost_of_transport":  entry["cost_of_transport"],
                    "crash_rate":         entry["crash_rate"],
                },
            )
            print(f"[HebbianCMAES] Best {tracker_key}: {entry['value']:.6g} "
                  f"(gen {entry['gen']}, ind {entry['idx']}) → {subdir}")

        # Final CMA-ES state
        import pickle
        with open(self.run_dir / "cmaes_final_state.pkl", "wb") as f:
            pickle.dump(es, f)

        print(f"[HebbianCMAES] Best genome + rules saved to: {best_dir}")

    # ------------------------------------------------------------------
    #  Resume support
    # ------------------------------------------------------------------

    def _load_generation(self, gen: int):
        """Restore solutions, fitnesses, and RNG from a generation checkpoint."""
        gen_path = self.gen_dir / f"gen_{gen:03d}"

        solutions_arr = np.load(gen_path / "solutions.npy")
        solutions = list(solutions_arr)
        fitnesses = np.load(gen_path / "fitnesses.npy")

        load_rng_state(gen_path / "rng_state.pkl")

        return solutions, fitnesses

    def _restore_cmaes(self, gen: int) -> "cma.CMAEvolutionStrategy":
        """Restore a CMA-ES instance from a pickled checkpoint."""
        import pickle
        gen_path = self.gen_dir / f"gen_{gen:03d}"
        state_path = gen_path / "cmaes_state.pkl"
        with open(state_path, "rb") as f:
            es = pickle.load(f)
        print(f"[HebbianCMAES] Restored CMA-ES state from gen {gen} "
              f"(sigma={es.sigma:.4g})")
        return es

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        resume_from_gen: Optional[int] = None,
        x0_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run the full CMA-ES optimisation.

        Parameters
        ----------
        resume_from_gen : int, optional
            If given, restores the CMA-ES state and population from the
            checkpoint saved at that generation and continues from there.
        x0_override : np.ndarray, optional
            When starting fresh (no resume), initialise the CMA-ES mean from
            this genome instead of the config default. Used by the outer
            loop to carry the best rules from the previous outer generation
            forward as the starting point for the new CMA-ES search.

        Returns
        -------
        best_genome : np.ndarray, shape (n_genes,)
            The genome with the highest fitness found during the run.
        best_fitness : float
            Corresponding scalar fitness value.
        """
        import time

        print(f"\n[HebbianCMAES] Run directory: {self.run_dir}")
        print(f"[HebbianCMAES] Genome dim: {self.n_genes}")
        print(f"[HebbianCMAES] Generations: {self.cfg.evolution.num_generations}")
        if self._use_multi_urdf:
            print(f"[HebbianCMAES] URDFs: {len(self._urdf_paths)} (multi-URDF path)")
        else:
            print(f"[HebbianCMAES] URDFs: 1 (single-URDF legacy path)")
        n_pop_cfg = self.cfg.cmaes.population_size
        expected_pop = (
            n_pop_cfg if n_pop_cfg > 0
            else int(4 + 3 * np.log(self.n_genes))
        )
        print(f"[HebbianCMAES] Population size: "
              f"{'auto ≈ ' + str(expected_pop) if n_pop_cfg == 0 else n_pop_cfg}")

        self._save_reproducibility()
        self._build_env_once()  # Pre-build environment for rules-only evolution
        if self.cfg.validation.enable:
            self._build_validation_env_once()  # Reserve the held-out validation pool
        self._run_start_time = time.perf_counter()

        # ------------------------------------------------------------------
        #  Initialise CMA-ES
        # ------------------------------------------------------------------

        # pycma options
        opts = cma.CMAOptions()
        opts["bounds"] = [0.0, 1.0]
        opts["maxiter"] = self.cfg.evolution.num_generations
        opts["tolupsigma"] = self.cfg.cmaes.tol_sigma
        opts["tolfun"] = self.cfg.cmaes.tol_fun
        opts["tolfunhist"] = -1.0     # disable history-based fitness tolerance
        opts["tolflatfitness"] = -1.0 # disable flat fitness detection
        opts["verbose"] = -9         # suppress pycma's own output

        algo = self.cfg.cmaes.algorithm.lower().strip()
        if algo in ("cmaes", "cma-es", "cma"):
            opts["CMA_diagonal"] = 0          # full covariance (standard CMA-ES)
        elif algo in ("sep-cmaes", "sep-cma-es", "sep_cmaes", "sep"):
            opts["CMA_diagonal"] = True       # diagonal-only (separable CMA-ES)
        else:
            raise ValueError(
                f"[HebbianCMAES] Unknown cmaes.algorithm '{self.cfg.cmaes.algorithm}'. "
                f"Supported: 'cmaes', 'sep-cmaes'."
            )
        print(f"[HebbianCMAES] Algorithm: {algo} "
              f"(CMA_diagonal={opts['CMA_diagonal']})")

        if self.cfg.cmaes.population_size > 0:
            opts["popsize"] = self.cfg.cmaes.population_size

        if bool(getattr(self.cfg.cmaes, "uniform_weighting", None) or False):
            # Uniform weights 1/mu on the top half, 0 on the bottom half:
            # only top-half membership matters, not the noise-sensitive
            # ordering within it (and active CMA's negative weights are
            # dropped). The weights list fixes popsize to its length, so it
            # must match the configured/auto lambda.
            lam = (
                self.cfg.cmaes.population_size
                if self.cfg.cmaes.population_size > 0
                else int(4 + 3 * np.log(self.n_genes))
            )
            mu = lam // 2
            opts["CMA_recombination_weights"] = [1.0 / mu] * mu + [0.0] * (lam - mu)
            print(f"[HebbianCMAES] Uniform recombination weights: "
                  f"1/{mu} on top {mu} of {lam} (mu_eff={mu}, active-CMA off)")

        if resume_from_gen is not None:
            # Restore CMA-ES from checkpoint
            es = self._restore_cmaes(resume_from_gen)
            start_gen = resume_from_gen + 1
            # Recover last solutions + fitnesses for display purposes
            last_solutions, last_fitnesses = self._load_generation(resume_from_gen)
        else:
            # Fresh start: choose initial mean
            if x0_override is not None:
                x0 = np.clip(np.asarray(x0_override, dtype=float), 0.0, 1.0)
                if x0.shape != (self.n_genes,):
                    raise ValueError(
                        f"x0_override shape {x0.shape} != expected ({self.n_genes},)"
                    )
                print(f"[HebbianCMAES] Using x0_override (carried-over rules)")
            elif self.cfg.hebbian.initialize_rules_to_zero:
                # 0.5 in normalised space maps to 0.0 for symmetric [-1,1] ranges
                x0 = np.full(self.n_genes, 0.5)
            else:
                # Random initialisation uniformly sampled from [0, 1]^n
                x0 = np.random.uniform(0.0, 1.0, self.n_genes)

            es = cma.CMAEvolutionStrategy(x0, self.cfg.cmaes.sigma0, opts)
            start_gen = 0
            last_solutions = None
            last_fitnesses = None

        # ------------------------------------------------------------------
        #  Uncertainty handling (UH-CMA-ES, Hansen et al. 2009; σ-only arm)
        # ------------------------------------------------------------------
        # On resume the handler is recreated fresh (noiseS resets to 0); this
        # only loses a little cumulated history and self-corrects within a few
        # measurement generations.
        self._nh = None
        self._last_uh = None
        if bool(getattr(self.cfg.cmaes, "uh_enabled", False)):
            reev = float(getattr(self.cfg.cmaes, "uh_reevals", 0.0) or 0.0)
            self._nh = cma.NoiseHandler(
                es.N,
                maxevals=[1, 1, 1],          # σ-only: never raise per-individual evals
                reevals=(reev if reev > 0 else None),
            )
            self._nh.theta = float(getattr(self.cfg.cmaes, "uh_theta", 0.5))
            alpha = float(getattr(self.cfg.cmaes, "uh_alphasigma", 0.0) or 0.0)
            if alpha > 0:
                self._nh.alphasigma = alpha
            print(
                f"[HebbianCMAES] UH-CMA-ES enabled: every={max(1, int(self.cfg.cmaes.uh_every))} gen, "
                f"reevals={'full-pop' if reev <= 0 else reev}, "
                f"alphasigma={self._nh.alphasigma:.4g}, theta={self._nh.theta:.4g}"
            )

        # ------------------------------------------------------------------
        #  Evolution loop
        # ------------------------------------------------------------------

        gen = start_gen
        refresh_every = int(getattr(self.cfg.catalog, "refresh_urdfs_every", 0) or 0)
        while not es.stop() and gen <= self.cfg.evolution.num_generations:
            self._gen_start_time = time.perf_counter()
            verbose = (gen == 0)

            # Periodic URDF refresh: resample the morphology pool every K gens.
            # Disabled when refresh_every == 0 (default); also skipped on the
            # single-URDF legacy path (handled inside _refresh_urdfs).
            if (
                refresh_every > 0
                and gen > 0
                and gen % refresh_every == 0
            ):
                self._refresh_urdfs(gen)
                # Carry the CMA-ES state (mean + covariance) across the
                # morphology change, but re-inflate the step size so the search
                # re-explores around the carried state for the new URDFs. The
                # objective just shifted by more than the typical signal, so a
                # collapsed sigma would otherwise leave the search stuck near the
                # previous optimum. The factor multiplies the CURRENT sigma, so
                # the kick scales with how far the search has converged; values
                # <= 1.0 are a no-op (sigma is only ever raised, never lowered).
                reinflate = float(getattr(self.cfg.cmaes, "sigma_reinflate", 0.0) or 0.0)
                if reinflate > 1.0:
                    target = reinflate * es.sigma
                    print(
                        f"[HebbianCMAES] Morphology changed → re-inflating "
                        f"sigma {es.sigma:.4g} → {target:.4g} "
                        f"(×{reinflate:g} current sigma; carry mean+covariance, re-explore)"
                    )
                    es.sigma = target

            # Sample new candidate solutions
            solutions = es.ask()   # list of np.ndarray, each shape (n_genes,)

            # Per-generation linear ramp on dens_min: dens_min(g) = base + slope * g.
            # Only applied when dens_min is explicitly set so the WP1-config
            # default is preserved when the override is null.
            _forest = getattr(self.cfg, "forest", None)
            slope = float(getattr(_forest, "dens_min_slope", 0.0) or 0.0)
            # forest.dens_min (if set) wins over evaluation.dens_min, matching
            # the precedence in evaluate._apply_forest_overrides.
            base_dens_min = (
                _forest.dens_min
                if _forest is not None and _forest.dens_min is not None
                else self.cfg.evaluation.dens_min
            )
            if (
                slope != 0.0
                and base_dens_min is not None
                and self._env is not None
                and hasattr(self._env, "set_dens_min")
            ):
                current_dens_min = float(base_dens_min) + slope * gen
                self._env.set_dens_min(current_dens_min)
                if verbose or gen % 10 == 0:
                    print(
                        f"[HebbianCMAES] gen {gen}: dens_min = {current_dens_min:.4f} "
                        f"(base={base_dens_min} + slope={slope} * gen={gen})"
                    )

            # Optionally refresh forest layouts so each generation sees new trees.
            if self.cfg.evaluation.refresh_forests_per_generation and self._env is not None:
                self._env.refresh_forests()

            # Evaluate (returns scalar fitness per individual)
            try:
                # Pass pre-built environment if available (rules-only)
                existing_env = (
                    (self._env, self._env_urdf_path)
                    if self._env is not None
                    else None
                )
                if self._use_multi_urdf:
                    fitnesses, metrics = evaluate_population_multi_urdf(
                        solutions,
                        self.cfg,
                        self._model_and_layer,
                        self._wp1_cfg,
                        urdf_paths=self._urdf_paths,
                        existing_env=existing_env,
                        verbose=verbose,
                    )
                else:
                    fitnesses, metrics = evaluate_population_cma_batched(
                        solutions,
                        self.cfg,
                        self._model_and_layer,
                        self._wp1_cfg,
                        catalog=self._catalog,
                        existing_env=existing_env,
                        verbose=verbose,
                    )
            except Exception as exc:
                print(f"[HebbianCMAES] Evaluation failed at gen {gen}: {exc}")
                # Tell CMA-ES that all solutions have zero fitness (don't crash run)
                fitnesses = np.zeros(len(solutions))
                metrics = {
                    "reward_sums": fitnesses,
                    "velocities": fitnesses,
                    "progresses": fitnesses,
                    "crash_flags": fitnesses,
                    "cots": fitnesses,
                    "v_deviations": fitnesses,
                }

            # CMA-ES minimises — negate fitness to maximise reward.
            # Optional min-max normalization rescales what CMA-ES sees to
            # [0, 1] (best → 1, worst → 0); logging/validation keep raw values.
            if getattr(self.cfg.cmaes, "normalize_fitness", False):
                tell_fitnesses = minmax_normalize_fitness(fitnesses)
            else:
                tell_fitnesses = fitnesses
            es.tell(solutions, (-tell_fitnesses).tolist())

            # UH-CMA-ES: measure residual rank-noise and bump sigma if found.
            # Done AFTER tell (mirrors the canonical pycma loop) and on a
            # cadence, since each measurement costs ≈ one extra full-pop eval.
            if self._nh is not None:
                uh_every = max(1, int(getattr(self.cfg.cmaes, "uh_every", 1)))
                if gen % uh_every == 0:
                    self._apply_uncertainty_handling(es, solutions, fitnesses, gen)

            # Baseline: evaluate frozen WP1 actor with zero Hebbian rules.
            # Cadence-gated: runs on gen 0 and every `baseline_every` generations.
            baseline_every = max(1, int(self.cfg.evaluation.baseline_every))
            should_run_baseline = (
                self.cfg.evaluation.run_baseline and (gen % baseline_every == 0)
            )
            baseline = self._evaluate_baseline(verbose=verbose) if should_run_baseline else None

            # Specialist: same idea as baseline, but with its own checkpoint.
            specialist_every = max(1, int(self.cfg.evaluation.specialist_every))
            should_run_specialist = (
                self.cfg.evaluation.run_specialist and (gen % specialist_every == 0)
            )
            specialist = (
                self._evaluate_specialist(verbose=verbose) if should_run_specialist else None
            )

            # Bookkeeping
            self._after_generation(
                gen, solutions, fitnesses, metrics, es,
                baseline=baseline, specialist=specialist,
            )
            self._update_best_trackers(gen, solutions, fitnesses, metrics)

            # Held-out validation: evaluate this generation's best individual and
            # the zero-rules baseline on a separate pool of fresh forests.
            # Cadence-gated: runs on gen 0 and every `validation.period` gens.
            if self.cfg.validation.enable and self._val_env is not None:
                val_period = max(1, int(self.cfg.validation.period))
                if gen % val_period == 0:
                    self._run_validation_pass(
                        gen, solutions, fitnesses, verbose=verbose
                    )

            last_solutions = solutions
            last_fitnesses = fitnesses
            gen += 1

            # Debug: check stopping criteria
            if es.stop():
                print(f"[HebbianCMAES] CMA-ES stop() triggered after gen {gen-1}:")
                print(f"  Stop dict: {es.stop()}")

        # ------------------------------------------------------------------
        #  Finalise
        # ------------------------------------------------------------------

        # Cleanup environments if they were pre-built. Release the validation
        # env first (shuts down its workers if parallel); the shared Genesis
        # runtime is then destroyed by _cleanup_env().
        self._cleanup_validation_env()
        self._cleanup_env()

        fitness_entry = self._best_trackers["fitness"]
        if fitness_entry is not None:
            best_genome  = fitness_entry["genome"]
            best_fitness = fitness_entry["fitness"]
        elif last_fitnesses is not None:
            best_idx     = int(np.argmax(last_fitnesses))
            best_genome  = last_solutions[best_idx]
            best_fitness = float(last_fitnesses[best_idx])
        else:
            best_genome  = np.full(self.n_genes, 0.5)
            best_fitness = 0.0

        self._finalize(
            es,
            last_solutions if last_solutions is not None else [],
            last_fitnesses if last_fitnesses is not None else np.array([])
        )

        return best_genome, best_fitness
