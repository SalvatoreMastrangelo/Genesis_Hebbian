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

def _format_time(seconds: float) -> Tuple[int, int, int]:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return h, m, s


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
            "fitness", "velocity", "progress", "crash_rate",
        ])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "pop_size",
            "best_fitness", "mean_fitness", "worst_fitness", "std_fitness",
            "sigma",
            "mean_velocity", "mean_progress", "mean_crash_rate",
        ])


def _init_baseline_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "fitness", "velocity", "progress", "crash_rate",
        ])


def _append_baseline_csv(
    path: Path,
    gen: int,
    baseline: Dict[str, float],
) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            gen,
            f"{baseline['fitness']:.6g}",
            f"{baseline['velocity']:.6g}",
            f"{baseline['progress']:.6g}",
            f"{baseline['crash_rate']:.6g}",
        ])


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
            ])


def _append_summary_csv(
    path: Path,
    gen: int,
    fitnesses: np.ndarray,
    metrics: Dict[str, np.ndarray],
    sigma: float,
) -> None:
    P = len(fitnesses)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            gen, P,
            f"{fitnesses.max():.6g}",
            f"{fitnesses.mean():.6g}",
            f"{fitnesses.min():.6g}",
            f"{fitnesses.std():.6g}",
            f"{sigma:.6g}",
            f"{metrics['velocities'].mean():.6g}",
            f"{metrics['progresses'].mean():.6g}",
            f"{metrics['crash_flags'].mean():.6g}",
        ])


def _print_generation_table(
    gen: int,
    fitnesses: np.ndarray,
    metrics: Dict[str, np.ndarray],
    sigma: float,
    total_elapsed: Optional[float] = None,
    iter_elapsed: Optional[float] = None,
    eta_seconds: Optional[float] = None,
    baseline: Optional[Dict[str, float]] = None,
) -> None:
    timing_parts = [f"CMA-ES Generation {gen}", f"Population={len(fitnesses)}"]

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

    header_str = " | ".join(timing_parts)
    print(f"\n{'═' * 90}")
    print(f"  {header_str}")
    print(f"{'═' * 90}")

    # Metric rows: (display name, array, baseline_key)
    rows = [
        ("Fitness (reward)", fitnesses, "fitness"),
        ("Velocity [m/s]",   metrics["velocities"], "velocity"),
        ("Progress [m]",     metrics["progresses"],  "progress"),
        ("Crash Rate",       metrics["crash_flags"],  "crash_rate"),
    ]

    lower_is_better = {"Crash Rate"}

    if baseline is not None:
        headers = [
            "Metric",
            "Best (Hebb)", "Mean (Hebb)", "Worst (Hebb)",
            "Std", "Baseline",
        ]
        table_rows = []
        for name, arr, key in rows:
            if name in lower_is_better:
                best, worst = arr.min(), arr.max()
            else:
                best, worst = arr.max(), arr.min()
            bv = baseline.get(key, float("nan"))
            table_rows.append([
                name,
                f"{best:.4g}",
                f"{arr.mean():.4g}",
                f"{worst:.4g}",
                f"{arr.std():.4g}",
                f"{bv:.4g}",
            ])
    else:
        headers = ["Metric", "Best", "Mean", "Worst", "Std"]
        table_rows = []
        for name, arr, key in rows:
            if name in lower_is_better:
                best, worst = arr.min(), arr.max()
            else:
                best, worst = arr.max(), arr.min()
            table_rows.append([
                name,
                f"{best:.4g}",
                f"{arr.mean():.4g}",
                f"{worst:.4g}",
                f"{arr.std():.4g}",
            ])

    print("\n  Metrics Summary (Hebb | Base):" if baseline else "\n  Metrics Summary:")
    print(tabulate(
        table_rows,
        headers=headers,
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

        # Parse catalog (may be empty → single default URDF)
        self._catalog: Optional[List[Tuple[str, str]]] = None
        if cfg.catalog.path:
            self._catalog = _load_catalog(cfg.catalog.path)
            print(f"[HebbianCMAES] Catalog: {len(self._catalog)} URDFs "
                  f"from {cfg.catalog.path}")
        else:
            print("[HebbianCMAES] No catalog — using default/fixed morphology URDF")

        # For rules-only evolution: pre-build environment once (will be reused)
        self._env = None
        self._env_urdf_path = None

        # Run directory
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{stamp}_{cfg.exp_name}"
        self.run_dir = Path(cfg.base_dir) / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.repro_dir = self.run_dir / "reproducibility"
        self.gen_dir = self.run_dir / "generations"
        self.results_dir = self.run_dir / "results"
        for d in (self.repro_dir, self.gen_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        # CSV paths
        self.pop_csv_path = self.results_dir / "cma_population.csv"
        self.summary_csv_path = self.results_dir / "cma_summary.csv"
        self.baseline_csv_path = self.results_dir / "baseline_summary.csv"
        _init_csvs(self.pop_csv_path, self.summary_csv_path)
        _init_baseline_csv(self.baseline_csv_path)

        # Timing
        self._run_start_time: Optional[float] = None
        self._gen_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    #  Environment setup (rules-only only)
    # ------------------------------------------------------------------

    def _build_env_once(self) -> None:
        """Build the environment once and keep it alive across generations.

        When a catalog is used, the environment is built per-URDF inside
        ``evaluate_population_cma_batched``.  For single-URDF evaluation,
        we build it once here and reuse it every generation.
        """
        if self._catalog is not None:
            # Catalog case: will be handled per-generation by evaluate function
            print("[HebbianCMAES] Using catalog — environment will be built per-generation")
            return

        # Rules-only case: pre-build environment
        import genesis as gs
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
        """Destroy the pre-built environment at run end."""
        if self._env is not None:
            import genesis as gs
            print("[HebbianCMAES] Destroying environment...")
            try:
                gs.destroy()
                self._env = None
                self._env_urdf_path = None
            except Exception as exc:
                print(f"[HebbianCMAES] Warning: failed to destroy environment: {exc}")

    # ------------------------------------------------------------------
    #  Baseline evaluation (frozen WP1 actor, zero Hebbian rules)
    # ------------------------------------------------------------------

    def _evaluate_baseline(self, verbose: bool = False) -> Optional[Dict[str, float]]:
        """Evaluate the frozen WP1 actor with zero Hebbian rules (A=B=C=D=0, decay=0).

        ABCD=0 (genome=0.5 for symmetric [-1,1] ranges) means no plasticity
        update.  Decay is also zeroed so weights stay exactly at the WP1
        checkpoint values throughout the episode.

        Two paths:
        - evolve_decay=False: decay comes from cfg.hebbian.decay → pass a
          config copy with decay=0.
        - evolve_decay=True:  decay genes sit at positions [4n:5n] in the
          genome; setting them to 0.0 maps to decay_range[0]=0.0.
        """
        import copy

        n_weights = self.cfg.hebbian.num_actions * self.cfg.hebbian.hidden_dim

        # Build baseline genome: ABCD=0.5 (→ 0 for symmetric ranges), rest=0.5
        baseline_genome = np.full(self.n_genes, 0.5)

        # Zero out evolved decay genes so they decode to decay_range[0]=0
        if self.cfg.hebbian.evolve_decay:
            decay_start = 4 * n_weights
            baseline_genome[decay_start: decay_start + n_weights] = 0.0

        # Config copy with fixed decay forced to 0 (covers non-evolved case)
        baseline_cfg = copy.deepcopy(self.cfg)
        baseline_cfg.hebbian.decay = 0.0

        existing_env = (
            (self._env, self._env_urdf_path)
            if self._env is not None
            else None
        )
        try:
            if verbose:
                print("[HebbianCMAES] Evaluating baseline (zero-Hebbian, zero-decay)...")
            fitnesses, metrics = evaluate_population_cma_batched(
                [baseline_genome],
                baseline_cfg,
                self._model_and_layer,
                self._wp1_cfg,
                catalog=self._catalog,
                existing_env=existing_env,
                verbose=verbose,
            )
            result = {
                "fitness":    float(fitnesses[0]),
                "velocity":   float(metrics["velocities"][0]),
                "progress":   float(metrics["progresses"][0]),
                "crash_rate": float(metrics["crash_flags"][0]),
            }
            if verbose:
                print(
                    f"[HebbianCMAES] Baseline: fitness={result['fitness']:.4g}  "
                    f"vel={result['velocity']:.4g}  prog={result['progress']:.4g}  "
                    f"crash={result['crash_rate']:.4g}"
                )
            return result
        except Exception as exc:
            print(f"[HebbianCMAES] Baseline evaluation failed: {exc}")
            return None

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
    ) -> None:
        import time

        sigma = float(es.sigma)

        # CSV logging
        _append_population_csv(self.pop_csv_path, gen, fitnesses, metrics)
        _append_summary_csv(self.summary_csv_path, gen, fitnesses, metrics, sigma)
        if baseline is not None:
            _append_baseline_csv(self.baseline_csv_path, gen, baseline)

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
        )

    # ------------------------------------------------------------------
    #  Finalisation
    # ------------------------------------------------------------------

    def _finalize(
        self,
        es,
        solutions: List[np.ndarray],
        fitnesses: np.ndarray,
    ) -> None:
        import yaml

        best_idx = int(np.argmax(fitnesses))
        best_genome = solutions[best_idx]
        best_fitness = float(fitnesses[best_idx])

        print(f"\n[HebbianCMAES] Done. Best fitness: {best_fitness:.6g} "
              f"(individual {best_idx})")
        print(f"[HebbianCMAES] Results saved to: {self.run_dir}")

        # Save best individual
        best_dir = self.run_dir / "best_individual"
        best_dir.mkdir(parents=True, exist_ok=True)

        np.save(best_dir / "genome.npy", best_genome)

        # Decode and save Hebbian rules
        rules = decode_hebbian_genes(
            list(np.clip(best_genome, 0.0, 1.0)),
            self.cfg.hebbian,
            out_features=self.cfg.hebbian.num_actions,
            in_features=self.cfg.hebbian.hidden_dim,
        )
        rules_dict = {k: v.cpu().numpy().tolist() for k, v in rules.items()}
        with open(best_dir / "hebbian_rules.yaml", "w") as f:
            yaml.dump(rules_dict, f, sort_keys=False)

        with open(best_dir / "fitness.yaml", "w") as f:
            yaml.dump({"fitness": best_fitness, "individual_idx": best_idx}, f)

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
    ) -> Tuple[np.ndarray, float]:
        """Run the full CMA-ES optimisation.

        Parameters
        ----------
        resume_from_gen : int, optional
            If given, restores the CMA-ES state and population from the
            checkpoint saved at that generation and continues from there.

        Returns
        -------
        best_genome : np.ndarray, shape (n_genes,)
            The genome with the highest fitness found during the run.
        best_fitness : float
            Corresponding scalar fitness value.
        """
        import time

        print(f"\n[HebbianCMAES] Run directory: {self.run_dir}")
        print(f"[HebbianCMAES] Genome dim: {self.n_genes} (Hebbian rules only)")
        print(f"[HebbianCMAES] Generations: {self.cfg.evolution.num_generations}")
        print(f"[HebbianCMAES] Catalog URDFs: "
              f"{len(self._catalog) if self._catalog else 1} "
              f"({'catalog' if self._catalog else 'default'})")
        n_pop_cfg = self.cfg.cmaes.population_size
        expected_pop = (
            n_pop_cfg if n_pop_cfg > 0
            else int(4 + 3 * np.log(self.n_genes))
        )
        print(f"[HebbianCMAES] Population size: "
              f"{'auto ≈ ' + str(expected_pop) if n_pop_cfg == 0 else n_pop_cfg}")

        self._save_reproducibility()
        self._build_env_once()  # Pre-build environment for rules-only evolution
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

        if self.cfg.cmaes.population_size > 0:
            opts["popsize"] = self.cfg.cmaes.population_size

        if resume_from_gen is not None:
            # Restore CMA-ES from checkpoint
            es = self._restore_cmaes(resume_from_gen)
            start_gen = resume_from_gen + 1
            # Recover last solutions + fitnesses for display purposes
            last_solutions, last_fitnesses = self._load_generation(resume_from_gen)
        else:
            # Fresh start: initial mean at 0.5 (centre of [0,1]^n)
            if self.cfg.hebbian.initialize_rules_to_zero:
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
        #  Evolution loop
        # ------------------------------------------------------------------

        gen = start_gen
        while not es.stop() and gen <= self.cfg.evolution.num_generations:
            self._gen_start_time = time.perf_counter()
            verbose = (gen == 0)

            # Sample new candidate solutions
            solutions = es.ask()   # list of np.ndarray, each shape (n_genes,)

            # Evaluate (returns scalar fitness per individual)
            try:
                # Pass pre-built environment if available (rules-only)
                existing_env = (
                    (self._env, self._env_urdf_path)
                    if self._env is not None
                    else None
                )
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
                }

            # CMA-ES minimises — negate fitness to maximise reward
            es.tell(solutions, (-fitnesses).tolist())

            # Baseline: evaluate frozen WP1 actor with zero Hebbian rules
            baseline = self._evaluate_baseline(verbose=verbose) if self.cfg.evaluation.run_baseline else None

            # Bookkeeping
            self._after_generation(gen, solutions, fitnesses, metrics, es, baseline=baseline)

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

        # Cleanup environment if it was pre-built
        self._cleanup_env()

        if last_solutions is None or last_fitnesses is None:
            # Edge case: stopped before completing gen 0
            best_genome = np.full(self.n_genes, 0.5)
            best_fitness = 0.0
        else:
            best_idx = int(np.argmax(last_fitnesses))
            best_genome = last_solutions[best_idx]
            best_fitness = float(last_fitnesses[best_idx])

        self._finalize(
            es,
            last_solutions if last_solutions is not None else [],
            last_fitnesses if last_fitnesses is not None else np.array([])
        )

        return best_genome, best_fitness
