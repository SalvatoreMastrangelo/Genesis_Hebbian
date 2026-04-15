"""
HebbianCodesignDEAP — NSGA-II loop for Hebbian + morphology co-optimisation.
=============================================================================

Follows the ``CodesignDEAP`` pattern from ``morph_evolution/evolution_nsga.py``
but with key differences:

- No training (backprop) — the actor is frozen.
- Fitness = multi-episode forward rollout with Hebbian plasticity.
- Genome = per-weight Hebbian rules (ABCD+lambda) + optional morphology.
- Crossover is toggleable (can be disabled for high-dimensional genomes).
"""

from __future__ import annotations

import csv
import datetime
import os
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from deap import base, creator, tools
from tabulate import tabulate

from WP2.config import HebbianEvolutionConfig
from WP2.objectives import default_fitness
from WP2.utils import (
    save_git_info,
    save_environment_info,
    save_pareto_front,
    save_rng_state,
    load_rng_state,
    split_genome,
)


# ============================================================================
#  Parallelism
# ============================================================================

def _want_parallel() -> bool:
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1


USE_PARALLEL = _want_parallel()

if USE_PARALLEL:
    import ray


# ============================================================================
#  CSV logging helpers
# ============================================================================

def _init_csvs(pop_path: Path, pareto_path: Path, gen_summary_path: Path, obj_names: List[str]) -> None:
    """Create CSV files with headers."""
    for p in (pop_path, pareto_path, gen_summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Population history
    with open(pop_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["generation", "uid", "genome", "front_rank"] + \
                 [f"fitness_{name}" for name in obj_names]
        writer.writerow(header)

    # Pareto history
    with open(pareto_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["generation", "uid", "genome"] + \
                 [f"fitness_{name}" for name in obj_names]
        writer.writerow(header)

    # Generation summary
    with open(gen_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["generation", "pareto_size", "pop_size"] + \
                 [f"best_{name}" for name in obj_names] + \
                 [f"mean_{name}" for name in obj_names] + \
                 ["crash_rate", "success_rate", "progress_mean", "progress_std", "progress_max"]
        writer.writerow(header)


def _append_population_csv(path: Path, gen: int, pop: list, fronts: list, obj_names: List[str]) -> None:
    """Append all individuals for this generation."""
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Build rank map
    rank_map = {}
    for rank_idx, front in enumerate(fronts):
        for ind in front:
            rank_map[id(ind)] = rank_idx

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for ind in pop:
            uid = getattr(ind, "uid", -1)
            rank = rank_map.get(id(ind), -1)
            row = [gen, uid, str(list(ind)), rank]
            if ind.fitness.valid:
                row.extend([f"{v:.6g}" for v in ind.fitness.values])
            else:
                row.extend(["NaN"] * len(obj_names))
            writer.writerow(row)


def _append_pareto_csv(path: Path, gen: int, front: list, obj_names: List[str]) -> None:
    """Append Pareto front individuals for this generation."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for ind in front:
            uid = getattr(ind, "uid", -1)
            row = [gen, uid, str(list(ind))]
            if ind.fitness.valid:
                row.extend([f"{v:.6g}" for v in ind.fitness.values])
            else:
                row.extend(["NaN"] * len(obj_names))
            writer.writerow(row)


def _append_gen_summary_csv(
    path: Path,
    gen: int,
    pop: list,
    pareto_size: int,
    obj_names: List[str],
    pop_stats: Dict[str, float] = None,
) -> None:
    """Append per-generation summary statistics."""
    valid = [ind for ind in pop if ind.fitness.valid]
    if not valid:
        return

    fitnesses = np.array([ind.fitness.values for ind in valid])  # (N, n_obj)
    bests = fitnesses.max(axis=0)
    means = fitnesses.mean(axis=0)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [gen, pareto_size, len(pop)]
        row.extend([f"{v:.6g}" for v in bests])
        row.extend([f"{v:.6g}" for v in means])

        # Append termination statistics
        if pop_stats:
            row.append(f"{pop_stats.get('crash_rate', ''):.6g}" if 'crash_rate' in pop_stats else "")
            row.append(f"{pop_stats.get('success_rate', ''):.6g}" if 'success_rate' in pop_stats else "")
            row.append(f"{pop_stats.get('progress_mean', ''):.6g}" if 'progress_mean' in pop_stats else "")
            row.append(f"{pop_stats.get('progress_std', ''):.6g}" if 'progress_std' in pop_stats else "")
            row.append(f"{pop_stats.get('progress_max', ''):.6g}" if 'progress_max' in pop_stats else "")
        else:
            row.extend([""] * 5)

        writer.writerow(row)


def _print_fitness_table(
    gen: int,
    pop: list,
    pareto_size: int,
    obj_names: List[str],
    gen_elapsed_time: float = None,
    iter_elapsed_time: float = None,
    total_elapsed_time: float = None,
    eta_seconds: float = None,
    pop_stats: Dict[str, float] = None,
) -> None:
    """Print formatted tables of fitness statistics and timing for the generation."""
    valid = [ind for ind in pop if ind.fitness.valid]
    if not valid:
        return

    fitnesses = np.array([ind.fitness.values for ind in valid])  # (N, n_obj)

    # Build timing info
    timing_parts = [f"Generation {gen}", f"Population={len(pop)}", f"Pareto Front={pareto_size}"]

    if total_elapsed_time is not None:
        elapsed_h, elapsed_m, elapsed_s = _format_time(total_elapsed_time)
        timing_parts.append(f"Elapsed: {elapsed_h}:{elapsed_m:02d}:{elapsed_s:02d}")

    if gen_elapsed_time is not None:
        gen_m, gen_s = divmod(int(gen_elapsed_time), 60)
        timing_parts.append(f"Gen time: {gen_m}:{gen_s:02d}")

    if eta_seconds is not None:
        eta_h, eta_m, eta_s = _format_time(eta_seconds)
        timing_parts.append(f"ETA: {eta_h}:{eta_m:02d}:{eta_s:02d}")

    header_str = " | ".join(timing_parts)
    print(f"\n{'═' * 80}")
    print(f"  {header_str}")
    print(f"{'═' * 80}")

    # Build fitness statistics table
    fitness_rows = []
    for i, name in enumerate(obj_names):
        obj_fitnesses = fitnesses[:, i]
        fitness_rows.append([
            name,
            f"{obj_fitnesses.max():.6g}",
            f"{obj_fitnesses.min():.6g}",
            f"{obj_fitnesses.mean():.6g}",
            f"{obj_fitnesses.std():.6g}",
        ])

    print("\n  Fitness Statistics:")
    print(tabulate(
        fitness_rows,
        headers=["Objective", "Best", "Min", "Mean", "Std"],
        tablefmt="grid",
        numalign="center",
        stralign="left",
    ))

    # Print population statistics (crash, spatial termination)
    if pop_stats:
        stats_rows = []

        if 'crash_rate' in pop_stats:
            crash_pct = pop_stats['crash_rate'] * 100
            success_pct = pop_stats['success_rate'] * 100
            stats_rows.append(["Crash Rate", f"{crash_pct:6.1f}%"])
            stats_rows.append(["Success Rate", f"{success_pct:6.1f}%"])

        if 'progress_mean' in pop_stats:
            stats_rows.append(["Position Mean", f"{pop_stats['progress_mean']:.4f}"])
            stats_rows.append(["Position Std", f"{pop_stats['progress_std']:.4f}"])
            stats_rows.append(["Position Max", f"{pop_stats['progress_max']:.4f}"])

        if stats_rows:
            print("\n  Population Statistics:")
            print(tabulate(
                stats_rows,
                headers=["Metric", "Value"],
                tablefmt="simple",
                stralign="left",
            ))

    print()


def _format_time(seconds: float) -> Tuple[int, int, int]:
    """Convert seconds to (hours, minutes, seconds)."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return hours, minutes, secs


def _compute_population_stats(pop: list) -> Dict[str, float]:
    """Compute aggregate statistics from population metrics.

    Returns crash rate, success rate, and spatial termination stats.
    """
    valid = [ind for ind in pop if ind.fitness.valid]
    if not valid:
        return {}

    stats = {}

    # Collect metrics from all individuals
    crash_flags = []
    progresses = []

    for ind in valid:
        if hasattr(ind, 'metrics'):
            metrics = ind.metrics
            if 'crash_flags' in metrics:
                crash_flags.append(metrics['crash_flags'])
            if 'progresses' in metrics:
                progresses.append(metrics['progresses'])

    # Compute crash statistics
    if crash_flags:
        crash_arr = np.array(crash_flags)
        crash_rate = np.mean(crash_arr)
        stats['crash_rate'] = crash_rate
        stats['success_rate'] = 1.0 - crash_rate

    # Compute spatial termination statistics (progress = distance traveled)
    if progresses:
        prog_arr = np.array(progresses)
        stats['progress_mean'] = np.mean(prog_arr)
        stats['progress_std'] = np.std(prog_arr)
        stats['progress_max'] = np.max(prog_arr)

    return stats


# ============================================================================
#  Main NSGA-II class
# ============================================================================

class HebbianCodesignDEAP:
    """NSGA-II loop for Hebbian plasticity + morphology co-optimisation.

    Usage
    -----
    >>> cfg = HebbianEvolutionConfig.from_yaml("configs/full_codesing.yaml")
    >>> ga = HebbianCodesignDEAP(cfg)
    >>> final_pop = ga.run()
    """

    def __init__(self, config: HebbianEvolutionConfig) -> None:
        self.cfg = config

        if self.cfg.evolution.population_size % 4 != 0:
            raise ValueError("population_size must be a multiple of 4 for tournamentDCD.")

        self.n_pop = self.cfg.evolution.population_size
        self.n_gen = self.cfg.evolution.num_generations
        self.cx_pb = self.cfg.evolution.crossover_probability
        self.mut_pb = self.cfg.evolution.mutation_probability
        self.enable_crossover = self.cfg.evolution.enable_crossover

        n_genes = self.cfg.total_genome_dim()
        if n_genes == 0:
            raise ValueError("Total genome dimension is 0. Enable Hebbian and/or morphology evolution.")

        self.n_genes = n_genes
        obj_names = self.cfg.active_objective_names()
        self.obj_names = obj_names
        n_obj = len(obj_names)

        # Pre-load frozen actor for reuse across generations
        from WP2.frozen_actor import load_frozen_actor
        from WP1.config import RunConfig
        self._model, self._last_layer, _, _ = load_frozen_actor(
            self.cfg.checkpoint_path, self.cfg.checkpoint_config_path, device=self.cfg.device
        )
        self._wp1_cfg = RunConfig.from_yaml(self.cfg.checkpoint_config_path)

        # --- Run directory ---
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{stamp}_{self.cfg.exp_name}"
        self.run_dir = Path(self.cfg.base_dir) / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories (matching plan structure)
        self.repro_dir = self.run_dir / "reproducibility"
        self.gen_dir = self.run_dir / "generations"
        self.results_dir = self.run_dir / "results"
        self.plots_dir = self.run_dir / "plots"
        for d in (self.repro_dir, self.gen_dir, self.results_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        # CSV paths
        self.pop_history_path = self.results_dir / "population_history.csv"
        self.pareto_history_path = self.results_dir / "pareto_history.csv"
        self.gen_summary_path = self.results_dir / "generation_summary.csv"
        _init_csvs(self.pop_history_path, self.pareto_history_path, self.gen_summary_path, obj_names)

        # --- DEAP setup ---
        # Clear previous DEAP creator classes to avoid conflicts
        for name in ("FitMultiWP2", "ChromWP2"):
            if name in creator.__dict__:
                del creator.__dict__[name]

        weights = self.cfg.fitness_weights()
        creator.create("FitMultiWP2", base.Fitness, weights=weights)
        creator.create("ChromWP2", list, fitness=creator.FitMultiWP2)
        self.IndType = creator.ChromWP2

        self._uid_counter = 0

        # Timing tracking
        self._run_start_time = None
        self._gen_start_times = {}

        # Bounds: all genes in [0, 1]
        self._low = [0.0] * n_genes
        self._up = [1.0] * n_genes

        # Environment management for reuse across generations
        self._current_env = None
        self._current_env_urdf = None
        self._current_env_morph = None
        self._current_env_size = None

        # Toolbox
        self.tb = base.Toolbox()
        self.tb.register("attr_float", random.random)
        self.tb.register("ind", tools.initRepeat, self.IndType, self.tb.attr_float, n=n_genes)
        self.tb.register("pop", tools.initRepeat, list, self.tb.ind)

        # NSGA-II operators
        self.tb.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.evolution.eta_c,
        )
        # Register mutation operator based on config
        if self.cfg.evolution.mutation_operator == "gaussian":
            self.tb.register(
                "mutate",
                tools.mutGaussian,
                mu=0.0,
                sigma=self.cfg.evolution.mutation_sigma,
                indpb=1.0 / n_genes,
            )
        else:  # polynomial (default)
            self.tb.register(
                "mutate",
                tools.mutPolynomialBounded,
                low=self._low,
                up=self._up,
                eta=self.cfg.evolution.eta_m,
                indpb=1.0 / n_genes,
            )
        self.tb.register("select", tools.selNSGA2)

    # ------------------------------------------------------------------
    #  UID management
    # ------------------------------------------------------------------

    def _assign_uid(self, ind) -> int:
        self._uid_counter += 1
        ind.uid = int(self._uid_counter)
        return ind.uid

    def _ensure_uids(self, population: list) -> None:
        for ind in population:
            if not hasattr(ind, "uid") or ind.uid is None:
                self._assign_uid(ind)

    # ------------------------------------------------------------------
    #  Environment management for reuse
    # ------------------------------------------------------------------

    def _get_population_morphology(self, population: list):
        """Extract morphology genome from population (assumes all same if batched)."""
        if not population:
            return None
        from WP2.utils import split_genome
        # Get morphology from first individual
        hebb_part, morph_part = split_genome(list(population[0]), self.cfg)
        return morph_part

    def _destroy_current_env(self) -> None:
        """Destroy the current environment if it exists."""
        if self._current_env is not None:
            import genesis as gs
            try:
                gs.destroy()
                self._current_env = None
                self._current_env_urdf = None
                self._current_env_morph = None
                self._current_env_size = None
                print("[evolve] Destroyed current environment")
            except Exception as exc:
                print(f"[evolve] Warning: Failed to destroy environment: {exc}")

    def _should_rebuild_env(self, population: list, num_envs: int) -> bool:
        """Check if environment needs to be rebuilt."""
        if self._current_env is None:
            return True

        # Check if population morphology changed
        if self.cfg.morphology.evolve:
            morph = self._get_population_morphology(population)
            if morph != self._current_env_morph:
                return True

        # Check if environment size changed
        if self._current_env_size != num_envs:
            return True

        return False

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(self, population: list) -> None:
        """Evaluate all individuals that need evaluation."""
        from WP2.evaluate import evaluate_population_batched

        # Use batched evaluation (all invalid individuals in parallel)
        model_and_layer = (self._model, self._last_layer, self.cfg.hebbian.num_actions, self.cfg.hebbian.hidden_dim)

        # Determine expected environment size
        invalid = [ind for ind in population if not ind.fitness.valid]
        P = len(invalid)
        if P == 0:
            return

        total_envs = self.cfg.evaluation.num_eval_envs
        S = total_envs // P
        actual_envs = S * P

        # Check if we need to rebuild environment
        if self._should_rebuild_env(population, actual_envs):
            self._destroy_current_env()

        # Prepare environment tuple for reuse
        existing_env = None
        if self._current_env is not None:
            existing_env = (self._current_env, self._current_env_urdf)

        # Evaluate with environment reuse
        result = evaluate_population_batched(
            population, self.cfg, model_and_layer, self._wp1_cfg,
            existing_env=existing_env,
            keep_env_alive=True,  # Keep environment alive for next generation
        )

        # Store environment for next generation
        if result is not None:
            env, urdf_path, morph_genome = result
            self._current_env = env
            self._current_env_urdf = urdf_path
            self._current_env_morph = morph_genome
            self._current_env_size = actual_envs
            print(f"[evolve] Stored environment for reuse (morph={morph_genome is not None}, size={actual_envs})")

    # ------------------------------------------------------------------
    #  Variation operators
    # ------------------------------------------------------------------

    def _apply_variation(self, offspring: list) -> None:
        """Apply crossover and/or mutation to offspring."""
        for ch in offspring:
            for attr in ("_persisted",):
                if hasattr(ch, attr):
                    delattr(ch, attr)

        for i in range(0, len(offspring), 2):
            c1, c2 = offspring[i], offspring[i + 1]

            # Crossover (toggleable)
            if self.enable_crossover and random.random() < self.cx_pb:
                self.tb.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

            # Mutation
            if random.random() < self.mut_pb:
                self.tb.mutate(c1)
                del c1.fitness.values
            if random.random() < self.mut_pb:
                self.tb.mutate(c2)
                del c2.fitness.values

            # Snap morphology genes to valid NACA bins (if morphology is evolved)
            if self.cfg.morphology.evolve:
                from morph_evolution.chromosome_drone import Chromosome_Drone
                hebb_dim = self.cfg.hebbian_genome_dim()
                morph_dim = self.cfg.morphology_genome_dim()
                for child in (c1, c2):
                    morph_section = child[hebb_dim:hebb_dim + morph_dim]
                    snapped = Chromosome_Drone.snap_genome_norm(morph_section)
                    child[hebb_dim:hebb_dim + morph_dim] = snapped

    # ------------------------------------------------------------------
    #  Generation bookkeeping
    # ------------------------------------------------------------------

    def _save_generation(self, gen: int, pop: list, front0: list) -> None:
        """Save population, Pareto front, and RNG state for this generation."""
        gen_path = self.gen_dir / f"gen_{gen:03d}"
        gen_path.mkdir(parents=True, exist_ok=True)

        # Population pickle
        with open(gen_path / "population.pkl", "wb") as f:
            pickle.dump(pop, f)

        # Pareto front pickle
        with open(gen_path / "pareto_front.pkl", "wb") as f:
            pickle.dump(front0, f)

        # RNG state
        save_rng_state(gen_path / "rng_state.pkl")

    def _after_generation(self, gen: int, pop: list) -> None:
        """Post-generation: log, save, print summary."""
        import time

        fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
        front0 = fronts[0] if fronts else []

        # Compute population-level statistics (for logging and display)
        pop_stats = _compute_population_stats(pop)

        # CSV logging
        _append_population_csv(self.pop_history_path, gen, pop, fronts, self.obj_names)
        _append_pareto_csv(self.pareto_history_path, gen, front0, self.obj_names)
        _append_gen_summary_csv(self.gen_summary_path, gen, pop, len(front0), self.obj_names, pop_stats)

        # Save generation checkpoint
        self._save_generation(gen, pop, front0)

        # Calculate timing
        gen_now = time.perf_counter()
        gen_elapsed = gen_now - self._gen_start_times.get(gen, gen_now)
        total_elapsed = gen_now - self._run_start_time if self._run_start_time else None

        # Estimate ETA
        eta_secs = None
        if gen > 0 and total_elapsed:
            avg_gen_time = total_elapsed / gen
            remaining_gens = self.n_gen - gen
            eta_secs = avg_gen_time * remaining_gens

        # Print formatted table of fitness statistics
        _print_fitness_table(
            gen, pop, len(front0), self.obj_names,
            gen_elapsed_time=gen_elapsed,
            total_elapsed_time=total_elapsed,
            eta_seconds=eta_secs,
            pop_stats=pop_stats,
        )

    # ------------------------------------------------------------------
    #  Reproducibility setup
    # ------------------------------------------------------------------

    def _save_reproducibility(self) -> None:
        """Save all reproducibility artefacts at run start."""
        import shutil

        # Config snapshot
        self.cfg.to_yaml(self.repro_dir / "config.yaml")

        # WP1 config copy
        if self.cfg.checkpoint_config_path and Path(self.cfg.checkpoint_config_path).is_file():
            shutil.copy2(self.cfg.checkpoint_config_path, self.repro_dir / "wp1_config.yaml")

        # WP1 actor weights copy (only actor, strip critic for space)
        if self.cfg.checkpoint_path and Path(self.cfg.checkpoint_path).is_file():
            shutil.copy2(self.cfg.checkpoint_path, self.repro_dir / "wp1_actor.pt")

        # Git info
        save_git_info(self.repro_dir / "git_info.txt")

        # Environment
        save_environment_info(self.repro_dir / "environment.txt")

    # ------------------------------------------------------------------
    #  Resume support
    # ------------------------------------------------------------------

    def _load_generation(self, gen: int) -> list:
        """Load population and RNG state from a generation checkpoint."""
        gen_path = self.gen_dir / f"gen_{gen:03d}"
        with open(gen_path / "population.pkl", "rb") as f:
            pop = pickle.load(f)
        load_rng_state(gen_path / "rng_state.pkl")

        # Restore UID counter
        max_uid = 0
        for ind in pop:
            uid = getattr(ind, "uid", 0)
            if uid > max_uid:
                max_uid = uid
        self._uid_counter = max_uid

        return pop

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    def run(self, resume_from_gen: Optional[int] = None) -> list:
        """Run the full NSGA-II evolution and return the final population."""
        import time

        print(f"[HebbianCodesignDEAP] Run directory: {self.run_dir}")
        print(f"[HebbianCodesignDEAP] Genome dim: {self.n_genes} "
              f"(Hebbian={self.cfg.hebbian_genome_dim()}, "
              f"Morphology={self.cfg.morphology_genome_dim()})")
        print(f"[HebbianCodesignDEAP] Objectives: {self.obj_names}")
        print(f"[HebbianCodesignDEAP] Population: {self.n_pop}, "
              f"Generations: {self.n_gen}")

        self._save_reproducibility()

        # Initialize timing
        self._run_start_time = time.perf_counter()

        start_gen = 0
        if resume_from_gen is not None:
            print(f"[HebbianCodesignDEAP] Resuming from generation {resume_from_gen}")
            pop = self._load_generation(resume_from_gen)
            start_gen = resume_from_gen + 1
        else:
            # Generation 0: random initial population
            pop = self.tb.pop(self.n_pop)

            # Apply zero initialization if configured
            if self.cfg.hebbian.initialize_rules_to_zero:
                from WP2.utils import create_zero_initialized_genome
                zero_genome = create_zero_initialized_genome(self.cfg)
                for ind in pop:
                    for i in range(len(ind)):
                        ind[i] = zero_genome[i]
                print("[HebbianCodesignDEAP] Initialized population with zero Hebbian rules")

            self._ensure_uids(pop)
            self._gen = 0

            self._gen_start_times[0] = time.perf_counter()
            self._evaluate_population(pop)
            pop = tools.selNSGA2(pop, self.n_pop)
            self._after_generation(0, pop)
            start_gen = 1

        # Generations 1..N
        for g in range(start_gen, self.n_gen + 1):
            self._gen = g
            self._gen_start_times[g] = time.perf_counter()

            print(f"\n{'=' * 60}")
            print(f"  Generation {g}/{self.n_gen}")
            print(f"{'=' * 60}")

            self._ensure_uids(pop)

            # Parent selection
            parents = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]
            for child in offspring:
                self._assign_uid(child)

            # Variation
            self._apply_variation(offspring)

            # Evaluate offspring
            self._evaluate_population(offspring)

            # Survivor selection (mu + lambda)
            pop = tools.selNSGA2(pop + offspring, self.n_pop)

            # Bookkeeping
            self._after_generation(g, pop)

        # Final Pareto front
        fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)
        front0 = fronts[0] if fronts else []
        save_pareto_front(self.run_dir, front0, self.cfg)
        print(f"\n[HebbianCodesignDEAP] Done. Final Pareto front: {len(front0)} solutions.")
        print(f"[HebbianCodesignDEAP] Results saved to: {self.run_dir}")

        # Clean up environment
        self._destroy_current_env()

        return pop
