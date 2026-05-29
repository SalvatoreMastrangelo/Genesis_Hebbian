"""
OuterLoop — NSGA-II URDF co-evolution wrapped around inner CMA-ES.
===================================================================

Top-level orchestration for the WP2 outer loop:

    Gen 0:
      1. Sample P URDF genomes (random + optional standard drone seed).
      2. Materialise URDFs, write catalog.txt.
      3. Run inner CMA-ES (fresh start, x0 random in [0,1]^n_rules).
      4. Evaluate per-URDF NSGA-II objectives with best inner-loop rules.
      5. Assign objectives to individuals; NSGA-II sort for ranking.
      6. Checkpoint.

    Gen g ≥ 1:
      1. Use NSGA-II selection + SBX + polynomial mutation on the previous
         population (with their objectives) to produce offspring of size P.
      2. Evaluate offspring via the same per-URDF eval pass (requires the
         *previous* best rules to evaluate the new URDFs on). Then combine
         parents + offspring and apply NSGA-II environmental selection to
         get the outer population of size P for this generation.
      3. Materialise URDFs, write catalog.txt.
      4. Re-evaluate carried-over rules on the new URDFs (logged, not
         used by CMA-ES directly).
      5. Run inner CMA-ES with ``x0_override`` = best rules from gen g-1.
      6. Re-run per-URDF NSGA-II objective evaluation with the new best
         rules so the ranking reflects the rules actually evolved this
         generation.
      7. Checkpoint.

Carry-forward semantics: at the end of each outer generation, we carry the
single best-fitness Hebbian-rule genome from the inner CMA-ES. The CMA-ES
distribution (covariance, step size) is *not* carried — each inner loop
starts its covariance fresh with ``sigma0``.

Logging: CSV per outer generation + full YAML config, population genome
arrays, per-URDF objectives, and the carried-rules checkpoint. Inner-loop
checkpoints land under ``outer_generations/gen_XXX/inner/``.
"""

from __future__ import annotations

import csv
import datetime
import pickle
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from WP2.config import HebbianEvolutionConfig
from WP2.evolve_cma import HebbianCMAES
from WP2.utils import seed_everything, save_rng_state

from .config import OuterLoopConfig
from .evaluation import (
    evaluate_urdfs_per_individual,
    evaluate_carried_rules_on_new_population,
)
from .nsga2 import (
    arrays_to_individuals,
    environmental_select,
    individuals_to_array,
    make_offspring,
    make_toolbox,
)
from .urdf_population import (
    materialize_urdfs,
    sample_initial_population,
    write_catalog_txt,
)


# ============================================================================
#  Helpers
# ============================================================================

def _build_inner_cfg(
    template: HebbianEvolutionConfig,
    outer_cfg: OuterLoopConfig,
    gen_dir: Path,
    catalog_path: Path,
    seed: int,
) -> HebbianEvolutionConfig:
    """Derive a per-outer-generation ``HebbianEvolutionConfig``.

    Starts from the loaded template and overrides:
      * ``exp_name`` → outer gen tag
      * ``base_dir`` → inner-run root under this outer gen
      * ``seed`` → ``outer_seed + g`` for reproducible variation across gens
      * ``catalog.path`` → our materialised catalog.txt
      * ``evolution.num_generations`` → outer config's ``inner_generations``
      * ``evaluation.num_eval_envs`` → outer config's ``num_eval_envs``
    """
    cfg = HebbianEvolutionConfig._from_dict({})  # fresh defaults
    # Copy all fields from the template
    for name in vars(template):
        setattr(cfg, name, getattr(template, name))
    # Deep-copy the sub-dataclasses so edits here don't leak back to template.
    import copy as _copy
    cfg.hebbian = _copy.deepcopy(template.hebbian)
    cfg.evolution = _copy.deepcopy(template.evolution)
    cfg.evaluation = _copy.deepcopy(template.evaluation)
    cfg.cmaes = _copy.deepcopy(template.cmaes)
    cfg.catalog = _copy.deepcopy(template.catalog)

    cfg.exp_name = f"inner_cma"
    cfg.base_dir = str(gen_dir / "inner")
    cfg.seed = seed
    cfg.catalog.path = str(catalog_path)
    cfg.evolution.num_generations = outer_cfg.inner_generations
    cfg.evaluation.num_eval_envs = outer_cfg.num_eval_envs
    return cfg


def _save_outer_checkpoint(
    gen_dir: Path,
    population_genomes: np.ndarray,
    objectives: np.ndarray,
    best_rules: np.ndarray,
    best_fitness: float,
    extras: dict,
) -> None:
    """Write the outer-gen checkpoint: genomes, objectives, best rules, extras."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    np.save(gen_dir / "population_genomes.npy", population_genomes)
    np.save(gen_dir / "objectives.npy", objectives)
    np.save(gen_dir / "best_rules.npy", best_rules)
    with open(gen_dir / "summary.pkl", "wb") as f:
        pickle.dump(
            {
                "best_fitness": float(best_fitness),
                "population_shape": tuple(population_genomes.shape),
                "objectives_shape": tuple(objectives.shape),
                **extras,
            },
            f,
        )


def _init_outer_csv(path: Path, objective_names: Sequence[str]) -> None:
    header = ["outer_gen", "individual"] + list(objective_names)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def _append_outer_csv(
    path: Path,
    gen: int,
    objectives: np.ndarray,
) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for i, row in enumerate(objectives):
            w.writerow([gen, i] + [f"{v:.6f}" for v in row])


def _init_carryover_csv(path: Path) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "outer_gen", "fitness", "progress_m",
            "cost_of_transport", "velocity", "crash_rate",
        ])


def _append_carryover_csv(path: Path, gen: int, stats: dict) -> None:
    if not stats:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            gen,
            f"{stats.get('fitness', 0.0):.6f}",
            f"{stats.get('progress_m', 0.0):.6f}",
            f"{stats.get('cost_of_transport', 0.0):.6f}",
            f"{stats.get('velocity', 0.0):.6f}",
            f"{stats.get('crash_rate', 0.0):.6f}",
        ])


# ============================================================================
#  Main class
# ============================================================================

class OuterLoop:
    """NSGA-II outer loop + inner CMA-ES orchestration."""

    def __init__(self, cfg: OuterLoopConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        # Resolve the inner-loop template config early so we fail fast on
        # bad paths.
        template_path = Path(cfg.inner_cfg_template)
        if not template_path.is_file():
            raise FileNotFoundError(
                f"inner_cfg_template not found: {template_path}"
            )
        self.inner_template = HebbianEvolutionConfig.from_yaml(template_path)

        # WP1 controller overrides: outer config wins over whatever the inner
        # template YAML said (so a single outer YAML fully pins the run).
        if cfg.checkpoint_path:
            self.inner_template.checkpoint_path = cfg.checkpoint_path
        if cfg.checkpoint_config_path:
            self.inner_template.checkpoint_config_path = cfg.checkpoint_config_path

        # Inner-loop worker override (None = inherit from template).
        if cfg.inner_num_eval_workers is not None:
            if cfg.inner_num_eval_workers < 1:
                raise ValueError(
                    f"inner_num_eval_workers must be ≥ 1 "
                    f"(got {cfg.inner_num_eval_workers})"
                )
            self.inner_template.evaluation.num_eval_workers = (
                cfg.inner_num_eval_workers
            )

        # Inner-loop sigma re-inflation override (None = inherit from template).
        # The outer config wins, so a single outer YAML pins how aggressively the
        # inner CMA-ES re-explores when the morphology changes.
        if cfg.inner_sigma_reinflate is not None:
            if cfg.inner_sigma_reinflate < 0.0:
                raise ValueError(
                    f"inner_sigma_reinflate must be ≥ 0 "
                    f"(got {cfg.inner_sigma_reinflate})"
                )
            self.inner_template.cmaes.sigma_reinflate = cfg.inner_sigma_reinflate

        # Fail fast on missing WP1 checkpoint/config (common user error).
        ckpt = Path(self.inner_template.checkpoint_path or "")
        ckpt_cfg = Path(self.inner_template.checkpoint_config_path or "")
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"WP1 checkpoint_path not set or missing: {ckpt!r}\n"
                f"  Set via --cfg.checkpoint_path <path> or in the inner template."
            )
        if not ckpt_cfg.is_file():
            raise FileNotFoundError(
                f"WP1 checkpoint_config_path not set or missing: {ckpt_cfg!r}\n"
                f"  Set via --cfg.checkpoint_config_path <path> or in the inner template."
            )

        # Run directory: {base_dir}/{YYYY-MM-DD_HH-MM-SS_exp_name}
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(cfg.base_dir) / f"{stamp}_{cfg.exp_name}"
        self.outer_gen_dir = self.run_dir / "outer_generations"
        self.results_dir = self.run_dir / "results"
        self.repro_dir = self.run_dir / "reproducibility"
        for d in (self.outer_gen_dir, self.results_dir, self.repro_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Log files (per-outer-gen)
        self.objectives_csv = self.results_dir / "outer_objectives.csv"
        self.carryover_csv = self.results_dir / "carried_rules_eval.csv"
        _init_outer_csv(
            self.objectives_csv,
            [o.name for o in cfg.objectives],
        )
        _init_carryover_csv(self.carryover_csv)

        # Save config snapshot for reproducibility
        cfg.to_yaml(self.repro_dir / "outer_loop_config.yaml")
        self.inner_template.to_yaml(self.repro_dir / "inner_template.yaml")

        # DEAP toolbox (created once; creator is cached by objective weights).
        self.toolbox = make_toolbox(cfg, genome_dim=15)

    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg

        print(f"[OuterLoop] Run dir: {self.run_dir}")
        print(f"[OuterLoop] Objectives: "
              f"{[(o.name, o.direction) for o in cfg.objectives]}")
        print(f"[OuterLoop] outer_gens={cfg.outer_generations}, "
              f"inner_gens={cfg.inner_generations}, "
              f"pop={cfg.population_size}, envs={cfg.num_eval_envs}")

        seed_everything(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # ------------------------------------------------------------------
        #  Gen 0 initial population
        # ------------------------------------------------------------------
        population_genomes = sample_initial_population(
            pop_size=cfg.population_size,
            seed=cfg.seed,
            seed_standard_drone=cfg.seed_standard_drone,
        )
        best_rules: Optional[np.ndarray] = None  # filled after inner loop
        current_individuals = None  # DEAP list, populated after gen 0 eval

        run_start = time.perf_counter()

        for g in range(cfg.outer_generations):
            gen_start = time.perf_counter()
            gen_dir = self.outer_gen_dir / f"gen_{g:03d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            urdf_dir = gen_dir / "urdfs"

            print(f"\n{'='*78}\n[OuterLoop] === OUTER GEN {g} ===\n{'='*78}")

            # --------------------------------------------------------------
            #  Generate offspring + select (for g ≥ 1)
            # --------------------------------------------------------------
            if g >= 1:
                assert current_individuals is not None, \
                    "Expected parent individuals after gen 0"
                offspring = make_offspring(
                    self.toolbox,
                    current_individuals,
                    crossover_prob=cfg.nsga2.crossover_prob,
                )
                # Offspring need objectives before environmental selection,
                # which requires URDF materialisation + NSGA-II eval. We
                # therefore evaluate offspring first, then combine.
                offspring_genomes = individuals_to_array(offspring)
                offspring_dir = gen_dir / "offspring"
                offspring_dir.mkdir(parents=True, exist_ok=True)
                print(f"[OuterLoop] Materialising {len(offspring)} offspring URDFs")
                offspring_paths = materialize_urdfs(offspring_genomes, offspring_dir)
                print(f"[OuterLoop] Evaluating offspring per-URDF "
                      f"(using prev-gen best rules)")
                assert best_rules is not None, \
                    "best_rules must be set after gen 0 for offspring eval"
                offspring_objs = evaluate_urdfs_per_individual(
                    urdf_paths=offspring_paths,
                    best_rules=best_rules,
                    outer_cfg=cfg,
                    inner_cfg_template=self.inner_template,
                )
                offspring_inds = arrays_to_individuals(
                    offspring_genomes, offspring_objs,
                )
                # (P+Q)→P environmental selection
                current_individuals = environmental_select(
                    self.toolbox,
                    current_individuals,
                    offspring_inds,
                    k=cfg.population_size,
                )
                population_genomes = individuals_to_array(current_individuals)

            # --------------------------------------------------------------
            #  Materialise URDFs for this outer generation
            # --------------------------------------------------------------
            print(f"[OuterLoop] Materialising {cfg.population_size} URDFs "
                  f"into {urdf_dir}")
            urdf_paths = materialize_urdfs(population_genomes, urdf_dir)
            catalog_path = write_catalog_txt(urdf_paths, urdf_dir)

            # --------------------------------------------------------------
            #  Re-evaluate carried-over rules on new URDF pop (g ≥ 1)
            # --------------------------------------------------------------
            if g >= 1 and best_rules is not None:
                print(f"[OuterLoop] Re-evaluating carried-over rules on new "
                      f"URDF population (logged reference)")
                stats = evaluate_carried_rules_on_new_population(
                    urdf_paths=urdf_paths,
                    carried_rules=best_rules,
                    inner_cfg_template=self.inner_template,
                    num_eval_envs=cfg.num_eval_envs,
                )
                _append_carryover_csv(self.carryover_csv, g, stats)
                if stats:
                    print(
                        f"[OuterLoop]   fitness={stats.get('fitness', 0):.3f}  "
                        f"progress={stats.get('progress_m', 0):.3f}m  "
                        f"cot={stats.get('cost_of_transport', 0):.3f}"
                    )

            # --------------------------------------------------------------
            #  Inner CMA-ES loop
            # --------------------------------------------------------------
            inner_cfg = _build_inner_cfg(
                template=self.inner_template,
                outer_cfg=cfg,
                gen_dir=gen_dir,
                catalog_path=catalog_path,
                seed=cfg.seed + g,
            )
            print(f"[OuterLoop] Starting inner CMA-ES (gens="
                  f"{inner_cfg.evolution.num_generations})")
            inner_runner = HebbianCMAES(inner_cfg)
            best_rules, best_fitness = inner_runner.run(
                x0_override=best_rules if g >= 1 else None,
            )
            print(f"[OuterLoop] Inner loop done: best_fitness={best_fitness:.4f}")

            # --------------------------------------------------------------
            #  NSGA-II evaluation with newly-found best rules
            # --------------------------------------------------------------
            print(f"[OuterLoop] Evaluating per-URDF objectives with new best rules")
            objectives = evaluate_urdfs_per_individual(
                urdf_paths=urdf_paths,
                best_rules=best_rules,
                outer_cfg=cfg,
                inner_cfg_template=inner_cfg,
            )

            # Wrap population as DEAP individuals with the fresh objectives,
            # then run selNSGA2 once so each individual has a per-front
            # crowding distance set (needed by selTournamentDCD next gen).
            current_individuals = arrays_to_individuals(
                population_genomes, objectives,
            )
            current_individuals = self.toolbox.select(
                current_individuals, cfg.population_size,
            )

            # --------------------------------------------------------------
            #  Log + checkpoint
            # --------------------------------------------------------------
            population_genomes = individuals_to_array(current_individuals)
            objectives = np.asarray(
                [ind.fitness.values for ind in current_individuals],
                dtype=np.float64,
            )
            _append_outer_csv(self.objectives_csv, g, objectives)
            _save_outer_checkpoint(
                gen_dir=gen_dir,
                population_genomes=population_genomes,
                objectives=objectives,
                best_rules=best_rules,
                best_fitness=best_fitness,
                extras={
                    "outer_gen": g,
                    "gen_seconds": time.perf_counter() - gen_start,
                    "total_seconds": time.perf_counter() - run_start,
                    "objective_names": [o.name for o in cfg.objectives],
                },
            )
            save_rng_state(gen_dir / "rng_state.pkl")

            print(f"[OuterLoop] Gen {g} wall time: "
                  f"{time.perf_counter() - gen_start:.1f}s")

        print(f"\n[OuterLoop] Run complete in "
              f"{time.perf_counter() - run_start:.1f}s")
