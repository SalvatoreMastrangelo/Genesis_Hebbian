"""
NSGA2MorphCMAES — persistent CMA-ES with NSGA-II-driven URDF refresh.
=====================================================================

The efficient WP2 outer loop. Instead of nesting a fresh inner CMA-ES run
per outer generation (legacy ``outer_loop.py``), this subclasses the inner
loop's ``HebbianCMAES`` and turns its periodic URDF-refresh hook into an
NSGA-II morphology update:

* ONE CMA-ES over Hebbian rules runs for the whole experiment. Mean and
  covariance carry across morphology changes; ``cmaes.sigma_reinflate``
  re-inflates the step size after each change (the knob is finally live).
* The eval env holds N URDFs (one Genesis scene each, exactly like the
  ``catalog.mutate`` experiments). Every ``catalog.refresh_urdfs_every``
  inner generations the URDF population is *evolved* instead of blindly
  mutated: NSGA-II environmental selection keeps ``n_elites`` survivors,
  binary tournament + SBX + polynomial mutation produce the rest.
* Per-URDF objectives (fitness = WP1 reward sum, cost of transport, …)
  are harvested for free from the ``per_urdf_*`` matrices that
  ``evaluate_population_multi_urdf`` already returns — no extra rollouts,
  no extra scene builds. Cost per outer generation = exactly one env
  rebuild, the same as the existing mutation runs.

Fairness note: URDFs are only ever compared *within* a phase — all N see
the same CMA individuals, the same forests, and the same speed grid, so the
NSGA-II ranking is internally consistent even though rules evolve across
phases. Elites are re-scored every phase, so no stale objective survives.

Outputs (on top of everything ``HebbianCMAES`` already writes):

* ``results/outer_per_urdf_per_gen.csv`` — per inner gen × URDF diagnostics.
* ``results/outer_population.csv``       — per outer gen × URDF phase-mean
  objectives + normalized genome (the NSGA-II selection input; also the
  Pareto archive used by ``pareto_plots``).
* ``outer/gen_XXX/{genomes,objectives}.npy`` — per-phase snapshots.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from deap import tools

from WP2.evolve_cma import HebbianCMAES, _read_norm_genomes_from_catalog

from .config import OuterNSGA2Config
from .nsga2 import arrays_to_individuals, individuals_to_array, make_toolbox
from .urdf_population import materialize_urdfs, write_catalog_txt


# Outer-loop objective / diagnostic names → per-URDF matrix keys emitted by
# WP2.evaluate.evaluate_population_multi_urdf (each of shape (N, P)).
_PER_URDF_ALIASES: Dict[str, str] = {
    "fitness": "per_urdf_reward",
    "reward": "per_urdf_reward",
    "cost_of_transport": "per_urdf_cot",
    "cot": "per_urdf_cot",
    "progress_m": "per_urdf_progress",
    "progress": "per_urdf_progress",
    "velocity": "per_urdf_velocity",
    "crash_rate": "per_urdf_crash",
}

# Canonical diagnostics recorded per URDF each generation (superset of any
# sensible objective pair).
_DIAG_KEYS: Tuple[str, ...] = (
    "fitness", "cost_of_transport", "progress_m", "velocity", "crash_rate",
)


# ============================================================================
#  NSGA-II variation without selTournamentDCD
# ============================================================================
#
# DEAP's selTournamentDCD requires k (and the pop) divisible by 4, which is
# hostile to the small URDF populations this loop runs at (N=2..8). We do the
# same thing — binary tournament on (pareto rank, crowding distance) — with a
# plain implementation free of divisibility constraints.

def _assign_rank_and_crowding(individuals: Sequence) -> None:
    fronts = tools.sortNondominated(individuals, len(individuals))
    for rank, front in enumerate(fronts):
        tools.emo.assignCrowdingDist(front)
        for ind in front:
            ind.pareto_rank = rank


def _binary_tournament(individuals: Sequence):
    a, b = random.choice(individuals), random.choice(individuals)
    if a.pareto_rank != b.pareto_rank:
        return a if a.pareto_rank < b.pareto_rank else b
    ca = getattr(a.fitness, "crowding_dist", 0.0)
    cb = getattr(b.fitness, "crowding_dist", 0.0)
    if ca != cb:
        return a if ca > cb else b
    return a if random.random() < 0.5 else b


def make_offspring_tournament(
    toolbox,
    parents: Sequence,
    n_offspring: int,
    crossover_prob: float,
) -> List:
    """Binary tournament (rank, crowding) → SBX → polynomial mutation.

    ``parents`` must have valid multi-objective fitnesses; rank + crowding
    are (re)assigned here. Returns exactly ``n_offspring`` new individuals.
    """
    _assign_rank_and_crowding(parents)
    offspring: List = []
    while len(offspring) < n_offspring:
        p1 = toolbox.clone(_binary_tournament(parents))
        p2 = toolbox.clone(_binary_tournament(parents))
        if random.random() < crossover_prob:
            toolbox.mate(p1, p2)
        toolbox.mutate(p1)
        toolbox.mutate(p2)
        offspring.extend([p1, p2])
    return offspring[:n_offspring]


# ============================================================================
#  Main class
# ============================================================================

class NSGA2MorphCMAES(HebbianCMAES):
    """Persistent CMA-ES over Hebbian rules + NSGA-II over URDF morphology.

    Takes ONE ``OuterNSGA2Config`` — a plain inner-loop config plus the
    ``outer:`` NSGA-II section. Population size is ``catalog.num_urdfs``,
    phase length is ``catalog.refresh_urdfs_every``.
    """

    def __init__(self, cfg: OuterNSGA2Config) -> None:
        cfg.validate()   # cheap; fail before the expensive env build
        super().__init__(cfg)
        self.outer = cfg.outer

        if not self._use_multi_urdf:
            raise ValueError(
                "NSGA2MorphCMAES requires the multi-URDF path "
                "(catalog.num_urdfs > 1 or force_multi_urdf)."
            )

        # Recover the initial population's normalized genomes from the
        # auto-generated catalog's genomes.txt (always parsable for
        # build_catalog outputs; physical → normalized via from_physical).
        cat_dir = Path(self._urdf_paths[0]).parent
        norm = _read_norm_genomes_from_catalog(cat_dir)
        if norm is None:
            raise ValueError(
                f"Could not recover normalized genomes from {cat_dir}/genomes.txt "
                "— the outer loop needs them to run NSGA-II variation."
            )
        if len(norm) != len(self._urdf_paths):
            raise ValueError(
                f"genomes.txt rows ({len(norm)}) != catalog URDFs "
                f"({len(self._urdf_paths)})"
            )
        self._morph_genomes = np.asarray(norm, dtype=np.float64)  # (N, 15)

        self.toolbox = make_toolbox(self.outer, genome_dim=self._morph_genomes.shape[1])

        # Phase bookkeeping
        self._outer_gen = 0
        self._phase_samples: List[Dict[str, np.ndarray]] = []
        self._archive_rows: List[dict] = []

        # Output locations
        self.outer_dir = self.run_dir / "outer"
        self.outer_dir.mkdir(parents=True, exist_ok=True)
        self.outer_per_gen_csv = self.results_dir / "outer_per_urdf_per_gen.csv"
        self.outer_pop_csv = self.results_dir / "outer_population.csv"
        self._init_outer_csvs()

        phase_len = int(cfg.catalog.refresh_urdfs_every)
        n_phases = max(1, int(cfg.evolution.num_generations) // phase_len)
        print(f"[NSGA2MorphCMAES] N={len(self._urdf_paths)} URDFs, "
              f"{n_phases} phases × {phase_len} inner gens, "
              f"n_elites={self.outer.n_elites}, objectives="
              f"{[(o.name, o.direction) for o in self.outer.objectives]}")

    # ------------------------------------------------------------------
    #  CSV setup / append
    # ------------------------------------------------------------------

    def _init_outer_csvs(self) -> None:
        with open(self.outer_per_gen_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["generation", "outer_gen", "urdf_idx"] + list(_DIAG_KEYS)
            )
        n_genes = self._morph_genomes.shape[1]
        obj_cols = [f"obj_{o.name}" for o in self.outer.objectives]
        gene_cols = [f"g{i}" for i in range(n_genes)]
        with open(self.outer_pop_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["outer_gen", "urdf_idx", "urdf_file", "n_score_gens"]
                + obj_cols + list(_DIAG_KEYS) + gene_cols
            )

    def _append_per_gen_csv(self, gen: int, diag: Dict[str, np.ndarray]) -> None:
        N = len(self._urdf_paths)
        with open(self.outer_per_gen_csv, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(N):
                w.writerow(
                    [gen, self._outer_gen, i]
                    + [f"{diag[k][i]:.6g}" for k in _DIAG_KEYS]
                )

    # ------------------------------------------------------------------
    #  Per-generation hook: harvest per-URDF metrics
    # ------------------------------------------------------------------

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
        super()._after_generation(
            gen, solutions, fitnesses, metrics, es,
            baseline=baseline, specialist=specialist,
        )

        pu_reward = metrics.get("per_urdf_reward")
        if pu_reward is None:
            # Eval-failure fallback dict — nothing to score this generation.
            print(f"[NSGA2MorphCMAES] gen {gen}: no per-URDF metrics "
                  f"(eval failed?) — skipping URDF scoring for this gen")
            return

        # Score URDFs with the top fraction of this generation's CMA
        # individuals (by overall fitness): "what does this morphology achieve
        # under the current good rules", averaged for noise reduction.
        P = len(fitnesses)
        k = max(1, int(round(P * float(self.outer.score_top_frac))))
        top = np.argsort(np.asarray(fitnesses, dtype=float))[::-1][:k]

        diag: Dict[str, np.ndarray] = {}
        for name in _DIAG_KEYS:
            mat = metrics.get(_PER_URDF_ALIASES[name])
            diag[name] = (
                np.asarray(mat, dtype=np.float64)[:, top].mean(axis=1)
                if mat is not None
                else np.full(len(self._urdf_paths), np.nan)
            )
        self._phase_samples.append(diag)
        self._append_per_gen_csv(gen, diag)

    # ------------------------------------------------------------------
    #  Phase reduction + archive
    # ------------------------------------------------------------------

    def _reduce_phase(self) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
        """Mean per-URDF diagnostics over the phase's scoring window.

        Returns ``(objectives (N, n_obj), diagnostics dict of (N,), n_gens)``
        or ``None`` when the phase produced no valid samples.
        """
        if not self._phase_samples:
            return None
        w = int(self.outer.score_window)
        samples = self._phase_samples if w <= 0 else self._phase_samples[-w:]
        agg = {
            key: np.mean(np.stack([s[key] for s in samples], axis=0), axis=0)
            for key in _DIAG_KEYS
        }
        objs = np.stack(
            [agg[self._canonical_objective(o.name)] for o in self.outer.objectives],
            axis=1,
        )
        return objs, agg, len(samples)

    @staticmethod
    def _canonical_objective(name: str) -> str:
        """Map an objective name to its _DIAG_KEYS entry."""
        alias = _PER_URDF_ALIASES.get(name)
        for key in _DIAG_KEYS:
            if _PER_URDF_ALIASES[key] == alias:
                return key
        raise KeyError(
            f"Objective {name!r} not supported; choose from "
            f"{sorted(_PER_URDF_ALIASES)}"
        )

    def _record_outer_generation(
        self,
        objs: np.ndarray,
        agg: Dict[str, np.ndarray],
        n_gens: int,
    ) -> None:
        """Snapshot the ending phase: CSV rows + per-phase .npy dumps."""
        N = len(self._urdf_paths)
        with open(self.outer_pop_csv, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(N):
                w.writerow(
                    [self._outer_gen, i, Path(self._urdf_paths[i]).name, n_gens]
                    + [f"{v:.6g}" for v in objs[i]]
                    + [f"{agg[k][i]:.6g}" for k in _DIAG_KEYS]
                    + [f"{g:.6f}" for g in self._morph_genomes[i]]
                )
        for i in range(N):
            self._archive_rows.append({
                "outer_gen": self._outer_gen,
                "urdf_idx": i,
                "urdf_file": Path(self._urdf_paths[i]).name,
                "objectives": objs[i].tolist(),
                **{k: float(agg[k][i]) for k in _DIAG_KEYS},
                "genome": self._morph_genomes[i].tolist(),
            })

        snap_dir = self.outer_dir / f"gen_{self._outer_gen:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.save(snap_dir / "genomes.npy", self._morph_genomes)
        np.save(snap_dir / "objectives.npy", objs)

    # ------------------------------------------------------------------
    #  URDF refresh = NSGA-II morphology update
    # ------------------------------------------------------------------

    def _refresh_urdfs(self, gen: int) -> None:
        """Close the current phase and evolve the URDF population.

        Called by the parent run loop every ``catalog.refresh_urdfs_every``
        inner generations. Replaces random-resample / blind-mutation with:
        score → archive → NSGA-II select ``n_elites`` survivors → tournament
        + SBX + polynomial-mutation offspring → materialize → env rebuild.
        """
        if not self._use_multi_urdf:
            return

        reduced = self._reduce_phase()
        if reduced is None:
            print(f"[NSGA2MorphCMAES] gen {gen}: phase produced no scores — "
                  f"keeping current URDF population (no refresh)")
            return
        objs, agg, n_gens = reduced
        self._record_outer_generation(objs, agg, n_gens)

        outer = self.outer
        N = len(self._morph_genomes)
        n_elites = min(outer.n_elites, N - 1)

        inds = arrays_to_individuals(self._morph_genomes, objs)
        elites = self.toolbox.select(inds, n_elites)  # selNSGA2 (rank+crowding)
        offspring = make_offspring_tournament(
            self.toolbox, inds,
            n_offspring=N - n_elites,
            crossover_prob=outer.crossover_prob,
        )
        new_genomes = np.clip(
            np.vstack([
                individuals_to_array(elites),
                individuals_to_array(offspring),
            ]),
            0.0, 1.0,
        )

        elite_src = [inds.index(e) for e in elites]
        print(
            f"\n[NSGA2MorphCMAES] === Outer gen {self._outer_gen} → "
            f"{self._outer_gen + 1} (inner gen {gen}) ===\n"
            f"[NSGA2MorphCMAES] Phase scores over {n_gens} gens:\n"
            + "\n".join(
                f"    urdf {i}: "
                + "  ".join(
                    f"{o.name}={objs[i, j]:.4g}"
                    for j, o in enumerate(outer.objectives)
                )
                for i in range(N)
            )
            + f"\n[NSGA2MorphCMAES] Elites kept (indices): {elite_src}; "
              f"{N - n_elites} offspring via tournament+SBX+PM"
        )

        # Materialize new URDFs while the Genesis runtime is still alive,
        # then rebuild the eval (and validation) envs — mirrors _mutate_urdfs.
        from general_policy.catalog import _write_genomes_txt
        from morph_evolution.chromosome_drone import Chromosome_Drone

        new_dir = self.run_dir / f"urdfs_gen_{gen:03d}"
        new_paths = materialize_urdfs(new_genomes.tolist(), new_dir)
        write_catalog_txt(new_paths, new_dir)
        phys = [
            Chromosome_Drone.to_physical(Chromosome_Drone.snap_genome_norm(list(g)))
            for g in new_genomes
        ]
        _write_genomes_txt(new_dir, [Path(p) for p in new_paths], phys)

        self._cleanup_validation_env()
        self._cleanup_env()
        self._urdf_paths = new_paths
        self._morph_genomes = new_genomes
        self._outer_gen += 1
        self._phase_samples = []
        self._build_env_once()
        if self.cfg.validation.enable:
            self._build_validation_env_once()

    # ------------------------------------------------------------------
    #  Run wrapper: flush the final phase
    # ------------------------------------------------------------------

    def run(
        self,
        resume_from_gen: Optional[int] = None,
        x0_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        if resume_from_gen is not None:
            raise NotImplementedError(
                "NSGA2MorphCMAES does not support resume yet (outer-loop "
                "population state is not checkpointed for restore)."
            )
        result = super().run(resume_from_gen=None, x0_override=x0_override)

        # The last phase ends without a refresh — record it so the archive
        # holds outer_generations complete entries.
        reduced = self._reduce_phase()
        if reduced is not None:
            objs, agg, n_gens = reduced
            self._record_outer_generation(objs, agg, n_gens)

        import pickle
        with open(self.outer_dir / "pareto_archive.pkl", "wb") as f:
            pickle.dump(self._archive_rows, f)
        print(f"[NSGA2MorphCMAES] Outer archive: {len(self._archive_rows)} "
              f"records over {self._outer_gen + 1} outer generations "
              f"→ {self.outer_pop_csv}")
        return result
