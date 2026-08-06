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
  come from a phase-end "exam" rollout (``outer.rescore``, default on):
  the top ``rescore_top_frac`` of the last generation's CMA individuals
  re-fly the URDF population on fresh forests, reusing the live env. With
  k controllers sharing the E slots per URDF each flies E/k forests
  (64 pop, F=60, k=8 → 480), so morphology scores rest on a much wider
  forest sample than any single generation. Cost: one extra rollout per
  phase. With ``rescore: false`` the objectives instead fall back to the
  per-generation harvest of the ``per_urdf_*`` matrices that
  ``evaluate_population_multi_urdf`` already returns (top
  ``score_top_frac`` of each generation, phase-averaged) — no extra
  rollouts; that harvest is always recorded as diagnostics either way.

Fairness note: URDFs are only ever compared *within* a phase — all N see
the same CMA individuals, the same forests, and the same speed grid, so the
NSGA-II ranking is internally consistent even though rules evolve across
phases. Elites are re-scored every phase, so no stale objective survives.

Outputs (on top of everything ``HebbianCMAES`` already writes):

* ``results/outer_per_urdf_per_gen.csv`` — per inner gen × URDF diagnostics.
* ``results/outer_population.csv``       — per outer gen × URDF objectives
  (exam or phase-mean, see ``obj_source`` column) + phase-mean diagnostics
  + normalized genome (the NSGA-II selection input; also the Pareto archive
  used by ``pareto_plots``).
* ``results/outer_exam_baseline.csv``    — per outer gen, the standard
  mydrone flown by the zero-rules generalist on THAT phase's exam forests
  (``outer.exam_baseline``, reusing the held-out validation env). It is the
  reference point ``pareto_plots`` draws as the gold star: the only
  measurement of the unevolved morphology on the exam distribution.
* ``results/pareto_front.csv``           — per outer gen, the Pareto-front
  members of ``outer_population.csv`` (exam-filtered, min-progress-gated),
  with front size, hypervolume and the normalized genome so any front
  morphology can be rebuilt with ``urdf_population.materialize_urdfs``.
  Rewritten at every phase end; backfillable for old runs via
  ``python -m WP2_Outer_Loop.pareto_fronts <run_dir>``.
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
from .pareto_fronts import build_pareto_front_csv_safe
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

# results/outer_exam_baseline.csv metric columns → the keys HebbianCMAES's
# _pack_eval_result uses. Canonical objective names on the CSV side so
# pareto_plots can look up a plotted objective directly by name.
_EXAM_BASELINE_COLS: Tuple[Tuple[str, str], ...] = (
    ("fitness", "fitness"),
    ("velocity", "velocity"),
    ("progress_m", "progress"),
    ("crash_rate", "crash_rate"),
    ("cost_of_transport", "cot"),
    ("v_deviation", "v_deviation"),
)


# ============================================================================
#  Phase-end exam: dedicated scoring rollout for NSGA-II objectives
# ============================================================================

def exam_top_k(pop_size: int, envs_per_drone: int, frac: float) -> int:
    """Number of controllers for the phase-end exam rollout.

    ``round(pop_size * frac)`` clamped to ``[1, pop_size]``, then lowered to
    the nearest divisor of ``envs_per_drone`` so the exam reuses the existing
    env exactly (``evaluate_population_multi_urdf`` requires ``k | E``).
    k=1 always divides, so this terminates.
    """
    k = max(1, min(int(pop_size), int(round(pop_size * frac))))
    while envs_per_drone % k != 0:
        k -= 1
    return k


def reduce_exam_metrics(
    metrics: Dict[str, np.ndarray],
    objectives: Sequence,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Reduce exam ``per_urdf_*`` matrices ``(N, k)`` → per-URDF scores ``(N,)``.

    Returns ``(objs (N, n_obj), diag dict over _DIAG_KEYS)``: the mean over
    the k exam controllers, with objective columns in config order. Missing
    diagnostic matrices become NaN columns; a missing *objective* matrix
    raises (the exam is then unusable).
    """
    n_urdfs = None
    for name in _DIAG_KEYS:
        mat = metrics.get(_PER_URDF_ALIASES[name])
        if mat is not None:
            n_urdfs = np.asarray(mat).shape[0]
            break
    if n_urdfs is None:
        raise ValueError("reduce_exam_metrics: no per_urdf_* matrices in metrics")

    diag: Dict[str, np.ndarray] = {}
    for name in _DIAG_KEYS:
        mat = metrics.get(_PER_URDF_ALIASES[name])
        diag[name] = (
            np.asarray(mat, dtype=np.float64).mean(axis=1)
            if mat is not None
            else np.full(n_urdfs, np.nan)
        )
    objs = np.stack(
        [diag[NSGA2MorphCMAES._canonical_objective(o.name)] for o in objectives],
        axis=1,
    )
    if np.isnan(objs).any():
        raise ValueError("reduce_exam_metrics: an objective matrix is missing")
    return objs, diag


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


def gated_select(
    toolbox,
    inds: Sequence,
    gate_progress: np.ndarray,
    min_progress: float,
    n_elites: int,
) -> Tuple[List, List]:
    """Hard minimum-progress admission filter around NSGA-II selection.

    Returns ``(elites, parent_pool)``:

    * gate off (``min_progress <= 0``) → plain ``selNSGA2`` over all
      individuals, pool = everyone (legacy behavior);
    * elites: ``selNSGA2`` over feasible individuals only, keeping
      ``min(n_elites, n_feasible)`` — an elite deficit becomes extra
      offspring slots, never infeasible survivors;
    * parent pool: the feasible individuals, topped up with the
      highest-progress infeasible ones to reach 2 when fewer than 2 are
      feasible (SBX needs a pair);
    * zero feasible → loud warning, refresh runs ungated;
    * NaN progress counts as feasible — missing gate data must never
      exclude a morphology.
    """
    inds = list(inds)
    if min_progress is None or float(min_progress) <= 0.0:
        return toolbox.select(inds, n_elites), inds

    gate = np.asarray(gate_progress, dtype=float)
    infeasible = gate < float(min_progress)  # NaN compares False → feasible
    feasible_idx = np.flatnonzero(~infeasible)
    if len(feasible_idx) == 0:
        print(f"[NSGA2MorphCMAES] WARNING: no morphology reached "
              f"min_progress_m={float(min_progress):g} — refresh runs UNGATED")
        return toolbox.select(inds, n_elites), inds

    feasible = [inds[i] for i in feasible_idx]
    elites = toolbox.select(feasible, min(n_elites, len(feasible)))
    pool = list(feasible)
    if len(pool) < 2:
        for i in sorted(np.flatnonzero(infeasible), key=lambda i: -gate[i]):
            if len(pool) >= 2:
                break
            pool.append(inds[i])
    return elites, pool


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
        # Last successfully-evaluated generation, cached for the phase-end
        # exam (its top rescore_top_frac controllers re-fly the URDFs).
        self._last_solutions: Optional[List[np.ndarray]] = None
        self._last_fitnesses: Optional[np.ndarray] = None
        self._last_gen: int = -1

        # Output locations
        self.outer_dir = self.run_dir / "outer"
        self.outer_dir.mkdir(parents=True, exist_ok=True)
        self.outer_per_gen_csv = self.results_dir / "outer_per_urdf_per_gen.csv"
        self.outer_pop_csv = self.results_dir / "outer_population.csv"
        self.exam_baseline_csv = self.results_dir / "outer_exam_baseline.csv"
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
                ["outer_gen", "urdf_idx", "urdf_file", "n_score_gens", "obj_source"]
                + obj_cols + list(_DIAG_KEYS) + gene_cols
            )
        if self.outer.exam_baseline:
            with open(self.exam_baseline_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["outer_gen", "inner_gen", "n_forests"]
                    + [col for col, _key in _EXAM_BASELINE_COLS]
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
        self._last_gen = gen

        pu_reward = metrics.get("per_urdf_reward")
        if pu_reward is None:
            # Eval-failure fallback dict — nothing to score this generation.
            print(f"[NSGA2MorphCMAES] gen {gen}: no per-URDF metrics "
                  f"(eval failed?) — skipping URDF scoring for this gen")
            return

        self._last_solutions = [np.asarray(s, dtype=np.float64) for s in solutions]
        self._last_fitnesses = np.asarray(fitnesses, dtype=np.float64)

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

    # ------------------------------------------------------------------
    #  Phase-end exam rollout
    # ------------------------------------------------------------------

    def _run_exam(self) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
        """One dedicated scoring rollout on the live eval env.

        The top ``rescore_top_frac`` of the last generation's controllers
        (ranked by overall fitness) re-fly the current URDF population on
        freshly generated forests. With k controllers sharing the E slots
        per URDF each flies E/k forests — a far wider forest sample than the
        per-generation harvest. Returns ``(objs (N, n_obj), diag, k)`` or
        ``None`` when the exam cannot run (disabled, env gone, no cached
        generation, eval failure) — callers then fall back to phase means.
        """
        if not getattr(self.outer, "rescore", False):
            return None
        if self._env is None or self._last_solutions is None:
            return None

        from WP2.evaluate import evaluate_population_multi_urdf

        E = int(self._env.E)
        k = exam_top_k(
            len(self._last_solutions), E, float(self.outer.rescore_top_frac)
        )
        top = np.argsort(self._last_fitnesses)[::-1][:k]
        sols = [self._last_solutions[i] for i in top]

        exam_forest = getattr(self.outer, "exam_forest", None)
        overrides = exam_forest.overrides() if exam_forest is not None else {}
        prev_forest: Optional[Dict] = None
        try:
            if overrides:
                prev_forest = self._env.apply_forest_overrides(overrides)
                print(f"[NSGA2MorphCMAES] Exam forest overrides: {overrides}")
            # Fresh layouts so the exam's E/k forests are new, not the ones
            # that ranked these controllers (avoids selection bias).
            self._env.refresh_forests()
            _fit, metrics = evaluate_population_multi_urdf(
                sols,
                self.cfg,
                self._model_and_layer,
                self._wp1_cfg,
                urdf_paths=self._urdf_paths,
                existing_env=(self._env, self._env_urdf_path),
                verbose=False,
            )
            objs, diag = reduce_exam_metrics(metrics, self.outer.objectives)
        except Exception as exc:
            print(f"[NSGA2MorphCMAES] Exam rollout failed ({exc}) — "
                  f"falling back to phase-mean scores")
            return None
        finally:
            if prev_forest:
                # Put the env back on nominal forests: it usually gets torn
                # down right after the exam, but on the no-refresh fallback
                # path it survives into the next generation. Regeneration is
                # pure tensor work — milliseconds.
                try:
                    self._env.apply_forest_overrides(prev_forest)
                    self._env.refresh_forests()
                except Exception as exc:
                    print(f"[NSGA2MorphCMAES] Exam forest restore failed: {exc}")
        print(f"[NSGA2MorphCMAES] Exam: top {k} controllers × {E // k} fresh "
              f"forests per URDF")
        self._run_exam_baseline(overrides)
        return objs, diag, k

    def _run_exam_baseline(self, overrides: Dict) -> None:
        """Fly the standard mydrone on the exam forests (the Pareto star).

        The held-out validation env already holds the unevolved reference
        morphology (``validation.validation_catalog`` empty → the standard
        mydrone) and is still alive here — ``_refresh_urdfs`` only tears it
        down after ``_score_phase`` returns. So the zero-rules generalist can
        re-fly it under the SAME forest overrides the exam just used, giving
        ``pareto_plots`` a reference point measured on the exam distribution
        rather than the (easier) nominal one. No scene is rebuilt.

        Self-contained: applies the overrides, evaluates, restores, and
        swallows every failure — the caller's exam objectives must never
        depend on the reference point.

        Cold-start caveat: the validation env is rebuilt at every URDF
        refresh, so with ``validation.period > catalog.refresh_urdfs_every``
        this can be its first flight of the phase — cold Taichi aero state
        (``_thr_flt`` starts at 0), which biases the star pessimistic. Keep
        ``validation.period <= catalog.refresh_urdfs_every``.
        """
        if not getattr(self.outer, "exam_baseline", False):
            return
        val_env = getattr(self, "_val_env", None)
        if val_env is None:
            return

        prev_forest: Optional[Dict] = None
        try:
            if overrides:
                prev_forest = val_env.apply_forest_overrides(overrides)
            val_env.refresh_forests()
            result = self._evaluate_reference_actor(
                "Exam-Baseline",
                ckpt_path=self.cfg.baseline_checkpoint_path or None,
                ckpt_cfg_path=self.cfg.baseline_checkpoint_config_path or None,
                env_override=val_env,
                urdf_paths_override=self._val_urdf_paths,
            )
            if result is None:
                print("[NSGA2MorphCMAES] Exam baseline produced no result — "
                      "no reference point for this phase")
                return
            self._append_exam_baseline_csv(result, val_env)
        except Exception as exc:
            print(f"[NSGA2MorphCMAES] Exam baseline failed ({exc}) — no "
                  f"reference point for this phase (objectives unaffected)")
        finally:
            if prev_forest:
                try:
                    val_env.apply_forest_overrides(prev_forest)
                    val_env.refresh_forests()
                except Exception as exc:
                    print(f"[NSGA2MorphCMAES] Exam baseline forest restore "
                          f"failed: {exc}")

    def _append_exam_baseline_csv(self, result: Dict[str, float], val_env) -> None:
        """One row of the standard-mydrone reference for the ending phase."""
        n_forests = int(getattr(val_env, "E", 0) or 0)
        vals = [float(result.get(key, float("nan")))
                for _col, key in _EXAM_BASELINE_COLS]
        with open(self.exam_baseline_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [self._outer_gen, self._last_gen, n_forests]
                + [f"{v:.6g}" for v in vals]
            )
        print(
            f"[NSGA2MorphCMAES] Exam baseline (standard mydrone, zero rules, "
            f"{n_forests} exam forests): "
            + "  ".join(f"{col}={v:.4g}"
                        for (col, _key), v in zip(_EXAM_BASELINE_COLS, vals))
        )

    def _score_phase(
        self,
    ) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int, str, np.ndarray]]:
        """Final per-URDF scores for the ending phase.

        Objectives come from the exam rollout when it runs, else from the
        phase-mean harvest; diagnostics stay phase means whenever available.
        ``gate_progress`` is the per-URDF progress from the SAME source as
        the objectives — it feeds the ``outer.min_progress_m`` admission
        filter. Returns ``(objs, diag, n_score_gens, obj_source,
        gate_progress)`` or ``None``.
        """
        reduced = self._reduce_phase()
        exam = self._run_exam()
        if exam is not None:
            objs, exam_diag, _k = exam
            agg = reduced[1] if reduced is not None else exam_diag
            n_gens = reduced[2] if reduced is not None else 0
            gate = np.asarray(exam_diag["progress_m"], dtype=np.float64)
            return objs, agg, n_gens, "exam", gate
        if reduced is not None:
            objs, agg, n_gens = reduced
            gate = np.asarray(agg["progress_m"], dtype=np.float64)
            return objs, agg, n_gens, "phase_mean", gate
        return None

    def _record_outer_generation(
        self,
        objs: np.ndarray,
        agg: Dict[str, np.ndarray],
        n_gens: int,
        obj_source: str = "phase_mean",
    ) -> None:
        """Snapshot the ending phase: CSV rows, .npy dumps, front CSV."""
        N = len(self._urdf_paths)
        with open(self.outer_pop_csv, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(N):
                w.writerow(
                    [self._outer_gen, i, Path(self._urdf_paths[i]).name,
                     n_gens, obj_source]
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
                "obj_source": obj_source,
                **{k: float(agg[k][i]) for k in _DIAG_KEYS},
                "genome": self._morph_genomes[i].tolist(),
            })

        snap_dir = self.outer_dir / f"gen_{self._outer_gen:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.save(snap_dir / "genomes.npy", self._morph_genomes)
        np.save(snap_dir / "objectives.npy", objs)

        # Refresh results/pareto_front.csv from the file we just appended to.
        # A full rebuild (ms on a few-thousand-row CSV, against an hours-long
        # phase) keeps the live file byte-identical to a backfilled one and
        # self-heals a resumed or hand-recovered run. Objectives and the gate
        # come from the in-memory config, so this never depends on
        # reproducibility/config.yaml being parseable.
        build_pareto_front_csv_safe(
            self.run_dir,
            specs=[(o.name, o.direction) for o in self.outer.objectives],
            min_progress=float(getattr(self.outer, "min_progress_m", 0.0) or 0.0),
        )

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

        scored = self._score_phase()
        if scored is None:
            print(f"[NSGA2MorphCMAES] gen {gen}: phase produced no scores — "
                  f"keeping current URDF population (no refresh)")
            return
        objs, agg, n_gens, obj_source, gate_progress = scored
        self._record_outer_generation(objs, agg, n_gens, obj_source=obj_source)

        outer = self.outer
        N = len(self._morph_genomes)
        n_elites = min(outer.n_elites, N - 1)
        min_progress = float(getattr(outer, "min_progress_m", 0.0))

        inds = arrays_to_individuals(self._morph_genomes, objs)
        elites, parent_pool = gated_select(
            self.toolbox, inds, gate_progress, min_progress, n_elites,
        )
        offspring = make_offspring_tournament(
            self.toolbox, parent_pool,
            n_offspring=N - len(elites),
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
        gate = np.asarray(gate_progress, dtype=float)
        infeasible = (gate < min_progress) if min_progress > 0.0 else np.zeros(N, bool)
        print(
            f"\n[NSGA2MorphCMAES] === Outer gen {self._outer_gen} → "
            f"{self._outer_gen + 1} (inner gen {gen}) ===\n"
            f"[NSGA2MorphCMAES] Phase scores ({obj_source}, {n_gens} gens"
            + (f", min_progress_m={min_progress:g}" if min_progress > 0.0 else "")
            + "):\n"
            + "\n".join(
                f"    urdf {i}: "
                + "  ".join(
                    f"{o.name}={objs[i, j]:.4g}"
                    for j, o in enumerate(outer.objectives)
                )
                + (f"  [INFEASIBLE: progress {gate[i]:.4g} < {min_progress:g}]"
                   if infeasible[i] else "")
                for i in range(N)
            )
            + f"\n[NSGA2MorphCMAES] Elites kept (indices): {elite_src}; "
              f"{N - len(elites)} offspring via tournament+SBX+PM"
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
        # holds outer_generations complete entries. The env is already torn
        # down here, so _score_phase falls back to phase-mean objectives.
        scored = self._score_phase()
        if scored is not None:
            # Gate progress unused here: the final flush only records — no
            # selection happens on it.
            objs, agg, n_gens, obj_source, _gate = scored
            self._record_outer_generation(objs, agg, n_gens, obj_source=obj_source)

        import pickle
        with open(self.outer_dir / "pareto_archive.pkl", "wb") as f:
            pickle.dump(self._archive_rows, f)
        print(f"[NSGA2MorphCMAES] Outer archive: {len(self._archive_rows)} "
              f"records over {self._outer_gen + 1} outer generations "
              f"→ {self.outer_pop_csv}")
        return result
