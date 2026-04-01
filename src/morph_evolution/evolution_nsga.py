#!/usr/bin/env python3
"""
evolution_nsga.py

NSGA-II co-design of drone morphology + control policy in a continuous
genome space, with:

  - Simulated Binary Crossover (SBX) in [0, 1]^D.
  - Polynomial mutation (bounded).
  - Policy inheritance between generations.
  - Fitness caching in a CSV "database".
  - Optional multi-GPU parallelism via Ray.
  - Post-hoc analysis and plotting utilities.

The genome lives in [0, 1]^D and is mapped to physical parameters by
`Chromosome_Drone.to_physical()`, which in turn drives `UrdfMaker`.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import os
import pickle
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from deap import base, creator, tools

from morph_evolution.chromosome_drone import Chromosome_Drone
from morph_evolution.utils.evaluation import (
    default_fitness as _default_fitness_impl,
    eval_only_custom as _eval_only_custom_impl,
    failure_result as _failure_result_impl,
    load_rep_payload as _load_rep_payload,
    train_and_eval_sync as _train_and_eval_sync_impl,
)
from morph_evolution.utils.reporting import (
    FitnessDB,
    PostAnalyzer,
    Stats,
    append_generation_summary,
    append_pareto_history,
    append_population_history,
    append_selection_pool_history,
    ckpt_idx_from_train_it as _ckpt_idx_from_train_it,
    init_report_csvs,
    invalid_objective_masks,
    parse_fail_reason as _parse_fail_reason,
    write_run_manifest,
)
from morph_evolution.utils.runtime import resolve_urdf_dir as _resolve_urdf_dir


# =============================================================================
#  GLOBAL GA CONFIGURATION (single place for all hyper-parameters)
# =============================================================================


DEFAULT_RESUME_SUFFIX = "second"
DEFAULT_POLICY_PATHS = [
    "/home/andrea/Documents/Genesis/src/logs/training_general/foundation-mixture_2655024/logs/ea/foundation-mixture/model_1900.pt",
    "/home/andrea/Documents/Genesis/src/logs/training_general/foundation-mixture_2663796/logs/ea/1/model_1900.pt",
    "/home/andrea/Documents/Genesis/src/logs/training_general/foundation-mixture_2663633/logs/ea/1/model_1900.pt",
]


@dataclass
class GAConfig:
    """
    All GA / NSGA-II hyper-parameters in one place.

    Edit this dataclass (or override fields from the CLI) to configure
    the search behaviour.
    """

    # --- Population & evolutionary budget ---------------------------------
    population_size: int = 40       # number of individuals per generation
    num_generations: int = 25       # number of generations to run

    # --- NSGA-II operators (continuous) -----------------------------------
    crossover_probability: float = 0.9   # probability of SBX crossover
    mutation_probability: float = 1.0 # 0.12    # probability of applying mutation
    eta_c: float = 15.0                  # SBX "spread" parameter (higher = more local)
    eta_m: float = 12.0                  # polynomial mutation parameter
    mutation_indpb_numerator: float = 1.5  # effective indpb = numerator / n_genes

    # --- RL training / evaluation -----------------------------------------
    gen_policy: bool = False
    policy_path: Optional[str] = None  # path to initial policy checkpoint
    policy_paths: List[str] = field(default_factory=lambda: list(DEFAULT_POLICY_PATHS))
    train_iters_new: int = 850       # iterations for NEW morphologies
    train_iters_inherit: int = 200  # iterations when inheriting from a parent
    train_repetition: int = 1       # repeat train+eval N times (gen_policy=0)
    train_envs: int = 16384           # number of envs during training
    eval_envs: int = 8192            # number of envs during evaluation
    vmin: float = 5.0               # min commanded speed in evaluation
    vmax: float = 25.0              # max commanded speed in evaluation

    # --- Fitness shaping / invalid individuals ----------------------------
    # fail_value removed; fallback uses INVALID_* sentinels
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (vel, -energy, progress)

    # --- Progress threshold (minimal_p) -----------------------------------
    use_dynamic_p: bool = False      # if True: percentile-based threshold
    fixed_p: float = 250.0          # fallback / fixed threshold [m]
    pct_above: float = 50.0         # fraction of individuals above minimal_p

    # --- Policy inheritance -----------------------------------------------
    inherit_policy: bool = False    # if True: offspring can inherit parent policy

    # --- Output / logging -------------------------------------------------
    csv_basename: str = "nsga"       # CSV filename (".csv" added automatically)
    run_name: Optional[str] = None   # Optional custom run name (defaults to timestamp)
    base_dir: str = "nsga"           # Root directory for all artifacts
    device: str = "cuda:0"           # Device passed to training/evaluation
    resume_from: Optional[str] = None  # Existing run folder to resume from
    resume_suffix: str = DEFAULT_RESUME_SUFFIX        # Suffix for the copied resume folder


# Default config used when no custom config is provided
DEFAULT_GA_CONFIG = GAConfig()


# =============================================================================
#  PARALLELISM SELECTION (Ray or serial)
# =============================================================================


def _want_parallel() -> bool:
    """
    Decide whether to run in parallel (Ray) or serial.

    Rules
    -----
    - GA_PARALLEL=0 / "false" / "no"  → force serial.
    - GA_PARALLEL=1 / "true" / "yes"  → force parallel.
    - GA_PARALLEL unset / "auto"      → parallel only if ≥ 2 GPUs.
    """
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1


USE_PARALLEL = _want_parallel()

if USE_PARALLEL:
    import ray  # type: ignore[import]

    ray_address = os.getenv("RAY_ADDRESS", "").strip()
    ray_log_to_driver = os.getenv("RAY_LOG_TO_DRIVER", "").strip().lower() in ("1", "true", "yes")
    if ray_address:
        ray.init(address=ray_address, log_to_driver=ray_log_to_driver)
    else:
        ray.init(log_to_driver=ray_log_to_driver)


# =============================================================================
#  SENTINEL VALUES FOR INVALID INDIVIDUALS
# =============================================================================

INVALID_V = {0.0}       # invalid average velocity
INVALID_E = {-10.0}     # negative energy sentinel (equivalent to +10 before flip)
INVALID_P = {0.0}       # invalid progress / maneuverability


def _default_fitness(_: Optional[Dict[str, Any]] = None) -> List[float]:
    return _default_fitness_impl(INVALID_V, INVALID_E, INVALID_P)


def _failure_result(
    reason: str,
    cfg: Dict[str, Any],
    exp_name: Optional[str] = None,
    train_it: Optional[int] = None,
):
    return _failure_result_impl(
        reason,
        cfg,
        INVALID_V,
        INVALID_E,
        INVALID_P,
        exp_name=exp_name,
        train_it=train_it,
    )


def _eval_only_custom(
    genome_norm,
    policy_path,
    tag,
    cfg,
    return_arrays=True,
):
    return _eval_only_custom_impl(
        genome_norm,
        policy_path,
        tag,
        cfg,
        INVALID_V,
        INVALID_E,
        INVALID_P,
        return_arrays=return_arrays,
    )


def _train_and_eval_sync(
    genome_norm: Sequence[float],
    parent_info: Tuple[Optional[str], Optional[int]],
    tag: str,
    cfg: Dict[str, Any],
    return_arrays: bool = True,
):
    return _train_and_eval_sync_impl(
        genome_norm,
        parent_info,
        tag,
        cfg,
        INVALID_V,
        INVALID_E,
        INVALID_P,
        return_arrays=return_arrays,
    )
# Parallel / serial dispatch wrapper
if USE_PARALLEL:

    # max_calls=1 forces Ray to recycle the worker process after each task,
    # which helps avoid memory growth across many train+eval runs.
    _ray_max_calls_raw = os.getenv("RAY_MAX_CALLS", "1").strip().lower()
    _ray_max_calls: Optional[int]
    if _ray_max_calls_raw in ("", "0", "none", "inf", "infinite"):
        _ray_max_calls = None
    else:
        try:
            _ray_max_calls = max(1, int(_ray_max_calls_raw))
        except Exception:
            _ray_max_calls = 1

    _ray_remote_kwargs = {"num_gpus": 1}
    if _ray_max_calls is not None:
        _ray_remote_kwargs["max_calls"] = _ray_max_calls

    @ray.remote(**_ray_remote_kwargs)
    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)

    @ray.remote(**_ray_remote_kwargs)
    def eval_only_remote(*args, **kwargs):
        return _eval_only_custom(*args, **kwargs)

else:

    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)

    def eval_only_remote(*args, **kwargs):
        return _eval_only_custom(*args, **kwargs)


IndType = TypeVar("IndType")


# =============================================================================
#  NSGA-II MAIN CLASS
# =============================================================================


class CodesignDEAP:
    """
    NSGA-II loop for morphology + controller co-design.

    Usage (programmatic)
    --------------------
        cfg = GAConfig()
        cfg.population_size = 40
        cfg.num_generations = 30
        ...
        ga = CodesignDEAP(cfg)
        final_pop = ga.run()
    """

    def __init__(self, config: GAConfig = DEFAULT_GA_CONFIG) -> None:
        self.cfg = config

        # Basic checks
        if self.cfg.population_size % 4 != 0:
            raise ValueError("population_size must be a multiple of 4 for tournamentDCD.")

        self.n_pop = self.cfg.population_size
        self.n_gen = self.cfg.num_generations
        self.cx_pb = self.cfg.crossover_probability
        self.mut_pb = self.cfg.mutation_probability
        self.inherit_policy = self.cfg.inherit_policy

        self.gen_policy = self.cfg.gen_policy
        self.policy_path = self.cfg.policy_path
        self.policy_paths = [str(p) for p in (self.cfg.policy_paths or []) if str(p).strip()]
        if not self.policy_paths and self.policy_path:
            self.policy_paths = [self.policy_path]
        if self.policy_paths:
            self.policy_path = self.policy_paths[0]

        self.tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.resume_source_dir: Optional[Path] = None
        self.resume_start_generation: Optional[int] = None
        self._resume_population: Optional[List["IndType"]] = None
        self._is_resume = bool(self.cfg.resume_from)

        if self._is_resume:
            self.resume_source_dir = Path(str(self.cfg.resume_from)).expanduser().resolve()
            self._validate_resume_source(self.resume_source_dir)
            self.base_dir = self._prepare_resume_directory(self.resume_source_dir)
            self.run_name = self.base_dir.name
        else:
            self.run_name = self.cfg.run_name or self.tag
            self.base_dir = (Path(self.cfg.base_dir) / self.run_name).expanduser().resolve()
        self.urdf_dir = self.base_dir / "urdf_generated"
        self.logs_dir = self.base_dir / "logs"
        self.analysis_dir = self.base_dir / "analysis"
        for d in (self.urdf_dir, self.logs_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log_root = (self.logs_dir / "ea").resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.exp_prefix = self.run_name
        self.stats_path = (self.analysis_dir / "stats.pkl").resolve()
        self.run_manifest_path = (self.analysis_dir / "run_manifest.csv").resolve()
        self.population_history_path = (self.analysis_dir / "population_history.csv").resolve()
        self.pareto_history_path = (self.analysis_dir / "pareto_history.csv").resolve()
        self.generation_summary_path = (self.analysis_dir / "generation_summary.csv").resolve()
        self.selection_pool_history_path = (self.analysis_dir / "selection_pool_history.csv").resolve()
        self._last_minimal_p = float(self.cfg.fixed_p)
        self._init_report_csvs()
        if self._is_resume:
            self.resume_start_generation = self._find_last_completed_generation()
            self._prune_truncated_resume_rows(int(self.resume_start_generation))

        self.db = FitnessDB(self.cfg.csv_basename, 3, root=self.analysis_dir)
        self.stats = Stats(self.n_pop, self.n_gen, 3)

        # Create DEAP fitness and individual types (only once)
        if "FitMulti" not in creator.__dict__:
            creator.create("FitMulti", base.Fitness, weights=self.cfg.weights)
        if "Chrom" not in creator.__dict__:
            creator.create("Chrom", list, fitness=creator.FitMulti)

        self.IndType = creator.Chrom

        self._uid_counter = 0

        # Bounds in normalized space (all genes ∈ [0, 1])
        low, up = Chromosome_Drone.get_bounds()
        self._low = low
        self._up = up
        n_genes = Chromosome_Drone.num_genes()
        self.n_genes = n_genes
        self.mutation_indpb = min(1.0, float(self.cfg.mutation_indpb_numerator) / float(n_genes))
        self._write_run_manifest()
        if self._is_resume:
            self._resume_population, self.resume_start_generation = self._load_resume_population()

        # DEAP toolbox
        self.tb = base.Toolbox()
        self.tb.register("attr_float", random.random)
        self.tb.register("ind", tools.initRepeat, self.IndType, self.tb.attr_float, n=n_genes)
        self.tb.register("pop", tools.initRepeat, list, self.tb.ind)

        # Typical NSGA-II operators for continuous decision variables
        self.tb.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_c,
        )
        self.tb.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_m,
            indpb=self.mutation_indpb,
        )
        self.tb.register("select", tools.selNSGA2)
        self.tb.register("evaluate", self._evaluate)

    @staticmethod
    def _coerce_int(value: Any, default: int = -1) -> int:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float = np.nan) -> float:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _coerce_str(value: Any, default: str = "") -> str:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        if isinstance(value, str):
            txt = value.strip().lower()
            if txt in ("1", "true", "yes"):
                return True
            if txt in ("0", "false", "no", ""):
                return False
        try:
            return bool(int(value))
        except Exception:
            return bool(value) if value is not None else default

    def _validate_resume_source(self, source_dir: Path) -> None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"resume_from directory not found: {source_dir}")

        required_paths = (
            source_dir / "analysis" / "run_manifest.csv",
            source_dir / "analysis" / "nsga.csv",
            source_dir / "analysis" / "population_history.csv",
            source_dir / "analysis" / "selection_pool_history.csv",
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "resume_from is missing required files: " + ", ".join(missing)
            )

    def _prepare_resume_directory(self, source_dir: Path) -> Path:
        suffix = (self.cfg.resume_suffix or "second").strip() or "second"
        requested_name = (self.cfg.run_name or "").strip()
        if requested_name:
            target_name = requested_name
        else:
            target_name = f"{source_dir.name}_{suffix}"

        target_dir = source_dir.parent / target_name
        counter = 2
        while target_dir.exists():
            target_dir = source_dir.parent / f"{target_name}_{counter}"
            counter += 1

        shutil.copytree(source_dir, target_dir)
        return target_dir.resolve()

    def _load_resume_population(self) -> Tuple[List["IndType"], int]:
        last_generation = int(
            self.resume_start_generation
            if self.resume_start_generation is not None
            else self._find_last_completed_generation()
        )
        if last_generation >= self.n_gen:
            raise ValueError(
                "Resume source already reached the configured num_generations: "
                f"last_completed_generation={last_generation}, num_generations={self.n_gen}"
            )

        population = self._rebuild_population_from_history(last_generation)
        self._restore_stats_from_history(last_generation)
        return population, last_generation

    def _find_last_completed_generation(self) -> int:
        selection_df = pd.read_csv(self.selection_pool_history_path)
        if selection_df.empty:
            raise ValueError("selection_pool_history.csv is empty; cannot resume")

        selection_df["generation"] = pd.to_numeric(selection_df["generation"], errors="coerce")
        selection_df["selected"] = pd.to_numeric(selection_df["selected"], errors="coerce").fillna(0).astype(int)
        valid_selection = selection_df.dropna(subset=["generation"]).copy()
        valid_selection["generation"] = valid_selection["generation"].astype(int)
        selected_counts = valid_selection.groupby("generation")["selected"].sum()
        completed_gens = selected_counts[selected_counts == self.n_pop].index.tolist()
        if not completed_gens:
            raise ValueError("No completed generation found in selection_pool_history.csv")

        last_generation = int(max(completed_gens))

        population_df = pd.read_csv(self.population_history_path)
        if population_df.empty:
            raise ValueError("population_history.csv is empty; cannot resume")
        population_df["generation"] = pd.to_numeric(population_df["generation"], errors="coerce")
        pop_count = int((population_df["generation"] == last_generation).sum())
        if pop_count != self.n_pop:
            raise ValueError(
                "Inconsistent population_history.csv for resume: "
                f"generation {last_generation} has {pop_count} rows, expected {self.n_pop}"
            )

        summary_df = pd.read_csv(self.generation_summary_path)
        if summary_df.empty:
            raise ValueError("generation_summary.csv is empty; cannot resume")
        summary_df["generation"] = pd.to_numeric(summary_df["generation"], errors="coerce")
        if int((summary_df["generation"] == last_generation).sum()) != 1:
            raise ValueError(
                "Inconsistent generation_summary.csv for resume: "
                f"generation {last_generation} not found exactly once"
            )

        return last_generation

    def _prune_truncated_resume_rows(self, last_completed_generation: int) -> None:
        analysis_csvs = [
            self.analysis_dir / f"{self.cfg.csv_basename}.csv",
            self.population_history_path,
            self.pareto_history_path,
            self.generation_summary_path,
            self.selection_pool_history_path,
        ]
        for csv_path in analysis_csvs:
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "generation" not in df.columns:
                continue
            gens = pd.to_numeric(df["generation"], errors="coerce")
            keep_mask = gens.isna() | (gens <= int(last_completed_generation))
            if bool((~keep_mask).any()):
                df.loc[keep_mask].to_csv(csv_path, index=False)

    def _rebuild_population_from_history(self, generation: int) -> List["IndType"]:
        population_df = pd.read_csv(self.population_history_path)
        population_df["generation"] = pd.to_numeric(population_df["generation"], errors="coerce")
        gen_df = population_df[population_df["generation"] == generation].copy()
        if len(gen_df) != self.n_pop:
            raise ValueError(
                f"Cannot rebuild generation {generation}: found {len(gen_df)} rows, expected {self.n_pop}"
            )

        pop: List["IndType"] = []
        max_uid = 0
        for row in gen_df.itertuples(index=False):
            try:
                chromo = ast.literal_eval(str(row.chromosome))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse chromosome for generation {generation}, uid={getattr(row, 'uid', 'NA')}: {exc}"
                ) from exc

            ind = self.IndType(chromo)
            ind.fitness.values = (
                self._coerce_float(getattr(row, "ff_0", np.nan)),
                self._coerce_float(getattr(row, "ff_1", np.nan)),
                self._coerce_float(getattr(row, "ff_2", np.nan)),
            )
            ind.uid = self._coerce_int(getattr(row, "uid", -1))
            ind.parent_uid_a = self._coerce_int(getattr(row, "parent_uid_a", -1))
            ind.parent_uid_b = self._coerce_int(getattr(row, "parent_uid_b", -1))
            ind.parent_gen_a = self._coerce_int(getattr(row, "parent_gen_a", -1))
            ind.parent_gen_b = self._coerce_int(getattr(row, "parent_gen_b", -1))
            ind.lineage_id = self._coerce_int(getattr(row, "lineage_id", -1))
            ind.lineage_root_uid = self._coerce_int(getattr(row, "lineage_root_uid", -1))
            ind.lineage_depth = self._coerce_int(getattr(row, "lineage_depth", -1))
            ind.primary_parent_uid = self._coerce_int(getattr(row, "primary_parent_uid", -1))
            ind.primary_parent_generation = self._coerce_int(getattr(row, "primary_parent_generation", -1))
            ind.reproduction_operator = self._coerce_str(getattr(row, "reproduction_operator", ""))
            ind.crossover_applied = self._coerce_int(getattr(row, "crossover_applied", 0), default=0)
            ind.mutation_applied = self._coerce_int(getattr(row, "mutation_applied", 0), default=0)
            ind.mutation_changed_genome = self._coerce_int(getattr(row, "mutation_changed_genome", 0), default=0)
            ind.topology_mutation = self._coerce_int(getattr(row, "topology_mutation", 0), default=0)
            ind.topology_mutation_magnitude = self._coerce_int(
                getattr(row, "topology_mutation_magnitude", 0), default=0
            )
            ind.topology_signature = self._coerce_str(getattr(row, "topology_signature", ""))
            ind.parent_a_topology_signature = self._coerce_str(
                getattr(row, "parent_a_topology_signature", "")
            )
            ind.parent_b_topology_signature = self._coerce_str(
                getattr(row, "parent_b_topology_signature", "")
            )
            ind.primary_parent_topology_signature = self._coerce_str(
                getattr(row, "primary_parent_topology_signature", "")
            )
            ind.successful_topology_mutation = self._coerce_int(
                getattr(row, "successful_topology_mutation", 0), default=0
            )
            ind.beneficial_topology_event = self._coerce_int(
                getattr(row, "beneficial_topology_event", 0), default=0
            )
            ind.selected_next_generation = self._coerce_int(getattr(row, "selected_next_generation", 1), default=1)
            ind.parent_best_scalar_fitness = self._coerce_float(
                getattr(row, "parent_best_scalar_fitness", np.nan)
            )
            ind.offspring_scalar_fitness = self._coerce_float(
                getattr(row, "offspring_scalar_fitness", np.nan)
            )
            ind.lineage_event = self._coerce_str(getattr(row, "lineage_event", ""))
            ind.parent_a_lineage_id = self._coerce_int(getattr(row, "parent_a_lineage_id", -1))
            ind.parent_b_lineage_id = self._coerce_int(getattr(row, "parent_b_lineage_id", -1))
            ind.cross_lineage_mating = self._coerce_int(getattr(row, "cross_lineage_mating", 0), default=0)
            ind.airfoil_signature = self._coerce_str(getattr(row, "airfoil_signature", ""))
            ind.parent_a_airfoil_signature = self._coerce_str(
                getattr(row, "parent_a_airfoil_signature", "")
            )
            ind.parent_b_airfoil_signature = self._coerce_str(
                getattr(row, "parent_b_airfoil_signature", "")
            )
            ind.primary_parent_airfoil_signature = self._coerce_str(
                getattr(row, "primary_parent_airfoil_signature", "")
            )
            ind.airfoil_mutation = self._coerce_int(getattr(row, "airfoil_mutation", 0), default=0)
            ind.successful_airfoil_mutation = self._coerce_int(
                getattr(row, "successful_airfoil_mutation", 0), default=0
            )
            ind.beneficial_airfoil_event = self._coerce_int(
                getattr(row, "beneficial_airfoil_event", 0), default=0
            )
            ind.offspring_vs_best_parent_scalar_delta = self._coerce_float(
                getattr(row, "offspring_vs_best_parent_scalar_delta", np.nan)
            )
            ind.max_p = self._coerce_float(getattr(row, "max_p", np.nan))
            ind.exp_name = self._coerce_str(getattr(row, "exp_name", ""))
            ind.train_it = self._coerce_int(getattr(row, "train_it", self.cfg.train_iters_new), self.cfg.train_iters_new)
            ind.failed = self._coerce_bool(getattr(row, "failed", False))
            ind._failed = bool(ind.failed)
            ind.fail_category = self._coerce_str(getattr(row, "fail_category", ""))
            ind.fail_reason = self._coerce_str(getattr(row, "fail_reason", ""))
            ind.cache_hit = False
            ind.evaluated_fresh = False
            ind.cache_source_uid = -1
            ind.cache_source_generation = -1
            ind.generation_origin = generation
            ind.parent_idx_a = -1
            ind.parent_idx_b = -1
            pop.append(ind)
            max_uid = max(max_uid, int(ind.uid))

        self._uid_counter = max(self._uid_counter, max_uid)
        self._ensure_uids(pop)
        return pop

    def _restore_stats_from_history(self, last_generation: int) -> None:
        stats_loaded = False
        if self.stats_path.is_file():
            try:
                with self.stats_path.open("rb") as f:
                    old_stats = pickle.load(f)
                arr = np.asarray(getattr(old_stats, "arr", np.array([])))
                if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] == self.n_pop:
                    max_gen = min(arr.shape[1], self.stats.arr.shape[1], last_generation + 1)
                    self.stats.arr[:, :max_gen, :] = arr[:, :max_gen, :]
                    stats_loaded = True
            except Exception:
                stats_loaded = False

        if stats_loaded:
            return

        population_df = pd.read_csv(self.population_history_path)
        population_df["generation"] = pd.to_numeric(population_df["generation"], errors="coerce")
        for gen in range(0, last_generation + 1):
            gen_df = population_df[population_df["generation"] == gen].copy()
            if len(gen_df) != self.n_pop:
                raise ValueError(
                    f"Cannot restore stats: generation {gen} has {len(gen_df)} rows, expected {self.n_pop}"
                )
            self.stats.arr[0, gen, :] = gen_df["ff_0"].to_numpy(dtype=float)
            self.stats.arr[1, gen, :] = gen_df["ff_1"].to_numpy(dtype=float)
            self.stats.arr[2, gen, :] = gen_df["ff_2"].to_numpy(dtype=float)

    def _init_report_csvs(self) -> None:
        init_report_csvs(
            self.population_history_path,
            self.pareto_history_path,
            self.generation_summary_path,
            self.selection_pool_history_path,
        )

    def _write_run_manifest(self) -> None:
        cfg_dict = asdict(self.cfg)
        cfg_dict["mutation_indpb_effective"] = getattr(self, "mutation_indpb", np.nan)
        cfg_dict["n_genes"] = getattr(self, "n_genes", np.nan)
        write_run_manifest(
            self.run_manifest_path,
            self.run_name,
            self.base_dir,
            self.analysis_dir,
            self.logs_dir,
            self.urdf_dir,
            cfg_dict,
            USE_PARALLEL,
        )

    def _invalid_objective_masks(self, pop: Sequence["IndType"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return invalid_objective_masks(pop, INVALID_V, INVALID_E, INVALID_P)

    def _append_population_history(self, pop: Sequence["IndType"], fronts: Sequence[Sequence["IndType"]]) -> None:
        append_population_history(
            self.population_history_path,
            self._gen,
            pop,
            fronts,
        )

    def _append_pareto_history(self, front0: Sequence["IndType"]) -> None:
        append_pareto_history(
            self.pareto_history_path,
            self._gen,
            front0,
        )

    def _append_generation_summary(self, pop: Sequence["IndType"], pareto_size: int) -> None:
        append_generation_summary(
            self.generation_summary_path,
            self._gen,
            pop,
            pareto_size,
            self._last_minimal_p,
            INVALID_V,
            INVALID_E,
            INVALID_P,
        )

    def _append_selection_pool_history(
        self,
        pool: Sequence["IndType"],
        fronts: Sequence[Sequence["IndType"]],
        selected: Sequence["IndType"],
        origin_map: Dict[int, str],
    ) -> None:
        append_selection_pool_history(
            self.selection_pool_history_path,
            self._gen,
            pool,
            fronts,
            [getattr(ind, "uid", -1) for ind in selected],
            origin_map,
        )

    @staticmethod
    def _fitness_scalar(values: Sequence[float] | None) -> float:
        if values is None:
            return float("-inf")
        try:
            v0, v1, v2 = [float(v) for v in values]
        except Exception:
            return float("-inf")
        return float(v0 + v1 + v2)

    @staticmethod
    def _topology_signature_from_genome(genome_norm: Sequence[float]) -> str:
        try:
            phys = Chromosome_Drone.to_physical(genome_norm)
            return f"{int(phys[12])}{int(phys[13])}{int(phys[14]):02d}"
        except Exception:
            return ""

    def _init_founder_logging(self, ind: "IndType") -> None:
        uid = int(getattr(ind, "uid", -1))
        topology_signature = self._topology_signature_from_genome(ind)
        ind.lineage_id = uid
        ind.lineage_root_uid = uid
        ind.lineage_depth = 0
        ind.primary_parent_uid = -1
        ind.primary_parent_generation = -1
        ind.reproduction_operator = "initial"
        ind.crossover_applied = 0
        ind.mutation_applied = 0
        ind.mutation_changed_genome = 0
        ind.topology_mutation = 0
        ind.topology_mutation_magnitude = 0
        ind.topology_signature = topology_signature
        ind.parent_a_topology_signature = ""
        ind.parent_b_topology_signature = ""
        ind.primary_parent_topology_signature = ""
        ind.successful_topology_mutation = 0
        ind.beneficial_topology_event = 0
        ind.selected_next_generation = 1
        ind.parent_best_scalar_fitness = np.nan
        ind.offspring_scalar_fitness = np.nan
        ind.lineage_event = "founder"
        ind.parent_a_lineage_id = -1
        ind.parent_b_lineage_id = -1
        ind.cross_lineage_mating = 0
        ind.airfoil_signature = topology_signature
        ind.parent_a_airfoil_signature = ""
        ind.parent_b_airfoil_signature = ""
        ind.primary_parent_airfoil_signature = ""
        ind.airfoil_mutation = 0
        ind.successful_airfoil_mutation = 0
        ind.beneficial_airfoil_event = 0
        ind.offspring_vs_best_parent_scalar_delta = np.nan

    def _annotate_variation(
        self,
        child: "IndType",
        parent_a: Optional["IndType"],
        parent_b: Optional["IndType"],
        *,
        crossover_applied: bool,
        mutation_applied: bool,
        pre_variation_genome: Sequence[float],
    ) -> None:
        p_primary = parent_a if parent_a is not None else parent_b
        primary_uid = int(getattr(p_primary, "uid", -1)) if p_primary is not None else -1
        primary_gen = int(getattr(p_primary, "parent_gen_a", -1))
        if p_primary is not None:
            primary_gen = int(getattr(p_primary, "generation_origin", getattr(self, "_gen", 0) - 1))
        lineage_id = int(getattr(p_primary, "lineage_id", primary_uid if primary_uid >= 0 else getattr(child, "uid", -1)))
        lineage_root_uid = int(getattr(p_primary, "lineage_root_uid", primary_uid if primary_uid >= 0 else getattr(child, "uid", -1)))
        lineage_depth = int(getattr(p_primary, "lineage_depth", -1)) + 1 if p_primary is not None else 0
        parent_a_lineage_id = int(getattr(parent_a, "lineage_id", -1)) if parent_a is not None else -1
        parent_b_lineage_id = int(getattr(parent_b, "lineage_id", -1)) if parent_b is not None else -1

        child_topology = self._topology_signature_from_genome(child)
        parent_a_topology = self._topology_signature_from_genome(parent_a) if parent_a is not None else ""
        parent_b_topology = self._topology_signature_from_genome(parent_b) if parent_b is not None else ""
        primary_topology = parent_a_topology if parent_a is not None else parent_b_topology
        topology_magnitude = int(child_topology != primary_topology) if primary_topology else 0
        airfoil_mutation = int(child_topology != primary_topology) if primary_topology else 0
        mutation_changed_genome = int(
            np.max(np.abs(np.asarray(child, dtype=float) - np.asarray(pre_variation_genome, dtype=float))) > 1e-12
        )
        operator_parts = []
        if crossover_applied:
            operator_parts.append("crossover")
        if mutation_applied:
            operator_parts.append("mutation")
        if not operator_parts:
            operator_parts.append("clone")

        child.lineage_id = lineage_id
        child.lineage_root_uid = lineage_root_uid
        child.lineage_depth = lineage_depth
        child.primary_parent_uid = primary_uid
        child.primary_parent_generation = primary_gen
        child.reproduction_operator = "+".join(operator_parts)
        child.crossover_applied = int(crossover_applied)
        child.mutation_applied = int(mutation_applied)
        child.mutation_changed_genome = mutation_changed_genome
        child.topology_mutation = int(topology_magnitude > 0)
        child.topology_mutation_magnitude = topology_magnitude
        child.topology_signature = child_topology
        child.parent_a_topology_signature = parent_a_topology
        child.parent_b_topology_signature = parent_b_topology
        child.primary_parent_topology_signature = primary_topology
        child.successful_topology_mutation = 0
        child.beneficial_topology_event = 0
        child.selected_next_generation = 0
        best_parent_scalar = max(
            self._fitness_scalar(getattr(parent_a, "fitness", None).values if parent_a is not None and hasattr(parent_a, "fitness") else None),
            self._fitness_scalar(getattr(parent_b, "fitness", None).values if parent_b is not None and hasattr(parent_b, "fitness") else None),
        )
        child.parent_best_scalar_fitness = best_parent_scalar if np.isfinite(best_parent_scalar) else np.nan
        child.offspring_scalar_fitness = np.nan
        child.lineage_event = "topology_split" if int(topology_magnitude > 0) else "offspring"
        child.parent_a_lineage_id = parent_a_lineage_id
        child.parent_b_lineage_id = parent_b_lineage_id
        child.cross_lineage_mating = int(
            parent_a_lineage_id >= 0 and parent_b_lineage_id >= 0 and parent_a_lineage_id != parent_b_lineage_id
        )
        child.airfoil_signature = child_topology
        child.parent_a_airfoil_signature = parent_a_topology
        child.parent_b_airfoil_signature = parent_b_topology
        child.primary_parent_airfoil_signature = primary_topology
        child.airfoil_mutation = airfoil_mutation
        child.successful_airfoil_mutation = 0
        child.beneficial_airfoil_event = 0
        child.offspring_vs_best_parent_scalar_delta = np.nan

    def _annotate_selection_outcomes(
        self,
        pool: Sequence["IndType"],
        selected: Sequence["IndType"],
    ) -> None:
        selected_uids = {int(getattr(ind, "uid", -1)) for ind in selected}
        for ind in pool:
            uid = int(getattr(ind, "uid", -1))
            ind.selected_next_generation = int(uid in selected_uids)
            ind.offspring_scalar_fitness = self._fitness_scalar(getattr(ind.fitness, "values", None))
            parent_best = float(getattr(ind, "parent_best_scalar_fitness", np.nan))
            offspring_scalar = float(getattr(ind, "offspring_scalar_fitness", np.nan))
            ind.offspring_vs_best_parent_scalar_delta = (
                offspring_scalar - parent_best if np.isfinite(parent_best) and np.isfinite(offspring_scalar) else np.nan
            )
            beneficial = (
                int(getattr(ind, "topology_mutation", 0)) == 1
                and int(getattr(ind, "selected_next_generation", 0)) == 1
                and np.isfinite(float(getattr(ind, "parent_best_scalar_fitness", np.nan)))
                and float(getattr(ind, "offspring_scalar_fitness", np.nan))
                > float(getattr(ind, "parent_best_scalar_fitness", np.nan))
            )
            beneficial_airfoil = (
                int(getattr(ind, "airfoil_mutation", 0)) == 1
                and int(getattr(ind, "selected_next_generation", 0)) == 1
                and np.isfinite(parent_best)
                and np.isfinite(offspring_scalar)
                and offspring_scalar > parent_best
            )
            ind.successful_topology_mutation = int(
                int(getattr(ind, "topology_mutation", 0)) == 1 and int(getattr(ind, "selected_next_generation", 0)) == 1
            )
            ind.beneficial_topology_event = int(bool(beneficial))
            ind.successful_airfoil_mutation = int(
                int(getattr(ind, "airfoil_mutation", 0)) == 1 and int(getattr(ind, "selected_next_generation", 0)) == 1
            )
            ind.beneficial_airfoil_event = int(bool(beneficial_airfoil))
            ind.generation_origin = int(getattr(self, "_gen", 0))

    @staticmethod
    def _logging_meta(ind: "IndType") -> Dict[str, Any]:
        return dict(
            lineage_id=int(getattr(ind, "lineage_id", -1)),
            lineage_root_uid=int(getattr(ind, "lineage_root_uid", -1)),
            lineage_depth=int(getattr(ind, "lineage_depth", -1)),
            primary_parent_uid=int(getattr(ind, "primary_parent_uid", -1)),
            primary_parent_generation=int(getattr(ind, "primary_parent_generation", -1)),
            reproduction_operator=str(getattr(ind, "reproduction_operator", "")),
            crossover_applied=int(getattr(ind, "crossover_applied", 0)),
            mutation_applied=int(getattr(ind, "mutation_applied", 0)),
            mutation_changed_genome=int(getattr(ind, "mutation_changed_genome", 0)),
            topology_mutation=int(getattr(ind, "topology_mutation", 0)),
            topology_mutation_magnitude=int(getattr(ind, "topology_mutation_magnitude", 0)),
            topology_signature=str(getattr(ind, "topology_signature", "")),
            parent_a_topology_signature=str(getattr(ind, "parent_a_topology_signature", "")),
            parent_b_topology_signature=str(getattr(ind, "parent_b_topology_signature", "")),
            primary_parent_topology_signature=str(getattr(ind, "primary_parent_topology_signature", "")),
            successful_topology_mutation=int(getattr(ind, "successful_topology_mutation", 0)),
            beneficial_topology_event=int(getattr(ind, "beneficial_topology_event", 0)),
            selected_next_generation=int(getattr(ind, "selected_next_generation", 0)),
            parent_best_scalar_fitness=float(getattr(ind, "parent_best_scalar_fitness", np.nan)),
            offspring_scalar_fitness=float(getattr(ind, "offspring_scalar_fitness", np.nan)),
            lineage_event=str(getattr(ind, "lineage_event", "")),
            parent_a_lineage_id=int(getattr(ind, "parent_a_lineage_id", -1)),
            parent_b_lineage_id=int(getattr(ind, "parent_b_lineage_id", -1)),
            cross_lineage_mating=int(getattr(ind, "cross_lineage_mating", 0)),
            airfoil_signature=str(getattr(ind, "airfoil_signature", "")),
            parent_a_airfoil_signature=str(getattr(ind, "parent_a_airfoil_signature", "")),
            parent_b_airfoil_signature=str(getattr(ind, "parent_b_airfoil_signature", "")),
            primary_parent_airfoil_signature=str(getattr(ind, "primary_parent_airfoil_signature", "")),
            airfoil_mutation=int(getattr(ind, "airfoil_mutation", 0)),
            successful_airfoil_mutation=int(getattr(ind, "successful_airfoil_mutation", 0)),
            beneficial_airfoil_event=int(getattr(ind, "beneficial_airfoil_event", 0)),
            offspring_vs_best_parent_scalar_delta=float(getattr(ind, "offspring_vs_best_parent_scalar_delta", np.nan)),
        )

    # ------------------------------------------------------------------ #
    # Fitness evaluation                                                 #
    # ------------------------------------------------------------------ #

    def _assign_uid(self, ind: "IndType") -> int:
        """Assign a new globally unique UID to an individual."""
        self._uid_counter += 1
        ind.uid = int(self._uid_counter)
        return ind.uid

    def _ensure_uids(self, population: Sequence["IndType"]) -> None:
        """Ensure all individuals have a UID; assign if missing or invalid."""
        max_uid = int(self._uid_counter)
        for ind in population:
            uid = getattr(ind, "uid", None)
            if uid is None:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            try:
                uid_val = int(uid)
            except Exception:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            if uid_val < 0:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            max_uid = max(max_uid, uid_val)
        if max_uid > self._uid_counter:
            self._uid_counter = max_uid

    def _evaluate(self, indiv: "IndType") -> Tuple[float, float, float]:
        """
        DEAP evaluation hook – possibly spawns Ray jobs.

        The individual is a list of floats in [0, 1] (normalized genome).
        """
        chromo = list(indiv)
        mode = "gen_policy eval-only" if self.gen_policy else "train+eval"
        if not hasattr(indiv, "uid") or getattr(indiv, "uid") is None:
            self._assign_uid(indiv)
        print(
            f"[evaluate] gen={getattr(self, '_gen', 0)} mode={mode} "
            f"uid={getattr(indiv, 'uid', -1)} "
            f"parent_uid=({getattr(indiv, 'parent_uid_a', -1)}, "
            f"{getattr(indiv, 'parent_uid_b', -1)}) "
            f"chr={chromo}"
        )

        if self.gen_policy and not self.policy_paths:
            raise ValueError("gen_policy requires --policy_path or --policy_paths")
        if self.gen_policy and self.cfg.train_repetition > 1:
            print(
                "   ↪ GEN_POLICY active → train_repetition ignored "
                f"(cfg={self.cfg.train_repetition})"
            )
        if self.gen_policy and len(self.policy_paths) > 1:
            print(
                "   ↪ GEN_POLICY active → averaging over "
                f"{len(self.policy_paths)} provided policies"
            )
        try:
            cfg_reps = int(self.cfg.train_repetition)
        except Exception:
            cfg_reps = 1
        if cfg_reps < 1:
            print("[evaluate][warn] train_repetition < 1; forcing to 1")
            cfg_reps = 1

        # CSV cache: if we have seen this chromosome before, reuse its fitness.
        if not self.gen_policy and self.cfg.use_dynamic_p:
            print("   ↪ cache bypassed (dynamic minimal_p enabled)")
        elif not self.gen_policy:
            cached_row = self.db.get_row(chromo)
            if cached_row is not None:
                cached_rep = cached_row.get("train_repetition", 1)
                try:
                    cached_rep = int(cached_rep)
                except Exception:
                    cached_rep = 1
                if cached_rep != cfg_reps:
                    print(
                        "   ↪ cache-hit skipped (train_repetition mismatch: "
                        f"cached={cached_rep} cfg={cfg_reps})"
                    )
                else:
                    ff_cached = [cached_row[f"ff_{i}"] for i in range(3)]
                    indiv.fitness.values = tuple(ff_cached)
                    indiv.max_p = cached_row.get("max_p", np.nan)
                    indiv.exp_name = cached_row.get("exp_name", None)
                    indiv.train_it = cached_row.get("train_it", self.cfg.train_iters_new)
                    indiv.failed = bool(cached_row.get("failed", False))
                    indiv.fail_reason = str(cached_row.get("fail_reason", "") or "")
                    indiv.fail_category = str(cached_row.get("fail_category", "") or "")
                    indiv._failed = bool(cached_row.get("failed", False))
                    indiv.cache_hit = True
                    indiv.evaluated_fresh = False
                    indiv.cache_source_uid = int(cached_row.get("uid", -1))
                    indiv.cache_source_generation = int(cached_row.get("generation", -1))
                    indiv._meta_raw = {
                        "exp_name": cached_row.get("exp_name", None),
                        "train_it": cached_row.get("train_it", self.cfg.train_iters_new),
                        "train_repetition": cached_row.get("train_repetition", 1),
                        "rep_exp_names": cached_row.get("rep_exp_names", ""),
                        "max_p": cached_row.get("max_p", np.nan),
                        "failed": bool(cached_row.get("failed", False)),
                        "fail_reason": str(cached_row.get("fail_reason", "") or ""),
                        "fail_category": str(cached_row.get("fail_category", "") or ""),
                        "eval_reward_mean": cached_row.get("eval_reward_mean", 0.0),
                        "train_duration_s": 0.0,
                        "eval_duration_s": 0.0,
                        "vel_v": cached_row.get("vel_v", np.nan),
                        "vel_E": cached_row.get("vel_E", np.nan),
                        "vel_P": cached_row.get("vel_P", np.nan),
                        "eff_v": cached_row.get("eff_v", np.nan),
                        "eff_E": cached_row.get("eff_E", np.nan),
                        "eff_P": cached_row.get("eff_P", np.nan),
                        "prog_v": cached_row.get("prog_v", np.nan),
                        "prog_E": cached_row.get("prog_E", np.nan),
                        "prog_P": cached_row.get("prog_P", np.nan),
                        "final_reward": cached_row.get("final_reward", np.nan),
                        "steps90_pct": cached_row.get("steps90_pct", np.nan),
                        "cache_hit": 1,
                        "evaluated_fresh": 0,
                        "cache_source_uid": int(cached_row.get("uid", -1)),
                        "cache_source_generation": int(cached_row.get("generation", -1)),
                        "lineage_id": cached_row.get("lineage_id", np.nan),
                        "lineage_root_uid": cached_row.get("lineage_root_uid", np.nan),
                        "lineage_depth": cached_row.get("lineage_depth", np.nan),
                        "primary_parent_uid": cached_row.get("primary_parent_uid", np.nan),
                        "primary_parent_generation": cached_row.get("primary_parent_generation", np.nan),
                        "reproduction_operator": cached_row.get("reproduction_operator", ""),
                        "crossover_applied": cached_row.get("crossover_applied", np.nan),
                        "mutation_applied": cached_row.get("mutation_applied", np.nan),
                        "mutation_changed_genome": cached_row.get("mutation_changed_genome", np.nan),
                        "topology_mutation": cached_row.get("topology_mutation", np.nan),
                        "topology_mutation_magnitude": cached_row.get("topology_mutation_magnitude", np.nan),
                        "topology_signature": cached_row.get("topology_signature", ""),
                        "parent_a_topology_signature": cached_row.get("parent_a_topology_signature", ""),
                        "parent_b_topology_signature": cached_row.get("parent_b_topology_signature", ""),
                        "primary_parent_topology_signature": cached_row.get("primary_parent_topology_signature", ""),
                        "successful_topology_mutation": cached_row.get("successful_topology_mutation", np.nan),
                        "beneficial_topology_event": cached_row.get("beneficial_topology_event", np.nan),
                        "selected_next_generation": cached_row.get("selected_next_generation", np.nan),
                        "parent_best_scalar_fitness": cached_row.get("parent_best_scalar_fitness", np.nan),
                        "offspring_scalar_fitness": cached_row.get("offspring_scalar_fitness", np.nan),
                        "lineage_event": cached_row.get("lineage_event", ""),
                        "parent_a_lineage_id": cached_row.get("parent_a_lineage_id", np.nan),
                        "parent_b_lineage_id": cached_row.get("parent_b_lineage_id", np.nan),
                        "cross_lineage_mating": cached_row.get("cross_lineage_mating", np.nan),
                        "airfoil_signature": cached_row.get("airfoil_signature", ""),
                        "parent_a_airfoil_signature": cached_row.get("parent_a_airfoil_signature", ""),
                        "parent_b_airfoil_signature": cached_row.get("parent_b_airfoil_signature", ""),
                        "primary_parent_airfoil_signature": cached_row.get("primary_parent_airfoil_signature", ""),
                        "airfoil_mutation": cached_row.get("airfoil_mutation", np.nan),
                        "successful_airfoil_mutation": cached_row.get("successful_airfoil_mutation", np.nan),
                        "beneficial_airfoil_event": cached_row.get("beneficial_airfoil_event", np.nan),
                        "offspring_vs_best_parent_scalar_delta": cached_row.get("offspring_vs_best_parent_scalar_delta", np.nan),
                    }
                    print(
                        f"   ↪ cache-hit uid={getattr(indiv, 'uid', -1)} "
                        f"parent_uid=({getattr(indiv, 'parent_uid_a', -1)}, "
                        f"{getattr(indiv, 'parent_uid_b', -1)}) "
                        f"exp={cached_row.get('exp_name', 'NA')} "
                        f"ff={ff_cached}"
                    )
                    return tuple(ff_cached)
        else:
            print("   ↪ GEN_POLICY → cache bypassed")

        # New chromosome → full train + eval pipeline (or eval-only when gen_policy)
        if self.gen_policy:
            print("   ↪ NEW chromosome → eval-only (gen policy, no training)")
        else:
            print(
                "   ↪ NEW chromosome → training for "
                f"{self.cfg.train_iters_new} iterations "
                f"(or {self.cfg.train_iters_inherit} if inheritance is triggered)."
            )

        parent_info = (
            getattr(indiv, "parent_exp", None),
            getattr(indiv, "parent_ckpt", None),
        )
        urdf_dir = _resolve_urdf_dir(self.urdf_dir)
        cfg = dict(
            TRAIN_ITERS=self.cfg.train_iters_new,
            TRAIN_ITERS_INHERIT=self.cfg.train_iters_inherit,
            TRAIN_REPETITION=cfg_reps,
            TRAIN_ENVS=self.cfg.train_envs,
            EVAL_ENVS=self.cfg.eval_envs,
            VMIN=self.cfg.vmin,
            VMAX=self.cfg.vmax,
            LOG_ROOT=str(self.log_root),
            EXP_PREFIX=self.exp_prefix,
            DEVICE=self.cfg.device,
            BASE_DIR=str(self.base_dir),
            URDF_DIR=str(urdf_dir),
            LOGS_DIR=str(self.logs_dir),
            GENERATION=getattr(self, "_gen", 0),
            IND_UID=getattr(indiv, "uid", -1),
        )

        if USE_PARALLEL:
            if self.gen_policy:
                print("   ↪ Ray eval-only job launched")
                fut = eval_only_remote.remote(chromo, self.policy_paths, self.tag, cfg, True)
            else:
                fut = train_and_eval_remote.remote(chromo, parent_info, self.tag, cfg, True)
            indiv._pending_future = fut
            # Placeholder; real fitness will be set after Ray returns.
            return (0.0, 0.0, 0.0)

        if self.gen_policy:
            print(
                "   ↪ GEN_POLICY active → skipping training "
                f"(policies={len(self.policy_paths)})"
            )
            
            ff, meta, extra = _eval_only_custom(
                chromo,
                self.policy_paths,
                self.tag,
                cfg,
                return_arrays=True,
            )
            print(f"   ✔ sync-eval ff={ff} max_p={meta['max_p']:.2f}")

            indiv._meta_raw = meta
            if extra and "rep_payloads" in extra:
                indiv._rep_payloads = extra["rep_payloads"]
                indiv._failed = False
            else:
                indiv._failed = bool(meta.get("failed"))
            indiv.max_p = meta["max_p"]
            indiv.exp_name = meta["exp_name"]
            indiv.train_it = meta["train_it"]
            indiv.cache_hit = False
            indiv.evaluated_fresh = True
            indiv.cache_source_uid = -1
            indiv.cache_source_generation = -1
            
            if extra and "rep_payloads" not in extra:
                indiv._p_s = extra["p_s"]
                indiv._v_s = extra["v_s"]
                indiv._E_s = extra["E_s"]

            indiv.fitness.values = tuple(ff)
            return tuple(ff)

        ff, meta, extra = _train_and_eval_sync(chromo, parent_info, self.tag, cfg, True)
        print(f"   ✔ sync-train+eval ff={ff} max_p={meta['max_p']:.2f}")

        indiv._meta_raw = meta
        if extra and "rep_payloads" in extra:
            indiv._rep_payloads = extra["rep_payloads"]
            indiv._failed = False
        else:
            indiv._failed = bool(meta.get("failed"))
        indiv.max_p = meta["max_p"]
        indiv.exp_name = meta["exp_name"]
        indiv.train_it = meta["train_it"]
        indiv.cache_hit = False
        indiv.evaluated_fresh = True
        indiv.cache_source_uid = -1
        indiv.cache_source_generation = -1
        if extra and "rep_payloads" not in extra:
            indiv._p_s = extra["p_s"]
            indiv._v_s = extra["v_s"]
            indiv._E_s = extra["E_s"]

        indiv.fitness.values = tuple(ff)
        return tuple(ff)

    # ------------------------------------------------------------------ #
    # Fitness post-processing (progress threshold)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_triples(
        p_s: np.ndarray,
        v_s: np.ndarray,
        E_s: np.ndarray,
        minimal_p: float,
    ):
        """
        Given smoothed progress (p_s), smoothed mean velocities (v_s),
        and smoothed energy (E_s), aligned on v_cmd,
        extract three representative operating points:

          - vel: max velocity with p >= minimal_p
          - eff: min energy with p >= minimal_p
          - prog: global max progress (unfiltered)
        """
        if len(p_s) == 0:
            zero = dict(mean_v=0.0, mean_E=10.0, mean_progress=0.0)
            return zero, zero, zero

        idx_p = int(np.argmax(p_s))
        mask = np.where(p_s >= minimal_p)[0]

        if mask.size == 0:
            vel = dict(mean_v=0.0, mean_E=10.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=10.0, mean_progress=0.0)
        else:
            idx_v = int(mask[np.argmax(v_s[mask])])
            idx_e = int(mask[np.argmin(E_s[mask])])
            vel = dict(
                mean_v=float(v_s[idx_v]),
                mean_E=float(E_s[idx_v]),
                mean_progress=float(p_s[idx_v]),
            )
            eff = dict(
                mean_v=float(v_s[idx_e]),
                mean_E=float(E_s[idx_e]),
                mean_progress=float(p_s[idx_e]),
            )

        prog = dict(
            mean_v=float(v_s[idx_p]),
            mean_E=float(E_s[idx_p]),
            mean_progress=float(p_s[idx_p]),
        )
        return vel, eff, prog

    def _finalize_and_persist(self, ind: "IndType", minimal_p: float) -> None:
        """
        Compute final fitness for an individual, apply the progress threshold,
        and write a row in the DB.
        """
        uid = getattr(ind, "uid", -1)
        parent_idx_a = getattr(ind, "parent_idx_a", -1)
        parent_idx_b = getattr(ind, "parent_idx_b", -1)
        parent_uid_a = getattr(ind, "parent_uid_a", -1)
        parent_uid_b = getattr(ind, "parent_uid_b", -1)
        parent_gen_a = getattr(ind, "parent_gen_a", -1)
        parent_gen_b = getattr(ind, "parent_gen_b", -1)
        if hasattr(ind, "_rep_payloads"):
            rep_payloads = list(getattr(ind, "_rep_payloads", []))
            if not rep_payloads:
                return

            def _avg_adjust(val: Any, invalid: set[float], replacement: float) -> Any:
                try:
                    val_f = float(val)
                except Exception:
                    return val
                return replacement if val_f in invalid else val_f

            def _accum(acc: Dict[str, List[float]], key: str, val: Any) -> None:
                try:
                    val_f = float(val)
                except Exception:
                    return
                acc.setdefault(key, []).append(val_f)

            rep_ff: List[Tuple[float, float, float]] = []
            rep_exp_names: List[str] = []
            acc: Dict[str, List[float]] = {}
            any_failed = False
            failed_count = 0
            train_repetition = len(rep_payloads)
            avg_failed_ff = (
                float(min(INVALID_V)),
                float(min(INVALID_E)),
                float(min(INVALID_P)),
            )
            avg_all_invalid = False

            for payload in rep_payloads:
                rep_meta = dict(payload.get("meta", {}))
                rep_extra = payload.get("extra", {})
                rep_idx = payload.get("rep_idx", None)
                rep_exp_name = rep_meta.get("exp_name", None)
                if rep_exp_name:
                    rep_exp_names.append(rep_exp_name)

                rep_failed = bool(rep_meta.get("failed"))
                if not rep_failed:
                    payload_path = rep_extra.get("payload_path")
                    if payload_path:
                        try:
                            payload_data = _load_rep_payload(payload_path)
                            p_s = np.asarray(payload_data.get("p_s", []))
                            v_s = np.asarray(payload_data.get("v_s", []))
                            E_s = np.asarray(payload_data.get("E_s", []))
                        except Exception as exc:
                            rep_failed = True
                            rep_meta["failed"] = True
                            rep_meta["fail_reason"] = f"payload_load_failed: {exc}"
                            p_s = np.array([])
                            v_s = np.array([])
                            E_s = np.array([])
                    else:
                        p_s = np.asarray(rep_extra.get("p_s", []))
                        v_s = np.asarray(rep_extra.get("v_s", []))
                        E_s = np.asarray(rep_extra.get("E_s", []))
                    if p_s.size == 0 or v_s.size == 0 or E_s.size == 0:
                        rep_failed = True
                        rep_meta["failed"] = True
                        rep_meta["fail_reason"] = rep_meta.get("fail_reason", "empty_eval_arrays")
                    elif not (
                        np.isfinite(p_s).all()
                        and np.isfinite(v_s).all()
                        and np.isfinite(E_s).all()
                    ):
                        rep_failed = True
                        rep_meta["failed"] = True
                        rep_meta["fail_reason"] = rep_meta.get(
                            "fail_reason", "non_finite_eval_arrays"
                        )

                if rep_failed:
                    fail_category, fail_reason = _parse_fail_reason(
                        rep_meta.get("fail_reason", "rep_failed")
                    )
                    rep_meta["fail_category"] = fail_category
                    rep_meta["fail_reason"] = fail_reason
                    ff_rep = tuple(_default_fitness())
                    ff_rep_avg = avg_failed_ff
                    any_failed = True
                    failed_count += 1
                    rep_meta.update(
                        dict(
                            vel_v=float(min(INVALID_V)),
                            vel_E=float(min(INVALID_E)),
                            vel_P=float(min(INVALID_P)),
                            eff_v=float(min(INVALID_V)),
                            eff_E=float(min(INVALID_E)),
                            eff_P=float(min(INVALID_P)),
                            prog_v=float(min(INVALID_V)),
                            prog_E=float(min(INVALID_E)),
                            prog_P=float(min(INVALID_P)),
                        )
                    )
                else:
                    vel_d, eff_d, prog_d = self._pick_triples(
                        p_s,
                        v_s,
                        E_s,
                        minimal_p,
                    )
                    ff_rep = (
                        vel_d["mean_v"],
                        -eff_d["mean_E"],
                        prog_d["mean_progress"],
                    )
                    ff_rep_avg = ff_rep
                    rep_meta.update(
                        dict(
                            vel_v=vel_d["mean_v"],
                            vel_E=-vel_d["mean_E"],
                            vel_P=vel_d["mean_progress"],
                            eff_v=eff_d["mean_v"],
                            eff_E=-eff_d["mean_E"],
                            eff_P=eff_d["mean_progress"],
                            prog_v=prog_d["mean_v"],
                            prog_E=-prog_d["mean_E"],
                            prog_P=prog_d["mean_progress"],
                        )
                    )

                rep_meta.update(
                    dict(
                        max_p=rep_meta.get("max_p", np.nan),
                        minimal_p=minimal_p,
                        uid=uid,
                        parent_idx_a=parent_idx_a,
                        parent_idx_b=parent_idx_b,
                        parent_uid_a=parent_uid_a,
                        parent_uid_b=parent_uid_b,
                        parent_gen_a=parent_gen_a,
                        parent_gen_b=parent_gen_b,
                        row_kind="rep",
                        rep_idx=rep_idx,
                        train_repetition=train_repetition,
                        rep_exp_names=rep_meta.get("exp_name", ""),
                        cache_hit=0,
                        evaluated_fresh=1,
                        cache_source_uid=-1,
                        cache_source_generation=-1,
                    )
                )
                rep_meta.update(self._logging_meta(ind))
                self.db.insert(list(ind), ff_rep, dict(generation=self._gen, **rep_meta))
                rep_ff.append(ff_rep_avg)

                for key in (
                    "train_duration_s",
                    "eval_duration_s",
                    "vel_v",
                    "vel_E",
                    "vel_P",
                    "eff_v",
                    "eff_E",
                    "eff_P",
                    "prog_v",
                    "prog_E",
                    "prog_P",
                    "max_p",
                    "final_reward",
                    "steps90_pct",
                    "eval_reward_mean",
                ):
                    if key in rep_meta:
                        if key in ("vel_v", "eff_v", "prog_v"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_V, 0.0))
                        elif key in ("vel_E", "eff_E", "prog_E"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_E, float(min(INVALID_E))))
                        elif key in ("vel_P", "eff_P", "prog_P"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_P, float(min(INVALID_P))))
                        else:
                            _accum(acc, key, rep_meta[key])
                for i in range(1, 11):
                    key = f"rew_{i * 10}pct"
                    if key in rep_meta:
                        _accum(acc, key, rep_meta[key])

            avg_all_invalid = train_repetition > 0 and failed_count == train_repetition
            if rep_ff and not avg_all_invalid:
                ff_final = tuple(np.nanmean(np.asarray(rep_ff, dtype=float), axis=0))
            else:
                ff_final = tuple(_default_fitness())

            meta = dict(getattr(ind, "_meta_raw", {}))
            meta.update(
                dict(
                    max_p=getattr(ind, "max_p", np.nan),
                    minimal_p=minimal_p,
                    ckpt_idx=_ckpt_idx_from_train_it(
                        meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))
                    ),
                    uid=uid,
                    parent_idx_a=parent_idx_a,
                    parent_idx_b=parent_idx_b,
                    parent_uid_a=parent_uid_a,
                    parent_uid_b=parent_uid_b,
                    parent_gen_a=parent_gen_a,
                    parent_gen_b=parent_gen_b,
                    row_kind="agg",
                    rep_idx=-1,
                    train_repetition=train_repetition,
                    rep_exp_names="|".join(rep_exp_names),
                    cache_hit=int(bool(getattr(ind, "cache_hit", False))),
                    evaluated_fresh=int(bool(getattr(ind, "evaluated_fresh", True))),
                    cache_source_uid=int(getattr(ind, "cache_source_uid", -1)),
                    cache_source_generation=int(getattr(ind, "cache_source_generation", -1)),
                )
            )
            meta.update(self._logging_meta(ind))
            for key, vals in acc.items():
                if vals:
                    meta[key] = float(np.nanmean(vals))
            if avg_all_invalid:
                meta.update(
                    dict(
                        vel_v=float(min(INVALID_V)),
                        vel_E=float(min(INVALID_E)),
                        vel_P=float(min(INVALID_P)),
                        eff_v=float(min(INVALID_V)),
                        eff_E=float(min(INVALID_E)),
                        eff_P=float(min(INVALID_P)),
                        prog_v=float(min(INVALID_V)),
                        prog_E=float(min(INVALID_E)),
                        prog_P=float(min(INVALID_P)),
                    )
                )
            if any_failed:
                meta["failed"] = True
                meta["fail_reason"] = "rep_failed"
                meta["fail_category"] = "rep_failed"

            ind.exp_name = meta.get("exp_name", getattr(ind, "exp_name", None))
            ind.train_it = meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))

            eff_e_pos = -float(meta.get("eff_E", 0.0))
            print(
                f"[finalize][rep-avg] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"vel={float(meta.get('vel_v', 0.0)):.2f} "
                f"effE={eff_e_pos:.2f} "
                f"prog={float(meta.get('prog_P', 0.0)):.2f} "
                f"reps={train_repetition} failed={failed_count}"
            )

            self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
            ind.failed = bool(meta.get("failed", False))
            ind.fail_reason = str(meta.get("fail_reason", ""))
            ind.fail_category = str(meta.get("fail_category", ""))
            ind.fitness.values = ff_final
            return

        if getattr(ind, "cache_hit", False):
            ff_final = tuple(ind.fitness.values)
            meta = dict(getattr(ind, "_meta_raw", {}))
            meta.update(
                dict(
                    max_p=getattr(ind, "max_p", np.nan),
                    minimal_p=minimal_p,
                    ckpt_idx=_ckpt_idx_from_train_it(
                        meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))
                    ),
                    uid=uid,
                    parent_idx_a=parent_idx_a,
                    parent_idx_b=parent_idx_b,
                    parent_uid_a=parent_uid_a,
                    parent_uid_b=parent_uid_b,
                    parent_gen_a=parent_gen_a,
                    parent_gen_b=parent_gen_b,
                    row_kind="agg",
                    rep_idx=-1,
                    train_repetition=int(meta.get("train_repetition", 1) or 1),
                    rep_exp_names=meta.get("rep_exp_names", meta.get("exp_name", "")),
                    cache_hit=1,
                    evaluated_fresh=0,
                    cache_source_uid=int(getattr(ind, "cache_source_uid", -1)),
                    cache_source_generation=int(getattr(ind, "cache_source_generation", -1)),
                )
            )
            meta.update(self._logging_meta(ind))
            self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
            ind.failed = bool(meta.get("failed", False))
            ind.fail_reason = str(meta.get("fail_reason", ""))
            ind.fail_category = str(meta.get("fail_category", ""))
            ind.fitness.values = ff_final
            return

        if not hasattr(ind, "_p_s") and not getattr(ind, "_failed", False):
            return

        meta = dict(getattr(ind, "_meta_raw", {}))
        meta.update(
            dict(
                max_p=getattr(ind, "max_p", np.nan),
                minimal_p=minimal_p,
                ckpt_idx=_ckpt_idx_from_train_it(
                    meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))
                ),
                uid=uid,
                parent_idx_a=parent_idx_a,
                parent_idx_b=parent_idx_b,
                parent_uid_a=parent_uid_a,
                parent_uid_b=parent_uid_b,
                parent_gen_a=parent_gen_a,
                parent_gen_b=parent_gen_b,
                row_kind="agg",
                rep_idx=-1,
                train_repetition=int(meta.get("train_repetition", 1) or 1),
                rep_exp_names=meta.get("rep_exp_names", meta.get("exp_name", "")),
                cache_hit=int(bool(getattr(ind, "cache_hit", False))),
                evaluated_fresh=int(bool(getattr(ind, "evaluated_fresh", True))),
                cache_source_uid=int(getattr(ind, "cache_source_uid", -1)),
                cache_source_generation=int(getattr(ind, "cache_source_generation", -1)),
            )
        )
        meta.update(self._logging_meta(ind))

        if getattr(ind, "_failed", False):
            ff_final = tuple(_default_fitness())
            fail_category, fail_reason = _parse_fail_reason(meta.get("fail_reason", "unknown"))
            meta["fail_category"] = fail_category
            meta["fail_reason"] = fail_reason
        else:
            vel_d, eff_d, prog_d = self._pick_triples(
                np.asarray(ind._p_s),
                np.asarray(ind._v_s),
                np.asarray(ind._E_s),
                minimal_p,
            )

            ff_final = (
                vel_d["mean_v"],
                -eff_d["mean_E"],
                prog_d["mean_progress"],
            )

            meta.update(
                dict(
                    vel_v=vel_d["mean_v"],
                    vel_E=-vel_d["mean_E"],
                    vel_P=vel_d["mean_progress"],
                    eff_v=eff_d["mean_v"],
                    eff_E=-eff_d["mean_E"],
                    eff_P=eff_d["mean_progress"],
                    prog_v=prog_d["mean_v"],
                    prog_E=-prog_d["mean_E"],
                    prog_P=prog_d["mean_progress"],
                )
            )

        ind.exp_name = meta.get("exp_name", getattr(ind, "exp_name", None))
        ind.train_it = meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))

        if getattr(ind, "_failed", False):
            print(
                f"[finalize][failed] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"reason={meta.get('fail_reason', 'unknown')} ff={ff_final}"
            )
        else:
            print(
                f"[finalize] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"vel={vel_d['mean_v']:.2f} effE={eff_d['mean_E']:.2f} "
                f"prog={prog_d['mean_progress']:.2f}"
            )

        self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
        ind.failed = bool(meta.get("failed", False))
        ind.fail_reason = str(meta.get("fail_reason", ""))
        ind.fail_category = str(meta.get("fail_category", ""))
        ind.fitness.values = ff_final

    # ------------------------------------------------------------------ #
    # Evolution helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cleanup_individual_payloads(ind: "IndType") -> None:
        """
        Drop heavy per-individual payloads from memory.

        Persisted evaluation payload CSVs live in the eval folders and are
        intentionally kept on disk for later inspection.
        """
        if hasattr(ind, "_rep_payloads"):
            try:
                delattr(ind, "_rep_payloads")
            except Exception:
                pass

        for attr in ("_p_s", "_v_s", "_E_s", "_meta_raw"):
            if hasattr(ind, attr):
                try:
                    delattr(ind, attr)
                except Exception:
                    pass

    def _train_eval_population(self, population: List["IndType"]) -> None:
        """
        Train + evaluate all individuals with invalid fitness.
        Handles:
          - launching Ray jobs,
          - waiting for results,
          - computing minimal_p,
          - finalizing and logging fitness values.
        """
        minimal_p_fixed = float(self.cfg.fixed_p)
        minimal_p_known = not self.cfg.use_dynamic_p

        # 1) Launch training/evaluation where needed
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.tb.evaluate(ind)
                if not USE_PARALLEL and minimal_p_known:
                    self._finalize_and_persist(ind, minimal_p_fixed)
                    ind._persisted = True
                    self._cleanup_individual_payloads(ind)

        # 2) Wait for Ray jobs
        if USE_PARALLEL:
            pend = [ind for ind in population if hasattr(ind, "_pending_future")]
            if pend:
                print(f"  ⏳ waiting for {len(pend)} Ray jobs…")
                fail_cfg = dict(
                    TRAIN_ITERS=self.cfg.train_iters_new,
                    TRAIN_REPETITION=self.cfg.train_repetition,
                )
                for ind in pend:
                    try:
                        ff, meta, extra = ray.get(ind._pending_future)
                    except Exception as exc:
                        reason = f"ray_job_failed: {exc}"
                        print(f"[safe_mode] {reason}")
                        ff, meta, extra = _failure_result(reason, fail_cfg)

                    ind._meta_raw = meta
                    if extra and "rep_payloads" in extra:
                        ind._rep_payloads = extra["rep_payloads"]
                        ind._failed = False
                    else:
                        ind._failed = bool(meta.get("failed"))
                    ind.max_p = meta.get("max_p", np.nan)
                    ind.exp_name = meta.get("exp_name", None)
                    ind.train_it = meta.get("train_it", self.cfg.train_iters_new)
                    ind.cache_hit = False
                    ind.evaluated_fresh = True
                    ind.cache_source_uid = -1
                    ind.cache_source_generation = -1
                    if extra and "rep_payloads" not in extra:
                        ind._p_s = extra.get("p_s", np.array([]))
                        ind._v_s = extra.get("v_s", np.array([]))
                        ind._E_s = extra.get("E_s", np.array([]))
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(
                        f"   ✅ Ray done uid={getattr(ind, 'uid', -1)} "
                        f"parent_uid=({getattr(ind, 'parent_uid_a', -1)}, "
                        f"{getattr(ind, 'parent_uid_b', -1)}) "
                        f"chr={list(ind)} "
                        f"ff={ff} max_p={ind.max_p:.2f}"
                    )

                    # When minimal_p is fixed, we can finalize immediately and
                    # drop all repetition payloads to keep memory bounded.
                    if minimal_p_known:
                        self._finalize_and_persist(ind, minimal_p_fixed)
                        ind._persisted = True
                        self._cleanup_individual_payloads(ind)

        # 3) minimal_p dynamic/fixed
        peaks: List[float] = []
        if self.cfg.use_dynamic_p:
            peaks = [getattr(ind, "max_p", np.nan) for ind in population]
            peaks = [p for p in peaks if not np.isnan(p)]
        if self.cfg.use_dynamic_p and peaks:
            perc = 100.0 - self.cfg.pct_above
            minimal_p = 0.9 * np.percentile(peaks, perc)
        else:
            minimal_p = self.cfg.fixed_p
        self._last_minimal_p = float(minimal_p)
        print(
            f"[Gen {self._gen}] minimal_p = {minimal_p:.2f} "
            f"(dynamic={self.cfg.use_dynamic_p}, pct_above={self.cfg.pct_above}%)"
        )

        # 4) finalize → CSV
        for ind in population:
            if not hasattr(ind, "_persisted"):
                self._finalize_and_persist(ind, minimal_p)
                ind._persisted = True
            # Regardless of dynamic/fixed minimal_p, once persisted we no
            # longer need per-repetition payloads in memory or on disk.
            if hasattr(ind, "_persisted"):
                self._cleanup_individual_payloads(ind)

    def _apply_variation(
        self,
        offspring: List["IndType"],
        parents: List["IndType"],
        parent_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Crossover, mutation, and optional inheritance **before** training.
        """
        # Clean up custom attributes on offspring
        for ch in offspring:
            for a in (
                "exp_name",
                "parent_exp",
                "parent_ckpt",
                "parent_idx_a",
                "parent_idx_b",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "_pending_future",
                "_meta_raw",
                "_p_s",
                "_v_s",
                "_E_s",
                "_evaluated",
                "_persisted",
                "max_p",
                "failed",
                "fail_reason",
                "fail_category",
                "cache_hit",
                "evaluated_fresh",
                "cache_source_uid",
                "cache_source_generation",
                "lineage_id",
                "lineage_root_uid",
                "lineage_depth",
                "primary_parent_uid",
                "primary_parent_generation",
                "reproduction_operator",
                "crossover_applied",
                "mutation_applied",
                "mutation_changed_genome",
                "topology_mutation",
                "topology_mutation_magnitude",
                "topology_signature",
                "parent_a_topology_signature",
                "parent_b_topology_signature",
                "primary_parent_topology_signature",
                "successful_topology_mutation",
                "beneficial_topology_event",
                "selected_next_generation",
                "parent_best_scalar_fitness",
                "offspring_scalar_fitness",
                "lineage_event",
                "parent_a_lineage_id",
                "parent_b_lineage_id",
                "cross_lineage_mating",
                "airfoil_signature",
                "parent_a_airfoil_signature",
                "parent_b_airfoil_signature",
                "primary_parent_airfoil_signature",
                "airfoil_mutation",
                "successful_airfoil_mutation",
                "beneficial_airfoil_event",
                "offspring_vs_best_parent_scalar_delta",
            ):
                if hasattr(ch, a):
                    delattr(ch, a)

        # Apply SBX + polynomial mutation pairwise
        for i in range(0, len(offspring), 2):
            c1, c2 = offspring[i], offspring[i + 1]
            idx_a = parent_indices[i] if parent_indices is not None else -1
            idx_b = parent_indices[i + 1] if parent_indices is not None else -1
            p_a = parents[i] if i < len(parents) else None
            p_b = parents[i + 1] if (i + 1) < len(parents) else None
            p_uid_a = getattr(p_a, "uid", -1) if p_a is not None else -1
            p_uid_b = getattr(p_b, "uid", -1) if p_b is not None else -1
            p_gen = getattr(self, "_gen", 0) - 1
            for child in (c1, c2):
                child.parent_idx_a = idx_a
                child.parent_idx_b = idx_b
                child.parent_uid_a = p_uid_a
                child.parent_uid_b = p_uid_b
                child.parent_gen_a = p_gen
                child.parent_gen_b = p_gen
                child.generation_origin = int(getattr(self, "_gen", 0))

            # crossover
            crossover_applied = False
            if random.random() < self.cx_pb:
                self.tb.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values
                crossover_applied = True

            # Keep discrete genes aligned to valid bins after crossover
            c1[:] = Chromosome_Drone.snap_genome_norm(c1)
            c2[:] = Chromosome_Drone.snap_genome_norm(c2)

            # mutation
            c1_before = list(c1)
            c2_before = list(c2)
            c1_mutated = False
            c2_mutated = False
            if random.random() < self.mut_pb:
                before = list(c1)
                self.tb.mutate(c1)
                del c1.fitness.values
                c1[:] = Chromosome_Drone.apply_discrete_mutation(before, c1)
                c1_mutated = True
            if random.random() < self.mut_pb:
                before = list(c2)
                self.tb.mutate(c2)
                del c2.fitness.values
                c2[:] = Chromosome_Drone.apply_discrete_mutation(before, c2)
                c2_mutated = True

            self._annotate_variation(
                c1,
                p_a,
                p_b,
                crossover_applied=crossover_applied,
                mutation_applied=c1_mutated,
                pre_variation_genome=c1_before,
            )
            self._annotate_variation(
                c2,
                p_a,
                p_b,
                crossover_applied=crossover_applied,
                mutation_applied=c2_mutated,
                pre_variation_genome=c2_before,
            )

            # inheritance → assign experiment + checkpoint suffix BEFORE training
            if self.inherit_policy:
                infos = []
                for p in (parents[i], parents[i + 1]):
                    if hasattr(p, "exp_name"):
                        ck_it = int(getattr(p, "train_it", self.cfg.train_iters_new))
                        ck_idx = max(0, ck_it - 1)
                        ck = self.log_root / p.exp_name / f"model_{ck_idx}.pt"
                        if ck.is_file():
                            infos.append((p.exp_name, ck_idx))
                if infos:
                    c1.parent_exp, c1.parent_ckpt = random.choice(infos)
                    c2.parent_exp, c2.parent_ckpt = random.choice(infos)

    def _after_generation(self, pop: List["IndType"]) -> None:
        """Update stats, generate plots, and print generation summary."""
        g = self._gen
        self.stats.record(g, pop, INVALID_V, INVALID_E, INVALID_P)
        fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
        front0 = fronts[0] if fronts else []
        self._append_population_history(pop, fronts)
        self._append_pareto_history(front0)
        self._append_generation_summary(pop, len(front0))
        if g % 3 == 0 or g == self.n_gen:
            out_dir = self.analysis_dir / f"g{g:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=str(out_dir / "gen"))

        def _safe_nanmax(arr: np.ndarray) -> float:
            return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")

        best_v = _safe_nanmax(self.stats.V[g])
        best_e_raw = _safe_nanmax(self.stats.E[g])
        best_e = -best_e_raw if np.isfinite(best_e_raw) else float("nan")
        best_p = _safe_nanmax(self.stats.M[g])
        print(
            f"--- Gen {g} summary  "
            f"best_vel={best_v:.2f}  best_eff={best_e:.2f}  best_prog={best_p:.2f}"
        )

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> List["IndType"]:
        """
        Run the full NSGA-II evolution and return the final population.
        """
        print(f"[setup] Run directory: {self.base_dir}")
        print(f"[setup] CSV cache: {self.db.path}")
        if self._is_resume:
            print(f"[setup] Resume source: {self.resume_source_dir}")
            print(
                "[setup] Resume mode: continuing from generation "
                f"{self.resume_start_generation} into copied run directory"
            )

        if self._is_resume:
            if self._resume_population is None or self.resume_start_generation is None:
                raise RuntimeError("Resume state was not initialized correctly")
            # Rebuild DEAP's transient NSGA-II attributes (e.g. crowding_dist)
            # before the first resumed tournament selection.
            pop = tools.selNSGA2(self._resume_population, self.n_pop)
            start_generation = int(self.resume_start_generation) + 1
        else:
            # GEN 0
            pop = self.tb.pop(self.n_pop)
            self._ensure_uids(pop)
            for ind in pop:
                ind[:] = Chromosome_Drone.snap_genome_norm(ind)
                ind.parent_idx_a = -1
                ind.parent_idx_b = -1
                ind.parent_uid_a = -1
                ind.parent_uid_b = -1
                ind.parent_gen_a = -1
                ind.parent_gen_b = -1
                ind.generation_origin = 0
                self._init_founder_logging(ind)
            self._gen = 0
            self._train_eval_population(pop)
            gen0_fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
            pop = tools.selNSGA2(pop, self.n_pop)
            self._append_selection_pool_history(
                pop,
                gen0_fronts,
                pop,
                {id(ind): "initial" for ind in pop},
            )
            self._after_generation(pop)
            start_generation = 1

        # GEN ≥ 1 or resumed generation
        for g in range(start_generation, self.n_gen + 1):
            self._gen = g
            print(f"\n════════ Generation {g}/{self.n_gen} ════════")
            self._ensure_uids(pop)

            # 1) parent selection (requires crowding_dist)
            parents = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]
            for child in offspring:
                self._assign_uid(child)
            parent_idx_map = {id(ind): idx for idx, ind in enumerate(pop)}
            parent_indices = [parent_idx_map.get(id(p), -1) for p in parents]

            # 2) variation (+ inheritance) before training
            self._apply_variation(offspring, parents, parent_indices)

            # 3) train + eval offspring
            self._train_eval_population(offspring)

            # 4) survivor-selection NSGA-II → new population
            pool = pop + offspring
            pool_fronts = tools.sortNondominated(pool, len(pool), first_front_only=False)
            selected = tools.selNSGA2(pool, self.n_pop)
            origin_map = {id(ind): "parent" for ind in pop}
            origin_map.update({id(ind): "offspring" for ind in offspring})
            self._annotate_selection_outcomes(pool, selected)
            self._append_selection_pool_history(pool, pool_fronts, selected, origin_map)
            pop = selected

            # 5) logging / plots
            self._after_generation(pop)

        # Save global stats
        with self.stats_path.open("wb") as f:
            import pickle

            pickle.dump(self.stats, f)
        print(f"Statistics saved ✔ → {self.stats_path}")
        return pop


# =============================================================================
#  CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=40, help="Population size.")
    parser.add_argument("--gen", type=int, default=25, help="Number of generations.")
    parser.add_argument(
        "--train_it",
        type=int,
        default=750,
        help="Training iterations for NEW morphologies.",
    )
    parser.add_argument(
        "--train_repetition",
        type=int,
        default=0,
        help="Repeat train+eval N times (gen_policy=0) and average final fitness.",
    )
    parser.add_argument(
        "--eta_m",
        type=float,
        default=None,
        help="Polynomial mutation eta parameter.",
    )
    parser.add_argument(
        "--eta_c",
        type=float,
        default=None,
        help="SBX crossover eta parameter.",
    )
    parser.add_argument(
        "--mutation_indpb_num",
        type=float,
        default=None,
        help="Per-gene mutation probability numerator; effective indpb = value / n_genes.",
    )
    parser.add_argument(
        "--inherit",
        action="store_true",
        default=False,
        help="Enable policy inheritance for offspring.",
    )
    parser.add_argument(
        "--dynamic_p",
        type=int,
        choices=(0, 1),
        default=0,
        help="Dynamic minimal_p (1=on, 0=off).",
    )
    parser.add_argument(
        "--no_dynamic_p",
        action="store_true",
        help="Disable dynamic minimal_p and use fixed_p instead.",
    )
    parser.add_argument(
        "--fixed_p",
        type=float,
        default=250.0,
        help="Fixed minimal_p threshold (used if --no_dynamic_p).",
    )
    parser.add_argument(
        "--pct_above",
        type=float,
        default=50.0,
        help="Percentage of individuals above minimal_p when dynamic.",
    )
    parser.add_argument(
        "--gen_policy",
        type=int,
        choices=(0, 1),
        default=1,
        help="0=train+eval, 1=eval-only with a custom pre-trained policy",
    )
    parser.add_argument(
        "--policy_path", type=str, default=None,
        help="Path to a pre-trained policy to use together with --gen_policy"
    )
    parser.add_argument(
        "--policy_paths",
        type=str,
        nargs="+",
        default=list(DEFAULT_POLICY_PATHS),
        help="Optional list of pre-trained policies to average in eval-only mode.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/andrea/Documents/Genesis/src/data_processing",
        help="Root directory for artifacts (defaults to cfg.base_dir).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training/eval (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from an existing run folder by copying it and continuing from the last completed generation.",
    )
    parser.add_argument(
        "--resume_suffix",
        type=str,
        default=DEFAULT_RESUME_SUFFIX,
        help="Suffix used for the copied run folder when resuming.",
    )

    args = parser.parse_args()

    # Build GA configuration from defaults + CLI overrides
    cfg = GAConfig()
    cfg.population_size = args.pop
    cfg.num_generations = args.gen
    cfg.train_iters_new = args.train_it
    cfg.train_repetition = args.train_repetition
    if args.eta_m is not None:
        cfg.eta_m = args.eta_m
    if args.eta_c is not None:
        cfg.eta_c = args.eta_c
    if args.mutation_indpb_num is not None:
        cfg.mutation_indpb_numerator = args.mutation_indpb_num
    cfg.inherit_policy = args.inherit
    cfg.csv_basename = "nsga"
    if args.no_dynamic_p:
        cfg.use_dynamic_p = False
    else:
        cfg.use_dynamic_p = bool(args.dynamic_p)
    cfg.fixed_p = args.fixed_p
    cfg.pct_above = args.pct_above
    cfg.gen_policy = bool(args.gen_policy)
    cfg.policy_path = args.policy_path
    cfg.policy_paths = args.policy_paths
    cfg.run_name = args.run_name
    if args.base_dir is not None:
        cfg.base_dir = args.base_dir
    cfg.device = args.device
    cfg.resume_from = args.resume_from
    cfg.resume_suffix = args.resume_suffix

    ga = CodesignDEAP(cfg)
    ga.run()


if __name__ == "__main__":
    main()
