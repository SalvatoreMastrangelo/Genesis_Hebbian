"""
Configuration for NSGA-II URDF co-evolution (WP2 outer loop).
=============================================================

Two config families live here:

* ``OuterNSGA2Config`` — the CURRENT single-file config for
  ``NSGA2MorphCMAES`` (``nsga_cma.py``). It *is* a WP2
  ``HebbianEvolutionConfig`` (same YAML layout as
  ``src/WP2/configs/cma_es_rules_only.yaml``) extended with one extra
  ``outer:`` section holding the NSGA-II-specific knobs. Everything else is
  expressed through the existing inner-loop fields:

  - NSGA-II population size N  = ``catalog.num_urdfs``
  - phase length (refresh)     = ``catalog.refresh_urdfs_every``
  - total CMA generations      = ``evolution.num_generations``
  - env budget                 = ``evaluation.num_eval_envs`` (N × H × F)
  - sigma kick on refresh      = ``cmaes.sigma_reinflate``

* ``OuterLoopConfig`` (+ ``NSGA2Config``) — the DEPRECATED legacy nested
  loop's config (``legacy_run.py`` / ``outer_loop.py``), kept for
  reference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from WP2.config import HebbianEvolutionConfig

# Sentinel that survives YAML round-trips and CLI overrides the same way
# real ints do, but signals "use the inner template's value". Implemented
# as Optional[int] (None).


# ============================================================================
#  Sub-configs
# ============================================================================

@dataclass
class NSGA2Config:
    """Parameters for the NSGA-II selection + variation step.

    Attributes
    ----------
    crossover_prob : float
        Probability of applying SBX crossover to a parent pair.
    mutation_prob : float
        Per-gene probability for polynomial mutation (DEAP convention:
        ``tools.mutPolynomialBounded(indpb=...)``). Default 1/15 = 0.0667.
    sbx_eta : float
        Distribution index for SBX. Larger values produce offspring closer
        to their parents. DEAP default 15.
    pm_eta : float
        Distribution index for polynomial mutation. DEAP default 20.
    """

    crossover_prob: float = 0.9
    mutation_prob: float = 1.0 / 15.0
    sbx_eta: float = 15.0
    pm_eta: float = 20.0


@dataclass
class ObjectiveSpec:
    """Declarative objective for NSGA-II.

    ``name`` must match a key emitted by the evaluation pipeline
    (see ``WP2_Outer_Loop.evaluation`` for the supported set).
    ``direction`` is ``"maximize"`` or ``"minimize"``.
    """

    name: str = "progress_m"
    direction: str = "maximize"


# ============================================================================
#  Top-level config
# ============================================================================

@dataclass
class OuterLoopConfig:
    """Top-level configuration for the NSGA-II outer loop."""

    exp_name: str = "outer_loop"

    # Loop sizing
    outer_generations: int = 20
    inner_generations: int = 50
    population_size: int = 16
    num_eval_envs: int = 256
    seed: int = 0

    # Population init
    seed_standard_drone: bool = True

    # Paths
    inner_cfg_template: str = "src/WP2/configs/cma_es_rules_only.yaml"
    base_dir: str = "logs/runs_outer"

    # WP1 controller overrides — applied to the loaded inner template.
    # Leave empty to use whatever the inner template YAML specifies.
    checkpoint_path: str = ""           # frozen WP1 actor .pt
    checkpoint_config_path: str = ""    # matching WP1 run config.yaml

    # Inner-loop worker override. ``None`` = inherit from inner template;
    # any int ≥ 1 replaces the template's ``evaluation.num_eval_workers``.
    inner_num_eval_workers: Optional[int] = None

    # Inner-loop CMA sigma re-inflation override. ``None`` = inherit from the
    # inner template's ``cmaes.sigma_reinflate``; any float ≥ 0 replaces it.
    # When the morphology changes, the inner CMA-ES re-inflates its step size to
    # ``sigma_reinflate × sigma0`` (carry mean+covariance, re-explore); 0 = off.
    inner_sigma_reinflate: Optional[float] = None

    # Sub-configs
    nsga2: NSGA2Config = field(default_factory=NSGA2Config)
    objectives: List[ObjectiveSpec] = field(default_factory=lambda: [
        ObjectiveSpec(name="progress_m", direction="maximize"),
        ObjectiveSpec(name="cost_of_transport", direction="minimize"),
    ])

    # ------------------------------------------------------------------
    #  YAML I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: Optional[Path] = None) -> str:
        data = asdict(self)
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OuterLoopConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OuterLoopConfig":
        cfg = cls()
        for key, val in data.items():
            if key == "nsga2" and isinstance(val, dict):
                sub = NSGA2Config()
                for sk, sv in val.items():
                    if hasattr(sub, sk):
                        setattr(sub, sk, sv)
                cfg.nsga2 = sub
            elif key == "objectives" and isinstance(val, list):
                specs: List[ObjectiveSpec] = []
                for item in val:
                    if isinstance(item, dict):
                        specs.append(ObjectiveSpec(
                            name=str(item.get("name", "progress_m")),
                            direction=str(item.get("direction", "maximize")),
                        ))
                if specs:
                    cfg.objectives = specs
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    # ------------------------------------------------------------------
    #  CLI override support
    # ------------------------------------------------------------------

    def apply_cli_overrides(self, argv: Optional[List[str]] = None) -> None:
        """Apply ``--cfg.section.key value`` overrides from argv.

        Examples
        --------
        .. code-block:: bash

            --cfg.outer_generations 30
            --cfg.population_size 32
            --cfg.num_eval_envs 512
            --cfg.nsga2.sbx_eta 20
            --cfg.nsga2.mutation_prob 0.1
        """
        if argv is None:
            argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith("--cfg.") and i + 1 < len(argv):
                key_path = arg[len("--cfg."):].split(".")
                val_str = argv[i + 1]
                if self._set_path(key_path, val_str):
                    i += 2
                    continue
            i += 1

    def _set_path(self, key_path: List[str], val_str: str) -> bool:
        if len(key_path) == 1:
            k = key_path[0]
            if hasattr(self, k):
                cur = getattr(self, k)
                setattr(self, k, _cast(val_str, cur))
                return True
            return False
        if len(key_path) == 2:
            sec, k = key_path
            sub = getattr(self, sec, None)
            if sub is not None and hasattr(sub, k):
                cur = getattr(sub, k)
                setattr(sub, k, _cast(val_str, cur))
                return True
        return False

    # ------------------------------------------------------------------
    #  Validation helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be ≥ 2 (got {self.population_size})")
        if self.population_size % 2 != 0:
            raise ValueError(
                f"population_size must be even for NSGA-II pairwise mating "
                f"(got {self.population_size})"
            )
        if self.outer_generations < 1:
            raise ValueError("outer_generations must be ≥ 1")
        if self.inner_generations < 1:
            raise ValueError("inner_generations must be ≥ 1")
        if self.num_eval_envs < self.population_size:
            raise ValueError(
                f"num_eval_envs ({self.num_eval_envs}) < population_size "
                f"({self.population_size}): NSGA-II evaluation gives 0 envs/URDF"
            )
        if not self.objectives:
            raise ValueError("objectives list is empty")
        for obj in self.objectives:
            if obj.direction not in ("maximize", "minimize"):
                raise ValueError(
                    f"objective {obj.name!r}: direction must be "
                    f"'maximize' or 'minimize' (got {obj.direction!r})"
                )

    def objective_weights(self) -> List[float]:
        """DEAP fitness weights: +1 for maximize, -1 for minimize."""
        return [1.0 if o.direction == "maximize" else -1.0 for o in self.objectives]


# ============================================================================
#  Steady-state outer loop config (persistent CMA-ES + NSGA-II URDF refresh)
# ============================================================================

@dataclass
class ExamForestConfig:
    """Optional forest overrides for the phase-end exam rollout.

    When ``override_forest`` is true, the non-null fields are applied to the
    live eval env's forest generator right before the exam and restored right
    after (``nsga_cma._run_exam``), so the exam can fly denser / longer /
    differently-structured forests than the inner loop without an env rebuild
    — trees are pure tensor obstacles, no Genesis geometry. With
    ``override_forest: false`` (default) the values are inert and the exam
    flies the inner loop's forest distribution. ``x_upper`` additionally moves the eval success line
    (``check_success``); note the 60 s episode length still caps reachable
    distance at ~vmax·60 m. Perception-coupled knobs (``tree_radius``, the y
    corridor) are intentionally unsupported: the DepthSolver bakes them in at
    env build, so overriding them would desync what the policy sees from what
    it collides with.
    """

    override_forest: bool = False            # master switch for this section
    dens_min: Optional[float] = None
    dens_max: Optional[float] = None
    num_trees: Optional[int] = None          # uniform mode only
    forest_mode: Optional[str] = None        # uniform | growing | lattice | latin
    x_upper: Optional[float] = None          # forest end + eval success line
    forest_length: Optional[float] = None    # lattice/latin modes
    x_spacing_start: Optional[float] = None  # lattice/latin modes
    x_spacing_end: Optional[float] = None    # lattice/latin modes
    y_spacing_min: Optional[float] = None    # lattice/latin modes
    y_spacing_max: Optional[float] = None    # lattice/latin modes

    def overrides(self) -> Dict[str, Any]:
        """Dict of the fields to apply — the payload for
        ``env.apply_forest_overrides``. Empty (feature off) unless
        ``override_forest`` is true; the flag itself is never included."""
        if not self.override_forest:
            return {}
        return {
            k: v for k, v in asdict(self).items()
            if v is not None and k != "override_forest"
        }


@dataclass
class OuterConfig:
    """The ``outer:`` section — NSGA-II knobs on top of a plain inner config.

    Attributes
    ----------
    n_elites : int
        NSGA-II survivors kept (and re-scored next phase) at each morphology
        update; the remaining ``catalog.num_urdfs - n_elites`` slots are
        offspring produced by binary tournament + SBX + polynomial mutation.
    score_top_frac : float
        Fraction of the CMA-ES population (ranked by that generation's
        fitness) whose per-URDF metrics are averaged into the URDF's
        objective scores. 0.5 = top half.
    score_window : int
        Number of trailing inner generations of the phase used for URDF
        scoring. 0 = all generations of the phase.
    rescore : bool
        Run a dedicated scoring rollout ("exam") at the end of each phase:
        the top ``rescore_top_frac`` of the last generation's CMA individuals
        re-fly the current URDF population on freshly generated forests,
        reusing the existing eval env. With k controllers sharing the
        ``E = H * F`` slots per URDF, each flies ``E / k`` forests (e.g.
        H=64, F=60, k=8 → 480 forests) — the NSGA-II objectives come from
        this single wide measurement instead of the phase-mean harvest
        (which stays as diagnostics). Falls back to the phase mean when the
        exam cannot run (env gone at the final flush, eval failure).
    rescore_top_frac : float
        Fraction of the CMA population used for the exam, ranked by the
        last generation's fitness. Rounded down to the nearest divisor of
        the per-URDF slot count. 0.125 = top 8 of 64.
    exam_forest : ExamForestConfig
        Optional forest overrides (density, tree count, mode, length) that
        apply ONLY to the exam rollout — see ``ExamForestConfig``. All-null
        (the default) means the exam flies the same forest distribution as
        the inner loop.
    min_progress_m : float
        Minimum per-URDF progress (meters) required for a morphology to be
        admitted to NSGA-II selection (elites + tournament parents). Hard
        filter: sub-threshold morphs never become elites or parents while
        feasible ones exist; with zero feasible morphs the refresh runs
        ungated (with a warning). Judged on the progress from the same
        source as the objectives (exam when it ran, else phase mean).
        0 = gate off.
    crossover_prob : float
        Probability of applying SBX crossover to a parent pair.
    mutation_prob : float
        Per-gene probability for polynomial mutation (DEAP convention:
        ``tools.mutPolynomialBounded(indpb=...)``). Default 1/15 = 0.0667.
    sbx_eta : float
        Distribution index for SBX. Larger values produce offspring closer
        to their parents. DEAP default 15.
    pm_eta : float
        Distribution index for polynomial mutation. DEAP default 20.
    objectives : list[ObjectiveSpec]
        Per-URDF objectives, harvested from the CMA rollouts. Supported
        names: fitness, cost_of_transport, progress_m, velocity, crash_rate
        (plus aliases — see ``nsga_cma._PER_URDF_ALIASES``).
    """

    n_elites: int = 2
    score_top_frac: float = 0.5
    score_window: int = 0
    rescore: bool = True
    rescore_top_frac: float = 0.125
    exam_forest: ExamForestConfig = field(default_factory=ExamForestConfig)
    min_progress_m: float = 0.0
    crossover_prob: float = 0.9
    mutation_prob: float = 1.0 / 15.0
    sbx_eta: float = 15.0
    pm_eta: float = 20.0
    objectives: List[ObjectiveSpec] = field(default_factory=lambda: [
        ObjectiveSpec(name="fitness", direction="maximize"),
        ObjectiveSpec(name="cost_of_transport", direction="minimize"),
    ])

    def objective_weights(self) -> List[float]:
        """DEAP fitness weights: +1 for maximize, -1 for minimize."""
        return [1.0 if o.direction == "maximize" else -1.0 for o in self.objectives]

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OuterConfig":
        sec = cls()
        for key, val in data.items():
            if key == "objectives" and isinstance(val, list):
                specs: List[ObjectiveSpec] = []
                for item in val:
                    if isinstance(item, dict):
                        specs.append(ObjectiveSpec(
                            name=str(item.get("name", "fitness")),
                            direction=str(item.get("direction", "maximize")),
                        ))
                if specs:
                    sec.objectives = specs
            elif key == "exam_forest" and isinstance(val, dict):
                known = {f.name for f in dataclass_fields(ExamForestConfig)}
                unknown = sorted(set(val) - known)
                if unknown:
                    raise ValueError(
                        f"outer.exam_forest: unknown keys {unknown} "
                        f"(supported: {sorted(known)})"
                    )
                sec.exam_forest = ExamForestConfig(**val)
            elif hasattr(sec, key):
                setattr(sec, key, val)
        return sec


@dataclass
class OuterNSGA2Config(HebbianEvolutionConfig):
    """Single-file config for ``NSGA2MorphCMAES`` (``nsga_cma.py``).

    A plain WP2 ``HebbianEvolutionConfig`` YAML plus one ``outer:`` section
    (see ``OuterConfig``). The run behaves exactly like an inner-loop
    multi-URDF run with periodic URDF refresh, except each refresh is an
    NSGA-II update instead of random resampling — so all sizing lives in the
    standard inner fields:

    * ``catalog.num_urdfs``            → NSGA-II population size N
    * ``catalog.refresh_urdfs_every``  → CMA generations per phase
    * ``evolution.num_generations``    → total CMA generations
    * ``evaluation.num_eval_envs``     → total env slots (N × H × F)
    * ``cmaes.sigma_reinflate``        → sigma kick after each morphology change

    CLI overrides use the single inherited namespace, e.g.
    ``--cfg.catalog.num_urdfs 4`` or ``--cfg.outer.n_elites 3``.
    """

    outer: OuterConfig = field(default_factory=OuterConfig)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OuterNSGA2Config":
        data = dict(data or {})
        outer_raw = data.pop("outer", None)
        cfg = super()._from_dict(data)
        if isinstance(outer_raw, dict):
            cfg.outer = OuterConfig._from_dict(outer_raw)
        return cfg

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def cma_population_size(self) -> int:
        """Effective CMA-ES population size H (pycma default when auto)."""
        H = int(self.cmaes.population_size)
        if H <= 0:
            H = int(4 + 3 * np.log(max(2, self.hebbian_genome_dim())))
        return H

    def validate(self) -> None:
        N = int(self.catalog.num_urdfs)
        outer = self.outer
        if N < 2:
            raise ValueError(
                f"catalog.num_urdfs must be ≥ 2 for NSGA-II (got {N})"
            )
        if int(self.catalog.refresh_urdfs_every) < 1:
            raise ValueError(
                "catalog.refresh_urdfs_every must be ≥ 1 — it is the number "
                "of CMA generations per outer phase "
                f"(got {self.catalog.refresh_urdfs_every})"
            )
        if self.catalog.mutate:
            raise ValueError(
                "catalog.mutate must be false: the NSGA-II update replaces "
                "the blind-mutation refresh entirely"
            )
        if not (1 <= outer.n_elites < N):
            raise ValueError(
                f"outer.n_elites must be in [1, catalog.num_urdfs-1] "
                f"(got {outer.n_elites} with num_urdfs={N})"
            )
        if not (0.0 < outer.score_top_frac <= 1.0):
            raise ValueError(
                f"outer.score_top_frac must be in (0, 1] (got {outer.score_top_frac})"
            )
        if outer.score_window < 0:
            raise ValueError("outer.score_window must be ≥ 0 (0 = whole phase)")
        if not (0.0 < outer.rescore_top_frac <= 1.0):
            raise ValueError(
                f"outer.rescore_top_frac must be in (0, 1] "
                f"(got {outer.rescore_top_frac})"
            )
        if float(outer.min_progress_m) < 0.0:
            raise ValueError(
                f"outer.min_progress_m must be ≥ 0 (got {outer.min_progress_m})"
            )
        ef = outer.exam_forest
        _modes = ("uniform", "growing", "lattice", "latin")
        if ef.forest_mode is not None and ef.forest_mode not in _modes:
            raise ValueError(
                f"outer.exam_forest.forest_mode must be one of {_modes} "
                f"(got {ef.forest_mode!r})"
            )
        if ef.num_trees is not None and int(ef.num_trees) < 1:
            raise ValueError(
                f"outer.exam_forest.num_trees must be ≥ 1 (got {ef.num_trees})"
            )
        if ef.x_upper is not None and float(ef.x_upper) <= 0.0:
            raise ValueError(
                f"outer.exam_forest.x_upper must be > 0 (got {ef.x_upper})"
            )
        for _nm in ("dens_min", "dens_max"):
            _v = getattr(ef, _nm)
            if _v is not None and float(_v) < 0.0:
                raise ValueError(
                    f"outer.exam_forest.{_nm} must be ≥ 0 (got {_v})"
                )
        if (
            ef.dens_min is not None and ef.dens_max is not None
            and float(ef.dens_max) < float(ef.dens_min)
        ):
            raise ValueError(
                f"outer.exam_forest: dens_max ({ef.dens_max}) < "
                f"dens_min ({ef.dens_min})"
            )
        if not outer.objectives:
            raise ValueError("outer.objectives list is empty")
        for obj in outer.objectives:
            if obj.direction not in ("maximize", "minimize"):
                raise ValueError(
                    f"objective {obj.name!r}: direction must be "
                    f"'maximize' or 'minimize' (got {obj.direction!r})"
                )
        # Env budget: every (URDF, individual) pair needs ≥ 1 forest.
        H = self.cma_population_size()
        if (int(self.evaluation.num_eval_envs) // N) // H < 1:
            raise ValueError(
                f"evaluation.num_eval_envs ({self.evaluation.num_eval_envs}) "
                f"too small: need ≥ N×H = {N}×{H} = {N * H} so every "
                f"(URDF, individual) pair gets ≥ 1 forest"
            )


def _cast(val_str: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return val_str.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(val_str)
    if isinstance(reference, float):
        return float(val_str)
    if reference is None:
        # Optional field with no default type — treat as int → float → str.
        try:
            return int(val_str)
        except ValueError:
            pass
        try:
            return float(val_str)
        except ValueError:
            pass
    return val_str
