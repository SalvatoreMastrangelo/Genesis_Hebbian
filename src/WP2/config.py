"""
HebbianEvolutionConfig — unified configuration for WP2 CMA-ES runs.
====================================================================

WP2 evolves Hebbian plasticity rules (ABCD per-weight parameters) for a
frozen WP1 actor using CMA-ES.  Morphology is always fixed.

This module mirrors the WP1 ``RunConfig`` pattern: a nested dataclass
hierarchy that can be:

1. **serialised to / deserialised from YAML** (``to_yaml`` / ``from_yaml``),
2. **overridden from the CLI** with ``--cfg.section.key value`` syntax,
3. **queried** for derived properties like genome dimensions.

Hierarchy
---------
::

    HebbianEvolutionConfig
    ├── exp_name              str                experiment tag
    ├── checkpoint_path       str                path to frozen WP1 actor weights
    ├── checkpoint_config_path str               path to matching WP1 config YAML
    ├── hebbian               HebbianConfig      ABCD plasticity rule ranges & eta
    ├── evolution             EvolutionConfig    number of generations / pop size hint
    ├── evaluation            EvaluationConfig   rollout episodes, envs, speed commands
    ├── cmaes                 CMAESConfig        CMA-ES step size, population, tolerances
    ├── catalog               CatalogConfig      optional multi-URDF catalog path
    ├── seed                  int                global random seed
    ├── device                str                torch device string
    └── base_dir              str                root directory for run logs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ============================================================================
#  Sub-configs
# ============================================================================

@dataclass
class HebbianConfig:
    """Per-weight ABCD Hebbian plasticity rule configuration.

    The ABCD rule modifies each weight in the frozen actor's last layer via:

        dW = eta * (A * outer(y, x) + B * x + C * y + D) - lambda * W

    where ``x`` is presynaptic activation (last hidden layer, shape ``hidden_dim``),
    ``y`` is postsynaptic activation (output before tanh, shape ``num_actions``), and
    ``A``, ``B``, ``C``, ``D``, ``lambda`` are per-weight learned parameters.

    All parameters are stored in [0, 1] within the genome and rescaled to their
    configured ranges during decoding (see ``decode_hebbian_genes()``).
    """

    enabled: bool = True
    eta: float = 0.01
    evolve_eta: bool = False
    decay: float = 0.01
    evolve_decay: bool = False
    use_oja_coefficient: bool = True
    initialize_rules_to_zero: bool = False
    num_actions: int = 7   # last-layer output dim (inferred from checkpoint)
    hidden_dim: int = 64   # last-layer input dim (actor MLP hidden size)
    w_max: float = 3.0
    A_range: Tuple[float, float] = (-1.0, 1.0)
    B_range: Tuple[float, float] = (-1.0, 1.0)
    C_range: Tuple[float, float] = (-1.0, 1.0)
    D_range: Tuple[float, float] = (-1.0, 1.0)
    decay_range: Tuple[float, float] = (0.0, 0.1)
    eta_range: Tuple[float, float] = (0.0, 0.1)


@dataclass
class EvolutionConfig:
    """High-level evolution loop parameters.

    ``num_generations`` is the stopping criterion for CMA-ES.
    ``population_size`` is a hint only — the actual CMA-ES population is
    controlled by ``CMAESConfig.population_size``.
    """

    population_size: int = 0       # hint only; actual pop set in cmaes.population_size
    num_generations: int = 50


@dataclass
class NoiseConfig:
    """Per-slot stochastic noise toggles.

    Each flag corresponds to a noise source that is sampled INDEPENDENTLY per
    env slot (i.e. NOT shared across individuals occupying the same forest
    slot).  ``True`` keeps the WP1 checkpoint's setting intact; ``False``
    force-disables the source for the WP2 evaluation env (the WP1 magnitude is
    overridden to 0 / off so the source produces no per-individual variance).

    Shared factors — forest layout, commanded speed, initial drone pose —
    are deterministic in eval mode and have no toggle here.

    Stochastic actor sampling has its own switch at ``evaluation.stochastic``.
    """

    action_latency: bool = True              # simulate_action_latency
    mass_shift: bool = True                  # property_randomization.mass_shift_std
    com_shift: bool = True                   # property_randomization.com_shift_std
    joint_target_episode_bias: bool = True   # property_randomization.joint_target_episode_bias_std
    joint_target_step_noise: bool = True     # property_randomization.joint_target_step_noise_std
    aero_noise: bool = True                  # aero_noise (aero solver param + magnitude/direction noise)


@dataclass
class EvaluationConfig:
    """Population-level vectorized rollout configuration for fitness evaluation.

    All individuals in a generation evaluate simultaneously using a shared
    pool of environments divided into slices (one per individual).

    Attributes
    ----------
    num_eval_episodes : int
        Ignored for CMA-ES (use ``catalog.num_episodes`` instead).
    num_eval_envs : int
        Total number of parallel Genesis environments.  Divided among all
        individuals: each individual gets ``num_eval_envs // pop_size`` envs.
    vmin, vmax : float
        Forward velocity command range [m/s] during rollout.
    stochastic : bool
        If True, sample from the policy distribution; otherwise use the mean.
    noise : NoiseConfig
        Per-slot noise toggles (see ``NoiseConfig``).  Sources flagged
        ``False`` are force-disabled so they cannot produce per-individual
        variance.
    """

    num_eval_episodes: int = 1
    num_eval_envs: int = 8192
    num_eval_workers: int = 1  # >1 enables ParallelMultiSceneEvalEnv (N worker processes)
    # Multi-GPU evaluation: when >1, workers are round-robin pinned to G GPUs
    # (URDFs are round-robin sharded across GPUs too, so each GPU handles
    # num_urdfs/G URDFs × envs_per_drone = num_eval_envs/G total env slots).
    # 0 or None → auto-detect via torch.cuda.device_count(). 1 → legacy
    # single-GPU behaviour. Clamped to min(num_gpus, num_eval_workers).
    num_gpus: int = 0
    vmin: float = 6.0
    vmax: float = 30.0
    stochastic: bool = True
    run_baseline: bool = False
    baseline_every: int = 1  # when run_baseline=True, evaluate every N generations (gen 0 always)
    run_specialist: bool = False
    specialist_every: int = 1  # when run_specialist=True, evaluate every N generations (gen 0 always)
    x_upper: Optional[float] = None  # override WP1 forest corridor length [m]
    dens_min: Optional[float] = None  # override forest density at x=0 [trees/m]
    dens_min_slope: float = 0.0  # per-generation linear ramp added to dens_min
    dens_max: Optional[float] = None  # override forest density at x=x_upper [trees/m]
    refresh_forests_per_generation: bool = False
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class CMAESConfig:
    """CMA-ES optimiser hyperparameters.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) treats the WP1
    reward sum as a scalar fitness and adapts a covariance matrix over the
    Hebbian-rule genome to guide search.

    Attributes
    ----------
    algorithm : str
        Which CMA-ES variant to use. Supported values:

        - ``"cmaes"`` (default): standard CMA-ES with a full covariance
          matrix. Scales as O(n^2) memory / O(n^3) per-generation work, so
          it is best for low-to-moderate genome dimensions (≲ a few
          hundred genes).
        - ``"sep-cmaes"``: separable CMA-ES (diagonal covariance
          throughout). Scales linearly with genome size and recommended
          for high-dimensional searches (thousands of genes). Cannot
          model correlations between dimensions but converges much
          faster in wall-clock terms for large n.

        Implemented via pycma's ``CMA_diagonal`` option (``True`` for
        sep-CMA-ES, ``0`` for standard).
    sigma0 : float
        Initial step size.  The genome lives in [0,1], so 0.3 spans ~30% of
        the domain.
    population_size : int
        Candidate solutions sampled each generation (CMA-ES "lambda").
        0 → auto (``4 + floor(3 * ln(n_genes))``).
    tol_sigma : float
        Convergence threshold on sigma.  0 disables.
    tol_fun : float
        Convergence threshold on fitness spread.  0 disables.
    """

    algorithm: str = "cmaes"   # "cmaes" or "sep-cmaes"
    sigma0: float = 0.3
    population_size: int = 0
    tol_sigma: float = 0.0
    tol_fun: float = 0.0


@dataclass
class CatalogConfig:
    """Optional multi-URDF catalog for robustness evaluation.

    Two ways to drive multi-URDF evaluation:
    1. ``path`` points to an existing ``catalog.txt`` (one URDF filename per
       line). The number of URDFs is inferred from the file.
    2. ``path`` is empty and ``num_urdfs > 1``: a catalog of that many random
       URDFs is auto-generated at run start (same sampler as WP1 training).

    When the effective number of URDFs is > 1, evaluation runs in the new
    multi-URDF path that builds a single Genesis scene holding all N URDFs
    as separate entities. When it is 1, the legacy single-URDF path is used
    unchanged (unless ``force_multi_urdf=True``, which routes a 1-URDF run
    through the multi-URDF path for verification).

    Attributes
    ----------
    path : str
        Path to a ``catalog.txt`` file.  Empty triggers auto-generation when
        ``num_urdfs > 1`` or ``force_multi_urdf=True``.
    num_urdfs : int
        Size of the auto-generated catalog when ``path`` is empty.  Ignored
        when ``path`` is set.
    num_episodes : int
        Rollout episodes per URDF per individual.
    force_multi_urdf : bool
        Verification flag. When True, use the multi-URDF evaluation path
        even with a single URDF (useful for confirming parity with the
        legacy single-URDF path).
    include_standard_mydrone : bool
        When True (default), the auto-generated catalog's first URDF is the
        fixed standard-mydrone baseline (``STANDARD_MYDRONE_GENOME``); the
        remaining ``num_urdfs - 1`` are sampled randomly. When False, all
        ``num_urdfs`` URDFs are sampled randomly. Ignored when ``path`` is
        set (the existing catalog is used as-is).
    refresh_urdfs_every : int
        When > 0, every ``refresh_urdfs_every`` inner CMA-ES generations the
        evaluation env is torn down and a brand-new random URDF population
        (same ``num_urdfs``, respecting ``include_standard_mydrone``) is
        sampled, written under ``urdfs_gen_XXX/``, and the env is rebuilt.
        ``0`` (default) disables the refresh — behaviour identical to before.
        Only effective in the multi-URDF path; ignored when running through
        the legacy single-URDF path.
    """

    path: str = ""
    num_urdfs: int = 1
    num_episodes: int = 1
    force_multi_urdf: bool = False
    include_standard_mydrone: bool = True
    refresh_urdfs_every: int = 0


# ============================================================================
#  Top-level config
# ============================================================================

@dataclass
class HebbianEvolutionConfig:
    """Top-level configuration for a WP2 CMA-ES evolution run."""

    exp_name: str = "hebbian_cma"
    checkpoint_path: str = ""
    checkpoint_config_path: str = ""

    # Optional separate checkpoint used ONLY for the zero-rules baseline eval.
    # When empty, the baseline reuses ``checkpoint_path`` / ``checkpoint_config_path``
    # (preserves previous behavior). When set, the baseline actor is built from
    # this checkpoint independently of the frozen-actor checkpoint that
    # Hebbian rules modulate. Architecture (MLP/LSTM/last-layer sizes) may
    # differ, but obs/action dims must still match the env.
    baseline_checkpoint_path: str = ""
    baseline_checkpoint_config_path: str = ""

    # Optional separate checkpoint used ONLY for the "specialist" comparison
    # curve.  Plays the same role as the baseline: evaluated with zero Hebbian
    # rules every ``evaluation.specialist_every`` generations on exactly the
    # same forests / speeds / URDFs as the population.  Architecture
    # (MLP/LSTM/last-layer sizes) may differ from the main frozen actor and
    # from the baseline; obs/action dims must still match the env.
    specialist_checkpoint_path: str = ""
    specialist_checkpoint_config_path: str = ""

    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cmaes: CMAESConfig = field(default_factory=CMAESConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)

    seed: int = 42
    device: str = "cuda:0"
    base_dir: str = "logs/runs_hebbian"

    # ------------------------------------------------------------------
    #  Derived helpers
    # ------------------------------------------------------------------

    def hebbian_genome_dim(self) -> int:
        """Number of Hebbian genes per individual.

        Base: 4 × n_weights (A, B, C, D).
        +n_weights if ``evolve_decay=True``.
        +n_weights if ``evolve_eta=True``.
        Returns 0 if Hebbian is disabled.
        """
        if not self.hebbian.enabled:
            return 0
        n_weights = self.hebbian.num_actions * self.hebbian.hidden_dim
        base = 4 * n_weights
        if self.hebbian.evolve_decay:
            base += n_weights
        if self.hebbian.evolve_eta:
            base += n_weights
        return base

    def total_genome_dim(self) -> int:
        """Total genome dimension (Hebbian rules only)."""
        return self.hebbian_genome_dim()

    # ------------------------------------------------------------------
    #  YAML I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialise this config to a YAML string."""
        data = _tuples_to_lists(asdict(self))
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HebbianEvolutionConfig":
        """Deserialise a ``HebbianEvolutionConfig`` from a YAML file.

        Missing keys fall back to dataclass defaults.  Unknown keys (e.g.
        leftover ``morphology`` or ``objectives`` sections from old configs)
        are silently ignored.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "HebbianEvolutionConfig":
        cfg = cls()
        sub_map = {
            "hebbian": HebbianConfig,
            "evolution": EvolutionConfig,
            "evaluation": EvaluationConfig,
            "cmaes": CMAESConfig,
            "catalog": CatalogConfig,
        }
        # Nested dataclass fields inside top-level sections (section -> {field: cls}).
        nested_sub_map = {
            "evaluation": {"noise": NoiseConfig},
        }
        for key, val in data.items():
            if key in sub_map and isinstance(val, dict):
                sub = sub_map[key]()
                nested = nested_sub_map.get(key, {})
                for sk, sv in val.items():
                    if sk in nested and isinstance(sv, dict):
                        nsub = nested[sk]()
                        for nk, nv in sv.items():
                            if hasattr(nsub, nk):
                                setattr(nsub, nk, nv)
                        setattr(sub, sk, nsub)
                        continue
                    if hasattr(sub, sk):
                        current = getattr(sub, sk)
                        if isinstance(current, tuple) and isinstance(sv, list):
                            sv = tuple(sv)
                        setattr(sub, sk, sv)
                setattr(cfg, key, sub)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
            # unknown keys (morphology, objectives, etc.) are silently skipped
        return cfg

    # ------------------------------------------------------------------
    #  CLI override support
    # ------------------------------------------------------------------

    def apply_cli_overrides(self, argv: Optional[List[str]] = None) -> None:
        """Apply ``--cfg.section.key value`` overrides from the command line.

        Examples
        --------
        .. code-block:: bash

            --cfg.hebbian.eta 0.05
            --cfg.evolution.num_generations 100
            --cfg.cmaes.sigma0 0.2
            --cfg.exp_name my-test
        """
        if argv is None:
            argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith("--cfg."):
                parts = arg[len("--cfg."):].split(".")
                if len(parts) == 3 and i + 1 < len(argv):
                    section, sub_section, key = parts
                    val_str = argv[i + 1]
                    sub = getattr(self, section, None)
                    nested = getattr(sub, sub_section, None) if sub is not None else None
                    if nested is not None and hasattr(nested, key):
                        current = getattr(nested, key)
                        setattr(nested, key, _cast(val_str, current))
                        i += 2
                        continue
                elif len(parts) == 2 and i + 1 < len(argv):
                    section, key = parts
                    val_str = argv[i + 1]
                    sub = getattr(self, section, None)
                    if sub is not None and hasattr(sub, key):
                        current = getattr(sub, key)
                        setattr(sub, key, _cast(val_str, current))
                        i += 2
                        continue
                elif len(parts) == 1 and i + 1 < len(argv):
                    key = parts[0]
                    if hasattr(self, key):
                        current = getattr(self, key)
                        setattr(self, key, _cast(argv[i + 1], current))
                        i += 2
                        continue
            i += 1


def _tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML-safe serialisation."""
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(item) for item in obj]
    return obj


def _cast(val_str: str, reference: Any) -> Any:
    """Cast a CLI string value to match the type of *reference*."""
    if isinstance(reference, bool):
        return val_str.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(val_str)
    if isinstance(reference, float):
        return float(val_str)
    if isinstance(reference, tuple):
        val_str = val_str.strip("()[]")
        items = [s.strip() for s in val_str.split(",")]
        return tuple(float(x) for x in items)
    if isinstance(reference, list):
        val_str = val_str.strip("[]")
        items = [s.strip() for s in val_str.split(",")]
        if reference and isinstance(reference[0], int):
            return [int(x) for x in items]
        return [float(x) for x in items]
    return val_str
