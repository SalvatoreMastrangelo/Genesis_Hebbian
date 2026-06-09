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
    ├── forest                ForestConfig       forest algorithm + parameters (null = WP1)
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
    rules_per_neuron: bool = False  # if True, A/B/C/D are shared by all input
                                    # weights of each output neuron → 4×num_actions
                                    # genes instead of 4×num_actions×hidden_dim.
                                    # decay/eta (if evolved) stay per-weight.
    num_actions: int = 7   # last-layer output dim (inferred from checkpoint)
    hidden_dim: int = 64   # last-layer input dim (actor MLP hidden size)
    w_max: float = 3.0
    A_range: Tuple[float, float] = (-1.0, 1.0)
    B_range: Tuple[float, float] = (-1.0, 1.0)
    C_range: Tuple[float, float] = (-1.0, 1.0)
    D_range: Tuple[float, float] = (-1.0, 1.0)
    decay_range: Tuple[float, float] = (0.0, 0.1)
    eta_range: Tuple[float, float] = (0.0, 0.1)

    def abcd_block_size(
        self,
        out_features: Optional[int] = None,
        in_features: Optional[int] = None,
    ) -> int:
        """Number of genes per A/B/C/D block.

        Per-weight (default): ``out × in``.  Per-neuron
        (``rules_per_neuron=True``): ``out`` — one shared rule for every input
        weight of each output neuron.  ``out``/``in`` default to
        ``num_actions``/``hidden_dim`` but may be overridden when the layer
        dims are inferred from a checkpoint.
        """
        o = self.num_actions if out_features is None else out_features
        i = self.hidden_dim if in_features is None else in_features
        return o if self.rules_per_neuron else o * i


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
    # Common random numbers: share per-slot domain-randomization draws (forest
    # assignment, mass/COM, joint episode bias + step noise, action latency,
    # aero parameters, observation noise) across individuals flying the same
    # forest, so CMA-ES ranks individuals by their Hebbian rules rather than by
    # independent DR luck. Full DR is preserved (the F scenarios still span the
    # distribution and refresh each generation). The per-step Taichi aero force
    # noise stays per-slot (generated in-kernel). Set False to reproduce the
    # legacy independent-per-slot behaviour (e.g. for the sigma_rank A/B test).
    crn: bool = True
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class ForestConfig:
    """Forest-generation overrides for the WP2 evaluation environment.

    Every field defaults to ``None``, meaning "inherit the value baked into the
    WP1 checkpoint config" (the ``env:`` block of ``checkpoint_config_path``).
    Set ``mode`` to pick the sampling algorithm and any other field to override
    that specific parameter; unset (``None``) fields keep the WP1 value, so an
    all-null ``forest`` section reproduces the WP1 forest exactly.

    ``mode`` selects the obstacle-sampling algorithm
    (``winged_drone_train.perception.forest``):

    - ``None`` / ``"wp1"`` — inherit the WP1 config's ``forest_mode`` (or its
      legacy ``growing_forest`` flag). Nothing about the mode is overridden.
    - ``"growing"`` — density ramps linearly from ``dens_min`` (at ``x_lower``)
      to ``dens_max`` (at ``x_upper``). Uses ``dens_min`` / ``dens_max`` (and
      optionally ``dens_min_min`` / ``dens_min_max`` to randomize the ramp
      start per forest).
    - ``"uniform"`` — constant density with exactly ``num_trees`` trees
      (``num_trees_eval`` is used in eval mode).
    - ``"lattice"`` — regular grid (fixed points + small jitter) that
      densifies with x. Uses ``x_spacing_start`` / ``x_spacing_end`` /
      ``forest_length`` / ``y_spacing_max`` / ``y_spacing_min``.
    - ``"latin"`` — stratified grid: same cell structure as ``lattice`` but
      one tree placed at a uniformly RANDOM position inside each cell (no fixed
      points). Uses the same ``x_spacing_*`` / ``forest_length`` /
      ``y_spacing_*`` keys.

    Geometry (``x_lower`` … ``tree_height``) applies to every mode.

    NOTE: ``x_upper`` / ``dens_min`` / ``dens_max`` can ALSO be set under
    ``evaluation`` (legacy location, where ``dens_min`` additionally drives the
    per-generation ``dens_min_slope`` ramp). When a value is set in BOTH
    places, the ``forest`` section wins. Prefer setting them here unless you
    need the ramp.
    """

    mode: Optional[str] = None  # None/"wp1" | "growing" | "uniform" | "lattice"

    # --- Corridor geometry (all modes) ---
    x_lower: Optional[float] = None
    x_upper: Optional[float] = None
    y_lower: Optional[float] = None
    y_upper: Optional[float] = None
    tree_radius: Optional[float] = None
    tree_height: Optional[float] = None

    # --- "growing" mode: linear density ramp dens_min -> dens_max ---
    dens_min: Optional[float] = None
    dens_max: Optional[float] = None
    dens_min_min: Optional[float] = None  # randomize ramp-start density per forest (low)
    dens_min_max: Optional[float] = None  # randomize ramp-start density per forest (high)

    # --- "uniform" mode: fixed tree count (num_trees_eval used in eval) ---
    num_trees: Optional[int] = None
    num_trees_eval: Optional[int] = None

    # --- "lattice" mode: grid spacings (densify with x) ---
    x_spacing_start: Optional[float] = None
    x_spacing_end: Optional[float] = None
    forest_length: Optional[float] = None
    y_spacing_max: Optional[float] = None
    y_spacing_min: Optional[float] = None

    def resolved_mode(self) -> Optional[str]:
        """Return the ``forest_mode`` to write into ``env_cfg``.

        ``None`` means "do not override — inherit the WP1 config". A non-null
        value is validated against the three supported algorithms.
        """
        if self.mode is None:
            return None
        m = str(self.mode).strip().lower()
        if m in ("", "wp1", "none", "null", "inherit", "default"):
            return None
        valid = ("uniform", "growing", "lattice", "latin")
        if m not in valid:
            raise ValueError(
                f"forest.mode must be one of {valid} (or null/'wp1' to inherit "
                f"the WP1 config); got {self.mode!r}"
            )
        return m


@dataclass
class ValidationConfig:
    """Held-out validation evaluation inside the inner CMA-ES loop.

    Independent of the population-eval pool and of the in-distribution
    ``run_baseline`` curve (which reuses the population's own forests). When
    ``enable`` is True, a SEPARATE pool of ``n_val_envs`` Genesis environments
    is reserved up front, split equally across the ``validation_catalog``
    URDFs. Every ``period`` generations — after the baseline phase — both the
    generation's best-fitness individual AND the zero-rules baseline are
    evaluated on those ``n_val_envs`` forests (freshly regenerated each pass),
    so the two curves can be plotted generation-by-generation to measure
    generalization to unseen forests.

    Attributes
    ----------
    enable : bool
        Master switch for the held-out validation pass. Default False.
    n_val_envs : int
        Total number of reserved validation environments, split equally across
        the validation URDFs (``n_val_envs // num_validation_urdfs`` per URDF).
    validation_catalog : str
        Path to a ``catalog.txt`` listing the validation URDFs. Empty or
        ``"none"`` falls back to the single standard-mydrone drone.
    period : int
        Cadence: run the validation pass on generation 0 and every ``period``
        generations thereafter.
    """

    enable: bool = False
    n_val_envs: int = 1024
    validation_catalog: str = ""
    period: int = 1


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
    sigma_reinflate : float
        On each morphology change (URDF refresh), re-inflate sigma to
        ``sigma_reinflate * sigma0`` (only ever raising it, never lowering).
        The mean and covariance are carried over — only the step size is reset
        so the search re-explores around the carried state for the new
        morphologies. 1.0 → reset to the initial step size each change; 0.0 →
        disabled (pure carry, sigma keeps shrinking). No effect when URDFs are
        not refreshed.
    uh_enabled : bool
        Enable UH-CMA-ES uncertainty handling (Hansen et al. 2009, the σ-only
        arm). After ``tell()`` the population is re-evaluated on the *same*
        forests (no refresh) to measure residual per-rollout rank noise — the
        stochastic-actor + per-step aero noise that CRN cannot remove. When
        rank-noise is detected (``noiseS > 0``) the step size is bumped by
        ``alphasigma``, counteracting σ-collapse/stall. The re-evaluation
        averaging arm is intentionally disabled (per-individual evals stay 1)
        because the forest count F is pinned by ``num_eval_envs``.
    uh_every : int
        Cadence: run the noise measurement every K generations. Each
        measurement costs ≈ one extra full-population evaluation (the env steps
        all slots regardless of how many individuals are re-evaluated, so a
        subset buys no savings on this batched GPU design). 1 → every gen
        (≈2× eval cost); 5 → +20%. Must be ≥ 1.
    uh_reevals : float
        Number of solutions used for the noise *measure*. 0 → use the whole
        population (robust, and free since the full re-eval is already paid
        for). >0 → restrict the rank-change mean to Hansen's subset of this
        size (``indices()`` policy). Does not change re-eval cost.
    uh_alphasigma : float
        Override for the per-measurement σ multiplier applied when noise is
        detected. 0 → pycma's principled default ``1 + 2/(N+10)``, which for a
        high-dim genome (e.g. N≈896) is tiny (~1.002) and only matters
        cumulatively. Raise it for a stronger kick — but mind the interaction
        with ``uh_every`` (fewer measurements → fewer kicks, so a larger value
        is affordable).
    uh_theta : float
        Rank-change tolerance threshold (Hansen's θ, default 0.5). Higher →
        more rank movement tolerated before noise is flagged.
    """

    algorithm: str = "cmaes"   # "cmaes" or "sep-cmaes"
    sigma0: float = 0.3
    population_size: int = 0
    tol_sigma: float = 0.0
    tol_fun: float = 0.0
    sigma_reinflate: float = 1.0

    # --- Uncertainty handling (UH-CMA-ES, Hansen et al. 2009; σ-only arm) ---
    uh_enabled: bool = False
    uh_every: int = 1
    uh_reevals: float = 0.0
    uh_alphasigma: float = 0.0
    uh_theta: float = 0.5


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
    mutate : bool
        Changes what a refresh does. When False (default) each refresh
        resamples a brand-new *random* URDF population (the behaviour above).
        When True, each refresh instead *mutates the current catalog*: every
        URDF's normalized genome is perturbed by Gaussian noise (std 0.1) per
        gene and clamped to ``[0, 1]``, then the URDFs are rebuilt from the
        mutated genomes. Mutation is cumulative across refreshes (gen 2K
        mutates the gen-K catalog) and every URDF is perturbed, including the
        standard-mydrone baseline when ``include_standard_mydrone`` is set.
        Requires a ``genomes.txt`` in the starting catalog dir (always present
        for auto-generated catalogs). Only effective in the multi-URDF path.
    mutation_std : float
        Std of the per-gene Gaussian noise applied to normalized genomes when
        ``mutate=True``. The genome lives in ``[0, 1]^D``, so ``0.1`` (default)
        is 10% of the domain. Ignored when ``mutate=False``.
    """

    path: str = ""
    num_urdfs: int = 1
    num_episodes: int = 1
    force_multi_urdf: bool = False
    include_standard_mydrone: bool = True
    refresh_urdfs_every: int = 0
    mutate: bool = False
    mutation_std: float = 0.1


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
    forest: ForestConfig = field(default_factory=ForestConfig)
    cmaes: CMAESConfig = field(default_factory=CMAESConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    seed: int = 42
    device: str = "cuda:0"
    base_dir: str = "logs/runs_hebbian"

    # ------------------------------------------------------------------
    #  Derived helpers
    # ------------------------------------------------------------------

    def hebbian_genome_dim(self) -> int:
        """Number of Hebbian genes per individual.

        Base: 4 × abcd_block_size (``out×in`` per-weight, or ``out`` per-neuron).
        +n_weights (per-weight) if ``evolve_decay=True``.
        +n_weights (per-weight) if ``evolve_eta=True``.
        Returns 0 if Hebbian is disabled.
        """
        if not self.hebbian.enabled:
            return 0
        n_weights = self.hebbian.num_actions * self.hebbian.hidden_dim
        base = 4 * self.hebbian.abcd_block_size()   # per-weight or per-neuron
        if self.hebbian.evolve_decay:
            base += n_weights   # decay stays per-weight
        if self.hebbian.evolve_eta:
            base += n_weights   # eta stays per-weight
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
            "forest": ForestConfig,
            "cmaes": CMAESConfig,
            "catalog": CatalogConfig,
            "validation": ValidationConfig,
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
