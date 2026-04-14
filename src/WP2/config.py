"""
HebbianEvolutionConfig — unified configuration for WP2 runs.
=============================================================

Motivation
----------
WP2 replaces PPO-based training with **Hebbian plasticity + evolutionary
co-optimisation**.  A frozen actor (pretrained in WP1) has its last-layer
weights modulated at runtime by per-weight ABCD Hebbian rules, while an
NSGA-II loop searches for the best combination of Hebbian rule-genes and
(optionally) morphology genes under multiple fitness objectives.

This module mirrors the WP1 ``RunConfig`` pattern: a nested dataclass
hierarchy that can be:

1. **serialised to / deserialised from YAML** (``to_yaml`` / ``from_yaml``),
2. **overridden from the CLI** with ``--cfg.section.key value`` syntax,
3. **queried** for derived properties like genome dimensions and active
   objective names.

Hierarchy
---------
::

    HebbianEvolutionConfig
    ├── exp_name              str                experiment tag (log folder name)
    ├── checkpoint_path       str                path to frozen WP1 actor weights
    ├── checkpoint_config_path str               path to matching WP1 config YAML
    ├── hebbian               HebbianConfig      ABCD plasticity rule ranges & eta
    ├── evolution             EvolutionConfig    NSGA-II population, crossover, mutation
    ├── evaluation            EvaluationConfig   rollout episodes, envs, speed commands
    ├── objectives            ObjectivesConfig   toggle individual fitness objectives
    ├── morphology            MorphologyConfig   enable/disable morphology co-evolution
    ├── seed                  int                global random seed
    ├── device                str                torch device string
    └── base_dir              str                root directory for run logs

Usage
-----
.. code-block:: python

    from WP2.config import HebbianEvolutionConfig

    # Defaults
    cfg = HebbianEvolutionConfig()

    # From YAML
    cfg = HebbianEvolutionConfig.from_yaml("configs/full_codesing.yaml")

    # CLI overrides
    cfg.apply_cli_overrides(["--cfg.evolution.population_size", "80"])

    # Snapshot for reproducibility
    cfg.to_yaml("logs/runs_hebbian/.../config.yaml")

    # Query derived properties
    print(cfg.active_objective_names())   # ['velocity', 'energy', 'progress']
    print(cfg.total_genome_dim())         # 1792 + 15 = 1807 (defaults: 7 actions, 64 hidden, no decay/eta evolution)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, asdict
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

    Attributes
    ----------
    enabled : bool
        Enable Hebbian plasticity.  If False, no Hebbian genome section is
        created and the actor's weights remain static.
    eta : float
        Global learning rate for Hebbian updates (scalar, applied to all
        weights uniformly).  Typical range: 0.001-0.1.  Ignored if
        ``evolve_eta=True`` (per-weight eta evolved instead).
    evolve_eta : bool
        Whether to evolve per-weight eta values (one per synaptic weight).
        If True, genome includes an additional ``n_weights`` genes
        (``num_actions × hidden_dim``, 448 by default) for per-weight learning
        rates rescaled from ``eta_range``.
    decay : float
        Global decay (lambda) coefficient for weight decay.  Typical range: 0.0-0.1.
        Ignored if ``evolve_decay=True`` (per-weight decay evolved instead).
    evolve_decay : bool
        Whether to evolve per-weight decay (lambda) values (one per synaptic weight).
        If True, genome includes an additional ``n_weights`` genes
        (``num_actions × hidden_dim``, 448 by default) for per-weight decay
        rescaled from ``decay_range``. If False, all weights use the global
        decay value. Default False.
    use_oja_coefficient : bool
        Whether to apply Oja-like coefficient k to modulate Hebbian plasticity.
        When True, dW = eta * k * [ABCD], where k accounts for weight drift from
        the frozen base controller.  When False, k = 1 (standard ABCD rule).
        Default True.
    initialize_rules_to_zero : bool
        Initialize all Hebbian rules (A, B, C, D) to 0.0 in the first generation.
        If True, the initial population has zero Hebbian plasticity; evolution
        then searches for non-zero rules. Default False (uniform random [0,1]).
    w_max : float
        Symmetric weight clipping bound; weights are clamped to [-w_max, w_max]
        after each Hebbian update to prevent instability.  Typical: 3.0.
    A_range : Tuple[float, float]
        Range [low, high] for the A coefficient (classical Hebbian term,
        proportional to pre×post activation product).  Default (-1.0, 1.0)
        allows both Hebbian and anti-Hebbian rules.
    B_range : Tuple[float, float]
        Range for the B coefficient (presynaptic-only term).
        Default (-1.0, 1.0).
    C_range : Tuple[float, float]
        Range for the C coefficient (postsynaptic-only term).
        Default (-1.0, 1.0).
    D_range : Tuple[float, float]
        Range for the D coefficient (bias/drift term, independent of activations).
        Default (-1.0, 1.0).
    decay_range : Tuple[float, float]
        Range for the decay (lambda) coefficient when ``evolve_decay=True``.
        Rescaled to [0, 0.1] by default (decay is always non-negative).
    eta_range : Tuple[float, float]
        Range for per-weight eta if ``evolve_eta=True``.  Default (0.0, 0.1).
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
    """NSGA-II multi-objective genetic algorithm hyperparameters.

    Configures the Non-dominated Sorting Genetic Algorithm II (NSGA-II) used
    to search the space of Hebbian rules and (optionally) morphologies.
    NSGA-II maintains a population, applies variation operators (crossover &
    mutation), and selects survivors based on Pareto dominance and crowding
    distance.

    Attributes
    ----------
    population_size : int
        Number of individuals per generation.  Must be a multiple of 4 for
        the tournament selection operator (tournamentDCD).  Typical: 40-100.
    num_generations : int
        Number of generations (outer NSGA-II loop iterations).  Each
        generation evaluates new individuals and updates the population.
        Typical: 30-100.
    enable_crossover : bool
        Whether to apply simulated-binary crossover (SBX) to parents.
        Can be disabled for very high-dimensional genomes (e.g., >1000)
        where crossover may be less beneficial.
    crossover_probability : float
        Probability that two selected parents undergo crossover in [0, 1].
        Default 0.9 means 90% of parent pairs are crossed over; 10% are
        cloned unchanged.
    mutation_probability : float
        Probability that an individual undergoes polynomial mutation.
        Default 1.0 means all offspring are mutated (standard in NSGA-II).
    eta_c : float
        Distribution index for simulated-binary crossover.  Controls how
        similar offspring are to their parents (lower → more variation).
        Typical: 10-30.  Default 20.0.
    eta_m : float
        Distribution index for polynomial mutation (only used if operator="polynomial").
        Controls mutation step sizes (lower → larger steps).  Typical: 10-30.  Default 20.0.
    mutation_operator : str
        Type of mutation operator: "polynomial" (eta-based) or "gaussian" (sigma-based).
        Default "polynomial" (NSGA-II standard). Use "gaussian" for direct std dev control.
    mutation_sigma : float
        Standard deviation for Gaussian mutation (only used if operator="gaussian").
        Controls mutation step sizes. Typical: 0.01-0.1.  Default 0.05.
    weights : Tuple[float, ...]
        Fitness weights for DEAP fitness objects (all maximisation by
        convention).  Must have at least as many entries as the number of
        active objectives.  Missing objectives default to 1.0.
        Example: (1.0, 1.0, 1.0) for 3 objectives with equal importance.
    """

    strategy: str = "nsga2"       # "nsga2" or "cma_es"
    population_size: int = 40
    num_generations: int = 30
    enable_crossover: bool = True
    crossover_probability: float = 0.9
    mutation_probability: float = 1.0
    eta_c: float = 20.0
    eta_m: float = 20.0
    mutation_operator: str = "polynomial"
    mutation_sigma: float = 0.05
    weights: Tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass
class EvaluationConfig:
    """Population-level vectorized rollout configuration for fitness evaluation.

    In batched evaluation mode (the default), all individuals in a generation
    evaluate simultaneously using a shared pool of environments. The pool is
    divided into slices, one per individual: each individual gets
    ``num_eval_envs // population_size`` environments. Fitness is aggregated
    over ``num_eval_episodes`` passes through the environment.

    Attributes
    ----------
    num_eval_episodes : int
        Number of independent passes through the environment pool.  Fitness is
        aggregated (mean) over all passes.  In batched mode, each individual
        effectively samples ``num_eval_envs // pop_size * num_eval_episodes``
        environment trajectories.  Higher values give more stable fitness
        estimates at the cost of longer evaluation time.  Typical: 1-5.
    num_eval_envs : int
        Total number of parallel Genesis environments in the shared pool.
        These are divided among all individuals in the population.
        Genesis simulates these vectorised, so larger values use more VRAM but
        evaluate faster (better GPU utilisation).  Typical: 4096-16384 on
        modern GPUs.
    vmin : float
        Minimum commanded forward velocity (m/s) during rollout.  Velocities
        are sampled uniformly from [vmin, vmax] and reset randomly during
        episodes to test diverse flight regimes.
    vmax : float
        Maximum commanded forward velocity (m/s).  Typical range: 5-40.
    stochastic : bool
        If True, actor samples from its policy distribution (exploration).
        If False, uses the policy mean (deterministic).  Default True
        (stochasticity helps discover diverse controller behaviors).
    """

    num_eval_episodes: int = 1
    num_eval_envs: int = 8192
    vmin: float = 6.0
    vmax: float = 30.0
    stochastic: bool = True


@dataclass
class ObjectivesConfig:
    """Multi-objective fitness function configuration.

    Each objective produces a single scalar value per individual (aggregated
    over episodes).  NSGA-II searches for a Pareto-optimal front trading off
    these objectives.  All objectives are oriented for maximisation; costs
    (energy, crash_rate) are negated during evaluation.

    Attributes
    ----------
    velocity : bool
        Mean forward velocity [m/s] across episodes.  Encourages fast flight.
        Computed as total_distance / total_time.
    energy : bool
        Negative total energy consumption [J].  Stored as -E_tot so
        maximisation minimises energy.  Encourages efficient control.
    progress : bool
        Mean forward distance [m] per episode.  Incentivises the controller
        to move the drone forward.
    smoothness : bool
        Negative action jerk (second time derivative).  Stored as -jerk so
        maximisation minimises jerky control and encourages graceful flight.
    crash_rate : bool
        Negative fraction of episodes ending in a crash.  Stored as -rate
        (in [0, 1]) so maximisation minimises crashes.  Encourages robust
        controllers.
    """

    velocity: bool = True
    energy: bool = True
    progress: bool = True
    smoothness: bool = False
    crash_rate: bool = False


@dataclass
class CMAESConfig:
    """CMA-ES optimiser hyperparameters (used when evolution.strategy = "cma_es").

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a single-objective
    optimiser.  It treats the WP1 reward sum as the scalar fitness and adapts a
    covariance matrix over the Hebbian-rule genome to guide search.

    Attributes
    ----------
    sigma0 : float
        Initial step size (standard deviation) for the search distribution.
        The genome lives in [0,1], so a starting sigma of ~0.3 spans a
        third of the domain.  Typical: 0.1–0.5.
    population_size : int
        Number of candidate solutions sampled each generation (CMA-ES "lambda").
        Set to 0 to let pycma choose automatically using the formula
        ``4 + floor(3 * ln(n_genes))``.  Override when you want more diversity
        (larger) or faster iterations (smaller).
    tol_sigma : float
        Convergence threshold: stop when sigma drops below this value.
        pycma default ``tolsigma = 1e-11``.
    tol_fun : float
        Convergence threshold: stop when the function-value spread across the
        current population drops below this value.
        pycma default ``tolfun = 1e-11``.
    """

    sigma0: float = 0.3
    population_size: int = 0     # 0 = auto (4 + floor(3*ln(n_genes)))
    tol_sigma: float = 1e-9
    tol_fun: float = 1e-11


@dataclass
class CatalogConfig:
    """URDF catalog for CMA-ES rules-only evaluation.

    When ``path`` points to a catalog file (one URDF filename per line, as
    produced by WP1 multi-morphology training), each CMA-ES candidate is
    evaluated against *all* catalog URDFs and the fitness is averaged.
    This encourages Hebbian rules that generalise across morphologies.

    If ``path`` is empty, a single default URDF (from
    ``morphology.fixed_genome`` or the WP1 standard genome) is used.

    Attributes
    ----------
    path : str
        Path to a catalog.txt file.  Each line is a filename of the form
        ``[p1, p2, ..., p15].urdf`` where the values are the physical genome
        parameters.  The URDF files are expected to live in the same directory
        as catalog.txt.  Empty string disables multi-URDF evaluation.
    num_episodes : int
        Number of independent rollout episodes per URDF per individual.
        Fitness is averaged across episodes and URDFs.  Default 1.
    """

    path: str = ""
    num_episodes: int = 1


@dataclass
class MorphologyConfig:
    """Morphology co-evolution configuration.

    Controls whether the genetic algorithm also optimises drone morphology
    (wing shape, mass distribution, etc.) alongside Hebbian rules, or
    evolves only Hebbian rules on a fixed morphology.

    Attributes
    ----------
    evolve : bool
        If True, include morphology genes in the genome (15 NACA parameters
        per ``Chromosome_Drone``), and the GA evolves morphology alongside
        Hebbian rules.  If False, morphology is fixed and only Hebbian
        parameters are evolved.
    fixed_genome : Optional[List[float]]
        Normalised morphology genome [0, 1]^15 to use if ``evolve=False``.
        If None, falls back to the WP1 default morphology (STANDARD_MYDRONE_GENOME).
    """

    evolve: bool = True
    fixed_genome: Optional[List[float]] = None


# ============================================================================
#  Top-level config
# ============================================================================

@dataclass
class HebbianEvolutionConfig:
    """Top-level configuration object for a complete WP2 evolution run.

    ``HebbianEvolutionConfig`` aggregates all sub-configs and provides
    derived properties for genome dimensions and active objectives.
    It replaces hardcoded parameter sets in the evolution loop and enables:

    1. **YAML persistence** — ``to_yaml()`` / ``from_yaml()`` for config
       snapshots and reproducibility.
    2. **CLI override** — ``apply_cli_overrides()`` parses ``--cfg.section.key``
       arguments for quick hyperparameter sweeps without editing YAML.
    3. **Derived queries** — ``active_objective_names()``, ``total_genome_dim()``,
       etc., to avoid recomputing these values in multiple places.

    Attributes
    ----------
    exp_name : str
        Experiment tag used in log folder names and run metadata.
        Default: "hebbian_codesing".
    checkpoint_path : str
        Path to the frozen WP1 actor checkpoint (.pt file).
        Empty string disables Hebbian evolution (not recommended).
    checkpoint_config_path : str
        Path to the matching WP1 RunConfig YAML.  Used to reconstruct
        the environment, observation, and reward configs for evaluation.
    hebbian : HebbianConfig
        Hebbian plasticity rule configuration (ABCD parameters, ranges, eta).
    evolution : EvolutionConfig
        NSGA-II hyperparameters (population size, generations, crossover, etc.).
    evaluation : EvaluationConfig
        Rollout settings (num episodes, num envs, velocity range, stochasticity).
    objectives : ObjectivesConfig
        Toggle flags for each fitness objective (velocity, energy, progress, etc.).
    morphology : MorphologyConfig
        Enable morphology evolution and/or specify a fixed morphology.
    seed : int
        Global random seed for reproducibility (numpy, torch, DEAP, genesis).
    device : str
        Torch device string (e.g., "cuda:0", "cpu").
    base_dir : str
        Root directory where timestamped run folders are created.
        Default: "logs/runs_hebbian".
    """

    exp_name: str = "hebbian_codesing"
    checkpoint_path: str = ""
    checkpoint_config_path: str = ""

    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    cmaes: CMAESConfig = field(default_factory=CMAESConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)

    seed: int = 42
    device: str = "cuda:0"
    base_dir: str = "logs/runs_hebbian"

    # ------------------------------------------------------------------
    #  Derived helpers
    # ------------------------------------------------------------------

    def active_objective_names(self) -> List[str]:
        """Return the list of active objective names in order.

        The order matches the order of objectives in the ``ObjectivesConfig``
        and is used by ``fitness_weights()`` and the DEAP fitness tuple.

        Returns
        -------
        List[str]
            List of objective names (e.g., ['velocity', 'energy', 'progress']).
        """
        names = []
        if self.objectives.velocity:
            names.append("velocity")
        if self.objectives.energy:
            names.append("energy")
        if self.objectives.progress:
            names.append("progress")
        if self.objectives.smoothness:
            names.append("smoothness")
        if self.objectives.crash_rate:
            names.append("crash_rate")
        return names

    def num_active_objectives(self) -> int:
        """Return the number of active objectives.

        Returns
        -------
        int
            Number of enabled objectives (e.g., 3 if velocity, energy, progress
            are enabled).
        """
        return len(self.active_objective_names())

    def fitness_weights(self) -> Tuple[float, ...]:
        """Return DEAP fitness weights matching active objectives.

        All objectives are maximised by convention in DEAP (energy and
        crash_rate are stored negated so that maximisation is semantically
        correct: maximising -cost minimises cost).

        The weights can be used to specify importance levels (e.g., prioritise
        velocity over smoothness by setting velocity weight > smoothness weight).

        Returns
        -------
        Tuple[float, ...]
            Fitness weights, one per active objective, in the same order as
            ``active_objective_names()``.  If the config specifies fewer
            weights than objectives, missing weights default to 1.0.
        """
        n = self.num_active_objectives()
        w = self.evolution.weights
        if len(w) >= n:
            return tuple(w[:n])
        # Pad with 1.0 if config has fewer weights than objectives
        return tuple(list(w) + [1.0] * (n - len(w)))

    def hebbian_genome_dim(self) -> int:
        """Number of Hebbian genes per individual.

        The Hebbian genome encodes per-weight ABCD rules for every weight in
        the frozen last layer (``num_actions`` outputs × ``hidden_dim`` inputs
        = ``n_weights`` weights; 7 × 64 = 448 with defaults).  Each weight
        contributes 4 base genes (A, B, C, D).  Lambda (decay) is either a
        global config value (if ``evolve_decay=False``) or evolved per-weight
        (if ``evolve_decay=True``). Similarly, eta (learning rate) is either
        global (if ``evolve_eta=False``) or per-weight (if ``evolve_eta=True``).

        Returns
        -------
        int
            Genome section size.  0 if Hebbian plasticity is disabled;
            1792 (4 × 448) with defaults if enabled and neither eta nor decay evolved;
            2240 (4 × 448 + 448) if eta or decay is evolved (but not both);
            2688 (4 × 448 + 448 + 448) if both eta and decay are evolved.
            (Values scale proportionally when ``num_actions`` or ``hidden_dim``
            differ from the 7 × 64 defaults.)
        """
        if not self.hebbian.enabled:
            return 0
        n_weights = self.hebbian.num_actions * self.hebbian.hidden_dim
        base = 4 * n_weights  # A, B, C, D per weight (always present)
        if self.hebbian.evolve_decay:
            base += n_weights  # per-weight decay (lambda) only if evolved
        if self.hebbian.evolve_eta:
            base += n_weights  # per-weight eta only if evolved
        return base

    def morphology_genome_dim(self) -> int:
        """Number of morphology genes per individual.

        The morphology genome encodes a 15-dimensional normalised vector
        representing wing shape, mass distribution, and aerodynamic parameters
        (``Chromosome_Drone`` encoding).  Each gene is in [0, 1] and is
        converted to physical parameters during phenotype construction.

        Returns
        -------
        int
            Genome section size.  0 if morphology evolution is disabled;
            15 if enabled.
        """
        if not self.morphology.evolve:
            return 0
        return 15  # Chromosome_Drone.num_genes()

    def total_genome_dim(self) -> int:
        """Total genome dimension across all sections.

        The full genome is laid out as:
            [Hebbian genes (0, 1792, 2240, 2688) | Morphology genes (0 or 15)]

        With default settings (num_actions=7, hidden_dim=64, evolve_eta=False,
        evolve_decay=False, morphology.evolve=True): 1792 + 15 = 1807.

        Returns
        -------
        int
            Total number of genes per individual (range: 0–2703 with defaults).
        """
        return self.hebbian_genome_dim() + self.morphology_genome_dim()

    # ------------------------------------------------------------------
    #  YAML I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialise this config to a YAML string.

        Tuples (e.g., ``A_range``) are converted to lists for YAML
        compatibility.

        Parameters
        ----------
        path : Path, optional
            If given, the YAML is also written to this file (parent dirs
            are created automatically).

        Returns
        -------
        str
            The YAML representation of the full config.
        """
        data = _tuples_to_lists(asdict(self))
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HebbianEvolutionConfig":
        """Deserialise a ``HebbianEvolutionConfig`` from a YAML file.

        Missing keys fall back to their dataclass defaults, so a YAML file
        only needs to specify the values that differ from the defaults.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        HebbianEvolutionConfig
            Fully-populated configuration object.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "HebbianEvolutionConfig":
        """Recursively construct a ``HebbianEvolutionConfig`` from a nested dict.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary (typically from ``yaml.safe_load()``).  Keys that do
            not correspond to a known field are silently ignored.

        Returns
        -------
        HebbianEvolutionConfig
            Populated config with defaults for any missing keys.
        """
        cfg = cls()
        sub_map = {
            "hebbian": HebbianConfig,
            "evolution": EvolutionConfig,
            "evaluation": EvaluationConfig,
            "objectives": ObjectivesConfig,
            "morphology": MorphologyConfig,
            "cmaes": CMAESConfig,
            "catalog": CatalogConfig,
        }
        for key, val in data.items():
            if key in sub_map and isinstance(val, dict):
                sub = sub_map[key]()
                for sk, sv in val.items():
                    if hasattr(sub, sk):
                        current = getattr(sub, sk)
                        # Convert lists back to tuples where the default is a tuple
                        if isinstance(current, tuple) and isinstance(sv, list):
                            sv = tuple(sv)
                        setattr(sub, sk, sv)
                setattr(cfg, key, sub)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    # ------------------------------------------------------------------
    #  CLI override support
    # ------------------------------------------------------------------

    def apply_cli_overrides(self, argv: Optional[List[str]] = None) -> None:
        """Apply ``--cfg.section.key value`` overrides from the command line.

        Scans *argv* for arguments of the form ``--cfg.<section>.<key>``
        followed by a value token.  The value is automatically cast to the
        type of the existing field (bool, int, float, list, or tuple).

        Top-level fields (e.g. ``exp_name``, ``seed``) can be set with a
        single dot: ``--cfg.exp_name my-experiment``.

        Parameters
        ----------
        argv : List[str], optional
            Argument list to scan.  Defaults to ``sys.argv[1:]``.

        Examples
        --------
        .. code-block:: bash

            --cfg.hebbian.eta 0.05
            --cfg.evolution.population_size 80
            --cfg.evolution.num_generations 50
            --cfg.objectives.smoothness true
            --cfg.exp_name my-test
        """
        if argv is None:
            argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith("--cfg."):
                parts = arg[len("--cfg."):].split(".")
                if len(parts) == 2 and i + 1 < len(argv):
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
    """Recursively convert tuples to lists for YAML-safe serialisation.

    YAML does not distinguish tuples from lists, so this helper converts
    all tuples to lists before dumping to YAML, ensuring lossless round-trip
    serialisation (lists are converted back to tuples during deserialization
    if the field's default is a tuple).

    Parameters
    ----------
    obj : Any
        Object (typically a nested dict from ``asdict(config)``).

    Returns
    -------
    Any
        Object with all tuples recursively replaced by lists.
    """
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(item) for item in obj]
    return obj


def _cast(val_str: str, reference: Any) -> Any:
    """Cast a CLI string value to match the type of *reference*.

    Handles ``bool``, ``int``, ``float``, ``tuple`` (comma-separated),
    ``list`` (comma-separated), and falls back to ``str``.

    Parameters
    ----------
    val_str : str
        Raw string from the command line.
    reference : Any
        The current value of the field — its type determines the cast.

    Returns
    -------
    Any
        The value cast to the appropriate Python type.

    Examples
    --------
    >>> _cast("true", False)
    True
    >>> _cast("42", 0)
    42
    >>> _cast("3.14", 0.0)
    3.14
    >>> _cast("1.0, 2.0", (-1.0, 1.0))
    (1.0, 2.0)
    >>> _cast("[64, 64]", [32, 32])
    [64, 64]
    """
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
