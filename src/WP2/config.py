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
    """

    num_eval_episodes: int = 1
    num_eval_envs: int = 8192
    vmin: float = 6.0
    vmax: float = 30.0
    stochastic: bool = True
    run_baseline: bool = False
    x_upper: Optional[float] = None  # override WP1 forest corridor length [m]
    refresh_forests_per_generation: bool = False


@dataclass
class CMAESConfig:
    """CMA-ES optimiser hyperparameters.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) treats the WP1
    reward sum as a scalar fitness and adapts a covariance matrix over the
    Hebbian-rule genome to guide search.

    Attributes
    ----------
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

    sigma0: float = 0.3
    population_size: int = 0
    tol_sigma: float = 0.0
    tol_fun: float = 0.0


@dataclass
class CatalogConfig:
    """Optional multi-URDF catalog for robustness evaluation.

    When ``path`` points to a catalog file (one URDF filename per line),
    each CMA-ES candidate is evaluated against *all* catalog URDFs and the
    fitness is averaged.  Empty string → single default URDF.

    Attributes
    ----------
    path : str
        Path to a ``catalog.txt`` file.  Empty disables multi-URDF evaluation.
    num_episodes : int
        Rollout episodes per URDF per individual.
    """

    path: str = ""
    num_episodes: int = 1


# ============================================================================
#  Top-level config
# ============================================================================

@dataclass
class HebbianEvolutionConfig:
    """Top-level configuration for a WP2 CMA-ES evolution run."""

    exp_name: str = "hebbian_cma"
    checkpoint_path: str = ""
    checkpoint_config_path: str = ""

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
        for key, val in data.items():
            if key in sub_map and isinstance(val, dict):
                sub = sub_map[key]()
                for sk, sv in val.items():
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
