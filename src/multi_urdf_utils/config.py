"""
BenchmarkConfig — configuration for WP2.5 multi-drone benchmark runs.
======================================================================

Mirrors the WP1/WP2 config pattern: nested dataclasses, YAML I/O, CLI overrides.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class BenchmarkParams:
    """Core benchmark parameters controlling scene layout and rollout budget.

    Attributes
    ----------
    N : int
        URDFs per scene.  Each scene places N distinct drone morphologies in
        the same Genesis scene.  Keep N ≤ 40 to avoid Taichi compilation limits.
    S : int
        Number of scenes (each scene is a separate gs.init/destroy cycle run
        sequentially or in parallel worker processes).
    E : int
        Environments per scene, shared by all N drone entities.  One scene
        has N × E total drone instances (each of the E slots has all N drones).
    num_episodes : int
        Rollout episodes per scene.  Fitness is averaged across episodes.
    max_steps : int
        Maximum timesteps per episode (hard cut-off if the episode doesn't
        terminate earlier via crash or success).
    seed : int
        Base random seed for URDF generation and environment resets.
    num_workers : int
        Number of parallel scene worker processes.  0 = auto (cpu_count //
        cpu_threads_per_worker).  Set to 1 for sequential execution.
    cpu_threads_per_worker : int
        CPU threads given to each Taichi compiler (TI_NUM_THREADS env var).
        2–4 is usually sufficient; tune for your CPU core count.
    """
    N: int = 4          # URDFs per scene
    S: int = 10         # number of scenes
    E: int = 256        # environments per scene (shared by all N entities)
    num_episodes: int = 3
    max_steps: int = 500
    seed: int = 42
    num_workers: int = 0              # parallel scene workers (0 = auto: cpu_count // cpu_threads_per_worker)
    cpu_threads_per_worker: int = 4   # CPU threads given to each Taichi compiler (TI_NUM_THREADS)


@dataclass
class CheckpointConfig:
    """Paths to a frozen WP1 actor checkpoint.

    Attributes
    ----------
    model_path : str
        Path to the .pt checkpoint file produced by WP1 training.
        e.g. ``logs/runs/2026-03-20_.../tb/model_999.pt``.
    config_path : str
        Path to the matching WP1 ``config.yaml`` snapshot (stored alongside
        the checkpoint).  Used to reconstruct env/obs/reward configs for
        evaluation.
    """
    model_path: str = ""
    config_path: str = ""


@dataclass
class HebbianBenchConfig:
    """Hebbian plasticity settings for the benchmark.

    Attributes
    ----------
    enabled : bool
        If False, use the frozen WP1 policy without any Hebbian updates
        (pure baseline evaluation).  If True, apply random Hebbian rules to
        each drone's last layer during rollout.
    eta : float
        Global Hebbian learning rate (scalar, applied to all weights).
    w_max : float
        Symmetric weight clipping bound applied after each Hebbian update.
    """
    enabled: bool = True
    eta: float = 0.01
    w_max: float = 3.0


@dataclass
class EnvBenchConfig:
    """Environment physics and forest parameters for the benchmark.

    These should match (or be compatible with) the WP1 training settings so
    that the loaded checkpoint is evaluated in a familiar regime.

    Attributes
    ----------
    dt : float
        Physics timestep in seconds.  Must equal WP1's control dt (0.04 s
        = 25 Hz control frequency; internally Genesis uses substeps=4 × 0.01 s).
    substeps : int
        Genesis substeps per control step.  dt × substeps gives the total
        physics integration time per step.
    episode_length_s : float
        Maximum episode duration in seconds.
    forest_density_min, forest_density_max : float
        Range for the Poisson forest density (trees/m²) sampled each reset.
    growing_forest : bool
        If True, tree density increases linearly with forward distance (matching
        WP1 curriculum training).
    tree_radius : float
        Cylinder radius for collision detection (m).
    tree_height : float
        Height of tree cylinders in the Genesis scene (m).
    base_init_pos : List[float]
        Initial drone position [x, y, z] in meters.
    base_init_quat : List[float]
        Initial orientation quaternion [w, x, y, z].
    y_lower, y_upper : float
        Lateral corridor bounds (m); drones are terminated if |y| exceeds this.
    forest_x_limit : float
        Forward distance at which the forest ends (m); crossing triggers success.
    x_upper : float
        Hard upper x-bound for episode termination (m).
    aero_noise : bool
        If True, add Gaussian noise to aerodynamic forces for robustness testing.
    vmin, vmax : float
        Forward velocity command range (m/s) sampled uniformly each episode reset.
    """
    dt: float = 0.04
    substeps: int = 4
    episode_length_s: float = 20.0
    forest_density_min: float = 0.0
    forest_density_max: float = 3.0
    growing_forest: bool = True
    tree_radius: float = 0.75
    tree_height: float = 100.0
    base_init_pos: List[float] = field(default_factory=lambda: [-30.0, 0.0, 15.0])
    base_init_quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    y_lower: float = -50.0
    y_upper: float = 50.0
    forest_x_limit: float = 150.0
    x_upper: float = 150.0
    aero_noise: bool = False
    vmin: float = 6.0
    vmax: float = 30.0


@dataclass
class CatalogBenchConfig:
    """URDF catalog settings for the benchmark.

    Attributes
    ----------
    catalog_dir : str
        Directory containing the URDF catalog.  Either a pre-existing folder
        with a ``catalog.txt`` listing URDF filenames, or an output directory
        where URDFs are generated on first run.
    urdf_seed : int
        Random seed used for URDF generation when creating a new catalog.
    """
    catalog_dir: str = "logs/.cache/wp2_5_urdfs"
    urdf_seed: int = 42


@dataclass
class BenchmarkConfig:
    """Top-level configuration for a WP2.5 benchmark run."""

    exp_name: str = "multi_drone_benchmark"
    benchmark: BenchmarkParams = field(default_factory=BenchmarkParams)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    hebbian: HebbianBenchConfig = field(default_factory=HebbianBenchConfig)
    env: EnvBenchConfig = field(default_factory=EnvBenchConfig)
    catalog: CatalogBenchConfig = field(default_factory=CatalogBenchConfig)
    device: str = "cuda:0"
    base_dir: str = "logs/runs_benchmark"

    @property
    def total_entities(self) -> int:
        """Total URDF entities across all scenes (N per scene × S scenes)."""
        return self.benchmark.N * self.benchmark.S

    @property
    def total_instances(self) -> int:
        """Total drone instances across all scenes (S scenes × N entities × E envs)."""
        return self.benchmark.S * self.benchmark.N * self.benchmark.E

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialise this config to a YAML string.

        Parameters
        ----------
        path : Path, optional
            If provided, write the YAML to this file (parent dirs created
            automatically).

        Returns
        -------
        str
            YAML representation of the full config.
        """
        data = _tuples_to_lists(asdict(self))
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Deserialise a ``BenchmarkConfig`` from a YAML file.

        Missing keys fall back to their dataclass defaults, so a YAML file
        only needs to specify the values that differ from the defaults.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        BenchmarkConfig
            Fully-populated configuration object.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Recursively construct a ``BenchmarkConfig`` from a nested dict.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary (typically from ``yaml.safe_load()``). Unknown keys are
            silently ignored; missing keys keep their dataclass defaults.

        Returns
        -------
        BenchmarkConfig
            Populated config with defaults for any missing keys.
        """
        cfg = cls()
        sub_map = {
            "benchmark": BenchmarkParams,
            "checkpoint": CheckpointConfig,
            "hebbian": HebbianBenchConfig,
            "env": EnvBenchConfig,
            "catalog": CatalogBenchConfig,
        }
        for key, val in data.items():
            if key in sub_map and isinstance(val, dict):
                sub = sub_map[key]()
                for sk, sv in val.items():
                    if hasattr(sub, sk):
                        setattr(sub, sk, sv)
                setattr(cfg, key, sub)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def apply_cli_overrides(self, argv: Optional[List[str]] = None) -> None:
        """Apply ``--cfg.section.key value`` overrides from the command line.

        Scans *argv* for arguments of the form ``--cfg.<section>.<key>``
        followed by a value token.  The value is automatically cast to the
        type of the existing field (bool, int, float, or list).

        Top-level fields (e.g. ``exp_name``, ``device``) can be overridden
        with a single dot: ``--cfg.exp_name my-run``.

        Parameters
        ----------
        argv : List[str], optional
            Argument list to scan.  Defaults to ``sys.argv[1:]``.

        Examples
        --------
        .. code-block:: bash

            --cfg.benchmark.N 8
            --cfg.benchmark.E 512
            --cfg.env.episode_length_s 30.0
            --cfg.hebbian.enabled false
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

    YAML does not distinguish tuples from lists.  This ensures lossless
    round-trip when the config is serialised and later reloaded.

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

    Handles ``bool``, ``int``, ``float``, ``list`` (comma-separated),
    and falls back to ``str``.

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
    >>> _cast("8", 1)
    8
    >>> _cast("0.04", 0.01)
    0.04
    >>> _cast("[64, 64]", [32, 32])
    [64, 64]
    """
    if isinstance(reference, bool):
        return val_str.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(val_str)
    if isinstance(reference, float):
        return float(val_str)
    if isinstance(reference, list):
        val_str = val_str.strip("[]")
        items = [s.strip() for s in val_str.split(",")]
        if reference and isinstance(reference[0], int):
            return [int(x) for x in items]
        return [float(x) for x in items]
    return val_str
