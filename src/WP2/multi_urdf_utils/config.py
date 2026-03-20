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
    """Core benchmark parameters controlling scene layout.

    N : URDFs per scene (keep ≤ 40 to avoid Taichi compilation limits)
    S : number of scenes (run sequentially; each gets its own gs.init/destroy)
    E : environments per scene shared by all N entities (each of the E envs
        contains all N drones; one scene has N × E total drone instances)
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
    """Paths to frozen WP1 actor checkpoint."""
    model_path: str = ""
    config_path: str = ""


@dataclass
class HebbianBenchConfig:
    """Hebbian plasticity settings for the benchmark."""
    enabled: bool = True
    eta: float = 0.01
    w_max: float = 3.0


@dataclass
class EnvBenchConfig:
    """Environment physics and forest parameters."""
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
    """URDF catalog generation settings."""
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
        data = _tuples_to_lists(asdict(self))
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
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
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(item) for item in obj]
    return obj


def _cast(val_str: str, reference: Any) -> Any:
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
