"""SNE benchmark configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class SNEBenchmarkConfig:
    """Configuration for the SNE (Scenes × N-URDFs × Envs) grid-search benchmark.

    The benchmark measures compilation and training throughput for every
    combination of ``(S, N, E)`` drawn from the three value lists below.
    """

    # ------------------------------------------------------------------ #
    # Grid dimensions — modify these in the YAML                         #
    # ------------------------------------------------------------------ #
    S_values: List[int] = field(default_factory=lambda: [1, 2])
    N_values: List[int] = field(default_factory=lambda: [1, 2])
    E_values: List[int] = field(default_factory=lambda: [64, 128])

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    num_iterations: int = 5
    """Number of PPO training iterations to run per (S, N, E) config."""

    base_cfg_path: str = "src/WP1/configs/foundation.yaml"
    """WP1 RunConfig YAML used as the base for all benchmark configurations."""

    # ------------------------------------------------------------------ #
    # Catalog                                                             #
    # ------------------------------------------------------------------ #
    catalog_dir: Optional[str] = None
    """Pre-built URDF catalog directory.  ``None`` → build a fresh one."""
    n_urdf: int = 32
    """URDFs to generate when building a fresh catalog (>= max(S)*max(N))."""
    urdf_seed: int = 42

    # ------------------------------------------------------------------ #
    # Resources                                                           #
    # ------------------------------------------------------------------ #
    device: str = "cuda:0"
    cpu_threads_per_worker: int = 4

    # ------------------------------------------------------------------ #
    # Output                                                              #
    # ------------------------------------------------------------------ #
    output_dir: str = "logs/runs_benchmark/sne_grid_search"
    exp_name: str = "sne_grid_search"

    # ------------------------------------------------------------------ #
    # Subprocess isolation timeout                                        #
    # ------------------------------------------------------------------ #
    subprocess_timeout_s: int = 1800
    """Max seconds to wait for one (S,N,E) subprocess before marking it timed-out."""

    @classmethod
    def from_yaml(cls, path: str) -> "SNEBenchmarkConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)
