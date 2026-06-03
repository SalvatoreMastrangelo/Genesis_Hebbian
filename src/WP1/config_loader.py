"""
YAML + CLI config loader for WP1.

The schema mirrors the dicts produced by ``winged_drone_train.train``:

    - ``ppo``            → train_cfg["algorithm"]
    - ``policy``         → train_cfg["policy"]
    - ``training``       → train_cfg top-level (num_steps_per_env, save_interval,
                           seed, empirical_normalization) + the local
                           ``num_envs`` / ``max_iterations`` / ``device`` knobs
    - ``env``            → env_cfg
    - ``obs``            → obs_cfg
    - ``reward``         → reward_cfg["reward_scales"]
    - ``command``        → command_cfg

A YAML file may specify any subset of these keys; missing values fall
back to the defaults defined by ``winged_drone_train.train.get_cfgs()``
and ``winged_drone_train.train.get_train_cfg()``.

CLI overrides use the standard ``--cfg.<section>.<key> <value>`` form
(dots may go arbitrarily deep, e.g. ``--cfg.obs.noise_std.depth 0.1``).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from winged_drone_train.train import get_cfgs, get_train_cfg


CFG_SECTIONS = (
    "ppo", "policy", "training", "env", "obs", "reward", "command", "catalog", "lss",
)

# Default catalog (multi-URDF training via general_policy.env_gen.Gen_Env):
#   n_urdf=None or 0          → single-URDF training (default winged_drone_train path)
#   n_urdf>0                  → build a fresh catalog of N URDFs inside the run folder
#   catalog_dir set           → reuse an existing catalog
#   include_standard_mydrone  → if True, the standard mydrone baseline morphology
#                               is inserted as the first catalog URDF; if False,
#                               the catalog is built purely from random genomes
DEFAULT_CATALOG: Dict[str, Any] = {
    "n_urdf": None,
    "catalog_dir": None,
    "urdf_seed": 0,
    "include_standard_mydrone": True,
}

# Default LSS (Logical Super-Scene = sharded multi-URDF worker training via
# general_policy.super_scene.run_logical_super_scene_training):
#   enabled=False           → use Gen_Env (single process) or single-URDF
#   enabled=True            → spawn one subprocess per URDF shard; requires a
#                             catalog (catalog.n_urdf > 0 or catalog.catalog_dir)
#                             and urdf_shard_size > 0
#   urdf_shard_size         → URDFs per worker / shard
#   num_workers=0           → auto (one worker per shard)
#   collection_gpus         → number of CUDA devices the workers cycle through
DEFAULT_LSS: Dict[str, Any] = {
    "enabled": False,
    "urdf_shard_size": 0,
    "num_workers": 0,
    "collection_gpus": 1,
}


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Read a YAML file into a plain dict. Empty files return ``{}``."""
    text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}: {path}")
    return data


def dump_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Write a dict to YAML, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


# --------------------------------------------------------------------------- #
# Deep merge
# --------------------------------------------------------------------------- #


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (returns a new dict).

    Nested dicts merge key-by-key. Any other value (lists, scalars, ``None``)
    replaces the base value outright — list concatenation is intentionally
    not supported, as overriding a list almost always means replacing it.
    """
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


# --------------------------------------------------------------------------- #
# CLI overrides
# --------------------------------------------------------------------------- #


def _coerce(value: str) -> Any:
    """Best-effort string-to-Python coercion for CLI override values."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = [_coerce(p.strip()) for p in inner.split(",")]
        return items
    return value


def parse_cli_overrides(argv: Optional[Iterable[str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Pull all ``--cfg.<a>.<b>... <value>`` pairs out of *argv*.

    Returns a (nested-dict, remaining-args) tuple. The nested dict is meant to
    be deep-merged on top of the YAML-loaded config.
    """
    args: List[str] = list(argv) if argv is not None else list(sys.argv[1:])
    nested: Dict[str, Any] = {}
    remaining: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--cfg.") and i + 1 < len(args):
            path = a[len("--cfg."):].split(".")
            value = _coerce(args[i + 1])
            cursor = nested
            for k in path[:-1]:
                cursor = cursor.setdefault(k, {})
                if not isinstance(cursor, dict):
                    raise ValueError(
                        f"--cfg override conflicts with non-dict at '{k}' in '{a}'"
                    )
            cursor[path[-1]] = value
            i += 2
            continue
        remaining.append(a)
        i += 1
    return nested, remaining


# --------------------------------------------------------------------------- #
# Build cfgs
# --------------------------------------------------------------------------- #


def _split_training_section(
    train_section: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split the YAML ``training:`` section into runner-cfg keys and "outer" keys.

    ``num_envs``, ``max_iterations``, ``device`` are consumed by the WP1 entry
    point itself; everything else (num_steps_per_env, save_interval, seed,
    empirical_normalization) is merged into the RSL-RL ``train_cfg`` dict.
    """
    outer_keys = {"num_envs", "max_iterations", "device"}
    outer: Dict[str, Any] = {}
    inner: Dict[str, Any] = {}
    for k, v in train_section.items():
        (outer if k in outer_keys else inner)[k] = v
    return outer, inner


def build_cfgs(
    yaml_path: Optional[str | Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
    *,
    exp_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose the full WP1 configuration from defaults + YAML + CLI overrides.

    Returns a single dict with the following entries:

        - ``exp_name``        : str
        - ``env_cfg``         : dict (consumed by WingedDroneEnv)
        - ``obs_cfg``         : dict (consumed by WingedDroneEnv)
        - ``reward_cfg``      : dict (consumed by WingedDroneEnv)
        - ``command_cfg``     : dict (consumed by WingedDroneEnv)
        - ``train_cfg``       : dict (consumed by OnPolicyRunner)
        - ``num_envs``        : int
        - ``max_iterations``  : int
        - ``device``          : str
        - ``yaml_payload``    : the merged YAML+CLI mapping (saved verbatim
                                inside the run folder for reproducibility)
    """
    # 1. Seed defaults straight from winged_drone_train (so any drift in
    #    those defaults is automatically picked up here).
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # train_cfg is built with placeholders; the YAML/CLI may override every
    # field, and the entry point fills in the real values at the end.
    train_cfg = get_train_cfg(exp_name="placeholder", max_iterations=1, seed=1)

    # 2. Load YAML overlay.
    yaml_data: Dict[str, Any] = {}
    if yaml_path is not None:
        yaml_data = load_yaml(yaml_path)

    # 3. Apply CLI overrides on top of the YAML payload, so CLI wins.
    if cli_overrides:
        yaml_data = deep_merge(yaml_data, cli_overrides)

    # 4. Pull out top-level scalars (exp_name) and the per-section overrides.
    exp_name = (
        exp_name_override
        or yaml_data.get("exp_name")
        or "drone-forest"
    )
    if not isinstance(exp_name, str):
        raise TypeError(f"exp_name must be a string, got {type(exp_name).__name__}")

    # 5. Merge per-section overrides into the dicts coming from
    #    winged_drone_train.
    env_cfg = deep_merge(env_cfg, yaml_data.get("env", {}) or {})
    obs_cfg = deep_merge(obs_cfg, yaml_data.get("obs", {}) or {})

    reward_override = yaml_data.get("reward", {}) or {}
    if reward_override:
        scales = dict(reward_cfg.get("reward_scales", {}))
        scales = deep_merge(scales, reward_override)
        reward_cfg = {"reward_scales": scales}

    command_cfg = deep_merge(command_cfg, yaml_data.get("command", {}) or {})

    # Catalog (multi-URDF training). Defaults to single-URDF mode.
    catalog_cfg = deep_merge(DEFAULT_CATALOG, yaml_data.get("catalog", {}) or {})

    # LSS (sharded multi-URDF training across worker subprocesses).
    lss_cfg = deep_merge(DEFAULT_LSS, yaml_data.get("lss", {}) or {})

    # 6. Apply PPO / policy / training overrides to the RSL-RL train_cfg.
    ppo_override = yaml_data.get("ppo", {}) or {}
    policy_override = yaml_data.get("policy", {}) or {}
    training_section = yaml_data.get("training", {}) or {}

    outer_training, inner_training = _split_training_section(training_section)

    train_cfg["algorithm"] = deep_merge(train_cfg["algorithm"], ppo_override)
    train_cfg["policy"] = deep_merge(train_cfg["policy"], policy_override)

    # Top-level RSL-RL keys (num_steps_per_env, save_interval, seed, etc.)
    train_cfg = deep_merge(train_cfg, inner_training)

    # 7. Resolve the outer scalars (with sane fallbacks).
    num_envs = int(outer_training.get("num_envs", 16384))
    max_iterations = int(outer_training.get("max_iterations", 1000))
    device = str(outer_training.get("device", "cuda:0"))

    seed = int(train_cfg.get("seed", 1))
    train_cfg["seed"] = seed

    # Plumb experiment name / max_iterations through the runner block so
    # that RSL-RL records the right tags in TensorBoard.
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["experiment_name"] = exp_name
    train_cfg["runner"]["max_iterations"] = max_iterations

    # Did the user (YAML or CLI) explicitly pin the seed? If not, the entry
    # point will call ``seed_runtime_randomness`` for OS-entropy randomness,
    # matching the wrapped script's default behaviour.
    user_seed_explicit = "seed" in training_section

    return {
        "exp_name": exp_name,
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
        "catalog_cfg": catalog_cfg,
        "lss_cfg": lss_cfg,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "device": device,
        "yaml_payload": yaml_data,
        "user_seed_explicit": user_seed_explicit,
    }
