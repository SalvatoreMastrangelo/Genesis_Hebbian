"""Compatibility shim providing a ``RunConfig`` API on top of ``build_cfgs``.

The legacy ``WP1.config.RunConfig`` was a nested dataclass with per-section
sub-objects (``env``, ``obs``, ``reward``, ``command``, ...) and a family of
``to_*_cfg()`` methods. The current WP1 configuration pipeline is the
function-based ``WP1.config_loader.build_cfgs``, which returns a bundle of
plain dicts.

External callers (notably ``WP2.frozen_actor``, ``WP2.evaluate``,
``WP2.evolve_cma``) still depend on the dataclass-style API. This module
keeps that API alive without resurrecting the full dataclass tree: it loads
the YAML via ``build_cfgs``, exposes the resulting dicts via ``to_*_cfg()``
methods, and wraps each dict in a small attribute-access proxy so that
expressions like ``wp1_cfg.env.x_upper`` and ``wp1_cfg.obs.num_obs`` keep
working. Assigning back through the proxy (``wp1_cfg.env.x_upper = ...``)
writes through to the underlying dict so subsequent ``to_env_cfg()`` calls
see the change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class _DictProxy:
    """Read/write attribute access over an underlying dict."""

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name in d:
            return d[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        d = object.__getattribute__(self, "_d")
        d[name] = value

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_d")

    def __repr__(self) -> str:
        return f"_DictProxy({object.__getattribute__(self, '_d')!r})"


class RunConfig:
    """Thin wrapper that gives ``build_cfgs`` output a RunConfig-style API."""

    def __init__(self, bundle: Dict[str, Any]) -> None:
        self._bundle = bundle
        self.env = _DictProxy(bundle["env_cfg"])
        self.obs = _DictProxy(bundle["obs_cfg"])
        self.reward = _DictProxy(bundle["reward_cfg"])
        self.command = _DictProxy(bundle["command_cfg"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        from WP1.config_loader import build_cfgs
        return cls(build_cfgs(path))

    def to_env_cfg(self) -> Dict[str, Any]:
        return self._bundle["env_cfg"]

    def to_obs_cfg(self) -> Dict[str, Any]:
        return self._bundle["obs_cfg"]

    def to_reward_cfg(self) -> Dict[str, Any]:
        return self._bundle["reward_cfg"]

    def to_command_cfg(self) -> Dict[str, Any]:
        return self._bundle["command_cfg"]

    def to_train_cfg(self) -> Dict[str, Any]:
        return self._bundle["train_cfg"]

    def to_legacy_cfgs(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        return (
            self._bundle["env_cfg"],
            self._bundle["obs_cfg"],
            self._bundle["reward_cfg"],
            self._bundle["command_cfg"],
            self._bundle["train_cfg"],
        )
