"""
Tests for exam-time forest overrides (`outer.exam_forest`).

Covers: the ExamForestConfig section (defaults, YAML round-trip, validation),
runtime overrides on a real ForestGenerator (density / length / mode take
effect on the next generate() and restore by re-applying the returned
previous values), the WingedDroneEnv method incl. the eval success-line sync
for x_upper, and the _run_exam apply→refresh→eval→restore wiring. No Genesis
scene is ever built.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.perception.forest import (
    ForestGenerator,
    apply_generator_overrides,
    generate_forests,
)

from WP2_Outer_Loop.config import ExamForestConfig, OuterNSGA2Config
from WP2_Outer_Loop.nsga_cma import NSGA2MorphCMAES


def _cfg(**outer_kw):
    cfg = OuterNSGA2Config()
    cfg.catalog.num_urdfs = 4
    cfg.catalog.refresh_urdfs_every = 8
    for k, v in outer_kw.items():
        assert hasattr(cfg.outer, k)
        setattr(cfg.outer, k, v)
    cfg.validate()
    return cfg


def _make_generator(**env_cfg):
    _, _, gen = generate_forests(
        num_envs=64,
        evaluation=True,
        unique_forests_eval=True,
        growing_forest=True,
        env_cfg=env_cfg,
        device="cpu",
    )
    return gen


# ---------------------------------------------------------------- config

def test_exam_forest_defaults_disabled():
    cfg = _cfg()
    assert cfg.outer.exam_forest.override_forest is False
    assert cfg.outer.exam_forest.overrides() == {}


def test_exam_forest_disabled_flag_ignores_set_fields():
    # Values may sit in the config; only override_forest=True activates them.
    ef = ExamForestConfig(dens_max=8.0, x_upper=900.0)
    assert ef.overrides() == {}
    ef.override_forest = True
    assert ef.overrides() == {"dens_max": 8.0, "x_upper": 900.0}
    # The flag itself must never reach apply_forest_overrides (unknown key
    # for the generator).
    assert "override_forest" not in ef.overrides()


def test_exam_forest_yaml_roundtrip(tmp_path):
    src = tmp_path / "outer.yaml"
    src.write_text(
        "catalog:\n"
        "  num_urdfs: 4\n"
        "  refresh_urdfs_every: 8\n"
        "outer:\n"
        "  exam_forest:\n"
        "    override_forest: true\n"
        "    dens_min: 1.5\n"
        "    dens_max: 8.0\n"
        "    forest_mode: growing\n"
        "    x_upper: 900.0\n"
    )
    cfg = OuterNSGA2Config.from_yaml(src)
    assert cfg.outer.exam_forest.overrides() == {
        "dens_min": 1.5, "dens_max": 8.0,
        "forest_mode": "growing", "x_upper": 900.0,
    }
    cfg.validate()
    dumped = tmp_path / "dumped.yaml"
    cfg.to_yaml(dumped)
    cfg2 = OuterNSGA2Config.from_yaml(dumped)
    assert cfg2.outer.exam_forest.override_forest is True
    assert cfg2.outer.exam_forest.overrides() == cfg.outer.exam_forest.overrides()


def test_exam_forest_unknown_yaml_key_raises(tmp_path):
    src = tmp_path / "outer.yaml"
    src.write_text(
        "outer:\n"
        "  exam_forest:\n"
        "    tree_radius: 2.0\n"   # deliberately unsupported (DepthSolver desync)
    )
    with pytest.raises(ValueError, match="tree_radius"):
        OuterNSGA2Config.from_yaml(src)


def test_exam_forest_validation():
    with pytest.raises(ValueError, match="forest_mode"):
        _cfg(exam_forest=ExamForestConfig(forest_mode="spiral"))
    with pytest.raises(ValueError, match="dens"):
        _cfg(exam_forest=ExamForestConfig(dens_min=5.0, dens_max=2.0))
    with pytest.raises(ValueError, match="num_trees"):
        _cfg(exam_forest=ExamForestConfig(num_trees=0))
    with pytest.raises(ValueError, match="x_upper"):
        _cfg(exam_forest=ExamForestConfig(x_upper=-100.0))
    _cfg(exam_forest=ExamForestConfig(dens_max=10.0, x_upper=1200.0))  # valid


# ---------------------------------------------------- generator overrides

def test_generator_denser_and_longer_then_restore():
    gen = _make_generator()
    base_cyl, _ = gen.generate()
    base_trees = base_cyl.shape[1]
    base_max_x = float(base_cyl[..., 0].max())

    base_x_upper = float(gen.config.x_upper)
    assert base_max_x <= base_x_upper

    prev = apply_generator_overrides(
        gen, {"dens_min": 4.0, "dens_max": 12.0, "x_upper": 1200.0}
    )
    exam_cyl, _ = gen.generate()
    assert exam_cyl.shape[1] > base_trees              # denser → more trees
    assert float(exam_cyl[..., 0].max()) > base_x_upper  # longer → past old end

    apply_generator_overrides(gen, prev)               # restore by re-applying
    back_cyl, _ = gen.generate()
    assert back_cyl.shape[1] == base_trees
    assert float(back_cyl[..., 0].max()) <= base_x_upper + 1e-3


def test_generator_mode_and_num_trees_override():
    gen = _make_generator()
    prev = apply_generator_overrides(
        gen, {"forest_mode": "uniform", "num_trees": 77}
    )
    cyl, _ = gen.generate()
    assert cyl.shape[1] == 77
    apply_generator_overrides(gen, prev)
    assert gen.forest_mode != "uniform" or gen.config.num_trees != 77


def test_generator_unknown_override_key_raises():
    gen = _make_generator()
    with pytest.raises(KeyError, match="tree_radius"):
        apply_generator_overrides(gen, {"tree_radius": 2.0})


# ---------------------------------------------------- env-level method

class _EnvStub:
    """Duck-typed stand-in for a WingedDroneEnv (no Genesis scene)."""

    def __init__(self, gen):
        self._forest_generator = gen
        self._success_x_limit_eval = 600.0


def test_env_x_upper_override_moves_success_line():
    stub = _EnvStub(_make_generator())
    prev = WingedDroneEnv.apply_forest_overrides(stub, {"x_upper": 900.0})
    assert stub._forest_generator.config.x_upper == 900.0
    assert stub._success_x_limit_eval == 900.0
    WingedDroneEnv.apply_forest_overrides(stub, prev)
    assert stub._success_x_limit_eval == stub._forest_generator.config.x_upper


def test_env_set_dens_min_exists_again():
    # Regression: MultiSceneEvalEnv.set_dens_min calls sub.set_dens_min,
    # which was lost in the winged_drone_train rewrite (dens_min ramp crashed).
    stub = _EnvStub(_make_generator())
    WingedDroneEnv.set_dens_min(stub, 3.5)
    assert stub._forest_generator.config.dens_min == 3.5


# ---------------------------------------------------- _run_exam wiring

class _FakeExamEnv:
    """Records the order of forest calls made by _run_exam."""

    E = 8

    def __init__(self):
        self.events = []

    def apply_forest_overrides(self, overrides):
        self.events.append(("apply", dict(overrides)))
        return {k: "PREV" for k in overrides}

    def refresh_forests(self):
        self.events.append(("refresh", None))


def _bare_loop(fake_env, exam_forest):
    loop = object.__new__(NSGA2MorphCMAES)
    loop.outer = _cfg(exam_forest=exam_forest).outer
    loop.cfg = None
    loop._env = fake_env
    loop._env_urdf_path = None
    loop._model_and_layer = None
    loop._wp1_cfg = None
    loop._urdf_paths = ["a.urdf", "b.urdf"]
    loop._last_solutions = [np.zeros(4) for _ in range(4)]
    loop._last_fitnesses = np.arange(4.0)
    return loop


def test_run_exam_applies_and_restores_forest(monkeypatch):
    fake = _FakeExamEnv()
    loop = _bare_loop(fake, ExamForestConfig(override_forest=True, dens_max=10.0))

    def fake_eval(sols, *a, **kw):
        fake.events.append(("eval", len(sols)))
        N, k = 2, len(sols)
        mats = ("per_urdf_reward", "per_urdf_progress", "per_urdf_velocity",
                "per_urdf_crash", "per_urdf_cot")
        return np.zeros(k), {m: np.ones((N, k)) for m in mats}

    import WP2.evaluate
    monkeypatch.setattr(WP2.evaluate, "evaluate_population_multi_urdf", fake_eval)

    result = loop._run_exam()
    assert result is not None
    kinds = [e[0] for e in fake.events]
    # exam overrides applied, forests refreshed, rollout, then restored and
    # regenerated so a surviving env is back to nominal forests
    assert kinds == ["apply", "refresh", "eval", "apply", "refresh"]
    assert fake.events[0][1] == {"dens_max": 10.0}
    assert fake.events[3][1] == {"dens_max": "PREV"}


def test_run_exam_restores_forest_when_eval_fails(monkeypatch):
    fake = _FakeExamEnv()
    loop = _bare_loop(fake, ExamForestConfig(override_forest=True, dens_max=10.0))

    def boom(*a, **kw):
        fake.events.append(("eval", None))
        raise RuntimeError("rollout died")

    import WP2.evaluate
    monkeypatch.setattr(WP2.evaluate, "evaluate_population_multi_urdf", boom)

    assert loop._run_exam() is None
    kinds = [e[0] for e in fake.events]
    assert kinds == ["apply", "refresh", "eval", "apply", "refresh"]


def test_run_exam_no_overrides_skips_forest_calls(monkeypatch):
    # Fields set but override_forest left False → the exam must not touch
    # the forest setup at all.
    fake = _FakeExamEnv()
    loop = _bare_loop(fake, ExamForestConfig(dens_max=10.0))

    def fake_eval(sols, *a, **kw):
        fake.events.append(("eval", len(sols)))
        N, k = 2, len(sols)
        mats = ("per_urdf_reward", "per_urdf_progress", "per_urdf_velocity",
                "per_urdf_crash", "per_urdf_cot")
        return np.zeros(k), {m: np.ones((N, k)) for m in mats}

    import WP2.evaluate
    monkeypatch.setattr(WP2.evaluate, "evaluate_population_multi_urdf", fake_eval)

    assert loop._run_exam() is not None
    kinds = [e[0] for e in fake.events]
    assert kinds == ["refresh", "eval"]   # no apply/restore, single refresh
