"""
Tests for exam-time forest overrides (`outer.exam_forest`).

Covers: the ExamForestConfig section (defaults, YAML round-trip, validation),
runtime overrides on a real ForestGenerator (density / length / mode take
effect on the next generate() and restore by re-applying the returned
previous values), the WingedDroneEnv method incl. the eval success-line sync
for x_upper, the _run_exam apply→refresh→eval→restore wiring, and the
exam-baseline rollout (`outer.exam_baseline`) that re-flies the standard
mydrone on the same exam forests to give the Pareto plot its star. No Genesis
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
from WP2_Outer_Loop.nsga_cma import _EXAM_BASELINE_COLS, NSGA2MorphCMAES


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


def _lead_in_density(gen, x_window=20.0):
    """Mean trees/meter in x ∈ [0, x_window) over all generated forests."""
    cyl = gen.cylinders
    c = gen.config
    xs, ys = cyl[..., 0], cyl[..., 1]
    active = (ys >= c.y_lower) & (ys <= c.y_upper)
    F = cyl.shape[0]
    return float(((xs < x_window) & active).sum()) / F / x_window


def test_generator_eval_dens_min_override_beats_randomization_keys():
    # Regression (2026-08-10, blind-exam batch): the run configs set
    # forest.dens_min_min: 0.0 / dens_min_max: 0.0 (training-only per-forest
    # randomization knobs), which silently shadowed `dens_min` in the growing
    # sampler — the exam's dens_min override (3.0 in batch_2, 1.0 in batch_3)
    # was ignored and both batches flew identical near-zero-floor forests.
    # In evaluation mode the explicit dens_min floor must always win.
    gen = _make_generator(
        dens_min=0.0, dens_max=5.0,
        dens_min_min=0.0, dens_min_max=0.0,
        x_lower=0.0, x_upper=100.0,
    )
    prev = apply_generator_overrides(
        gen, {"dens_min": 3.0, "dens_max": 5.5, "x_upper": 300.0}
    )
    gen.generate()
    lead_in = _lead_in_density(gen)
    # floor 3.0 → ≈3.1 trees/m in the lead-in; the shadow bug gives ≈0.19
    assert lead_in > 2.0, (
        f"exam dens_min override ignored: lead-in density {lead_in:.2f} "
        f"trees/m (expected ≈3.1)"
    )

    # restore puts the nominal near-zero lead-in back
    apply_generator_overrides(gen, prev)
    gen.generate()
    assert _lead_in_density(gen) < 1.0


def test_generator_training_dens_min_randomization_still_active():
    # The eval-mode fix must not touch training: with a non-degenerate
    # [dens_min_min, dens_min_max] range, training forests still randomize
    # their ramp floor per forest (WP1 domain randomization).
    _, _, gen = generate_forests(
        num_envs=64,
        evaluation=False,
        unique_forests_eval=False,
        growing_forest=True,
        env_cfg=dict(
            dens_min=0.0, dens_max=5.0,
            dens_min_min=0.5, dens_min_max=3.0,
            x_lower=0.0, x_upper=100.0,
        ),
        device="cpu",
    )
    cyl = gen.cylinders
    c = gen.config
    ys = cyl[..., 1]
    active = (ys >= c.y_lower) & (ys <= c.y_upper)
    # per-forest active tree counts vary because each forest drew its own
    # ramp floor in [0.5, 3.0] (identical counts ⇒ randomization dead)
    counts = active.sum(dim=1).float()
    assert float(counts.std()) > 1.0


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


def _bare_loop(fake_env, exam_forest, val_env=None, **outer_kw):
    loop = object.__new__(NSGA2MorphCMAES)
    cfg = _cfg(exam_forest=exam_forest, **outer_kw)
    loop.outer = cfg.outer
    loop.cfg = cfg
    loop._env = fake_env
    loop._env_urdf_path = None
    loop._model_and_layer = None
    loop._wp1_cfg = None
    loop._urdf_paths = ["a.urdf", "b.urdf"]
    loop._last_solutions = [np.zeros(4) for _ in range(4)]
    loop._last_fitnesses = np.arange(4.0)
    # Exam-baseline state (see the _run_exam_baseline block below).
    loop._val_env = val_env
    loop._val_urdf_paths = ["standard.urdf"]
    loop._outer_gen = 3
    loop._last_gen = 31
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


# ------------------------------------------- exam baseline (Pareto star)

class _FakeValEnv(_FakeExamEnv):
    """Held-out validation env stand-in (standard mydrone, 4096 slots)."""

    E = 4096


_REF_RESULT = {
    "fitness": 12.5, "velocity": 14.0, "progress": 71.0,
    "crash_rate": 0.25, "cot": 0.8, "v_deviation": 1.5,
}


def _exam_loop(tmp_path, val_env, **outer_kw):
    """A bare loop wired for _run_exam + _run_exam_baseline, with the
    reference-actor rollout stubbed out (no Genesis)."""
    loop = _bare_loop(
        _FakeExamEnv(),
        ExamForestConfig(override_forest=True, dens_max=10.0),
        val_env=val_env,
        **outer_kw,
    )
    loop.exam_baseline_csv = tmp_path / "outer_exam_baseline.csv"
    return loop


def _stub_population_exam(monkeypatch, fake_env):
    def fake_eval(sols, *a, **kw):
        fake_env.events.append(("eval", len(sols)))
        N, k = 2, len(sols)
        mats = ("per_urdf_reward", "per_urdf_progress", "per_urdf_velocity",
                "per_urdf_crash", "per_urdf_cot")
        return np.zeros(k), {m: np.ones((N, k)) for m in mats}

    import WP2.evaluate
    monkeypatch.setattr(WP2.evaluate, "evaluate_population_multi_urdf", fake_eval)


def test_exam_baseline_flies_val_env_on_exam_forests(tmp_path, monkeypatch):
    val = _FakeValEnv()
    loop = _exam_loop(tmp_path, val)
    _stub_population_exam(monkeypatch, loop._env)

    seen = {}

    def fake_ref(label, **kw):
        val.events.append(("eval", label))
        seen.update(kw)
        return dict(_REF_RESULT)

    loop._evaluate_reference_actor = fake_ref

    assert loop._run_exam() is not None

    # The validation env flew the SAME overrides as the exam, then restored.
    assert [e[0] for e in val.events] == [
        "apply", "refresh", "eval", "apply", "refresh"]
    assert val.events[0][1] == {"dens_max": 10.0}
    assert val.events[3][1] == {"dens_max": "PREV"}
    # ...on the validation env, not the population env.
    assert seen["env_override"] is val
    assert seen["urdf_paths_override"] == ["standard.urdf"]

    import csv as _csv
    rows = list(_csv.reader(open(loop.exam_baseline_csv)))
    assert len(rows) == 1
    assert rows[0][:3] == ["3", "31", "4096"]          # outer_gen, inner_gen, E
    assert [float(v) for v in rows[0][3:]] == [
        _REF_RESULT[key] for _col, key in _EXAM_BASELINE_COLS]


def test_exam_baseline_columns_are_canonical_objective_names():
    # pareto_plots looks these up by plotted-objective name, so the CSV must
    # spell them progress_m / cost_of_transport, not progress / cot.
    cols = [col for col, _key in _EXAM_BASELINE_COLS]
    assert "progress_m" in cols and "cost_of_transport" in cols
    assert "progress" not in cols and "cot" not in cols


def test_exam_baseline_failure_leaves_exam_objectives(tmp_path, monkeypatch):
    val = _FakeValEnv()
    loop = _exam_loop(tmp_path, val)
    _stub_population_exam(monkeypatch, loop._env)

    def boom(label, **kw):
        val.events.append(("eval", label))
        raise RuntimeError("reference rollout died")

    loop._evaluate_reference_actor = boom

    # The star is optional; the objectives are not.
    assert loop._run_exam() is not None
    assert [e[0] for e in val.events] == [
        "apply", "refresh", "eval", "apply", "refresh"]
    assert not loop.exam_baseline_csv.exists()


def test_exam_baseline_disabled_never_touches_val_env(tmp_path, monkeypatch):
    val = _FakeValEnv()
    loop = _exam_loop(tmp_path, val, exam_baseline=False)
    _stub_population_exam(monkeypatch, loop._env)
    loop._evaluate_reference_actor = lambda *a, **kw: pytest.fail(
        "reference actor flown with outer.exam_baseline=False")

    assert loop._run_exam() is not None
    assert val.events == []
    assert not loop.exam_baseline_csv.exists()


def test_exam_baseline_without_validation_env_is_a_noop(tmp_path, monkeypatch):
    # validation.enable=false → no val env → no star, exam unaffected.
    loop = _exam_loop(tmp_path, None)
    _stub_population_exam(monkeypatch, loop._env)
    loop._evaluate_reference_actor = lambda *a, **kw: pytest.fail(
        "reference actor flown without a validation env")

    assert loop._run_exam() is not None
    assert not loop.exam_baseline_csv.exists()
