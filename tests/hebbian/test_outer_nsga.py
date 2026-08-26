"""
Tests for the WP2 outer loop (persistent CMA-ES + NSGA-II URDF refresh).

Pure-python: covers the NSGA-II variation helpers, the outer config
validation, the objective-name aliasing, the 2-D hypervolume, and the
per-URDF metric reductions added to WP2.evaluate — no Genesis required.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2_Outer_Loop.config import ObjectiveSpec, OuterNSGA2Config
from WP2_Outer_Loop.nsga2 import (
    arrays_to_individuals,
    individuals_to_array,
    make_toolbox,
)
from WP2_Outer_Loop.nsga_cma import (
    NSGA2MorphCMAES,
    exam_top_k,
    gated_select,
    make_offspring_tournament,
    reduce_exam_metrics,
)
from WP2_Outer_Loop.pareto_fronts import _DIAG_KEYS, build_pareto_front_csv
from WP2_Outer_Loop.pareto_plots import (
    _admission_mask,
    _baseline_is_constant,
    _cumulative_champions,
    _exam_flew_nominal_forests,
    _hv_reference,
    _hypervolume_2d,
    _load_exam_baseline,
    _load_exam_baseline_series,
    _load_min_progress,
    _load_refresh_every,
    _load_validation_baseline_series,
    _nondominated_mask,
    plot_champion_curves,
    plot_pareto_front,
)


def _cfg(num_urdfs=4, **outer_kw):
    """Single-file config: inner-loop fields + `outer:` NSGA-II section."""
    cfg = OuterNSGA2Config()
    cfg.catalog.num_urdfs = num_urdfs
    cfg.catalog.refresh_urdfs_every = 8
    for k, v in outer_kw.items():
        assert hasattr(cfg.outer, k)
        setattr(cfg.outer, k, v)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------- config

def test_config_defaults_validate():
    _cfg()


def test_config_rejects_bad_elites():
    with pytest.raises(ValueError):
        _cfg(num_urdfs=4, n_elites=4)
    with pytest.raises(ValueError):
        _cfg(num_urdfs=4, n_elites=0)


def test_config_allows_odd_and_non_multiple_of_four_pops():
    # The legacy loop required pop % 4 == 0 (DEAP selTournamentDCD); the
    # rewrite must not.
    for n in (2, 3, 5, 6):
        _cfg(num_urdfs=n, n_elites=1)


def test_config_rejects_single_urdf():
    with pytest.raises(ValueError):
        _cfg(num_urdfs=1)


def test_config_rejects_mutate_refresh():
    cfg = _cfg()
    cfg.catalog.mutate = True
    with pytest.raises(ValueError, match="mutate"):
        cfg.validate()


def test_config_rejects_disabled_refresh():
    cfg = _cfg()
    cfg.catalog.refresh_urdfs_every = 0
    with pytest.raises(ValueError, match="refresh_urdfs_every"):
        cfg.validate()


def test_config_rejects_too_small_env_budget():
    cfg = _cfg()
    cfg.cmaes.population_size = 16
    cfg.evaluation.num_eval_envs = cfg.catalog.num_urdfs * 16 - 1
    with pytest.raises(ValueError, match="num_eval_envs"):
        cfg.validate()


def test_config_single_file_yaml_roundtrip(tmp_path):
    """One YAML carries inner fields AND the outer section; to_yaml round-trips."""
    src = tmp_path / "outer.yaml"
    src.write_text(
        "exp_name: single_file_test\n"
        "seed: 7\n"
        "cmaes:\n"
        "  population_size: 8\n"
        "  sigma_reinflate: 1.5\n"
        "catalog:\n"
        "  num_urdfs: 4\n"
        "  refresh_urdfs_every: 8\n"
        "  include_standard_mydrone: false\n"
        "evaluation:\n"
        "  num_eval_envs: 512\n"
        "outer:\n"
        "  n_elites: 3\n"
        "  score_top_frac: 0.25\n"
        "  sbx_eta: 10.0\n"
        "  objectives:\n"
        "    - name: progress_m\n"
        "      direction: maximize\n"
        "    - name: crash_rate\n"
        "      direction: minimize\n"
    )
    cfg = OuterNSGA2Config.from_yaml(src)
    # inner fields land in the usual places
    assert cfg.exp_name == "single_file_test"
    assert cfg.seed == 7
    assert cfg.cmaes.population_size == 8
    assert cfg.catalog.num_urdfs == 4
    assert cfg.catalog.include_standard_mydrone is False
    # outer section is parsed into typed fields
    assert cfg.outer.n_elites == 3
    assert cfg.outer.score_top_frac == 0.25
    assert cfg.outer.sbx_eta == 10.0
    assert [(o.name, o.direction) for o in cfg.outer.objectives] == [
        ("progress_m", "maximize"), ("crash_rate", "minimize")
    ]
    assert cfg.outer.objective_weights() == [1.0, -1.0]
    cfg.validate()
    # the serialized config (what lands in reproducibility/config.yaml)
    # is itself a loadable single-file config
    dumped = tmp_path / "dumped.yaml"
    cfg.to_yaml(dumped)
    cfg2 = OuterNSGA2Config.from_yaml(dumped)
    assert cfg2.outer.n_elites == 3
    assert cfg2.catalog.num_urdfs == 4
    assert [(o.name, o.direction) for o in cfg2.outer.objectives] == [
        ("progress_m", "maximize"), ("crash_rate", "minimize")
    ]


# ---------------------------------------------------------------- variation

@pytest.mark.parametrize("N", [2, 3, 4, 6])
def test_offspring_counts_and_bounds(N):
    random.seed(0)
    cfg = _cfg(num_urdfs=max(N, 2), n_elites=1)
    tb = make_toolbox(cfg.outer, genome_dim=15)
    genomes = np.random.rand(N, 15)
    objs = np.column_stack([np.random.rand(N) * 100, np.random.rand(N) * 2])
    inds = arrays_to_individuals(genomes, objs)
    for n_elites in range(1, N):
        elites = tb.select(inds, n_elites)
        off = make_offspring_tournament(tb, inds, N - n_elites, 0.9)
        new = np.vstack([individuals_to_array(elites),
                         individuals_to_array(off)])
        assert new.shape == (N, 15)
        assert new.min() >= 0.0 and new.max() <= 1.0


def test_selnsga2_respects_objective_directions():
    """fitness maximize + cot minimize: the dominant point must win at k=1."""
    random.seed(0)
    cfg = _cfg()
    tb = make_toolbox(cfg.outer, genome_dim=15)
    g = np.random.rand(2, 15)
    objs = np.array([[100.0, 0.5],   # dominates on both objectives
                     [50.0, 1.5]])
    sel = tb.select(arrays_to_individuals(g, objs), 1)
    np.testing.assert_allclose(list(sel[0]), g[0])


# ---------------------------------------------------------------- aliases

def test_objective_aliases():
    f = NSGA2MorphCMAES._canonical_objective
    assert f("fitness") == "fitness"
    assert f("cot") == "cost_of_transport"
    assert f("cost_of_transport") == "cost_of_transport"
    assert f("progress") == "progress_m"
    with pytest.raises(KeyError):
        f("not_a_metric")


# ---------------------------------------------------------------- hypervolume

def test_hypervolume_simple_rectangle():
    front = np.array([[1.0, 1.0]])
    assert _hypervolume_2d(front, np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_hypervolume_two_point_front():
    front = np.array([[2.0, 1.0], [1.0, 2.0]])
    # (2-0)*(1-0) + (1-0)*(2-1) = 3
    assert _hypervolume_2d(front, np.array([0.0, 0.0])) == pytest.approx(3.0)


def test_hypervolume_grows_with_dominating_point():
    ref = np.array([0.0, 0.0])
    hv1 = _hypervolume_2d(np.array([[1.0, 1.0]]), ref)
    hv2 = _hypervolume_2d(np.array([[2.0, 2.0]]), ref)
    assert hv2 > hv1


def test_nondominated_mask():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.5]])  # maximization
    mask = _nondominated_mask(pts)
    assert mask.tolist() == [False, True, True]


# ------------------------------------------- fixed hypervolume reference

def test_hv_reference_fixed_for_progress_cot():
    # (progress_m max, cost_of_transport min) both have a fixed anchor —
    # the 80 m admission-gate progress and a CoT ceiling of 0.5 — so the
    # reference is fixed at (80, -0.5) in maximization space and
    # hypervolumes compare across runs without a shared dead rectangle.
    pts = np.array([[150.0, -0.2], [180.0, -0.3]])
    ref, fixed = _hv_reference(
        [("progress_m", "maximize"), ("cost_of_transport", "minimize")], pts)
    assert fixed is True
    np.testing.assert_allclose(ref, [80.0, -0.5])


def test_hv_reference_fixed_accepts_aliases():
    pts = np.array([[150.0, -0.2]])
    ref, fixed = _hv_reference(
        [("progress", "maximize"), ("cot", "minimize")], pts)
    assert fixed is True
    np.testing.assert_allclose(ref, [80.0, -0.5])


def test_hv_reference_falls_back_for_unbounded_objective():
    # fitness is reward-shaped (no absolute scale) → run-relative reference:
    # worst observed − 5% of span per objective.
    pts = np.array([[10.0, -0.2], [20.0, -0.4]])
    ref, fixed = _hv_reference(
        [("fitness", "maximize"), ("cost_of_transport", "minimize")], pts)
    assert fixed is False
    np.testing.assert_allclose(ref, [10.0 - 0.05 * 10.0, -0.4 - 0.05 * 0.2])


# ---------------------------------------------------------------- exam pass

def test_exam_top_k_batch0_geometry():
    # H=64 controllers, E=3840 slots/drone, top 12.5% → 8 controllers,
    # each flying E/8 = 480 forests.
    assert exam_top_k(64, 3840, 0.125) == 8


def test_exam_top_k_rounds_down_to_divisor_of_env_slots():
    # round(10 * 0.5) = 5 but 5 does not divide 64 → largest divisor ≤ 5 is 4.
    assert exam_top_k(10, 64, 0.5) == 4


def test_exam_top_k_bounds():
    assert exam_top_k(64, 3840, 0.0001) == 1   # floor at one controller
    assert exam_top_k(4, 64, 1.0) == 4         # whole population allowed
    assert exam_top_k(64, 61, 0.5) == 1        # prime slot count → only 1 divides


def test_reduce_exam_metrics_means_and_objective_order():
    N, k = 3, 4
    rng = np.random.default_rng(0)
    metrics = {
        key: rng.random((N, k))
        for key in ("per_urdf_reward", "per_urdf_progress", "per_urdf_velocity",
                    "per_urdf_crash", "per_urdf_cot")
    }
    objectives = [
        ObjectiveSpec(name="progress_m", direction="maximize"),
        ObjectiveSpec(name="cost_of_transport", direction="minimize"),
    ]
    objs, diag = reduce_exam_metrics(metrics, objectives)
    assert objs.shape == (N, 2)
    np.testing.assert_allclose(objs[:, 0], metrics["per_urdf_progress"].mean(axis=1))
    np.testing.assert_allclose(objs[:, 1], metrics["per_urdf_cot"].mean(axis=1))
    for key in ("fitness", "cost_of_transport", "progress_m", "velocity", "crash_rate"):
        assert diag[key].shape == (N,)
    np.testing.assert_allclose(diag["fitness"], metrics["per_urdf_reward"].mean(axis=1))


def test_reduce_exam_metrics_missing_matrix_gives_nan_diag():
    # A missing diagnostic matrix must not crash the reduction; its diag
    # column is NaN-filled while requested objectives still compute.
    N, k = 2, 3
    metrics = {
        "per_urdf_reward": np.ones((N, k)),
        "per_urdf_progress": np.ones((N, k)),
        "per_urdf_cot": np.ones((N, k)),
    }
    objs, diag = reduce_exam_metrics(
        metrics, [ObjectiveSpec(name="fitness", direction="maximize")]
    )
    assert objs.shape == (N, 1)
    assert np.isnan(diag["velocity"]).all()


def test_score_phase_falls_back_when_env_gone():
    # Final-flush path: the eval env is already torn down, so the exam is
    # impossible and _score_phase must return phase-mean objectives — and a
    # gate-progress vector taken from the same (phase-mean) source.
    loop = object.__new__(NSGA2MorphCMAES)
    loop.outer = _cfg().outer  # default objectives: fitness, cost_of_transport
    loop._env = None
    loop._last_solutions = None
    loop._last_fitnesses = None
    N = 4
    loop._urdf_paths = [f"u{i}.urdf" for i in range(N)]
    rng = np.random.default_rng(1)
    sample = {
        k: rng.random(N)
        for k in ("fitness", "cost_of_transport", "progress_m", "velocity", "crash_rate")
    }
    loop._phase_samples = [sample]
    scored = loop._score_phase()
    assert scored is not None
    objs, agg, n_gens, source, gate_progress = scored
    assert source == "phase_mean"
    assert n_gens == 1
    np.testing.assert_allclose(objs[:, 0], sample["fitness"])
    np.testing.assert_allclose(objs[:, 1], sample["cost_of_transport"])
    np.testing.assert_allclose(gate_progress, sample["progress_m"])


def test_score_phase_gate_progress_follows_exam_source():
    # When the exam scores the phase, feasibility must be judged on the
    # exam's progress (same source as the objectives), while the recorded
    # diagnostics stay phase means.
    loop = object.__new__(NSGA2MorphCMAES)
    loop.outer = _cfg().outer
    N = 3
    loop._urdf_paths = [f"u{i}.urdf" for i in range(N)]
    keys = ("fitness", "cost_of_transport", "progress_m", "velocity", "crash_rate")
    phase_sample = {k: np.full(N, 1.0) for k in keys}
    phase_sample["progress_m"] = np.array([10.0, 20.0, 30.0])
    loop._phase_samples = [phase_sample]
    exam_objs = np.arange(N * 2, dtype=float).reshape(N, 2)
    exam_diag = {k: np.full(N, 2.0) for k in keys}
    exam_diag["progress_m"] = np.array([50.0, 60.0, 70.0])
    loop._run_exam = lambda: (exam_objs, exam_diag, 4)
    scored = loop._score_phase()
    assert scored is not None
    objs, agg, n_gens, source, gate_progress = scored
    assert source == "exam"
    np.testing.assert_allclose(objs, exam_objs)
    np.testing.assert_allclose(gate_progress, exam_diag["progress_m"])
    np.testing.assert_allclose(agg["progress_m"], phase_sample["progress_m"])


def test_config_rescore_defaults_and_validation():
    cfg = _cfg()
    assert cfg.outer.rescore is True
    assert cfg.outer.rescore_top_frac == pytest.approx(0.125)
    with pytest.raises(ValueError, match="rescore_top_frac"):
        _cfg(rescore_top_frac=0.0)
    with pytest.raises(ValueError, match="rescore_top_frac"):
        _cfg(rescore_top_frac=1.5)


# ------------------------------------------------- minimum-progress gate

def test_config_min_progress_default_off():
    assert _cfg().outer.min_progress_m == 0.0


def test_config_min_progress_validation():
    _cfg(min_progress_m=40.0)
    with pytest.raises(ValueError, match="min_progress_m"):
        _cfg(min_progress_m=-1.0)


def test_config_min_progress_yaml_roundtrip(tmp_path):
    src = tmp_path / "outer.yaml"
    src.write_text(
        "catalog:\n  num_urdfs: 4\n  refresh_urdfs_every: 8\n"
        "outer:\n  min_progress_m: 35.5\n"
    )
    cfg = OuterNSGA2Config.from_yaml(src)
    assert cfg.outer.min_progress_m == 35.5
    dumped = tmp_path / "dumped.yaml"
    cfg.to_yaml(dumped)
    assert OuterNSGA2Config.from_yaml(dumped).outer.min_progress_m == 35.5


def _gate_fixture(progress, cot=None):
    """Individuals over (progress_m max, cost_of_transport min) where every
    point is nondominated ungated: progress ascending, cot ascending (the
    low-progress/low-cot rows are exactly the degenerate front arm)."""
    random.seed(0)
    cfg = _cfg(num_urdfs=max(2, len(progress)), n_elites=1)
    tb = make_toolbox(cfg.outer, genome_dim=15)
    progress = np.asarray(progress, dtype=float)
    N = len(progress)
    if cot is None:
        cot = np.linspace(0.05, 0.3, N)
    genomes = np.random.rand(N, 15)
    inds = arrays_to_individuals(genomes, np.column_stack([progress, cot]))
    return tb, inds, progress


def test_gated_select_off_matches_plain_nsga2():
    tb, inds, progress = _gate_fixture([5.0, 40.0, 80.0, 120.0])
    elites, pool = gated_select(tb, inds, progress, 0.0, n_elites=2)
    expected = tb.select(list(inds), 2)
    assert {id(e) for e in elites} == {id(e) for e in expected}
    assert pool == list(inds)


def test_gated_select_excludes_infeasible_from_elites_and_pool():
    # Ungated, the 5 m morph is nondominated (best cot) and would be kept.
    tb, inds, progress = _gate_fixture([5.0, 40.0, 80.0, 120.0])
    elites, pool = gated_select(tb, inds, progress, 30.0, n_elites=2)
    assert len(elites) == 2
    assert inds[0] not in elites
    assert inds[0] not in pool
    assert all(ind in inds[1:] for ind in elites)
    assert pool == inds[1:]


def test_gated_select_elite_deficit_not_filled_by_infeasible():
    tb, inds, progress = _gate_fixture([5.0, 10.0, 80.0, 120.0])
    elites, pool = gated_select(tb, inds, progress, 30.0, n_elites=3)
    assert len(elites) == 2                    # only 2 feasible
    assert all(ind in inds[2:] for ind in elites)
    assert pool == inds[2:]


def test_gated_select_single_feasible_tops_up_parent_pool():
    tb, inds, progress = _gate_fixture([5.0, 25.0, 10.0, 120.0])
    elites, pool = gated_select(tb, inds, progress, 30.0, n_elites=2)
    assert elites == [inds[3]]
    # Pool topped up to 2 with the highest-progress infeasible morph (25 m).
    assert pool == [inds[3], inds[1]]


def test_gated_select_zero_feasible_falls_back_ungated(capsys):
    tb, inds, progress = _gate_fixture([5.0, 10.0, 15.0, 20.0])
    elites, pool = gated_select(tb, inds, progress, 30.0, n_elites=2)
    assert len(elites) == 2
    assert pool == list(inds)
    assert "UNGATED" in capsys.readouterr().out


def test_gated_select_nan_progress_counts_feasible():
    # Missing gate data must never exclude a morphology.
    tb, inds, progress = _gate_fixture([np.nan, 40.0, 80.0, 120.0])
    elites, pool = gated_select(tb, inds, progress, 30.0, n_elites=2)
    assert pool == list(inds)


# ------------------------------------------------- plot admission gate

def test_load_min_progress_reads_saved_config(tmp_path):
    (tmp_path / "reproducibility").mkdir(parents=True)
    (tmp_path / "reproducibility" / "config.yaml").write_text(
        "outer:\n  min_progress_m: 30.0\n"
    )
    assert _load_min_progress(tmp_path) == 30.0


def test_load_min_progress_absent_means_off(tmp_path):
    # Old runs (pre-gate config) and missing config both → 0.0.
    assert _load_min_progress(tmp_path) == 0.0
    (tmp_path / "reproducibility").mkdir(parents=True)
    (tmp_path / "reproducibility" / "config.yaml").write_text("outer:\n  n_elites: 2\n")
    assert _load_min_progress(tmp_path) == 0.0


def test_admission_mask_from_objective_column():
    import pandas as pd
    df = pd.DataFrame({
        "obj_progress_m": [5.0, 40.0, np.nan, 120.0],
        "obj_cost_of_transport": [0.05, 0.1, 0.2, 0.3],
    })
    specs = [("progress_m", "maximize"), ("cost_of_transport", "minimize")]
    mask = _admission_mask(df, specs, 30.0)
    assert mask.tolist() == [False, True, True, True]  # NaN admitted


def test_admission_mask_from_diag_column_when_progress_not_objective():
    import pandas as pd
    df = pd.DataFrame({
        "obj_fitness": [1.0, 2.0, 3.0],
        "obj_cost_of_transport": [0.05, 0.1, 0.2],
        "progress_m": [5.0, 40.0, 120.0],
    })
    specs = [("fitness", "maximize"), ("cost_of_transport", "minimize")]
    mask = _admission_mask(df, specs, 30.0)
    assert mask.tolist() == [False, True, True]


def test_admission_mask_gate_off_or_no_progress_column():
    import pandas as pd
    df = pd.DataFrame({"obj_fitness": [1.0, 2.0]})
    specs = [("fitness", "maximize"), ("cost_of_transport", "minimize")]
    assert _admission_mask(df, specs, 0.0).all()
    assert _admission_mask(df, specs, 30.0).all()  # no progress info → no gate


def _synthetic_run_dir(tmp_path, min_progress_yaml=None, obj_source="exam",
                       with_obj_source=True):
    """Minimal run dir: outer_population.csv + saved single-file config.

    ``with_obj_source=False`` reproduces the legacy CSV layout (runs
    predating the column, whose objectives are all phase means).
    """
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "reproducibility").mkdir(parents=True)
    rows = [
        # gen 0: degenerate low-progress/low-cot morph + two real flyers
        (0, 0, "a.urdf", 8, obj_source, 5.0, 0.05),
        (0, 1, "b.urdf", 8, obj_source, 80.0, 0.20),
        (0, 2, "c.urdf", 8, obj_source, 100.0, 0.30),
        # gen 1: entirely infeasible generation (robustness edge)
        (1, 0, "d.urdf", 8, obj_source, 4.0, 0.04),
        (1, 1, "e.urdf", 8, obj_source, 6.0, 0.06),
        (1, 2, "f.urdf", 8, obj_source, 8.0, 0.08),
    ]
    if not with_obj_source:
        rows = [r[:4] + r[5:] for r in rows]
    lines = ["outer_gen,urdf_idx,urdf_file,n_score_gens"
             + (",obj_source" if with_obj_source else "")
             + ",obj_progress_m,obj_cost_of_transport"]
    lines += [",".join(map(str, r)) for r in rows]
    (tmp_path / "results" / "outer_population.csv").write_text("\n".join(lines) + "\n")
    cfg = (
        "outer:\n"
        "  objectives:\n"
        "    - {name: progress_m, direction: maximize}\n"
        "    - {name: cost_of_transport, direction: minimize}\n"
    )
    if min_progress_yaml is not None:
        cfg += f"  min_progress_m: {min_progress_yaml}\n"
    (tmp_path / "reproducibility" / "config.yaml").write_text(cfg)
    return tmp_path


def test_plot_pareto_front_gated_smoke(tmp_path):
    run_dir = _synthetic_run_dir(tmp_path, min_progress_yaml=30.0)
    plot_pareto_front(run_dir)  # threshold read from the saved config
    assert (run_dir / "plots" / "pareto_front_evolution.png").is_file()
    assert (run_dir / "plots" / "pareto_hypervolume.png").is_file()


def test_plot_pareto_front_explicit_override_smoke(tmp_path):
    # Old runs without the knob: gate passed explicitly (CLI --min-progress).
    run_dir = _synthetic_run_dir(tmp_path, min_progress_yaml=None)
    plot_pareto_front(run_dir, min_progress=30.0)
    assert (run_dir / "plots" / "pareto_front_evolution.png").is_file()


def test_plot_pareto_front_returns_fixed_ref_hypervolume(tmp_path):
    # progress/cot run, gate off → every point admitted; the cumulative
    # front is all 6 points (progress and cot both ascending) but only
    # (100, 0.30) lies strictly beyond the (80 m, CoT 0.5) anchor:
    # (100-80)*(0.5-0.30) = 4.0
    run_dir = _synthetic_run_dir(tmp_path, min_progress_yaml=None)
    res = plot_pareto_front(run_dir)
    assert res["ref_fixed"] is True
    np.testing.assert_allclose(res["ref"], [80.0, 0.5])  # raw objective space
    assert res["hypervolume"] == pytest.approx(4.0)


def test_plot_pareto_front_fixed_ref_hv_respects_gate(tmp_path):
    # min_progress 30 admits only (80, 0.20) and (100, 0.30); the anchor
    # already clips everything at or below 80 m, so as long as the anchor
    # sits at/above the gate, gated and ungated runs give the SAME fixed-ref
    # hypervolume: (100-80)*(0.5-0.30) = 4.0
    run_dir = _synthetic_run_dir(tmp_path, min_progress_yaml=30.0)
    res = plot_pareto_front(run_dir)
    assert res["ref_fixed"] is True
    assert res["hypervolume"] == pytest.approx(4.0)


# ------------------------------------- exam-baseline star (pareto_plots)

def _exam_baseline_run_dir(
    tmp_path, rows=((0, 20.0, 0.4), (1, 30.0, 0.6)),
    enable=True, validation_catalog="", write_csv=True, exam_forest=None,
):
    """Run dir carrying results/outer_exam_baseline.csv + a validation config.

    ``exam_forest`` writes an ``outer.exam_forest`` block (e.g.
    ``{"override_forest": True, "dens_max": 5.5}``) — an exam that flew a
    forest distribution of its own. The default (``None``) is a run whose exam
    flew the inner loop's forests, so the validation pass measured the
    reference on the very distribution the exam scored on.
    """
    run_dir = _synthetic_run_dir(tmp_path)
    if write_csv:
        lines = ["outer_gen,inner_gen,n_forests,fitness,velocity,progress_m,"
                 "crash_rate,cost_of_transport,v_deviation"]
        lines += [f"{g},{(g + 1) * 8},4096,1.0,12.0,{prog},0.1,{cot},1.0"
                  for g, prog, cot in rows]
        (run_dir / "results" / "outer_exam_baseline.csv").write_text(
            "\n".join(lines) + "\n")
    cfg = (run_dir / "reproducibility" / "config.yaml").read_text()
    if exam_forest is not None:
        cfg += "  exam_forest:\n"
        for key, val in exam_forest.items():
            if val is None:
                text = "null"
            elif isinstance(val, bool):
                text = "true" if val else "false"
            else:
                text = str(val)
            cfg += f"    {key}: {text}\n"
    cfg += (f"validation:\n"
            f"  enable: {'true' if enable else 'false'}\n"
            f"  validation_catalog: '{validation_catalog}'\n")
    (run_dir / "reproducibility" / "config.yaml").write_text(cfg)
    return run_dir


def test_load_exam_baseline_means_over_phases(tmp_path):
    # The standard drone + frozen generalist is a fixed system: per-phase
    # spread is forest noise, so the star is the mean.
    run_dir = _exam_baseline_run_dir(tmp_path)
    star = _load_exam_baseline(run_dir, "progress_m", "cost_of_transport")
    assert star == pytest.approx((25.0, 0.5))


def test_load_exam_baseline_maps_velocity_deviation_alias(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path)
    star = _load_exam_baseline(run_dir, "progress_m", "velocity_deviation")
    assert star == pytest.approx((25.0, 1.0))


def test_load_exam_baseline_none_for_custom_validation_catalog(tmp_path):
    # A custom validation catalog means the reference drone is NOT the
    # standard mydrone — no star rather than a mislabelled one.
    run_dir = _exam_baseline_run_dir(tmp_path, validation_catalog="my_cat.txt")
    assert _load_exam_baseline(run_dir, "progress_m", "cost_of_transport") is None


def test_load_exam_baseline_none_when_validation_disabled(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path, enable=False)
    assert _load_exam_baseline(run_dir, "progress_m", "cost_of_transport") is None


def test_load_exam_baseline_none_for_runs_predating_the_flag(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path, write_csv=False)
    assert _load_exam_baseline(run_dir, "progress_m", "cost_of_transport") is None


def test_load_exam_baseline_none_for_unplotted_objective(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path)
    assert _load_exam_baseline(run_dir, "progress_m", "nonsense") is None


def test_plot_pareto_front_draws_exam_star(tmp_path):
    # End to end: exam-only rows + the baseline CSV → the star is back.
    run_dir = _exam_baseline_run_dir(tmp_path)
    res = plot_pareto_front(run_dir)
    assert res["star"] == pytest.approx((25.0, 0.5))
    assert (run_dir / "plots" / "pareto_front_evolution.png").is_file()


def test_plot_pareto_front_exam_run_without_baseline_csv_has_no_star(tmp_path):
    # Exam flew forests of its own (denser/longer) and the run predates
    # outer.exam_baseline: no star rather than the validation baseline, which
    # flew the easier nominal forests and would sit in the wrong place.
    run_dir = _exam_baseline_run_dir(
        tmp_path, write_csv=False,
        exam_forest={"override_forest": True, "dens_min": 3.0, "x_upper": 300.0},
    )
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n0,150.0,0.05\n")
    assert plot_pareto_front(run_dir)["star"] is None


# --- exam without forest overrides: validation IS the exam distribution -----

def test_exam_flew_nominal_forests_for_runs_predating_exam_forest(tmp_path):
    # No exam_forest block at all ⇒ the exam could only have flown the inner
    # loop's forests.
    assert _exam_flew_nominal_forests(_exam_baseline_run_dir(tmp_path)) is True


def test_exam_flew_nominal_forests_when_master_switch_is_off(tmp_path):
    # override_forest false makes the values inert (ExamForestConfig.overrides
    # returns {}), so a populated block with the switch off still flies nominal.
    run_dir = _exam_baseline_run_dir(
        tmp_path, exam_forest={"override_forest": False, "dens_min": 3.0,
                               "x_upper": 300.0},
    )
    assert _exam_flew_nominal_forests(run_dir) is True


def test_exam_flew_nominal_forests_when_every_override_is_null(tmp_path):
    run_dir = _exam_baseline_run_dir(
        tmp_path, exam_forest={"override_forest": True, "dens_min": None,
                               "x_upper": None},
    )
    assert _exam_flew_nominal_forests(run_dir) is True


def test_exam_flew_nominal_forests_false_when_an_override_is_set(tmp_path):
    run_dir = _exam_baseline_run_dir(
        tmp_path, exam_forest={"override_forest": True, "dens_max": 5.5},
    )
    assert _exam_flew_nominal_forests(run_dir) is False


def test_plot_pareto_front_falls_back_to_validation_star_on_nominal_exam(tmp_path):
    # Exam-scored run with no forest overrides: the exam and the held-out
    # validation pass flew the SAME distribution, so the validation baseline
    # is the right star even though outer_exam_baseline.csv was never written.
    run_dir = _exam_baseline_run_dir(tmp_path, write_csv=False)
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n"
        "0,116.0,0.28\n1,118.0,0.30\n")
    assert plot_pareto_front(run_dir)["star"] == pytest.approx((117.0, 0.29))


def test_exam_baseline_csv_wins_over_the_validation_fallback(tmp_path):
    # When the run did write the exam baseline, that measurement is the star —
    # the fallback only fills a gap, it never overrides a real measurement.
    run_dir = _exam_baseline_run_dir(tmp_path)  # star = (25.0, 0.5)
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n0,116.0,0.28\n")
    assert plot_pareto_front(run_dir)["star"] == pytest.approx((25.0, 0.5))


def test_validation_star_fallback_still_needs_the_standard_drone(tmp_path):
    # Custom validation catalog ⇒ the reference drone is not the standard
    # mydrone; the fallback must not smuggle in a mislabelled star.
    run_dir = _exam_baseline_run_dir(
        tmp_path, write_csv=False, validation_catalog="my_cat.txt")
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n0,116.0,0.28\n")
    assert plot_pareto_front(run_dir)["star"] is None


# ------------------------------------- exam champion curves (pareto_plots)

_CHAMP_SPECS = [("progress_m", "maximize"), ("cost_of_transport", "minimize")]


def _champ_df():
    """The exam rows of `_synthetic_run_dir` as a frame: gen 0 holds the real
    flyers, gen 1 only degenerate low-progress/low-CoT morphs."""
    import pandas as pd
    return pd.DataFrame({
        "outer_gen": [0, 0, 0, 1, 1, 1],
        "urdf_file": ["a.urdf", "b.urdf", "c.urdf",
                      "d.urdf", "e.urdf", "f.urdf"],
        "obj_progress_m": [5.0, 80.0, 100.0, 4.0, 6.0, 8.0],
        "obj_cost_of_transport": [0.05, 0.20, 0.30, 0.04, 0.06, 0.08],
    })


def test_cumulative_champions_report_both_objectives_of_each_champion():
    ch = _cumulative_champions(_champ_df(), _CHAMP_SPECS)
    # c.urdf holds the progress record, a.urdf the CoT record; each champion
    # contributes BOTH of its objectives, so the CoT curve shows what the
    # progress champion actually costs (0.30) — not the population minimum.
    assert ch.loc[0, "a_obj0"] == pytest.approx(100.0)
    assert ch.loc[0, "a_obj1"] == pytest.approx(0.30)
    assert ch.loc[0, "b_obj0"] == pytest.approx(5.0)
    assert ch.loc[0, "b_obj1"] == pytest.approx(0.05)


def test_cumulative_champions_carry_the_record_holder_forward():
    ch = _cumulative_champions(_champ_df(), _CHAMP_SPECS)
    # Nothing in gen 1 beats 100 m, so c.urdf still holds progress; d.urdf
    # does take the CoT record (0.04 < 0.05).
    assert ch.loc[1, "a_obj0"] == pytest.approx(100.0)
    assert ch.loc[1, "a_urdf"] == "c.urdf"
    assert ch.loc[1, "b_obj1"] == pytest.approx(0.04)
    assert ch.loc[1, "b_urdf"] == "d.urdf"


def test_cumulative_champions_honour_objective_directions():
    # Flipped directions must flip both champions — the sign comes from the
    # config spec, it is not hardcoded to "progress up, CoT down".
    specs = [("progress_m", "minimize"), ("cost_of_transport", "maximize")]
    ch = _cumulative_champions(_champ_df(), specs)
    assert ch.loc[1, "a_obj0"] == pytest.approx(4.0)
    assert ch.loc[1, "b_obj1"] == pytest.approx(0.30)


def test_cumulative_champions_ignore_nan_objectives():
    df = _champ_df()
    df.loc[2, "obj_progress_m"] = float("nan")  # c.urdf's exam produced no score
    ch = _cumulative_champions(df, _CHAMP_SPECS)
    assert ch.loc[0, "a_obj0"] == pytest.approx(80.0)
    assert ch.loc[0, "a_urdf"] == "b.urdf"


def test_plot_champion_curves_smoke(tmp_path):
    run_dir = _synthetic_run_dir(tmp_path)
    res = plot_champion_curves(run_dir)
    assert res is not None and res["source"] == "exam"
    assert res["baseline"] is False
    assert (run_dir / "plots" / "exam_champions.png").is_file()


def test_plot_champion_curves_falls_back_to_phase_mean_rows(tmp_path):
    # No exam rows → plot the phase-mean objectives instead of skipping, but
    # under a filename that cannot be mistaken for an exam figure.
    run_dir = _synthetic_run_dir(tmp_path, obj_source="phase_mean")
    res = plot_champion_curves(run_dir)
    assert res["source"] == "phase_mean"
    assert (run_dir / "plots" / "phase_mean_champions.png").is_file()
    assert not (run_dir / "plots" / "exam_champions.png").exists()


def test_plot_champion_curves_handles_legacy_csv_without_obj_source(tmp_path):
    # Runs predating the column are uniformly phase-mean scored.
    run_dir = _synthetic_run_dir(tmp_path, with_obj_source=False)
    res = plot_champion_curves(run_dir)
    assert res["source"] == "phase_mean"
    assert res["champions"].loc[1, "a_obj0"] == pytest.approx(100.0)
    assert (run_dir / "plots" / "phase_mean_champions.png").is_file()


def test_plot_champion_curves_prefer_exam_rows_over_phase_mean_fallback(tmp_path):
    # A run with both sources must score champions from the exam rows alone —
    # the trailing phase_mean phase flies an easier forest distribution and
    # would forge a fake record.
    run_dir = _synthetic_run_dir(tmp_path)
    csv = run_dir / "results" / "outer_population.csv"
    csv.write_text(csv.read_text()
                   + "2,0,z.urdf,8,phase_mean,999.0,0.001\n")
    res = plot_champion_curves(run_dir)
    assert res["source"] == "exam"
    assert res["champions"].index.max() == 1          # gen 2 dropped entirely
    assert res["champions"].loc[1, "a_obj0"] == pytest.approx(100.0)
    assert res["champions"].loc[1, "b_obj1"] == pytest.approx(0.04)


def test_plot_champion_curves_draw_per_gen_baseline(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path)
    res = plot_champion_curves(run_dir)
    assert res["baseline"] is True
    assert (run_dir / "plots" / "exam_champions.png").is_file()


def test_plot_champion_curves_no_baseline_for_custom_validation_catalog(tmp_path):
    # Custom catalog → the reference drone is not the standard mydrone, so
    # the dashed line would be mislabelled.
    run_dir = _exam_baseline_run_dir(tmp_path, validation_catalog="my_cat.txt")
    assert plot_champion_curves(run_dir)["baseline"] is False


def test_plot_champion_curves_fall_back_to_validation_on_nominal_exam(tmp_path):
    # Same fallback as the Pareto star: no exam baseline CSV, but the exam
    # flew the inner loop's forests, so the validation pass is the reference.
    run_dir = _exam_baseline_run_dir(tmp_path, write_csv=False)
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n0,116.0,0.28\n")
    assert plot_champion_curves(run_dir)["baseline"] is True


def test_plot_champion_curves_no_baseline_when_the_exam_forest_differed(tmp_path):
    run_dir = _exam_baseline_run_dir(
        tmp_path, write_csv=False,
        exam_forest={"override_forest": True, "dens_max": 5.5},
    )
    (run_dir / "results" / "validation_summary.csv").write_text(
        "generation,baseline_progress,baseline_cot\n0,116.0,0.28\n")
    assert plot_champion_curves(run_dir)["baseline"] is False


def test_exam_baseline_series_keeps_per_phase_rows(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path)  # rows at outer_gen 0 and 1
    base = _load_exam_baseline_series(run_dir, ["progress_m", "cost_of_transport"])
    assert base.index.tolist() == [0, 1]
    assert base.iloc[1].tolist() == pytest.approx([30.0, 0.6])
    assert not _baseline_is_constant(base)


def test_baseline_is_constant_for_whole_run_sentinel_row(tmp_path):
    # A standalone re-run (exam_baseline_rerun) writes one row tagged
    # outer_gen=-1: it measures the whole run, not a phase, so it belongs on
    # the panels as a horizontal line — never as a point at x=-1.
    run_dir = _exam_baseline_run_dir(tmp_path, rows=((-1, 224.0, 0.29),))
    base = _load_exam_baseline_series(run_dir, ["progress_m", "cost_of_transport"])
    assert _baseline_is_constant(base)


def test_plot_champion_curves_draws_whole_run_exam_baseline(tmp_path):
    run_dir = _exam_baseline_run_dir(tmp_path, rows=((-1, 224.0, 0.29),))
    assert plot_champion_curves(run_dir)["baseline"] is True


def _phase_mean_run_dir(tmp_path, refresh_every=2, enable=True,
                        validation_catalog="", write_csv=True):
    """Phase-mean run dir carrying results/validation_summary.csv — the
    standard-mydrone reference for runs with no exam rollout."""
    run_dir = _synthetic_run_dir(tmp_path, obj_source="phase_mean")
    if write_csv:
        # Four inner gens → two phases of two at refresh_every=2.
        lines = ["generation,baseline_progress,baseline_cot"]
        lines += [f"{g},{prog},{cot}" for g, prog, cot in
                  [(0, 100.0, 0.30), (1, 102.0, 0.30),
                   (2, 110.0, 0.20), (3, 114.0, 0.20)]]
        (run_dir / "results" / "validation_summary.csv").write_text(
            "\n".join(lines) + "\n")
    cfg = (run_dir / "reproducibility" / "config.yaml").read_text()
    cfg += (f"catalog:\n  refresh_urdfs_every: {refresh_every}\n"
            f"validation:\n"
            f"  enable: {'true' if enable else 'false'}\n"
            f"  validation_catalog: '{validation_catalog}'\n")
    (run_dir / "reproducibility" / "config.yaml").write_text(cfg)
    return run_dir


def test_load_refresh_every_reads_catalog_section(tmp_path):
    assert _load_refresh_every(_phase_mean_run_dir(tmp_path, refresh_every=7)) == 7


def test_validation_baseline_series_averages_within_phase_windows(tmp_path):
    # validation_summary.csv is indexed by INNER generation; the champion
    # curves are per OUTER generation, so each phase window collapses to its
    # mean: gens 0-1 → outer 0, gens 2-3 → outer 1.
    run_dir = _phase_mean_run_dir(tmp_path, refresh_every=2)
    base = _load_validation_baseline_series(
        run_dir, ["progress_m", "cost_of_transport"])
    assert base.index.tolist() == [0, 1]
    assert base.iloc[0].tolist() == pytest.approx([101.0, 0.30])
    assert base.iloc[1].tolist() == pytest.approx([112.0, 0.20])


def test_validation_baseline_series_falls_back_to_whole_run_mean(tmp_path):
    # Unreadable refresh period → the inner gens cannot be mapped to phases,
    # so collapse to one whole-run row under the same outer_gen=-1 sentinel
    # the exam re-run uses, and let the plot draw it flat.
    run_dir = _phase_mean_run_dir(tmp_path, refresh_every=0)
    base = _load_validation_baseline_series(
        run_dir, ["progress_m", "cost_of_transport"])
    assert base.index.tolist() == [-1]
    assert _baseline_is_constant(base)
    assert base.iloc[0].tolist() == pytest.approx([106.5, 0.25])


def test_validation_baseline_series_none_for_custom_validation_catalog(tmp_path):
    run_dir = _phase_mean_run_dir(tmp_path, validation_catalog="my_cat.txt")
    assert _load_validation_baseline_series(
        run_dir, ["progress_m", "cost_of_transport"]) is None


def test_plot_champion_curves_draws_validation_baseline_on_phase_mean(tmp_path):
    run_dir = _phase_mean_run_dir(tmp_path)
    res = plot_champion_curves(run_dir)
    assert res["source"] == "phase_mean"
    assert res["baseline"] is True


def test_plot_champion_curves_no_baseline_when_validation_csv_absent(tmp_path):
    run_dir = _phase_mean_run_dir(tmp_path, write_csv=False)
    assert plot_champion_curves(run_dir)["baseline"] is False


def test_plot_champion_curves_no_exam_baseline_on_phase_mean_figure(tmp_path):
    # The baseline flew the EXAM forests; drawing it against phase-mean
    # champions would compare two different distributions on one axis.
    run_dir = _exam_baseline_run_dir(tmp_path)
    csv = run_dir / "results" / "outer_population.csv"
    csv.write_text(csv.read_text().replace(",exam,", ",phase_mean,"))
    res = plot_champion_curves(run_dir)
    assert res["source"] == "phase_mean"
    assert res["baseline"] is False


# ------------------------------------------------- per-URDF reductions

def test_per_urdf_reduction_consistency():
    """Mean over URDFs of the per-URDF matrix must equal the per-individual
    reduction used by the inner loop."""
    D, P, F = 3, 4, 5
    metric = torch.rand(D, P * F)
    per_urdf = metric.view(D, P, F).mean(dim=2)          # (D, P)
    per_ind = metric.view(D, P, F).mean(dim=(0, 2))      # (P,)
    assert torch.allclose(per_urdf.mean(dim=0), per_ind, atol=1e-6)


def test_per_urdf_cot_penalises_crashers():
    """Ratio-of-sums CoT: a slot with ~zero distance must yield a huge CoT,
    not a spuriously small one."""
    D, P, F = 1, 2, 4
    energy = torch.full((D, P, F), 5.0)
    dx = torch.tensor([[[20.0] * F, [1e-6] * F]])  # ind 0 flies, ind 1 crashes
    mg = torch.tensor([[9.81]])
    pu_energy = energy.mean(dim=2)
    pu_dx = dx.mean(dim=2)
    cot = pu_energy / (mg * pu_dx.clamp(min=1e-2))
    assert cot[0, 1] > 100 * cot[0, 0]


# ------------------------------------------------- per-generation front CSV

def _pop_csv(tmp_path, rows, *, with_source=True, n_genes=3):
    """Write a minimal results/outer_population.csv; return the run dir.

    Each row is a dict with at least ``outer_gen``, ``obj_progress_m`` and
    ``obj_cost_of_transport``; missing optional fields are filled in.
    """
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    gene_cols = [f"g{i}" for i in range(n_genes)]
    header = (["outer_gen", "urdf_idx", "urdf_file", "n_score_gens"]
              + (["obj_source"] if with_source else [])
              + ["obj_progress_m", "obj_cost_of_transport"]
              + list(_DIAG_KEYS) + gene_cols)
    lines = [",".join(header)]
    for i, r in enumerate(rows):
        rec = {
            "outer_gen": r["outer_gen"],
            "urdf_idx": r.get("urdf_idx", i),
            "urdf_file": r.get("urdf_file", f"ind_{r.get('urdf_idx', i):03d}.urdf"),
            "n_score_gens": r.get("n_score_gens", 4),
            "obj_source": r.get("obj_source", "exam"),
            "obj_progress_m": r["obj_progress_m"],
            "obj_cost_of_transport": r["obj_cost_of_transport"],
            "fitness": r.get("fitness", 1.0),
            "cost_of_transport": r["obj_cost_of_transport"],
            "progress_m": r["obj_progress_m"],
            "velocity": r.get("velocity", 10.0),
            "crash_rate": r.get("crash_rate", 0.5),
        }
        for j, c in enumerate(gene_cols):
            rec[c] = r.get(c, round(0.1 * (i + j), 4))
        lines.append(",".join(str(rec[c]) for c in header))
    (results / "outer_population.csv").write_text("\n".join(lines) + "\n")
    return tmp_path


_SPECS = [("progress_m", "maximize"), ("cost_of_transport", "minimize")]


def test_front_csv_keeps_only_nondominated_rows(tmp_path):
    """Front membership per generation, on a hand-checked population."""
    run = _pop_csv(tmp_path, [
        # gen 0: idx 0 and 2 are nondominated; idx 1 is dominated by idx 0.
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=90.0, obj_cost_of_transport=0.30),
        dict(outer_gen=0, urdf_idx=2, obj_progress_m=80.0, obj_cost_of_transport=0.10),
        # gen 1: only idx 0 survives (best on both objectives).
        dict(outer_gen=1, urdf_idx=0, obj_progress_m=120.0, obj_cost_of_transport=0.05),
        dict(outer_gen=1, urdf_idx=1, obj_progress_m=110.0, obj_cost_of_transport=0.09),
    ])
    out = build_pareto_front_csv(run, specs=_SPECS, min_progress=0.0)
    assert out == run / "results" / "pareto_front.csv"

    df = pd.read_csv(out)
    assert list(zip(df["outer_gen"], df["urdf_idx"])) == [(0, 0), (0, 2), (1, 0)]
    assert df.loc[df["outer_gen"] == 0, "front_size"].tolist() == [2, 2]
    # front_rank sorts by the first objective, best first.
    assert df.loc[df["outer_gen"] == 0, "front_rank"].tolist() == [0, 1]


def test_front_csv_applies_min_progress_gate(tmp_path):
    """A gated-out morph is excluded even when it is nondominated."""
    rows = [
        # Cheapest CoT in the population, but below the progress gate.
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=10.0, obj_cost_of_transport=0.01),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=0, urdf_idx=2, obj_progress_m=80.0, obj_cost_of_transport=0.10),
    ]
    ungated = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path / "a", rows), specs=_SPECS, min_progress=0.0))
    assert 0 in ungated["urdf_idx"].tolist()

    gated = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path / "b", rows), specs=_SPECS, min_progress=30.0))
    assert gated["urdf_idx"].tolist() == [1, 2]


def test_front_csv_missing_min_progress_means_gate_off(tmp_path):
    """No saved config (or no min_progress_m key) => nothing is barred."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=10.0, obj_cost_of_transport=0.01),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=100.0, obj_cost_of_transport=0.20),
    ]
    # min_progress=None => resolved from the run dir, which has no config.
    run = _pop_csv(tmp_path, rows)
    df = pd.read_csv(build_pareto_front_csv(run, specs=_SPECS, min_progress=None))
    assert df["urdf_idx"].tolist() == [1, 0]

    # An explicit config that omits the knob behaves identically.
    repro = run / "reproducibility"
    repro.mkdir()
    (repro / "config.yaml").write_text("outer:\n  n_elites: 2\n")
    assert _load_min_progress(run) == 0.0
    df2 = pd.read_csv(build_pareto_front_csv(run, specs=_SPECS, min_progress=None))
    assert df2["urdf_idx"].tolist() == [1, 0]


def test_front_csv_drops_phase_mean_when_exam_rows_exist(tmp_path):
    """The final-phase phase_mean fallback must not forge a front tail."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_source="exam",
             obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=1, urdf_idx=0, obj_source="phase_mean",
             obj_progress_m=500.0, obj_cost_of_transport=0.01),
    ]
    df = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path, rows), specs=_SPECS, min_progress=0.0))
    assert df["outer_gen"].tolist() == [0]


def test_front_csv_keeps_phase_mean_when_that_is_all_there_is(tmp_path):
    """Legacy / rescore-off runs: every row is phase_mean, keep them all."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_source="phase_mean",
             obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=1, urdf_idx=0, obj_source="phase_mean",
             obj_progress_m=120.0, obj_cost_of_transport=0.10),
    ]
    df = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path, rows), specs=_SPECS, min_progress=0.0))
    assert df["outer_gen"].tolist() == [0, 1]


def test_front_csv_handles_missing_obj_source_column(tmp_path):
    """Pre-obj_source runs (e.g. outer_full_run_4_64_64_r1) pass through."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=80.0, obj_cost_of_transport=0.10),
    ]
    df = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path, rows, with_source=False),
        specs=_SPECS, min_progress=0.0))
    assert sorted(df["urdf_idx"].tolist()) == [0, 1]


def test_front_csv_carries_genome_unchanged(tmp_path):
    """g* columns must round-trip, so materialize_urdfs can rebuild a front
    morphology straight from this file."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=100.0,
             obj_cost_of_transport=0.20, g0=0.125, g1=0.5, g2=0.875),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=90.0,
             obj_cost_of_transport=0.30, g0=0.25, g1=0.75, g2=0.5),
    ]
    run = _pop_csv(tmp_path, rows)
    front = pd.read_csv(build_pareto_front_csv(run, specs=_SPECS, min_progress=0.0))
    pop = pd.read_csv(run / "results" / "outer_population.csv")
    for _, fr in front.iterrows():
        src = pop[(pop["outer_gen"] == fr["outer_gen"])
                  & (pop["urdf_idx"] == fr["urdf_idx"])].iloc[0]
        for c in ("g0", "g1", "g2"):
            assert fr[c] == pytest.approx(src[c])


def test_front_csv_hypervolume_matches_direct_computation(tmp_path):
    """Per-gen hypervolume, against the fixed (80 m, CoT 0.5) reference."""
    rows = [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=80.0, obj_cost_of_transport=0.10),
    ]
    df = pd.read_csv(build_pareto_front_csv(
        _pop_csv(tmp_path, rows), specs=_SPECS, min_progress=0.0))
    ref, fixed = _hv_reference(_SPECS, np.array([[100.0, -0.20], [80.0, -0.10]]))
    expected = _hypervolume_2d(np.array([[100.0, -0.20], [80.0, -0.10]]), ref)
    assert bool(df["hv_reference_fixed"].iloc[0]) is fixed is True
    assert df["hypervolume"].iloc[0] == pytest.approx(expected)
    assert df["hypervolume"].nunique() == 1  # per-gen scalar, repeated


def test_front_csv_is_idempotent(tmp_path):
    """Rebuilding must not append or reorder — the live writer rewrites this
    file at every phase end."""
    run = _pop_csv(tmp_path, [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=100.0, obj_cost_of_transport=0.20),
        dict(outer_gen=0, urdf_idx=1, obj_progress_m=80.0, obj_cost_of_transport=0.10),
    ])
    first = build_pareto_front_csv(run, specs=_SPECS, min_progress=0.0).read_bytes()
    second = build_pareto_front_csv(run, specs=_SPECS, min_progress=0.0).read_bytes()
    assert first == second


def test_front_csv_missing_or_empty_source_returns_none(tmp_path):
    """No outer_population.csv, or a zero-byte one, must not raise."""
    assert build_pareto_front_csv(tmp_path, specs=_SPECS, min_progress=0.0) is None
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "outer_population.csv").write_text("")
    assert build_pareto_front_csv(tmp_path, specs=_SPECS, min_progress=0.0) is None


def test_front_csv_all_rows_gated_out_returns_none(tmp_path):
    """A generation with nothing admissible yields no rows, not a crash."""
    run = _pop_csv(tmp_path, [
        dict(outer_gen=0, urdf_idx=0, obj_progress_m=5.0, obj_cost_of_transport=0.20),
    ])
    assert build_pareto_front_csv(run, specs=_SPECS, min_progress=50.0) is None


def test_front_csv_safe_swallows_failures(tmp_path, monkeypatch):
    """The live writer must never take down an evolution run."""
    import WP2_Outer_Loop.pareto_fronts as pf

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(pf, "build_pareto_front_csv", boom)
    assert pf.build_pareto_front_csv_safe(tmp_path) is None
