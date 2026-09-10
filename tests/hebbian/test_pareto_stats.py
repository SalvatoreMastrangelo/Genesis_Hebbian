"""
Tests for the cross-condition Pareto-front significance tests
(``WP2_Outer_Loop.pareto_stats``).

Pure-python: synthetic ``outer_population.csv`` fixtures in ``tmp_path``,
no Genesis. Covers the exact permutation core (Mann-Whitney, Wilcoxon,
Holm, the Hodges-Lehmann shift CI), the along-the-front max-T test, the
objective-plane EAF test and p-value maps, hypervolume trajectories, the
per-run indicators, the seed-paired table, the orchestrator and the CLI.
"""

import itertools
import json
import sys
from math import comb
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

from WP2_Outer_Loop.pareto_fronts import _nondominated_mask  # noqa: E402
from WP2_Outer_Loop.pareto_stats import (  # noqa: E402
    StatsGroup,
    all_labelings,
    attain_matrix,
    attainment_test,
    eaf_pvalue_maps,
    eaf_test,
    fisher_table,
    holm,
    hv_trajectories,
    indicator_tests,
    load_run,
    main,
    mwu_exact,
    paired_tests,
    progress_attainment_test,
    progress_curve,
    run_analysis,
    sample_at_fraction,
    timewise_test,
    wilcoxon_exact,
)


# ----------------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------------

def _write_run(run_dir: Path, rows, *, seed=1, refresh=4, min_progress=80.0,
               baseline=None):
    """Minimal outer-loop run folder: config with the standard objective
    pair, gate, seed and refresh period; an ``outer_population.csv``; and
    optionally an exam baseline. ``rows`` are
    ``(outer_gen, progress, cot, obj_source)``."""
    (run_dir / "reproducibility").mkdir(parents=True)
    (run_dir / "results").mkdir()
    cfg = {
        "seed": seed,
        "catalog": {"refresh_urdfs_every": refresh},
        "outer": {
            "objectives": [
                {"name": "progress_m", "direction": "maximize"},
                {"name": "cost_of_transport", "direction": "minimize"},
            ],
            "min_progress_m": min_progress,
        },
    }
    with open(run_dir / "reproducibility" / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    df = pd.DataFrame(
        [{"outer_gen": g, "urdf_idx": i, "obj_source": src,
          "obj_progress_m": p, "obj_cost_of_transport": c,
          "progress_m": p, "cost_of_transport": c}
         for i, (g, p, c, src) in enumerate(rows)]
    )
    df.to_csv(run_dir / "results" / "outer_population.csv", index=False)
    if baseline is not None:
        pd.DataFrame(
            [{"outer_gen": g, "progress_m": p, "cost_of_transport": c}
             for g, (p, c) in enumerate(baseline)]
        ).to_csv(run_dir / "results" / "outer_exam_baseline.csv", index=False)
    return run_dir


def _synthetic_rows(rng, cheaper_above=None, delta=0.0, n_gens=3, n_per_gen=25):
    """Exam rows tracing a convex CoT(progress) curve with noise above
    120 m; group-A style runs are ``delta`` cheaper for progress >=
    ``cheaper_above``. Every run shares identical anchors at 85 m, 120 m
    (cheaper than any noisy point, so the attainment curve below 120 m is
    tied across runs by construction) and the same tip progress (222 m),
    so tiny synthetic groups cannot produce spurious perfect splits."""
    def curve(p):
        return 0.08 + 0.55 * (p / 200.0) ** 2

    rows = [(0, 85.0, 0.10, "exam"), (0, 120.0, 0.15, "exam")]
    tip_cot = curve(222.0) - (delta if cheaper_above is not None else 0.0)
    rows.append((0, 222.0, float(tip_cot), "exam"))
    for g in range(n_gens):
        prog = rng.uniform(121.0, 215.0, n_per_gen)
        cot = curve(prog) + rng.normal(0, 0.01, n_per_gen)
        if cheaper_above is not None:
            cot = np.where(prog >= cheaper_above, cot - delta, cot)
        rows += [(g, float(p), float(max(c, 0.19)), "exam") for p, c in zip(prog, cot)]
    return rows


def _two_condition_runs(tmp_path, n_a=5, n_b=5, delta=0.06, seeds_paired=True):
    """``(runs_a, runs_b)`` directories: A runs are ``delta`` cheaper above
    120 m; seeds pair A_i with B_i when ``seeds_paired``."""
    rng = np.random.default_rng(0)
    runs_a, runs_b = [], []
    for i in range(n_a):
        runs_a.append(_write_run(tmp_path / f"a_r{i}", _synthetic_rows(rng, 120.0, delta),
                                 seed=100 + i, baseline=[(150.0, 0.30)]))
    for i in range(n_b):
        seed = 100 + i if seeds_paired else 500 + i
        runs_b.append(_write_run(tmp_path / f"b_r{i}", _synthetic_rows(rng),
                                 seed=seed, refresh=1, baseline=[(150.0, 0.30)]))
    return runs_a, runs_b


def _brute_force_nondominated(pts):
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                keep[i] = False
                break
    return keep


# ----------------------------------------------------------------------------
#  Non-domination fast path (shared primitive in pareto_fronts)
# ----------------------------------------------------------------------------

def test_nondominated_mask_2d_matches_brute_force_with_ties_and_duplicates():
    rng = np.random.default_rng(3)
    for _ in range(30):
        pts = rng.integers(0, 5, size=(40, 2)).astype(float)   # many ties
        assert (_nondominated_mask(pts) == _brute_force_nondominated(pts)).all()
        pts = rng.normal(size=(60, 2))
        assert (_nondominated_mask(pts) == _brute_force_nondominated(pts)).all()


def test_nondominated_mask_2d_is_fast_on_thousands_of_points():
    import time
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(6000, 2))
    t0 = time.perf_counter()
    mask = _nondominated_mask(pts)
    assert time.perf_counter() - t0 < 2.0   # the O(n^2) loop took ~60 s
    assert 0 < mask.sum() < 200


def test_nondominated_mask_3d_still_brute_force_semantics():
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 0.5], [2.0, 2.0, 2.0]])
    assert _nondominated_mask(pts).tolist() == [False, False, True]


# ----------------------------------------------------------------------------
#  Exact permutation core
# ----------------------------------------------------------------------------

def test_all_labelings_enumerates_every_split_once():
    lab = all_labelings(6, 2)
    assert lab.shape == (comb(6, 2), 2)
    assert len({tuple(r) for r in lab.tolist()}) == comb(6, 2)
    assert lab[0].tolist() == [0, 1]   # the observed labelling comes first


def test_mwu_exact_matches_scipy_exact_two_sided():
    rng = np.random.default_rng(1)
    lab = all_labelings(12, 6)
    for _ in range(10):
        a, b = rng.normal(size=6), rng.normal(size=6) + rng.uniform(-1, 1)
        got = mwu_exact(a, b, lab, better="higher")["p_two_sided"]
        want = mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue
        assert abs(got - want) < 1e-12


def test_mwu_exact_complete_separation_hits_the_floor_with_delta_one():
    a = np.array([5.0, 6.0, 7.0, 8.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = mwu_exact(a, b, better="higher")
    assert res["p_two_sided"] == pytest.approx(2 / comb(8, 4))
    assert res["p_floor"] == pytest.approx(2 / comb(8, 4))
    assert res["cliffs_delta"] == 1.0
    assert res["prob_superiority"] == 1.0
    # 'lower is better' flips the sign of the effect, not the p-value
    res_low = mwu_exact(a, b, better="lower")
    assert res_low["cliffs_delta"] == -1.0
    assert res_low["p_two_sided"] == res["p_two_sided"]


def test_mwu_exact_shift_estimate_and_ci_bracket_a_pure_shift():
    rng = np.random.default_rng(2)
    b = rng.normal(size=8)
    a = b + 1.5                       # exact shift, same noise
    res = mwu_exact(a, b, better="higher")
    assert res["hl_shift"] == pytest.approx(1.5)
    assert res["hl_ci_lo"] <= 1.5 <= res["hl_ci_hi"]
    assert res["hl_ci_lo"] < res["hl_ci_hi"]


def test_holm_step_down_adjustment():
    got = holm([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06
    np.testing.assert_allclose(got, [0.03, 0.06, 0.06])


def test_wilcoxon_exact_matches_scipy():
    rng = np.random.default_rng(4)
    for _ in range(10):
        d = rng.normal(size=7) + 0.5
        got = wilcoxon_exact(d)["p_two_sided"]
        want = wilcoxon(d, alternative="two-sided", method="exact").pvalue
        assert abs(got - want) < 1e-12


def test_wilcoxon_exact_all_same_sign_hits_the_floor():
    res = wilcoxon_exact(np.array([1.0, 2.0, 0.5, 3.0, 1.5, 2.5]))
    assert res["n"] == 6
    assert res["p_two_sided"] == pytest.approx(2 / 2 ** 6)
    assert res["p_floor"] == pytest.approx(2 / 2 ** 6)


# ----------------------------------------------------------------------------
#  Along-the-front tests
# ----------------------------------------------------------------------------

def test_timewise_test_positive_t_means_group_a_better_and_constant_columns_are_null():
    lab = all_labelings(8, 4)
    vals = np.zeros((8, 3))
    vals[:4, 0] = [10, 11, 12, 13]      # A clearly higher in column 0
    vals[4:, 0] = [1, 2, 3, 4]
    vals[:, 1] = 5.0                    # all tied
    vals[:4, 2] = [1, 2, 3, 4]          # A clearly lower in column 2
    vals[4:, 2] = [10, 11, 12, 13]
    res = timewise_test(vals, 4, lab, better="higher")
    assert res["t_obs"][0] > 0 and res["t_obs"][2] < 0
    assert res["t_obs"][1] == 0.0 and res["p_raw"][1] == 1.0
    assert res["p_raw"][0] == pytest.approx(2 / comb(8, 4))
    assert res["p_maxT"][0] >= res["p_raw"][0]
    assert res["mean_diff"][0] == pytest.approx(9.0)
    res_low = timewise_test(vals, 4, lab, better="lower")
    assert res_low["t_obs"][2] > 0      # A lower = A better now


def test_attainment_test_finds_the_band_where_a_is_cheaper_and_censors_non_reaching_runs():
    lab = all_labelings(8, 4)
    grid = np.arange(90.0, 221.0, 10.0)
    # every run: cheap arm identical; above 150 m A is 0.05 cheaper; B tips at 200
    fronts_a = [np.array([[100.0, 0.10 + e], [150.0, 0.20 + e], [210.0, 0.30 + e]])
                for e in (0.000, 0.002, 0.004, 0.006)]
    fronts_b = [np.array([[100.0, 0.10 + e], [150.0, 0.25 + e], [200.0, 0.35 + e]])
                for e in (0.001, 0.003, 0.005, 0.007)]
    df = attainment_test(fronts_a, fronts_b, grid, lab)
    assert list(df.columns) == ["progress_m", "n_reach_a", "n_reach_b", "mean_diff_cot",
                                "hl_shift_cot", "t_obs", "p_raw", "p_maxT"]
    row = df.set_index("progress_m")
    # below 150 m the values interleave -> no evidence
    assert row.loc[100.0, "p_raw"] > 0.2
    # at 150 m and above A is completely separated -> floor, positive t
    assert row.loc[160.0, "p_raw"] == pytest.approx(2 / comb(8, 4))
    assert row.loc[160.0, "t_obs"] > 0
    # A offsets average 0.003, B offsets 0.004 -> -0.05 - 0.001
    assert row.loc[160.0, "mean_diff_cot"] == pytest.approx(-0.051)
    assert row.loc[160.0, "hl_shift_cot"] == pytest.approx(-0.051)
    # 210 m: only A reaches -> censored B ranks worst, still separated
    assert row.loc[210.0, "n_reach_a"] == 4 and row.loc[210.0, "n_reach_b"] == 0
    assert np.isnan(row.loc[210.0, "mean_diff_cot"])
    assert row.loc[210.0, "t_obs"] > 0
    # 220 m: nobody reaches -> null
    assert row.loc[220.0, "p_raw"] == 1.0


def test_progress_curve_is_farthest_point_within_the_cot_budget():
    front = np.array([[100.0, 0.1], [150.0, 0.2], [200.0, 0.3]])
    grid = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    got = progress_curve(front, grid)
    want = np.array([np.nan, 100.0, 100.0, 150.0, 150.0, 200.0, 200.0])
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert np.isnan(progress_curve(np.zeros((0, 2)), grid)).all()


def test_progress_attainment_test_finds_the_cot_band_where_a_flies_farther():
    lab = all_labelings(8, 4)
    grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    # same cheap arm; at CoT >= 0.2 A reaches 40 m farther; only A has a
    # point below CoT 0.10 (B's arm starts at 0.10)
    fronts_a = [np.array([[90.0 + e, 0.08], [150.0 + e, 0.20], [210.0 + e, 0.30]])
                for e in (0.0, 0.5, 1.0, 1.5)]
    fronts_b = [np.array([[90.0 + e, 0.10], [110.0 + e, 0.20], [170.0 + e, 0.30]])
                for e in (0.25, 0.75, 1.25, 1.75)]
    df = progress_attainment_test(fronts_a, fronts_b, grid, lab)
    assert list(df.columns) == ["cot", "n_have_a", "n_have_b", "mean_diff_progress",
                                "hl_shift_progress", "t_obs", "p_raw", "p_maxT"]
    row = df.set_index("cot")
    assert row.loc[0.05, "p_raw"] == 1.0                       # nobody has a point yet
    assert row.loc[0.08, "n_have_a"] == 4 if 0.08 in row.index else True
    assert row.loc[0.10, "n_have_b"] == 4 and row.loc[0.10, "n_have_a"] == 4
    assert row.loc[0.15, "p_raw"] > 0.2                        # arms interleave
    assert row.loc[0.20, "p_raw"] == pytest.approx(2 / comb(8, 4))
    assert row.loc[0.20, "t_obs"] > 0                          # A farther
    assert row.loc[0.20, "mean_diff_progress"] == pytest.approx(40.0 - 0.25)
    assert row.loc[0.40, "mean_diff_progress"] == pytest.approx(40.0 - 0.25)
    assert (df["p_maxT"] >= df["p_raw"]).all()


# ----------------------------------------------------------------------------
#  Objective-plane tests
# ----------------------------------------------------------------------------

def test_attain_matrix_marks_points_weakly_dominated_by_a_front_member():
    fronts = [np.array([[100.0, 0.10], [200.0, 0.30]])]
    p_grid = np.array([90.0, 150.0, 210.0])
    c_grid = np.array([0.05, 0.10, 0.35])
    m = attain_matrix(fronts, p_grid, c_grid).reshape(1, 3, 3)[0]
    # rows: progress 90/150/210 ; cols: cot 0.05/0.10/0.35
    assert m.tolist() == [[False, True, True],
                          [False, False, True],
                          [False, False, False]]


def test_eaf_test_identical_groups_give_p_one_and_separated_groups_reach_the_floor():
    lab = all_labelings(8, 4)
    p_grid = np.arange(90.0, 221.0, 5.0)
    c_grid = np.arange(0.05, 0.5, 0.01)
    same = [np.array([[100.0, 0.10], [200.0, 0.30]]) for _ in range(4)]
    res = eaf_test(same, [f.copy() for f in same], p_grid, c_grid, lab)
    assert res["ks"] == 0.0 and res["p"] == 1.0
    cheap = [np.array([[100.0, 0.10], [200.0, 0.20]]) for _ in range(4)]
    res = eaf_test(cheap, same, p_grid, c_grid, lab)
    assert res["ks"] == 1.0
    assert res["p"] == pytest.approx(2 / comb(8, 4))
    assert res["diff"].shape == (len(p_grid), len(c_grid))
    assert res["ks_null"].shape == (comb(8, 4),)
    assert 100.0 < res["at_progress"] <= 200.0 and 0.20 <= res["at_cot"] < 0.30


def test_fisher_table_two_sided_probabilities():
    tab = fisher_table(4, 4)
    assert tab.shape == (9, 5)
    assert tab[4, 4] == pytest.approx(2 / comb(8, 4))       # all-A split of 4
    assert tab[8, 4] == pytest.approx(1.0)                  # everyone attains
    assert tab[0, 0] == pytest.approx(1.0)                  # nobody attains
    assert np.isnan(tab[2, 3])                              # impossible cell


def test_eaf_pvalue_maps_min_equals_eaf_test_p_and_sign_follows_direction():
    lab = all_labelings(8, 4)
    p_grid = np.arange(90.0, 221.0, 5.0)
    c_grid = np.arange(0.05, 0.5, 0.01)
    cheap = [np.array([[100.0, 0.10], [200.0, 0.20]]) for _ in range(4)]
    dear = [np.array([[100.0, 0.10], [200.0, 0.30]]) for _ in range(4)]
    res = eaf_test(cheap, dear, p_grid, c_grid, lab)
    maps = eaf_pvalue_maps(cheap, dear, p_grid, c_grid, lab, res["ks_null"])
    for key in ("p_raw", "p_maxd", "p_minp", "sign"):
        assert maps[key].shape == (len(p_grid), len(c_grid))
    assert np.nanmin(maps["p_raw"]) == pytest.approx(2 / comb(8, 4))
    assert np.nanmin(maps["p_maxd"]) == pytest.approx(res["p"])
    assert np.nanmin(maps["p_minp"]) == pytest.approx(res["p"])
    sig = maps["p_maxd"] < 0.05
    assert sig.any() and (maps["sign"][sig] > 0).all()      # A attains more
    # swapping the groups flips the sign only
    swapped = eaf_pvalue_maps(dear, cheap, p_grid, c_grid, lab, res["ks_null"])
    np.testing.assert_allclose(swapped["p_raw"], maps["p_raw"], equal_nan=True)
    assert (swapped["sign"][sig] < 0).all()


# ----------------------------------------------------------------------------
#  Hypervolume over the run
# ----------------------------------------------------------------------------

def test_hv_trajectories_phase_and_cumulative_fronts(tmp_path):
    rows = [
        (0, 100.0, 0.20, "exam"), (0, 150.0, 0.30, "exam"),
        (1, 120.0, 0.35, "exam"),                      # dominated cumulatively
        (2, 180.0, 0.25, "exam"), (2, 60.0, 0.05, "exam"),   # 60 m gated out
        (3, 200.0, 0.10, "phase_mean"),                # not exam -> dropped
    ]
    run = _write_run(tmp_path / "run", rows)
    traj = hv_trajectories(run)
    assert traj["outer_gen"].tolist() == [0, 1, 2]
    # ref (80 m, 0.5): gen 0 front {(100,.2),(150,.3)} = 20*0.3 + 50*0.2 = 16
    assert traj.loc[0, "hv_phase"] == pytest.approx(16.0)
    assert traj.loc[0, "hv_cum"] == pytest.approx(16.0)
    # gen 1 alone: (120, .35) -> 40*0.15 = 6 ; cumulative unchanged
    assert traj.loc[1, "hv_phase"] == pytest.approx(6.0)
    assert traj.loc[1, "hv_cum"] == pytest.approx(16.0)
    # gen 2 alone: (180,.25) -> 100*0.25 = 25 ; cumulative {(100,.2),(180,.25)}
    assert traj.loc[2, "hv_phase"] == pytest.approx(25.0)
    assert traj.loc[2, "hv_cum"] == pytest.approx(20 * 0.3 + 80 * 0.25)
    assert (traj["hv_cum"].diff().dropna() >= 0).all()


def test_sample_at_fraction_picks_nearest_phase():
    traj = pd.DataFrame({"outer_gen": range(5), "hv_cum": [1.0, 2.0, 3.0, 4.0, 5.0]})
    got = sample_at_fraction(traj, "hv_cum", np.array([0.0, 0.5, 0.9, 1.0]))
    np.testing.assert_allclose(got, [1.0, 3.0, 5.0, 5.0])


# ----------------------------------------------------------------------------
#  Per-run indicators, tables
# ----------------------------------------------------------------------------

def test_load_run_reads_seed_refresh_front_and_indicators(tmp_path):
    rows = [(0, 100.0, 0.10, "exam"), (0, 160.0, 0.20, "exam"), (1, 200.0, 0.30, "exam")]
    run = _write_run(tmp_path / "run", rows, seed=42, refresh=6)
    rs = load_run(run, levels=(150.0, 190.0))
    assert rs.seed == 42 and rs.refresh == 6 and rs.name == "run"
    np.testing.assert_allclose(rs.front, [[100.0, 0.10], [160.0, 0.20], [200.0, 0.30]])
    assert rs.tip == 200.0 and rs.arm == 100.0
    assert rs.hv == pytest.approx(20 * 0.4 + 60 * 0.3 + 40 * 0.2)
    assert rs.indicators()["cot_at_150"] == pytest.approx(0.20)
    assert rs.indicators()["cot_at_190"] == pytest.approx(0.30)
    assert rs.indicators()["hv"] == pytest.approx(rs.hv)
    rs2 = load_run(run, levels=(150.0,), cot_levels=(0.15, 0.25))
    assert rs2.indicators()["prog_at_cot_0.15"] == pytest.approx(100.0)
    assert rs2.indicators()["prog_at_cot_0.25"] == pytest.approx(160.0)


def test_indicator_tests_table_has_one_row_per_indicator_with_holm(tmp_path):
    runs_a, runs_b = _two_condition_runs(tmp_path)
    a = [load_run(r) for r in runs_a]
    b = [load_run(r) for r in runs_b]
    df = indicator_tests(a, b)
    assert set(df["key"]) == {"hv", "cot_at_150", "cot_at_170", "cot_at_190",
                              "prog_at_cot_0.15", "prog_at_cot_0.2", "prog_at_cot_0.25",
                              "tip", "arm"}
    assert df.set_index("key").loc["prog_at_cot_0.2", "better"] == "higher"
    assert df.set_index("key").loc["prog_at_cot_0.25", "hl_shift"] > 0
    assert (df["p_holm"] >= df["p_two_sided"]).all()
    hv_row = df.set_index("key").loc["hv"]
    assert hv_row["n_a"] == 5 and hv_row["n_b"] == 5
    assert hv_row["better"] == "higher" and hv_row["hl_shift"] > 0
    assert df.set_index("key").loc["cot_at_190", "better"] == "lower"


def test_paired_tests_pairs_by_seed_and_averages_duplicates(tmp_path):
    runs_a, runs_b = _two_condition_runs(tmp_path, n_a=3, n_b=3)
    # a fourth A run duplicating seed 100 -> averaged with a_r0, not dropped
    rng = np.random.default_rng(9)
    dup = _write_run(tmp_path / "a_dup", _synthetic_rows(rng, 120.0, 0.06), seed=100)
    a = [load_run(r) for r in runs_a + [dup]]
    b = [load_run(r) for r in runs_b]
    df = paired_tests(a, b)
    assert df.set_index("key").loc["hv", "n_pairs"] == 3
    assert sorted(json.loads(df.set_index("key").loc["hv", "seeds"])) == [100, 101, 102]
    assert (df["p_two_sided"] >= df["p_floor"]).all()


def test_paired_tests_without_shared_seeds_is_empty(tmp_path):
    runs_a, runs_b = _two_condition_runs(tmp_path, n_a=2, n_b=2, seeds_paired=False)
    df = paired_tests([load_run(r) for r in runs_a], [load_run(r) for r in runs_b])
    assert df.empty


# ----------------------------------------------------------------------------
#  Orchestrator + CLI
# ----------------------------------------------------------------------------

def test_run_analysis_writes_tables_figures_and_summary(tmp_path):
    runs_a, runs_b = _two_condition_runs(tmp_path)
    ga = StatsGroup("alpha", "red", [load_run(r) for r in runs_a])
    gb = StatsGroup("beta", "blue", [load_run(r) for r in runs_b])
    out = tmp_path / "stats"
    summary = run_analysis(ga, gb, out, grid_step=5.0, cot_step=0.01)
    for name in ("indicators_per_run.csv", "indicator_tests.csv", "paired_tests.csv",
                 "attainment_test.csv", "progress_attainment_test.csv", "eaf_maps.npz",
                 "hv_trajectories.csv", "hv_time_test.csv", "summary.json",
                 "indicator_strips.png", "indicator_strips_progress.png",
                 "attainment_difference.png", "attainment_difference_progress.png",
                 "eaf_difference.png", "eaf_pvalue.png", "hypervolume_significance.png"):
        assert (out / name).is_file(), name
    assert summary["n_labelings"] == comb(10, 5)
    assert summary["groups"] == ["alpha", "beta"]
    # the synthetic A runs are cheaper above 120 m: the band must start near there
    band = summary["attainment"]["significant_band_m"]
    assert band is not None and 115.0 <= band[0] <= 135.0
    assert summary["eaf"]["ks"] == 1.0 and summary["eaf"]["p"] == pytest.approx(2 / comb(10, 5))
    # read the other way: A flies farther for CoT budgets above its 120 m anchor
    cband = summary["progress_attainment"]["significant_band_cot"]
    assert cband is not None and 0.15 <= cband[0] <= 0.30 and cband[1] >= 0.35
    assert summary["hypervolume"]["mwu"]["p_two_sided"] == pytest.approx(2 / comb(10, 5))
    assert summary["star"] == pytest.approx([150.0, 0.30])
    saved = json.load(open(out / "summary.json"))
    assert saved["eaf"]["p"] == summary["eaf"]["p"]


def test_cli_requires_exactly_two_groups_and_writes_output(tmp_path, capsys):
    runs_a, runs_b = _two_condition_runs(tmp_path, n_a=3, n_b=3)
    out = tmp_path / "cli_out"
    argv = (["--group", "alpha", "red"] + [str(r) for r in runs_a]
            + ["--group", "beta", "blue"] + [str(r) for r in runs_b]
            + ["--out", str(out), "--grid-step", "5", "--cot-step", "0.01"])
    assert main(argv) == 0
    assert (out / "summary.json").is_file()
    printed = capsys.readouterr().out
    assert "hv" in printed and "p-value" in printed
    with pytest.raises(SystemExit):
        main(["--group", "alpha", "red", str(runs_a[0]), "--out", str(out)])
