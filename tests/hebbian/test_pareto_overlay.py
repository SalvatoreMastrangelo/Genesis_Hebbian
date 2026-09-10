"""
Tests for the cross-run Pareto-front overlay (``WP2_Outer_Loop.pareto_overlay``).

Pure-python: synthetic ``outer_population.csv`` / ``outer_exam_baseline.csv``
fixtures in ``tmp_path``, no Genesis. Covers the cumulative front, the
attainment curve and its group mean, the exam star, the figure writer and
the ``--group`` CLI.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2_Outer_Loop.pareto_overlay import (  # noqa: E402
    FrontGroup,
    attainment_curve,
    cumulative_front,
    exam_star,
    main,
    mean_attainment,
    plot_front_overlay,
)


# ----------------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------------

def _write_run(run_dir: Path, rows, *, min_progress=80.0, baseline=None,
               validation=False):
    """Minimal outer-loop run folder: config with the standard objective
    pair + gate, an ``outer_population.csv`` and optionally an exam
    baseline. ``rows`` are ``(outer_gen, progress, cot, obj_source)``.
    ``validation=True`` records a default-catalog validation env, the gate
    ``pareto_plots`` requires before it draws the Bixler star."""
    (run_dir / "reproducibility").mkdir(parents=True)
    (run_dir / "results").mkdir()
    cfg = {"outer": {
        "objectives": [
            {"name": "progress_m", "direction": "maximize"},
            {"name": "cost_of_transport", "direction": "minimize"},
        ],
        "min_progress_m": min_progress,
    }}
    if validation:
        cfg["validation"] = {"enable": True, "validation_catalog": None}
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


# ----------------------------------------------------------------------------
#  Attainment curve
# ----------------------------------------------------------------------------

def test_attainment_curve_is_cheapest_point_reaching_at_least_P():
    front = np.array([[100.0, 0.1], [150.0, 0.2], [200.0, 0.3]])
    grid = np.array([80.0, 100.0, 120.0, 150.0, 180.0, 200.0, 210.0])
    got = attainment_curve(front, grid)
    want = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, np.nan])
    np.testing.assert_allclose(got, want, equal_nan=True)


def test_attainment_curve_does_not_depend_on_point_order():
    front = np.array([[200.0, 0.3], [100.0, 0.1], [150.0, 0.2]])
    grid = np.array([120.0, 160.0])
    np.testing.assert_allclose(attainment_curve(front, grid), [0.2, 0.3])


def test_attainment_curve_of_empty_front_is_all_nan():
    got = attainment_curve(np.zeros((0, 2)), np.array([100.0, 150.0]))
    assert np.isnan(got).all()


# ----------------------------------------------------------------------------
#  Group mean
# ----------------------------------------------------------------------------

def test_mean_attainment_averages_runs_and_stops_at_shortest_tip():
    long_run = np.array([[100.0, 0.1], [200.0, 0.3]])
    short_run = np.array([[100.0, 0.3], [150.0, 0.5]])
    grid = np.array([100.0, 150.0, 175.0, 200.0])
    got = mean_attainment([long_run, short_run], grid)
    # 100 m: (0.1 + 0.3)/2 ; 150 m: (0.3 + 0.5)/2 ; beyond the short tip: NaN
    np.testing.assert_allclose(got, [0.2, 0.4, np.nan, np.nan], equal_nan=True)


# ----------------------------------------------------------------------------
#  Cumulative front from a run folder
# ----------------------------------------------------------------------------

def test_cumulative_front_pools_generations_gates_and_drops_dominated(tmp_path):
    rows = [
        (0, 100.0, 0.10, "exam"),   # kept
        (0, 150.0, 0.25, "exam"),   # dominated by gen-1 (160, 0.20)
        (0,  60.0, 0.05, "exam"),   # below the 80 m gate
        (1, 160.0, 0.20, "exam"),   # kept
        (1, 200.0, 0.30, "exam"),   # kept
        (1, 220.0, 0.01, "phase_mean"),  # would dominate all, but not exam-scored
    ]
    run = _write_run(tmp_path / "run", rows)
    front = cumulative_front(run)
    np.testing.assert_allclose(
        front, [[100.0, 0.10], [160.0, 0.20], [200.0, 0.30]]
    )


def test_cumulative_front_honours_explicit_gate_override(tmp_path):
    rows = [(0, 90.0, 0.10, "exam"), (0, 150.0, 0.20, "exam")]
    run = _write_run(tmp_path / "run", rows)
    np.testing.assert_allclose(
        cumulative_front(run, min_progress=120.0), [[150.0, 0.20]]
    )


# ----------------------------------------------------------------------------
#  Exam star
# ----------------------------------------------------------------------------

def test_exam_star_is_mean_over_all_phases_of_all_runs(tmp_path):
    a = _write_run(tmp_path / "a", [(0, 100.0, 0.1, "exam")],
                   baseline=[(190.0, 0.28), (192.0, 0.30)])
    b = _write_run(tmp_path / "b", [(0, 100.0, 0.1, "exam")],
                   baseline=[(188.0, 0.26)])
    prog, cot = exam_star([a, b])
    assert prog == pytest.approx((190.0 + 192.0 + 188.0) / 3)
    assert cot == pytest.approx((0.28 + 0.30 + 0.26) / 3)


def test_exam_star_is_none_without_baseline_files(tmp_path):
    a = _write_run(tmp_path / "a", [(0, 100.0, 0.1, "exam")])
    assert exam_star([a]) is None


# ----------------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------------

def test_plot_front_overlay_writes_png_and_pdf(tmp_path):
    groups = [
        FrontGroup("co-design", "red",
                   [np.array([[100.0, 0.1], [200.0, 0.3]]),
                    np.array([[110.0, 0.12], [210.0, 0.32]])],
                   ["cod_r0", "cod_r1"]),
        FrontGroup("morphology-only", "blue",
                   [np.array([[100.0, 0.15], [190.0, 0.35]])],
                   ["morph_r0"]),
    ]
    out = plot_front_overlay(groups, tmp_path / "overlay.png",
                             star=(190.8, 0.288))
    assert out == tmp_path / "overlay.png"
    assert out.stat().st_size > 0
    assert (tmp_path / "overlay.pdf").stat().st_size > 0


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def test_main_builds_figure_from_group_args(tmp_path):
    base = [(190.0, 0.28)]
    c0 = _write_run(tmp_path / "c0", [(0, 100.0, 0.10, "exam"), (0, 200.0, 0.30, "exam")], baseline=base)
    c1 = _write_run(tmp_path / "c1", [(0, 100.0, 0.11, "exam"), (0, 205.0, 0.31, "exam")], baseline=base)
    m0 = _write_run(tmp_path / "m0", [(0, 100.0, 0.15, "exam"), (0, 195.0, 0.35, "exam")], baseline=base)
    out = tmp_path / "fig" / "overlay.png"
    rc = main([
        "--group", "co-design", "red", str(c0), str(c1),
        "--group", "morphology-only", "blue", str(m0),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0
    # The attainment grid used for the means is written next to the figure
    # so the thesis numbers can be quoted from it.
    means = pd.read_csv(out.with_name("overlay_mean_attainment.csv"))
    assert set(["progress_m", "co-design", "morphology-only"]) <= set(means.columns)
    row = means.loc[np.isclose(means["progress_m"], 150.0)].iloc[0]
    assert row["co-design"] == pytest.approx((0.30 + 0.31) / 2)
    assert row["morphology-only"] == pytest.approx(0.35)


def test_main_rejects_group_without_runs(tmp_path):
    with pytest.raises(SystemExit):
        main(["--group", "co-design", "red", "--out", str(tmp_path / "o.png")])


# ----------------------------------------------------------------------------
#  Run-dir resolution (globbing the synced ``outer_*_rX`` wrappers)
# ----------------------------------------------------------------------------

def test_resolve_run_dir_descends_into_single_timestamped_child(tmp_path):
    from WP2_Outer_Loop.pareto_overlay import resolve_run_dir
    wrapper = tmp_path / "outer_exam_r0"
    run = _write_run(wrapper / "2026-08-26_18-19-52_exam", [(0, 100.0, 0.1, "exam")])
    (wrapper / "logs").mkdir()  # sibling without results/ must be ignored
    assert resolve_run_dir(wrapper) == run
    assert resolve_run_dir(run) == run


def test_resolve_run_dir_rejects_folder_without_population_csv(tmp_path):
    from WP2_Outer_Loop.pareto_overlay import resolve_run_dir
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(tmp_path / "empty")


# ----------------------------------------------------------------------------
#  Dashed continuation past the shortest tip (partial mean + run counts)
# ----------------------------------------------------------------------------

def test_partial_mean_attainment_averages_defined_runs_and_counts_them():
    from WP2_Outer_Loop.pareto_overlay import partial_mean_attainment
    long_run = np.array([[100.0, 0.1], [200.0, 0.3]])
    short_run = np.array([[100.0, 0.3], [150.0, 0.5]])
    grid = np.array([100.0, 150.0, 175.0, 200.0, 250.0])
    mean, n = partial_mean_attainment([long_run, short_run], grid)
    np.testing.assert_allclose(mean, [0.2, 0.4, 0.3, 0.3, np.nan], equal_nan=True)
    np.testing.assert_array_equal(n, [2, 2, 1, 1, 0])


def test_mean_attainment_table_carries_partial_mean_and_counts():
    from WP2_Outer_Loop.pareto_overlay import mean_attainment_table
    g = FrontGroup("cod", "red",
                   [np.array([[100.0, 0.1], [200.0, 0.3]]),
                    np.array([[100.0, 0.3], [150.0, 0.5]])])
    t = mean_attainment_table([g], np.array([100.0, 175.0]))
    assert list(t.columns) == ["progress_m", "cod", "cod_partial", "cod_n"]
    np.testing.assert_allclose(t["cod"], [0.2, np.nan], equal_nan=True)
    np.testing.assert_allclose(t["cod_partial"], [0.2, 0.3])
    np.testing.assert_array_equal(t["cod_n"], [2, 1])


def test_plot_front_overlay_tail_variant_writes_separate_files(tmp_path):
    groups = [
        FrontGroup("co-design", "red",
                   [np.array([[100.0, 0.1], [200.0, 0.3]]),
                    np.array([[110.0, 0.12], [210.0, 0.32]])]),
        FrontGroup("morphology-only", "blue",
                   [np.array([[100.0, 0.15], [190.0, 0.35]]),
                    np.array([[100.0, 0.16], [220.0, 0.40]])]),
    ]
    out = plot_front_overlay(groups, tmp_path / "overlay_tail.png",
                             star=(190.8, 0.288), tail=True)
    assert out.stat().st_size > 0
    assert (tmp_path / "overlay_tail.pdf").stat().st_size > 0


def test_main_also_writes_tail_figure(tmp_path):
    c0 = _write_run(tmp_path / "c0", [(0, 100.0, 0.10, "exam"), (0, 200.0, 0.30, "exam")])
    c1 = _write_run(tmp_path / "c1", [(0, 100.0, 0.11, "exam"), (0, 220.0, 0.31, "exam")])
    m0 = _write_run(tmp_path / "m0", [(0, 100.0, 0.15, "exam"), (0, 195.0, 0.35, "exam")])
    out = tmp_path / "fig" / "overlay.png"
    assert main(["--group", "co-design", "red", str(c0), str(c1),
                 "--group", "morphology-only", "blue", str(m0),
                 "--out", str(out), "--no-star"]) == 0
    tail = out.with_name("overlay_tail.png")
    assert tail.stat().st_size > 0
    assert tail.with_suffix(".pdf").stat().st_size > 0
    means = pd.read_csv(out.with_name("overlay_mean_attainment.csv"))
    row = means.loc[np.isclose(means["progress_m"], 210.0)].iloc[0]
    assert np.isnan(row["co-design"])            # past the shortest co-design tip
    assert row["co-design_partial"] == pytest.approx(0.31)
    assert row["co-design_n"] == 1


# ----------------------------------------------------------------------------
#  Styling contract: legend = mean lines + star only, no count strip
# ----------------------------------------------------------------------------

def _two_groups():
    return [
        FrontGroup("co-design", "red",
                   [np.array([[100.0, 0.1], [200.0, 0.3]]),
                    np.array([[110.0, 0.12], [210.0, 0.32]])]),
        FrontGroup("morphology-only", "blue",
                   [np.array([[100.0, 0.15], [190.0, 0.35]]),
                    np.array([[100.0, 0.16], [220.0, 0.40]])]),
    ]


@pytest.mark.parametrize("tail", [False, True])
def test_legend_lists_only_group_means_and_star(tail):
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, _two_groups(), star=(190.8, 0.288), tail=tail)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert len(labels) == 3
    assert labels[0].startswith("co-design")
    assert labels[1].startswith("morphology-only")
    assert labels[2] == "Bixler (generalist controller)"


def test_legend_has_no_star_entry_without_star():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, _two_groups(), star=None)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert labels == ["co-design (2 runs)", "morphology-only (2 runs)"]


def test_tail_figure_is_a_single_axes(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    captured = {}
    real_savefig = plt.Figure.savefig

    def spy(self, *a, **k):
        captured["n_axes"] = len(self.axes)
        return real_savefig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    plot_front_overlay(_two_groups(), tmp_path / "t.png", tail=True)
    assert captured["n_axes"] == 1


def test_star_is_gold_with_black_edge_like_pareto_plots():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, _two_groups(), star=(190.8, 0.288))
    stars = [c for c in ax.collections if c.get_paths()]  # scatter → PathCollection
    plt.close(fig)
    assert len(stars) == 1
    assert tuple(stars[0].get_facecolor()[0]) == to_rgba("gold")
    assert tuple(stars[0].get_edgecolor()[0]) == to_rgba("black")


# ----------------------------------------------------------------------------
#  The star label is shared with the per-run outer-loop plots
# ----------------------------------------------------------------------------

def test_pareto_front_plot_legend_names_the_bixler(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_plots import plot_pareto_front
    run = _write_run(tmp_path / "run",
                     [(0, 100.0, 0.10, "exam"), (0, 200.0, 0.30, "exam"),
                      (1, 120.0, 0.12, "exam"), (1, 210.0, 0.32, "exam")],
                     baseline=[(190.0, 0.28)], validation=True)
    seen = []
    real_savefig = plt.Figure.savefig

    def spy(self, *a, **k):
        for ax in self.axes:
            leg = ax.get_legend()
            if leg is not None:
                seen.extend(t.get_text() for t in leg.get_texts())
        return real_savefig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    res = plot_pareto_front(run)
    assert res is not None and res["star"] == pytest.approx((190.0, 0.28))
    assert "Bixler (generalist controller)" in seen
    assert not any("mydrone" in s for s in seen)


# ----------------------------------------------------------------------------
#  Mirrored rule: solid only where every front covers P, dashed head + tail
#  = mean over the fronts that cover P
# ----------------------------------------------------------------------------

_LONG = np.array([[100.0, 0.10], [150.0, 0.20], [200.0, 0.30]])
_LATE = np.array([[140.0, 0.15], [200.0, 0.32]])   # no cheap arm: starts at 140 m


def test_partial_mean_attainment_excludes_runs_whose_front_starts_beyond_P():
    from WP2_Outer_Loop.pareto_overlay import partial_mean_attainment
    mean, n = partial_mean_attainment([_LONG, _LATE], np.array([120.0, 140.0, 160.0]))
    # 120 m: only the long run's front covers it (its 150 m body, CoT 0.20);
    # the late run's cheapest body would count 0.15 under the plain
    # attainment, but its front has not started, so it is left out.
    np.testing.assert_allclose(mean, [0.20, 0.175, 0.31])
    np.testing.assert_array_equal(n, [1, 2, 2])


def test_mean_attainment_is_nan_before_every_front_has_started():
    got = mean_attainment([_LONG, _LATE], np.array([120.0, 140.0, 200.0, 210.0]))
    np.testing.assert_allclose(got, [np.nan, 0.175, 0.31, np.nan], equal_nan=True)


def _dashed_x_ranges(ax):
    out = []
    for ln in ax.get_lines():
        if ln.get_linestyle() in ("-", "solid", "None"):
            continue
        x = np.asarray(ln.get_xdata(), dtype=float)
        y = np.asarray(ln.get_ydata(), dtype=float)
        x = x[np.isfinite(y)]
        if x.size:
            out.append((x.min(), x.max()))
    return out


def test_main_figure_has_dashed_head_but_no_dashed_tail():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, [FrontGroup("g", "red", [_LONG, _LATE])], tail=False)
    ranges = _dashed_x_ranges(ax)
    plt.close(fig)
    assert ranges, "expected a dashed head segment"
    assert all(hi <= 140.0 + 0.5 for _lo, hi in ranges), ranges   # head only
    assert all(lo <= 100.0 + 0.5 for lo, _hi in ranges), ranges


def test_tail_figure_has_dashed_head_and_dashed_tail():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    short = np.array([[100.0, 0.12], [180.0, 0.25]])
    fig, ax = plt.subplots()
    draw_front_overlay(ax, [FrontGroup("g", "red", [_LONG, _LATE, short])], tail=True)
    ranges = sorted(_dashed_x_ranges(ax))
    plt.close(fig)
    assert len(ranges) == 2, ranges
    (h_lo, h_hi), (t_lo, t_hi) = ranges
    assert h_lo <= 100.5 and 139.5 <= h_hi <= 140.5      # head: up to the last front start
    assert 179.5 <= t_lo <= 180.5 and t_hi >= 199.5      # tail: from the first tip on


def test_solid_mean_covers_exactly_the_range_all_fronts_span():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, [FrontGroup("g", "red", [_LONG, _LATE])], tail=True)
    solid = [ln for ln in ax.get_lines()
             if ln.get_linewidth() > 2 and ln.get_linestyle() in ("-", "solid")]
    plt.close(fig)
    assert len(solid) == 1
    y = np.asarray(solid[0].get_ydata(), float); x = np.asarray(solid[0].get_xdata(), float)
    x = x[np.isfinite(y)]
    assert x.min() == pytest.approx(140.0) and x.max() == pytest.approx(200.0)


# ----------------------------------------------------------------------------
#  Solid threshold: dashed only when fewer than ``min_runs`` fronts cover P
# ----------------------------------------------------------------------------

_LATE_LONG = np.array([[140.0, 0.15], [220.0, 0.32]])   # starts late, ends last
_SHORT = np.array([[100.0, 0.12], [180.0, 0.25]])        # ends first


def test_mean_attainment_honours_min_runs_threshold():
    grid = np.array([85.0, 120.0, 190.0, 210.0])
    fronts = [_LONG, _LATE_LONG, _SHORT]
    # covering counts: 85 → 0, 120 → 2 (long, short), 190 → 2 (long, late), 210 → 1 (late)
    # 120 m: long's cheapest body reaching 120 m is its 150 m one (0.20),
    # short's is its 180 m one (0.25) → 0.225
    default = mean_attainment(fronts, grid)                 # all 3 required
    np.testing.assert_allclose(default, [np.nan] * 4, equal_nan=True)
    two = mean_attainment(fronts, grid, min_runs=2)
    np.testing.assert_allclose(two, [np.nan, 0.225, 0.31, np.nan], equal_nan=True)


def test_min_runs_above_group_size_means_all_runs():
    grid = np.array([150.0])
    assert mean_attainment([_LONG, _SHORT], grid, min_runs=4)[0] == pytest.approx(0.225)


def test_solid_covers_where_at_least_min_runs_fronts_cover(tmp_path):
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pareto_overlay import draw_front_overlay
    fig, ax = plt.subplots()
    draw_front_overlay(ax, [FrontGroup("g", "red", [_LONG, _LATE_LONG, _SHORT])],
                       tail=True, min_runs=2)
    solid = [ln for ln in ax.get_lines()
             if ln.get_linewidth() > 2 and ln.get_linestyle() in ("-", "solid")]
    dashed = sorted(_dashed_x_ranges(ax))
    plt.close(fig)
    y = np.asarray(solid[0].get_ydata(), float); x = np.asarray(solid[0].get_xdata(), float)
    x = x[np.isfinite(y)]
    assert x.min() == pytest.approx(100.0) and x.max() == pytest.approx(200.0)
    # only the tail (one run left past 200 m) is dashed; the head has no
    # sub-threshold stretch because two fronts already cover 100 m
    assert len(dashed) == 1 and 199.5 <= dashed[0][0] <= 200.5 and dashed[0][1] >= 219.5


def test_cli_solid_min_runs_flag_reaches_the_csv(tmp_path):
    c0 = _write_run(tmp_path / "c0", [(0, 100.0, 0.10, "exam"), (0, 200.0, 0.30, "exam")])
    c1 = _write_run(tmp_path / "c1", [(0, 140.0, 0.15, "exam"), (0, 220.0, 0.32, "exam")])
    c2 = _write_run(tmp_path / "c2", [(0, 100.0, 0.12, "exam"), (0, 180.0, 0.25, "exam")])
    out = tmp_path / "fig" / "overlay.png"
    assert main(["--group", "co-design", "red", str(c0), str(c1), str(c2),
                 "--out", str(out), "--no-star", "--solid-min-runs", "2"]) == 0
    means = pd.read_csv(out.with_name("overlay_mean_attainment.csv"))
    at = lambda P: means.loc[np.isclose(means["progress_m"], P)].iloc[0]
    assert at(120.0)["co-design"] == pytest.approx(0.275)     # 2 of 3 cover → solid: (0.30 + 0.25) / 2
    assert np.isnan(at(210.0)["co-design"])                    # 1 of 3 → dashed only
    assert at(210.0)["co-design_partial"] == pytest.approx(0.32)
