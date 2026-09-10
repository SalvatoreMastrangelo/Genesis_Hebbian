"""
Tests for the cross-run exam hypervolume aggregate
(``WP2_Outer_Loop.hypervolume_aggregate``).

Pure-python: synthetic ``outer_population.csv`` fixtures in ``tmp_path``,
no Genesis. Covers the per-run per-phase / cumulative HV series (checked
by hand against the fixed (80 m, CoT 0.5) reference), agreement with
``pareto_plots.plot_pareto_front``, the group mean ± std aggregate over
ragged run lengths, the figure writer and the ``--group`` CLI.
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

from WP2_Outer_Loop.hypervolume_aggregate import (  # noqa: E402
    HVGroup,
    MEASURES,
    aggregate_hv,
    aggregate_table,
    draw_hv_aggregate,
    hv_series,
    main,
    plot_hv_aggregate,
)


# ----------------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------------

def _write_run(run_dir: Path, rows, *, min_progress=80.0):
    """Minimal outer-loop run folder: config with the standard objective
    pair + gate and an ``outer_population.csv``. ``rows`` are
    ``(outer_gen, progress, cot, obj_source)``."""
    (run_dir / "reproducibility").mkdir(parents=True)
    (run_dir / "results").mkdir()
    cfg = {"outer": {
        "objectives": [
            {"name": "progress_m", "direction": "maximize"},
            {"name": "cost_of_transport", "direction": "minimize"},
        ],
        "min_progress_m": min_progress,
    }}
    with open(run_dir / "reproducibility" / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    df = pd.DataFrame(
        [{"outer_gen": g, "urdf_idx": i, "obj_source": src,
          "obj_progress_m": p, "obj_cost_of_transport": c,
          "progress_m": p, "cost_of_transport": c}
         for i, (g, p, c, src) in enumerate(rows)]
    )
    df.to_csv(run_dir / "results" / "outer_population.csv", index=False)
    return run_dir


# Two exam phases with hand-computable hypervolumes w.r.t. ref (80 m, 0.5):
#   phase 0 front {(100, .3), (150, .4)}          -> 70*.1 + 20*.1 = 9
#   phase 1 front {(120,.2), (130,.3), (200,.45)} -> 120*.05 + 50*.15 + 40*.1 = 17.5
#   cumulative @1 {(120,.2), (130,.3), (150,.4), (200,.45)}
#                                                 -> 120*.05 + 70*.05 + 50*.1 + 40*.1 = 18.5
_TWO_PHASES = [
    (0, 100.0, 0.30, "exam"),
    (0, 150.0, 0.40, "exam"),
    (0,  60.0, 0.05, "exam"),          # below the 80 m gate: ignored
    (1, 120.0, 0.20, "exam"),
    (1, 200.0, 0.45, "exam"),
    (1, 130.0, 0.30, "exam"),          # on the front (further than (120, .2))
    (2, 250.0, 0.01, "phase_mean"),    # not exam-scored: dropped entirely
]


def _two_phase_run(tmp_path, name="run"):
    return _write_run(tmp_path / name, _TWO_PHASES)


def _series(phases, per_phase, cumulative):
    return pd.DataFrame({"phase": phases, "hv_per_phase": per_phase,
                         "hv_cumulative": cumulative})


# ----------------------------------------------------------------------------
#  Per-run series
# ----------------------------------------------------------------------------

def test_hv_series_matches_hand_computed_fixed_reference(tmp_path):
    s = hv_series(_two_phase_run(tmp_path))
    assert list(s.columns) == ["phase", "hv_per_phase", "hv_cumulative"]
    np.testing.assert_array_equal(s["phase"], [0, 1])
    np.testing.assert_allclose(s["hv_per_phase"], [9.0, 17.5])
    np.testing.assert_allclose(s["hv_cumulative"], [9.0, 18.5])


def test_hv_series_drops_non_exam_rows_and_phases(tmp_path):
    s = hv_series(_two_phase_run(tmp_path))
    assert 2 not in s["phase"].tolist()


def test_hv_series_honours_explicit_gate_override(tmp_path):
    s = hv_series(_two_phase_run(tmp_path), min_progress=140.0)
    # phase 0: only (150,.4) survives -> 70*.1 = 7
    assert s["hv_per_phase"].iloc[0] == pytest.approx(7.0)


def test_hv_series_cumulative_is_monotone_non_decreasing(tmp_path):
    rows = [(g, 100.0 + 10 * g, 0.3 - 0.01 * g, "exam") for g in range(6)]
    rows += [(3, 90.0, 0.45, "exam")]  # a weak phase does not lower the pool
    s = hv_series(_write_run(tmp_path / "run", rows))
    assert (np.diff(s["hv_cumulative"]) >= -1e-12).all()


def test_hv_series_agrees_with_pareto_plots(tmp_path):
    """The aggregate must reproduce exactly what each run's own
    ``pareto_hypervolume.png`` shows: same gate, same fixed reference."""
    from WP2_Outer_Loop.pareto_plots import plot_pareto_front

    run = _two_phase_run(tmp_path)
    info = plot_pareto_front(run)
    s = hv_series(run)
    assert info["ref_fixed"]
    assert s["hv_cumulative"].iloc[-1] == pytest.approx(info["hypervolume"])


# ----------------------------------------------------------------------------
#  Group aggregate
# ----------------------------------------------------------------------------

def test_aggregate_hv_mean_std_and_count_over_ragged_runs():
    a = _series([0, 1, 2], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    b = _series([0, 1], [3.0, 4.0], [3.0, 4.0])
    agg = aggregate_hv([a, b], "hv_per_phase")
    np.testing.assert_array_equal(agg["phase"], [0, 1, 2])
    np.testing.assert_allclose(agg["mean"], [2.0, 3.0, 3.0])
    # sample std (ddof=1): |1-3|/sqrt(2) = sqrt(2); a lone run has no std
    np.testing.assert_allclose(agg["std"][:2], [np.sqrt(2.0)] * 2)
    assert np.isnan(agg["std"].iloc[2])
    np.testing.assert_array_equal(agg["n"], [2, 2, 1])


def test_aggregate_hv_of_identical_runs_has_zero_std():
    a = _series([0, 1], [5.0, 6.0], [5.0, 7.0])
    agg = aggregate_hv([a, a.copy(), a.copy()], "hv_cumulative")
    np.testing.assert_allclose(agg["mean"], [5.0, 7.0])
    np.testing.assert_allclose(agg["std"], [0.0, 0.0])
    np.testing.assert_array_equal(agg["n"], [3, 3])


def test_aggregate_hv_rejects_unknown_measure():
    with pytest.raises(ValueError):
        aggregate_hv([_series([0], [1.0], [1.0])], "hv_bogus")


def test_aggregate_table_has_one_block_per_group_and_measure():
    g1 = HVGroup("co-design", "red",
                 [_series([0, 1], [1.0, 2.0], [1.0, 2.0])], ["a"])
    g2 = HVGroup("morph", "blue",
                 [_series([0, 1, 2], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]),
                  _series([0, 1, 2], [3.0, 4.0, 7.0], [3.0, 4.0, 7.0])],
                 ["b", "c"])
    table = aggregate_table([g1, g2])
    assert list(table["phase"]) == [0, 1, 2]
    for label in ("co-design", "morph"):
        for m in MEASURES:
            for stat in ("mean", "std", "n"):
                assert f"{label}_{m}_{stat}" in table.columns
    assert np.isnan(table["co-design_hv_per_phase_mean"].iloc[2])
    assert table["morph_hv_per_phase_n"].iloc[2] == 2
    assert table["morph_hv_per_phase_mean"].iloc[2] == pytest.approx(6.0)


# ----------------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------------

def _groups():
    cod = HVGroup("co-design", "#d62728",
                  [_series(range(3), [1, 2, 3], [1, 2, 3]),
                   _series(range(3), [2, 3, 4], [2, 3, 4])], ["c0", "c1"])
    mor = HVGroup("morphology-only", "#1f77b4",
                  [_series(range(5), [1, 1, 2, 2, 2], [1, 1, 2, 2, 2]),
                   _series(range(5), [0, 1, 1, 2, 3], [0, 1, 1, 2, 3])],
                  ["m0", "m1"])
    return [cod, mor]


def test_plot_writes_png_pdf_and_csv(tmp_path):
    out = plot_hv_aggregate(_groups(), tmp_path / "hv.png")
    assert out.is_file()
    assert out.with_suffix(".pdf").is_file()
    assert (tmp_path / "hv_aggregate.csv").is_file()
    table = pd.read_csv(tmp_path / "hv_aggregate.csv")
    assert "co-design_hv_cumulative_mean" in table.columns
    assert len(table) == 5


def test_draw_legend_names_groups_with_run_counts():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, _groups(), "hv_per_phase")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert labels == ["co-design (2 runs)", "morphology-only (2 runs)"]


def test_draw_band_is_mean_plus_minus_one_std():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, _groups()[:1], "hv_per_phase")
    (line,) = [l for l in ax.get_lines() if l.get_label() == "co-design (2 runs)"]
    np.testing.assert_allclose(line.get_ydata(), [1.5, 2.5, 3.5])
    (band,) = ax.collections
    verts = band.get_paths()[0].vertices
    std = np.sqrt(0.5)
    assert verts[:, 1].min() == pytest.approx(1.5 - std)
    assert verts[:, 1].max() == pytest.approx(3.5 + std)
    plt.close(fig)


def test_draw_show_runs_adds_one_faint_line_per_run():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, _groups(), "hv_cumulative", show_runs=True)
    faint = [l for l in ax.get_lines() if l.get_alpha() is not None
             and l.get_alpha() < 0.5]
    plt.close(fig)
    assert len(faint) == 4


def test_draw_without_show_runs_has_only_mean_lines():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, _groups(), "hv_cumulative")
    plt.close(fig)
    assert len(ax.get_lines()) == 2


def test_plot_single_measure_variant(tmp_path):
    out = plot_hv_aggregate(_groups(), tmp_path / "cum.png",
                            measures=("hv_cumulative",))
    assert out.is_file()


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def test_main_builds_groups_from_run_dirs(tmp_path, capsys):
    c0 = _two_phase_run(tmp_path, "cod_r0")
    c1 = _two_phase_run(tmp_path, "cod_r1")
    m0 = _two_phase_run(tmp_path, "morph_r0")
    out = tmp_path / "plots" / "exam_hv.png"
    assert main(["--group", "co-design", "red", str(c0), str(c1),
                 "--group", "morphology-only", "blue", str(m0),
                 "--out", str(out)]) == 0
    assert out.is_file() and out.with_suffix(".pdf").is_file()
    table = pd.read_csv(out.with_name("exam_hv_aggregate.csv"))
    assert table["co-design_hv_cumulative_n"].tolist() == [2, 2]
    assert table["co-design_hv_cumulative_mean"].iloc[-1] == pytest.approx(18.5)
    assert table["co-design_hv_cumulative_std"].iloc[-1] == pytest.approx(0.0)
    assert "co-design" in capsys.readouterr().out


def test_main_resolves_synced_wrapper_folders(tmp_path):
    wrapper = tmp_path / "outer_exam_r0"
    _write_run(wrapper / "2026-08-10_00-00-00_exam", _TWO_PHASES)
    out = tmp_path / "hv.png"
    assert main(["--group", "co-design", "red", str(wrapper),
                 "--out", str(out)]) == 0
    assert out.is_file()


def test_main_requires_a_group(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "hv.png")])


def test_main_min_progress_override_reaches_the_series(tmp_path):
    c0 = _two_phase_run(tmp_path, "cod_r0")
    out = tmp_path / "hv.png"
    assert main(["--group", "co-design", "red", str(c0),
                 "--out", str(out), "--min-progress", "140"]) == 0
    table = pd.read_csv(tmp_path / "hv_aggregate.csv")
    assert table["co-design_hv_per_phase_mean"].iloc[0] == pytest.approx(7.0)


# ----------------------------------------------------------------------------
#  Ragged groups: solid where every run contributes, dashed where fewer
# ----------------------------------------------------------------------------

def _ragged_group():
    return HVGroup("co-design", "#d62728",
                   [_series(range(5), [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    _series(range(5), [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]),
                    _series(range(3), [2, 3, 4], [2, 3, 4])], ["a", "b", "c"])


def _mean_lines(ax, label):
    solid = [l for l in ax.get_lines() if l.get_label() == label]
    dashed = [l for l in ax.get_lines() if l.get_linestyle() not in ("-", "solid")
              and l.get_alpha() is None]
    return solid, dashed


def test_mean_is_solid_only_where_every_run_contributes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, [_ragged_group()], "hv_per_phase")
    (solid,), (dashed,) = _mean_lines(ax, "co-design (3 runs)")
    plt.close(fig)
    # phases 0–2: all 3 runs -> solid (2, 3, 4); 3–4: two runs -> NaN
    np.testing.assert_allclose(solid.get_ydata(), [2, 3, 4, np.nan, np.nan],
                               equal_nan=True)
    # the dashed continuation starts on the last solid point: (4, 5, 6)
    np.testing.assert_allclose(dashed.get_ydata(), [np.nan, np.nan, 4, 5, 6],
                               equal_nan=True)


def test_no_dashed_segment_when_all_runs_have_equal_length():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, _groups(), "hv_per_phase")
    _solid, dashed = _mean_lines(ax, "co-design (2 runs)")
    plt.close(fig)
    assert dashed == []


def test_solid_min_runs_lowers_the_threshold():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_hv_aggregate(ax, [_ragged_group()], "hv_per_phase", min_runs=2)
    (solid,), dashed = _mean_lines(ax, "co-design (3 runs)")
    plt.close(fig)
    np.testing.assert_allclose(solid.get_ydata(), [2, 3, 4, 5, 6])
    assert dashed == []


def test_cli_solid_min_runs_flag(tmp_path):
    c0 = _write_run(tmp_path / "cod_r0", _TWO_PHASES)
    c1 = _write_run(tmp_path / "cod_r1", _TWO_PHASES[:3])  # one phase only
    out = tmp_path / "hv.png"
    assert main(["--group", "co-design", "red", str(c0), str(c1),
                 "--out", str(out), "--solid-min-runs", "1"]) == 0
    assert out.is_file()
