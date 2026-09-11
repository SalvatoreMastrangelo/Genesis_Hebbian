"""Tests for ``WP2_Outer_Loop.render_champions`` and its end-of-run hook.

Pure python: synthetic ``outer_population.csv`` runs, the Genesis renderer
itself is monkeypatched (it needs the simulator runtime). Covers champion
selection (gate applied to the CoT pick, progress pick ungated), the
multi-run batch, and ``pareto_plots.plot_outer_run`` calling the renderer by
default without ever failing on a render error.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from WP2_Outer_Loop import pareto_plots, render_champions as rc


# ----------------------------------------------------------------------------
#  Fixture
# ----------------------------------------------------------------------------

def _write_run(run_dir: Path, rows, *, min_progress=80.0, source="exam"):
    """Run folder with the standard objective pair + gate. ``rows`` are
    ``(outer_gen, urdf_idx, progress, cot)``; slot names repeat across
    generations like the real catalog (``ind_<idx>.urdf``)."""
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
    recs = []
    for g, i, p, c in rows:
        rec = {"outer_gen": g, "urdf_idx": i, "urdf_file": f"ind_{i:03d}.urdf",
               "obj_source": source, "obj_progress_m": p, "obj_cost_of_transport": c,
               "progress_m": p, "cost_of_transport": c}
        rec.update({f"g{k}": (g * 7 + i * 3 + k) % 11 / 10.0 for k in range(15)})
        recs.append(rec)
    pd.DataFrame(recs).to_csv(run_dir / "results" / "outer_population.csv", index=False)
    return run_dir


ROWS = [
    # gen, slot, progress, CoT
    (0, 0, 150.0, 0.30),
    (0, 1, 120.0, 0.20),
    (1, 0, 220.0, 0.40),   # progress record
    (1, 1, 5.0, 0.05),     # cheapest of all, but does not fly -> gated out
    (2, 0, 100.0, 0.15),   # cheapest admitted row
    (2, 1, 210.0, 0.35),
]


# ----------------------------------------------------------------------------
#  Champion selection
# ----------------------------------------------------------------------------

def test_cot_champion_is_cheapest_row_passing_the_gate(tmp_path):
    run = _write_run(tmp_path / "run", ROWS)
    targets, gate = rc.select_champions(run)
    assert gate == 80.0
    labels = [t[0] for t in targets]
    assert labels == ["progress_champion", "cot_champion_gated80m"]
    prog, cot = targets[0][1], targets[1][1]
    assert (int(prog["outer_gen"]), int(prog["urdf_idx"])) == (1, 0)
    assert (int(cot["outer_gen"]), int(cot["urdf_idx"])) == (2, 0)
    assert cot["obj_cost_of_transport"] == 0.15   # not the 0.05 non-flyer


def test_gate_off_lets_the_non_flyer_hold_the_cot_record(tmp_path):
    run = _write_run(tmp_path / "run", ROWS, min_progress=0.0)
    targets, gate = rc.select_champions(run)
    assert gate == 0.0
    assert targets[1][0] == "cot_champion_gated0m"
    assert targets[1][1]["obj_cost_of_transport"] == 0.05


def test_min_progress_override_beats_the_saved_gate(tmp_path):
    run = _write_run(tmp_path / "run", ROWS, min_progress=0.0)   # run predates the gate
    targets, gate = rc.select_champions(run, min_progress=80.0)
    assert gate == 80.0
    assert targets[1][0] == "cot_champion_gated80m"
    assert targets[1][1]["obj_cost_of_transport"] == 0.15


def test_progress_champion_is_identified_by_gen_and_slot_not_by_name(tmp_path):
    # slot 0 holds 150 m at gen 0 and 220 m at gen 1 — same urdf_file name
    run = _write_run(tmp_path / "run", ROWS)
    targets, _ = rc.select_champions(run)
    prog = targets[0][1]
    assert prog["urdf_file"] == "ind_000.urdf"
    assert int(prog["outer_gen"]) == 1
    assert rc.champion_stem("progress_champion", prog) == "progress_champion_gen01_ind_000"


def test_non_exam_run_is_refused(tmp_path):
    run = _write_run(tmp_path / "run", ROWS, source="phase_mean")
    with pytest.raises(RuntimeError, match="not exam-scored"):
        rc.select_champions(run)


def test_missing_population_csv_raises(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError):
        rc.select_champions(tmp_path / "run")


# ----------------------------------------------------------------------------
#  Framing
# ----------------------------------------------------------------------------

def test_common_distance_is_set_by_the_largest_body():
    small = np.array([0.8, 0.9, 0.4])
    big = np.array([1.1, 1.8, 0.5])
    d34_big, dtop_big = rc._distances([big], rc.DEFAULT_RES)
    d34_both, dtop_both = rc._distances([small, big], rc.DEFAULT_RES)
    assert (d34_both, dtop_both) == (d34_big, dtop_big)
    # the top view must fit the 1.8 m span across the 4:3 width
    t = math.tan(math.radians(rc.FOV_DEG / 2))
    assert dtop_big == pytest.approx(rc.MARGIN * 1.8 / (2 * t * (1200 / 900)))


def test_prune_stale_removes_outputs_of_other_stems_only(tmp_path):
    out = tmp_path / "champion_renders"; out.mkdir()
    keep = "cot_champion_gated80m_gen48_ind_001"
    stale = "cot_champion_gated0m_gen17_ind_048"
    for stem in (keep, stale):
        for suffix in ("_3quarter.png", "_top.png", ".urdf"):
            (out / (stem + suffix)).write_text("x")
    (out / "champions.csv").write_text("x")
    removed = rc._prune_stale(out, [keep])
    assert sorted(p.name for p in removed) == sorted(
        stale + s for s in ("_3quarter.png", "_top.png", ".urdf"))
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [keep + s for s in ("_3quarter.png", "_top.png", ".urdf")] + ["champions.csv"])


# ----------------------------------------------------------------------------
#  Batch over several runs
# ----------------------------------------------------------------------------

def test_render_runs_resolves_wrappers_and_survives_failures(tmp_path, monkeypatch):
    good = _write_run(tmp_path / "outer_a_r0" / "2026-01-01_run", ROWS)
    bad = _write_run(tmp_path / "outer_b_r0" / "2026-01-02_run", ROWS, source="phase_mean")
    (tmp_path / "outer_c_r0").mkdir()   # no CSV at all
    seen = []

    def fake_render(rd, work=None, res=None, min_progress=None):   # select like the real thing, skip Genesis
        rc.select_champions(rd, min_progress)
        seen.append(Path(rd))

    monkeypatch.setattr(rc, "render_run", fake_render)
    status = rc.render_runs([tmp_path / "outer_a_r0", tmp_path / "outer_b_r0",
                             tmp_path / "outer_c_r0"], work=None)
    assert seen == [good]
    assert status[str(tmp_path / "outer_a_r0")] == "ok"
    assert status[str(tmp_path / "outer_b_r0")].startswith("failed:")
    assert status[str(tmp_path / "outer_c_r0")].startswith("failed:")
    assert bad.exists()


def test_main_exits_nonzero_when_a_run_fails(tmp_path, monkeypatch):
    _write_run(tmp_path / "run", ROWS)
    monkeypatch.setattr(rc, "render_run", lambda rd, work=None, res=None, min_progress=None: None)
    rc.main([str(tmp_path / "run")])                       # all ok -> returns
    with pytest.raises(SystemExit):
        rc.main([str(tmp_path / "run"), str(tmp_path / "missing")])


# ----------------------------------------------------------------------------
#  End-of-run hook in pareto_plots
# ----------------------------------------------------------------------------

def _stub_plots(monkeypatch):
    for name in ("build_pareto_front_csv", "plot_pareto_front",
                 "plot_champion_curves", "plot_outer_metrics"):
        monkeypatch.setattr(pareto_plots, name, lambda *a, **k: None)


def test_plot_outer_run_renders_champions_by_default(tmp_path, monkeypatch):
    _stub_plots(monkeypatch)
    calls = []
    monkeypatch.setattr(rc, "render_run",
                        lambda rd, work=None, res=None, min_progress=None: calls.append((Path(rd), min_progress)))
    pareto_plots.plot_outer_run(tmp_path)
    assert calls == [(tmp_path, None)]
    pareto_plots.plot_outer_run(tmp_path, min_progress=80.0)   # gate override reaches the renderer
    assert calls[-1] == (tmp_path, 80.0)


def test_plot_outer_run_can_skip_rendering(tmp_path, monkeypatch):
    _stub_plots(monkeypatch)
    calls = []
    monkeypatch.setattr(rc, "render_run", lambda rd, work=None, res=None, min_progress=None: calls.append(rd))
    pareto_plots.plot_outer_run(tmp_path, render=False)
    assert calls == []


def test_render_failure_never_breaks_the_plots(tmp_path, monkeypatch, capsys):
    _stub_plots(monkeypatch)

    def boom(rd, work=None, res=None, min_progress=None):
        raise ImportError("No module named 'gstaichi'")

    monkeypatch.setattr(rc, "render_run", boom)
    pareto_plots.plot_outer_run(tmp_path)        # must not raise
    assert pareto_plots.render_champions_safe(tmp_path) is False
    assert "champion renders skipped" in capsys.readouterr().out


def test_render_champions_safe_reports_success(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "render_run", lambda rd, work=None, res=None, min_progress=None: tmp_path)
    assert pareto_plots.render_champions_safe(tmp_path) is True
