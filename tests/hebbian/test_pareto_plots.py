"""
Tests for the per-run outer-loop Pareto plots
(``WP2_Outer_Loop.pareto_plots.plot_pareto_front``): the star-free copies
written by ``no_star_subdir`` / ``--no-bixler-subdir``.

Pure-python: synthetic run folders in ``tmp_path`` (population CSV, exam
baseline CSV, config with the standard validation drone), no Genesis.
"""

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2_Outer_Loop.pareto_fronts import BIXLER_LABEL  # noqa: E402
from WP2_Outer_Loop.pareto_plots import (  # noqa: E402
    main,
    plot_outer_run,
    plot_pareto_front,
)


# ----------------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------------

_ROWS = [
    (0, 100.0, 0.30, "exam"), (0, 150.0, 0.40, "exam"), (0, 60.0, 0.20, "exam"),
    (1, 120.0, 0.20, "exam"), (1, 130.0, 0.30, "exam"), (1, 200.0, 0.45, "exam"),
]


def _write_run(run_dir: Path, rows=_ROWS, *, baseline=(190.0, 0.28),
               min_progress=80.0):
    """Minimal exam-scored run: config (objectives, gate, standard
    validation drone), ``outer_population.csv`` and, unless ``baseline`` is
    None, an ``outer_exam_baseline.csv`` so the Bixler star is drawn."""
    (run_dir / "reproducibility").mkdir(parents=True)
    (run_dir / "results").mkdir()
    cfg = {
        "validation": {"enable": True, "validation_catalog": None},
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
    pd.DataFrame(
        [{"outer_gen": g, "urdf_idx": i, "obj_source": src,
          "obj_progress_m": p, "obj_cost_of_transport": c,
          "progress_m": p, "cost_of_transport": c}
         for i, (g, p, c, src) in enumerate(rows)]
    ).to_csv(run_dir / "results" / "outer_population.csv", index=False)
    if baseline is not None:
        pd.DataFrame([{"outer_gen": 0, "inner_gen": 3, "n_forests": 8,
                       "progress_m": baseline[0],
                       "cost_of_transport": baseline[1]}]
                     ).to_csv(run_dir / "results" / "outer_exam_baseline.csv",
                              index=False)
    return run_dir


def _figure_spy(monkeypatch):
    """Patch ``Figure.savefig`` to record, per saved path, the legend
    entries, the number of gold (star) scatters, the title and the axis
    limits of the first axes."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    seen = {}
    real_savefig = plt.Figure.savefig
    gold = to_rgba("gold")

    def spy(self, fname, *a, **k):
        ax = self.axes[0]
        leg = ax.get_legend()
        labels = ([t.get_text() for t in leg.get_texts()]
                  if leg is not None else [])
        stars = sum(1 for c in ax.collections
                    if isinstance(c, PathCollection) and len(c.get_facecolor())
                    and tuple(c.get_facecolor()[0]) == gold)
        seen[Path(fname)] = dict(labels=labels, stars=stars,
                                 title=ax.get_title(),
                                 xlim=ax.get_xlim(), ylim=ax.get_ylim())
        return real_savefig(self, fname, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    return seen


# ----------------------------------------------------------------------------
#  Star-free copies
# ----------------------------------------------------------------------------

def test_no_star_subdir_replots_both_front_figures_without_the_star(
        tmp_path, monkeypatch, capsys):
    seen = _figure_spy(monkeypatch)
    run = _write_run(tmp_path / "run")
    res = plot_pareto_front(run, no_star_subdir="no_bixler")
    plots, sub = run / "plots", run / "plots" / "no_bixler"
    assert res["no_star_figures_dir"] == sub
    for name in ("pareto_front_evolution.png",
                 "pareto_front_evolution_zoomed.png"):
        orig, copy = seen[plots / name], seen[sub / name]
        assert (sub / name).stat().st_size > 0
        # originals keep the star, copies have neither the marker nor the entry
        assert BIXLER_LABEL in orig["labels"] and orig["stars"] == 1
        assert copy["stars"] == 0
        assert not any("Bixler" in l for l in copy["labels"])
        # everything else in the legend, the title and the crop are the same
        assert copy["labels"] == [l for l in orig["labels"] if l != BIXLER_LABEL]
        assert copy["title"] == orig["title"]
        assert copy["xlim"] == pytest.approx(orig["xlim"])
        assert copy["ylim"] == pytest.approx(orig["ylim"])
    # the two crops really differ from each other
    assert seen[sub / "pareto_front_evolution.png"]["xlim"] != pytest.approx(
        seen[sub / "pareto_front_evolution_zoomed.png"]["xlim"])
    # the subfolder holds the two front figures only — the hypervolume plot
    # has no star and stays beside the originals
    assert sorted(p.name for p in sub.iterdir()) == [
        "pareto_front_evolution.png", "pareto_front_evolution_zoomed.png"]
    assert (plots / "pareto_hypervolume.png").is_file()
    assert str(sub) in capsys.readouterr().out


def test_no_star_subdir_skipped_when_no_star_is_drawn(tmp_path, capsys):
    run = _write_run(tmp_path / "run", baseline=None)
    res = plot_pareto_front(run, no_star_subdir="no_bixler")
    assert res["star"] is None and res["no_star_figures_dir"] is None
    assert not (run / "plots" / "no_bixler").exists()
    assert "star-free copies skipped" in capsys.readouterr().out


def test_default_writes_no_star_free_copies(tmp_path):
    run = _write_run(tmp_path / "run")
    res = plot_pareto_front(run)
    assert res["no_star_figures_dir"] is None
    assert sorted(p.name for p in (run / "plots").iterdir()) == [
        "pareto_front_evolution.png", "pareto_front_evolution_zoomed.png",
        "pareto_hypervolume.png"]


def test_plot_outer_run_passes_the_subdir_through(tmp_path):
    run = _write_run(tmp_path / "run")
    plot_outer_run(run, render=False, no_star_subdir="star_free")
    assert (run / "plots" / "star_free" / "pareto_front_evolution.png").is_file()
    assert (run / "plots" / "star_free" / "pareto_front_evolution_zoomed.png").is_file()


def test_cli_no_bixler_subdir_flag_defaults_to_no_bixler(tmp_path, capsys):
    run = _write_run(tmp_path / "run")
    assert main([str(run), "--no-render", "--no-bixler-subdir"]) == 0
    sub = run / "plots" / "no_bixler"
    assert (sub / "pareto_front_evolution.png").is_file()
    assert (sub / "pareto_front_evolution_zoomed.png").is_file()
    assert str(sub) in capsys.readouterr().out


def test_cli_no_bixler_subdir_takes_a_custom_name_and_min_progress(tmp_path):
    run = _write_run(tmp_path / "run", min_progress=0.0)
    assert main([str(run), "--no-render", "--min-progress", "80",
                 "--no-bixler-subdir", "star_free"]) == 0
    assert (run / "plots" / "star_free" / "pareto_front_evolution_zoomed.png").is_file()


def test_cli_without_the_flag_writes_no_copies(tmp_path):
    run = _write_run(tmp_path / "run")
    assert main([str(run), "--no-render"]) == 0
    assert not (run / "plots" / "no_bixler").exists()
