"""
Tests for ``WP2_Outer_Loop.transfer_eval`` — the front-morphology re-fly tool.

Everything except the rollout itself is exercised here: front selection,
genome resolution, forest-setting validation, the result-CSV round trip, and
the overlay guard that stops two incomparable measurements being plotted
together. The rollout needs a GPU, so ``--dry-run`` covers the plumbing up to
the point of measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from WP2_Outer_Loop.transfer_eval import (  # noqa: E402
    _parse_forest_args,
    apply_forest_settings,
    comparability_conflicts,
    forest_field_names,
    paired_summary,
    read_results,
    resolve_genome,
    select_front,
    write_results,
    zero_rules_genome,
)


# ------------------------------------------------------------------ fixtures

def _front_csv(tmp_path, rows, n_genes=3) -> Path:
    """Minimal pareto_front.csv."""
    gene_cols = [f"g{i}" for i in range(n_genes)]
    recs = []
    for i, r in enumerate(rows):
        rec = {
            "outer_gen": r["outer_gen"],
            "urdf_idx": r.get("urdf_idx", i),
            "urdf_file": f"ind_{r.get('urdf_idx', i):03d}.urdf",
            "front_size": r.get("front_size", len(rows)),
            "front_rank": r.get("front_rank", i),
            "obj_progress_m": r.get("progress_m", 100.0),
            "obj_cost_of_transport": r.get("cost_of_transport", 0.2),
            "progress_m": r.get("progress_m", 100.0),
            "cost_of_transport": r.get("cost_of_transport", 0.2),
        }
        for j, c in enumerate(gene_cols):
            rec[c] = r.get(c, 0.1 * (i + j))
        recs.append(rec)
    path = tmp_path / "pareto_front.csv"
    pd.DataFrame(recs).to_csv(path, index=False)
    return path


class _Hebbian:
    num_actions = 7
    hidden_dim = 32
    evolve_decay = False

    @staticmethod
    def abcd_block_size():
        return 7 * 32


class _Cfg:
    """Just enough of HebbianEvolutionConfig for genome resolution."""

    def __init__(self, evolve_decay=False):
        self.hebbian = _Hebbian()
        self.hebbian.evolve_decay = evolve_decay

    def hebbian_genome_dim(self):
        base = 4 * self.hebbian.abcd_block_size()
        return base + (self.hebbian.num_actions * self.hebbian.hidden_dim
                       if self.hebbian.evolve_decay else 0)


# ------------------------------------------------------------- front select

def test_select_front_defaults_to_last_generation(tmp_path):
    csv = _front_csv(tmp_path, [
        dict(outer_gen=0, urdf_idx=0, g0=0.1),
        dict(outer_gen=1, urdf_idx=0, g0=0.2),
        dict(outer_gen=1, urdf_idx=3, g0=0.3),
    ])
    sel = select_front(csv, "last")
    assert set(sel["outer_gen"]) == {1}
    assert sorted(sel["urdf_idx"]) == [0, 3]


def test_select_front_explicit_generation(tmp_path):
    csv = _front_csv(tmp_path, [
        dict(outer_gen=0, urdf_idx=7, g0=0.1),
        dict(outer_gen=1, urdf_idx=0, g0=0.2),
    ])
    assert select_front(csv, 0)["urdf_idx"].tolist() == [7]
    with pytest.raises(ValueError, match="outer_gen 5 not in"):
        select_front(csv, 5)


def test_select_front_dedups_repeated_genomes(tmp_path):
    """Elites persist across generations, so identical genomes recur — flying
    the same morphology twice costs a scene and adds nothing."""
    csv = _front_csv(tmp_path, [
        dict(outer_gen=2, urdf_idx=0, g0=0.5, g1=0.5, g2=0.5),
        dict(outer_gen=2, urdf_idx=1, g0=0.5, g1=0.5, g2=0.5),  # duplicate
        dict(outer_gen=2, urdf_idx=2, g0=0.9, g1=0.1, g2=0.3),
    ])
    assert select_front(csv, "last")["urdf_idx"].tolist() == [0, 2]
    assert select_front(csv, "last", dedup=False)["urdf_idx"].tolist() == [0, 1, 2]


def test_select_front_rejects_csv_without_genomes(tmp_path):
    path = tmp_path / "no_genes.csv"
    pd.DataFrame([{"outer_gen": 0, "urdf_idx": 0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no g\\* genome columns"):
        select_front(path)


# ----------------------------------------------------------------- genomes

def test_zero_rules_genome_is_midrange():
    """ABCD = 0.5 normalized ⇒ 0 on symmetric bounds ⇒ ΔW = 0."""
    cfg = _Cfg()
    g = zero_rules_genome(cfg)
    assert g.shape == (cfg.hebbian_genome_dim(),)
    assert np.all(g == 0.5)


def test_zero_rules_genome_zeroes_evolved_decay():
    cfg = _Cfg(evolve_decay=True)
    g = zero_rules_genome(cfg)
    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    start = 4 * cfg.hebbian.abcd_block_size()
    assert np.all(g[:start] == 0.5)
    assert np.all(g[start:start + n_weights] == 0.0)


def test_resolve_genome_prefers_saved_best_individual(tmp_path):
    saved = tmp_path / "best_individual" / "fitness"
    saved.mkdir(parents=True)
    np.save(saved / "genome.npy", np.arange(896, dtype=np.float64))
    g, src = resolve_genome("best", _Cfg(), tmp_path)
    assert g[0] == 0.0 and g[-1] == 895.0
    assert "best_individual" in src


def test_resolve_genome_recovers_from_generations_when_run_timed_out(tmp_path):
    """No best_individual/ (the TIMEOUT case) — reconstruct from the CSV +
    per-generation solutions, the way WP2.recover_best does."""
    (tmp_path / "results").mkdir()
    pd.DataFrame([
        {"generation": 0, "individual_idx": 0, "fitness": 10.0},
        {"generation": 1, "individual_idx": 2, "fitness": 99.0},  # the best
        {"generation": 1, "individual_idx": 0, "fitness": 20.0},
    ]).to_csv(tmp_path / "results" / "cma_population.csv", index=False)
    gen_dir = tmp_path / "generations" / "gen_001"
    gen_dir.mkdir(parents=True)
    sols = np.zeros((4, 896))
    sols[2] = 0.75
    np.save(gen_dir / "solutions.npy", sols)

    g, src = resolve_genome("best", _Cfg(), tmp_path)
    assert np.all(g == 0.75)
    assert "gen_001" in src and "fitness 99" in src


def test_resolve_genome_explicit_path(tmp_path):
    p = tmp_path / "mine.npy"
    np.save(p, np.full(896, 0.25))
    g, src = resolve_genome(str(p), _Cfg(), None)
    assert np.all(g == 0.25) and src == str(p)


def test_resolve_genome_zero_needs_no_run_dir():
    g, src = resolve_genome("zero", _Cfg(), None)
    assert np.all(g == 0.5) and "generalist" in src


def test_resolve_genome_missing_sources_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_genome("best", _Cfg(), tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_genome(str(tmp_path / "nope.npy"), _Cfg(), None)


# ------------------------------------------------------------ forest fields

def test_apply_forest_settings_rejects_unknown_field():
    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig()
    with pytest.raises(KeyError, match="Unknown forest field"):
        apply_forest_settings(cfg, {"tree_colour": 3.0})


def test_apply_forest_settings_allows_perception_coupled_fields():
    """tree_radius and the y corridor are refused by the *runtime* generator
    override path (DepthSolver desync), but here they go in at env build, so
    the solver is constructed from the same values."""
    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig()
    resolved = apply_forest_settings(
        cfg, {"tree_radius": 0.25, "y_lower": -12.0, "y_upper": 12.0}
    )
    assert cfg.forest.tree_radius == 0.25
    assert resolved["y_lower"] == -12.0
    assert {"tree_radius", "y_lower", "y_upper"} <= set(forest_field_names())


def test_apply_forest_settings_clears_shadowing_legacy_overrides():
    """cfg.evaluation.x_upper sits between the WP1 config and cfg.forest; if a
    run set it, leaving it would beat an explicitly requested forest value."""
    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig()
    cfg.evaluation.x_upper = 100.0
    cfg.evaluation.dens_max = 5.0
    apply_forest_settings(cfg, {"x_upper": 400.0})
    assert cfg.forest.x_upper == 400.0
    assert cfg.evaluation.x_upper is None
    assert cfg.evaluation.dens_max == 5.0  # untouched: not requested


class _Args:
    def __init__(self, **kw):
        for name in ("x_lower", "x_upper", "y_lower", "y_upper", "dens_min",
                     "dens_max", "num_trees", "tree_radius", "tree_height",
                     "forest_length", "forest_mode"):
            setattr(self, name, None)
        self.forest = None
        for k, v in kw.items():
            setattr(self, k, v)


def test_parse_forest_args_maps_flags_and_kv():
    settings = _parse_forest_args(_Args(
        x_upper=400.0, dens_max=8.0, forest_mode="growing",
        forest=["num_trees=250", "y_spacing_min=2.5"],
    ))
    assert settings == {
        "x_upper": 400.0, "dens_max": 8.0, "mode": "growing",
        "num_trees": 250, "y_spacing_min": 2.5,
    }
    assert isinstance(settings["num_trees"], int)


def test_parse_forest_args_rejects_bad_kv():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_forest_args(_Args(forest=["x_upper"]))
    with pytest.raises(KeyError, match="Unknown forest field"):
        _parse_forest_args(_Args(forest=["nonsense=1"]))


# ------------------------------------------------------- results round trip

def _result_df():
    return pd.DataFrame([
        {"urdf_idx": 0, "progress_m": 120.0, "progress_m_se": 2.0,
         "cost_of_transport": 0.30, "cost_of_transport_se": 0.01,
         "crash_rate": 0.4},
        {"urdf_idx": 1, "progress_m": 95.0, "progress_m_se": 3.0,
         "cost_of_transport": 0.11, "cost_of_transport_se": 0.02,
         "crash_rate": 0.9},
    ])


def test_results_round_trip_preserves_provenance(tmp_path):
    prov = {"tag": "hebbian", "forest": {"x_upper": 400.0},
            "forest_seed": 0, "n_forests": 32}
    out = write_results(_result_df(), prov, tmp_path / "transfer_hebbian.csv")
    df, back = read_results(out)
    assert back == prov
    assert df["urdf_idx"].tolist() == [0, 1]
    assert df["progress_m"].tolist() == [120.0, 95.0]


def test_read_results_tolerates_missing_provenance(tmp_path):
    p = tmp_path / "plain.csv"
    _result_df().to_csv(p, index=False)
    df, prov = read_results(p)
    assert prov == {} and len(df) == 2


# ------------------------------------------------------------ overlay guard

def _prov(**kw):
    base = {"forest": {"x_upper": 400.0}, "forest_seed": 0, "n_forests": 32,
            "vmin": 10.0, "vmax": 20.0, "stochastic": True,
            "num_eval_episodes": 1}
    base.update(kw)
    return base


def test_comparability_accepts_matching_measurements():
    assert comparability_conflicts([_prov(tag="a"), _prov(tag="b")]) == []


@pytest.mark.parametrize("field,value", [
    ("forest", {"x_upper": 100.0}),
    ("forest_seed", 7),
    ("n_forests", 64),
    ("vmax", 25.0),
    ("stochastic", False),
])
def test_comparability_flags_each_measurement_difference(field, value):
    """A controller comparison is only a controller comparison if everything
    else matched — each of these differences must be caught."""
    conflicts = comparability_conflicts([_prov(), _prov(**{field: value})])
    assert len(conflicts) == 1 and conflicts[0].startswith(field + ":")


def test_comparability_ignores_the_controller_itself():
    """The checkpoint and genome are exactly what is *supposed* to differ."""
    a = _prov(checkpoint_md5="aaa", genome_source="zero rules (generalist)")
    b = _prov(checkpoint_md5="bbb", genome_source="best_individual/...")
    assert comparability_conflicts([a, b]) == []


# --------------------------------------------------------------- statistics

def test_paired_summary_reports_delta_and_flags_noise():
    a = _result_df()
    b = a.copy()
    b["progress_m"] = a["progress_m"] + 10.0     # consistent shift
    b["cost_of_transport"] = a["cost_of_transport"]  # no change at all
    out = paired_summary("generalist", a, "hebbian", b)
    assert "2 shared morphologies" in out
    assert "+10" in out
    # An exactly-zero delta has zero spread; the guard must not claim it is
    # resolved on the strength of a degenerate SE.
    cot_line = [ln for ln in out.splitlines() if "cost_of_transport" in ln][0]
    assert "not resolved" in cot_line


def test_paired_summary_handles_disjoint_morphologies():
    a = _result_df()
    b = _result_df()
    b["urdf_idx"] = [90, 91]
    assert "No shared urdf_idx" in paired_summary("a", a, "b", b)


# ------------------------------------------------------- standard-drone star

def test_standard_drone_row_uses_the_canonical_genome():
    """Built from STANDARD_MYDRONE_GENOME through from_physical — the same
    round trip that seeds outer-loop slot 0 — not a stale on-disk URDF."""
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
    from WP2_Outer_Loop.transfer_eval import standard_drone_row

    n = Chromosome_Drone.num_genes()
    row = standard_drone_row(n)
    assert len(row) == 1
    assert int(row["is_standard"].iloc[0]) == 1
    assert int(row["urdf_idx"].iloc[0]) == -1     # not a front member
    assert int(row["front_rank"].iloc[0]) == -1
    expected = Chromosome_Drone.from_physical(list(STANDARD_MYDRONE_GENOME))
    got = [float(row[f"g{i}"].iloc[0]) for i in range(n)]
    assert got == pytest.approx(list(expected))


def test_standard_drone_row_rejects_gene_count_mismatch():
    from WP2_Outer_Loop.transfer_eval import standard_drone_row
    with pytest.raises(ValueError, match="refusing to fly a mismatched"):
        standard_drone_row(3)


def test_with_standard_drone_appends_and_marks(tmp_path):
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from WP2_Outer_Loop.transfer_eval import with_standard_drone

    n = Chromosome_Drone.num_genes()
    front = select_front(_front_csv(tmp_path, [
        dict(outer_gen=5, urdf_idx=0, g0=0.1),
        dict(outer_gen=5, urdf_idx=1, g0=0.2),
    ], n_genes=n))
    out = with_standard_drone(front, n)
    assert len(out) == 3
    assert out["is_standard"].tolist() == [0, 0, 1]


def test_with_standard_drone_alone_when_no_front():
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from WP2_Outer_Loop.transfer_eval import with_standard_drone
    out = with_standard_drone(None, Chromosome_Drone.num_genes())
    assert len(out) == 1 and int(out["is_standard"].iloc[0]) == 1


def test_paired_summary_excludes_the_standard_drone():
    """The reference is not part of the evolved population, so it must not
    drag the paired mean over front morphologies."""
    a = _result_df()
    a["is_standard"] = [0, 0]
    b = a.copy()
    b["progress_m"] = a["progress_m"] + 10.0

    # An extreme reference row that would visibly skew the mean if counted.
    ref = pd.DataFrame([{"urdf_idx": -1, "progress_m": 0.0, "progress_m_se": 0.0,
                         "cost_of_transport": 9.0, "cost_of_transport_se": 0.0,
                         "crash_rate": 1.0, "is_standard": 1}])
    a2 = pd.concat([a, ref], ignore_index=True)
    b2 = pd.concat([b, ref], ignore_index=True)

    assert paired_summary("g", a, "h", b) == paired_summary("g", a2, "h", b2)
    assert "2 shared morphologies" in paired_summary("g", a2, "h", b2)


def test_plot_separates_star_from_scatter(tmp_path):
    """The star must not join the scatter or the nondominated line."""
    from WP2_Outer_Loop.transfer_eval import plot_transfer
    df = _result_df()
    df["is_standard"] = [0, 0]
    ref = pd.DataFrame([{"urdf_idx": -1, "progress_m": 60.0,
                         "progress_m_se": 1.0, "cost_of_transport": 0.5,
                         "cost_of_transport_se": 0.01, "crash_rate": 1.0,
                         "is_standard": 1}])
    out = plot_transfer([("hebbian", pd.concat([df, ref], ignore_index=True))],
                        tmp_path / "p.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_still_works_without_is_standard_column(tmp_path):
    """Result CSVs written before the flag existed have no such column."""
    from WP2_Outer_Loop.transfer_eval import plot_transfer
    out = plot_transfer([("generalist", _result_df())], tmp_path / "q.png")
    assert out.is_file() and out.stat().st_size > 0


# ------------------------------------------------------------- plot cosmetics

def test_progress_axis_uses_5m_ticks(tmp_path, monkeypatch):
    """The fronts span only ~35 m, so matplotlib's default 10 m ticks are too
    coarse to read a morph's position off. Asserts on the real axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.transfer_eval import plot_transfer

    captured = {}
    real_subplots = plt.subplots

    def spy(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    plot_transfer([("hebbian", _result_df())], tmp_path / "t.png")

    # The locator pads a tick either side of the requested range, so assert on
    # the spacing and coverage rather than exact endpoints.
    ticks = np.asarray(captured["ax"].xaxis.get_major_locator()
                       .tick_values(90.0, 110.0))
    assert np.allclose(np.diff(ticks), 5.0)
    assert ticks.min() <= 90.0 and ticks.max() >= 110.0


def test_front_generation_ignores_the_standard_drone():
    from WP2_Outer_Loop.transfer_eval import _front_generation
    df = pd.DataFrame([
        {"outer_gen": 48, "is_standard": 0},
        {"outer_gen": 48, "is_standard": 0},
        {"outer_gen": -1, "is_standard": 1},   # reference sentinel
    ])
    assert _front_generation(df) == 48


def test_front_generation_none_when_only_the_reference():
    from WP2_Outer_Loop.transfer_eval import _front_generation
    assert _front_generation(
        pd.DataFrame([{"outer_gen": -1, "is_standard": 1}])) is None
    assert _front_generation(pd.DataFrame([{"urdf_idx": 0}])) is None


# ------------------------------------------------------------ random morphs

def test_random_morphs_are_reproducible_from_the_seed():
    from WP2_Outer_Loop.transfer_eval import random_morph_rows
    a = random_morph_rows(8, seed=0)
    b = random_morph_rows(8, seed=0)
    c = random_morph_rows(8, seed=7)
    genes = [c_ for c_ in a.columns if c_.startswith("g") and c_[1:].isdigit()]
    assert a[genes].equals(b[genes])
    assert not a[genes].equals(c[genes])


def test_random_morphs_span_the_unit_box_and_are_not_front_members():
    """Uniform in [0,1]^15 — the outer loop's own gen-0 distribution."""
    from WP2_Outer_Loop.transfer_eval import random_morph_rows
    df = random_morph_rows(64, seed=1)
    genes = [c for c in df.columns if c.startswith("g") and c[1:].isdigit()]
    assert len(df) == 64 and len(genes) == 15
    vals = df[genes].to_numpy()
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert vals.min() < 0.05 and vals.max() > 0.95      # actually spans the box
    assert (df["outer_gen"] == -1).all()                # not from any front
    assert (df["is_standard"] == 0).all()


def test_random_morphs_exclude_the_standard_drone_seed():
    """seed_standard_drone=False: slot 0 must be random, not the mydrone."""
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
    from WP2_Outer_Loop.transfer_eval import random_morph_rows

    std = np.asarray(Chromosome_Drone.from_physical(list(STANDARD_MYDRONE_GENOME)))
    df = random_morph_rows(8, seed=0)
    genes = [f"g{i}" for i in range(len(std))]
    assert not np.allclose(df[genes].iloc[0].to_numpy(), std)


def test_random_morphs_plus_standard_drone():
    from WP2_Outer_Loop.transfer_eval import random_morph_rows, with_standard_drone
    df = with_standard_drone(random_morph_rows(16, seed=0), 15)
    assert len(df) == 17
    assert df["is_standard"].tolist() == [0] * 16 + [1]


def test_morph_seed_mismatch_blocks_the_overlay():
    """Two controllers must have flown the SAME random bodies."""
    a = _prov(random_morphs=128, morph_seed=0)
    b = _prov(random_morphs=128, morph_seed=1)
    conflicts = comparability_conflicts([a, b])
    assert len(conflicts) == 1 and conflicts[0].startswith("morph_seed:")

    c = _prov(random_morphs=64, morph_seed=0)
    assert any(x.startswith("random_morphs:")
               for x in comparability_conflicts([a, c]))


def test_default_title_distinguishes_the_three_sources():
    from WP2_Outer_Loop.transfer_eval import _default_title
    front = pd.DataFrame([{"outer_gen": 52, "is_standard": 0}])
    assert _default_title("hebbian", front) == "hebbian on gen-52 front morphologies"

    rand = pd.DataFrame([{"outer_gen": -1, "is_standard": 0}] * 4
                        + [{"outer_gen": -1, "is_standard": 1}])
    assert _default_title("hebbian", rand) == "hebbian on 4 random morphologies"

    only_std = pd.DataFrame([{"outer_gen": -1, "is_standard": 1}])
    assert _default_title("hebbian", only_std) == "hebbian on the standard mydrone"
