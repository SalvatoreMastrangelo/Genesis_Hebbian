"""
Tests for ``WP2_Outer_Loop.controller_generality`` — per-generation best
controllers re-flown on random and own-front morphologies.

Everything except the rollout is exercised here: generation discovery and
thinning, best-of-generation extraction, controller/body tables, the exam
forest mapping, pass grouping, metric reduction, resume bookkeeping, the
per-generation summary, the final-centroid companion, and the plots.
Runs inside the docker image (pandas/matplotlib):

    docker run --rm -v "$PWD":/workspace/bind -w /workspace/bind \
      -e PYTHONPATH=/workspace/bind/src mygenesis:latest \
      python -m pytest tests/hebbian/test_controller_generality.py -q \
      -o addopts="--import-mode=importlib"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from WP2_Outer_Loop.controller_generality import (  # noqa: E402
    best_of_generation,
    body_table,
    build_controllers,
    completed_bodies,
    controller_passes,
    exam_forest_settings,
    final_centroid,
    generation_indices,
    reduce_pass,
    select_generations,
    summarize,
)


# ------------------------------------------------------------------ fixtures

N_GENES = 6


def _fake_run(tmp_path: Path, gens: Sequence[int] = range(6), pop: int = 4,
              n_genes: int = N_GENES, refresh_every: int = 2) -> Path:
    """A run dir with ``generations/gen_XXX/{solutions,fitnesses}.npy`` where
    the best individual of gen g is index ``g % pop`` with fitness 100 + g,
    plus a two-phase ``outer_population.csv`` and a ``pareto_front.csv``."""
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    for g in gens:
        d = run / "generations" / f"gen_{g:03d}"
        d.mkdir(parents=True)
        sols = rng.uniform(0.2, 0.8, size=(pop, n_genes))
        fits = np.full(pop, 50.0)
        fits[g % pop] = 100.0 + g
        np.save(d / "solutions.npy", sols)
        np.save(d / "fitnesses.npy", fits)

    res = run / "results"
    res.mkdir()
    pop_rows = []
    for og, spread in ((0, 0.4), (1, 0.1)):
        for i in range(4):
            row = {"outer_gen": og, "urdf_idx": i, "urdf_file": f"ind_{i}.urdf",
                   "obj_source": "exam", "progress_m": 100.0 + i,
                   "cost_of_transport": 0.3}
            row.update({f"g{j}": 0.5 + spread * ((i % 2) - 0.5) * (1 + j % 2)
                        for j in range(15)})
            pop_rows.append(row)
    pd.DataFrame(pop_rows).to_csv(res / "outer_population.csv", index=False)

    front_rows = []
    for og, idxs in ((0, (0, 1)), (1, (1, 2, 3))):
        for rank, i in enumerate(idxs):
            row = {"outer_gen": og, "urdf_idx": i, "urdf_file": f"ind_{i}.urdf",
                   "obj_source": "exam", "front_size": len(idxs),
                   "front_rank": rank, "obj_progress_m": 100.0 + i,
                   "obj_cost_of_transport": 0.3, "progress_m": 100.0 + i,
                   "cost_of_transport": 0.3}
            row.update({f"g{j}": 0.1 * (i + 1) + 0.01 * j for j in range(15)})
            front_rows.append(row)
    pd.DataFrame(front_rows).to_csv(res / "pareto_front.csv", index=False)

    rep = run / "reproducibility"
    rep.mkdir()
    (rep / "config.yaml").write_text(
        "catalog:\n  refresh_urdfs_every: %d\n" % refresh_every
    )
    return run


class _ExamForest:
    def __init__(self, on=True):
        self.override_forest = on
        self.dens_min = 1.0
        self.dens_max = 5.5
        self.num_trees = None
        self.forest_mode = "growing"
        self.x_upper = 300.0
        self.forest_length = None
        self.x_spacing_start = None
        self.x_spacing_end = None
        self.y_spacing_min = None
        self.y_spacing_max = None

    def overrides(self):
        if not self.override_forest:
            return {}
        return {k: v for k, v in vars(self).items()
                if v is not None and k != "override_forest"}


class _Outer:
    def __init__(self, on=True):
        self.exam_forest = _ExamForest(on)


class _Cfg:
    def __init__(self, on=True):
        self.outer = _Outer(on)


def _metrics(N: int, P: int, F: int, base: float = 100.0) -> Dict[str, np.ndarray]:
    """Synthetic ``evaluate_population_multi_urdf`` metrics: per-slot value
    = base + 10*body + controller + forest/10 so every reduction is checkable."""
    body = np.arange(N)[:, None, None]
    ctrl = np.arange(P)[None, :, None]
    forest = np.arange(F)[None, None, :]
    slots = base + 10.0 * body + 1.0 * ctrl + 0.1 * forest
    m = {}
    for col, pu, ps in (("fitness", "per_urdf_reward", "per_slot_reward"),
                        ("progress_m", "per_urdf_progress", "per_slot_progress"),
                        ("cost_of_transport", "per_urdf_cot", "per_slot_cot"),
                        ("velocity", "per_urdf_velocity", "per_slot_velocity"),
                        ("crash_rate", "per_urdf_crash", "per_slot_crash")):
        m[ps] = slots.copy()
        m[pu] = slots.mean(axis=2)
    return m


# ------------------------------------------------------- generation discovery

def test_generation_indices_lists_complete_generations_only(tmp_path):
    run = _fake_run(tmp_path, gens=[0, 1, 2, 5])
    # A pruned generation (no solutions) must not be listed.
    d = run / "generations" / "gen_003"
    d.mkdir()
    np.save(d / "fitnesses.npy", np.zeros(4))
    assert generation_indices(run) == [0, 1, 2, 5]


def test_generation_indices_empty_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        generation_indices(tmp_path / "nowhere")


def test_select_generations_thins_and_keeps_the_last():
    assert select_generations([0, 1, 2, 3, 4, 5, 6], every=1) == [0, 1, 2, 3, 4, 5, 6]
    assert select_generations([0, 1, 2, 3, 4, 5, 6], every=4) == [0, 4, 6]
    assert select_generations([0, 1, 2, 3, 4, 5, 6, 7, 8], every=4) == [0, 4, 8]


def test_select_generations_rejects_bad_stride():
    with pytest.raises(ValueError):
        select_generations([0, 1], every=0)


# ------------------------------------------------------- best of generation

def test_best_of_generation_returns_argmax_genome(tmp_path):
    run = _fake_run(tmp_path)
    genome, idx, fit = best_of_generation(run, 3)
    sols = np.load(run / "generations" / "gen_003" / "solutions.npy")
    assert idx == 3 and fit == pytest.approx(103.0)
    np.testing.assert_allclose(genome, sols[3])


def test_build_controllers_puts_zero_rules_first(tmp_path):
    run = _fake_run(tmp_path)
    zero = np.full(N_GENES, 0.5)
    table, genomes = build_controllers(run, [0, 2, 5], zero)
    assert list(table["controller_id"]) == [0, 1, 2, 3]
    assert list(table["kind"]) == ["zero", "best", "best", "best"]
    assert list(table["gen"]) == [-1, 0, 2, 5]
    assert list(table["best_idx"]) == [-1, 0, 2, 1]
    assert table["in_run_fitness"].iloc[1:].tolist() == pytest.approx([100.0, 102.0, 105.0])
    assert genomes.shape == (4, N_GENES)
    np.testing.assert_allclose(genomes[0], zero)
    sols = np.load(run / "generations" / "gen_005" / "solutions.npy")
    np.testing.assert_allclose(genomes[3], sols[1])


def test_build_controllers_rejects_genome_length_mismatch(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(ValueError, match="genes"):
        build_controllers(run, [0], np.full(N_GENES + 1, 0.5))


# ------------------------------------------------------------- exam forests

def test_exam_forest_settings_maps_onto_forest_config_fields():
    got = exam_forest_settings(_Cfg(on=True))
    assert got == {"dens_min": 1.0, "dens_max": 5.5, "mode": "growing",
                   "x_upper": 300.0}


def test_exam_forest_settings_empty_when_exam_flew_nominal_forests():
    assert exam_forest_settings(_Cfg(on=False)) == {}


# ------------------------------------------------------------------- bodies

def test_body_table_random_then_front_with_global_ids(tmp_path):
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=5, morph_seed=0, own="front")
    assert list(bodies["body_set"]) == ["random"] * 5 + ["front"] * 3
    assert list(bodies["body_id"]) == list(range(8))
    # Front rows are the last generation's front, dedup'd, in front order.
    front = bodies[bodies["body_set"] == "front"]
    assert list(front["urdf_idx"]) == [1, 2, 3]
    assert list(front["outer_gen"]) == [1, 1, 1]
    assert [c for c in bodies.columns if c.startswith("g")] == [f"g{i}" for i in range(15)]
    rnd = bodies[bodies["body_set"] == "random"]
    assert rnd[[f"g{i}" for i in range(15)]].to_numpy().min() >= 0.0
    assert rnd[[f"g{i}" for i in range(15)]].to_numpy().max() <= 1.0


def test_body_table_random_only(tmp_path):
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=3, morph_seed=1, own="none")
    assert list(bodies["body_set"]) == ["random"] * 3


def test_body_table_random_is_reproducible_from_seed(tmp_path):
    run = _fake_run(tmp_path)
    a = body_table(run, random_n=3, morph_seed=7, own="none")
    b = body_table(run, random_n=3, morph_seed=7, own="none")
    c = body_table(run, random_n=3, morph_seed=8, own="none")
    genes = [f"g{i}" for i in range(15)]
    np.testing.assert_allclose(a[genes].to_numpy(), b[genes].to_numpy())
    assert not np.allclose(a[genes].to_numpy(), c[genes].to_numpy())


def test_body_table_needs_at_least_one_set(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(ValueError):
        body_table(run, random_n=0, morph_seed=0, own="none")


# ------------------------------------------------------------------- passes

def test_controller_passes_single_pass_when_per_pass_is_zero():
    assert controller_passes(5, per_pass=0) == [([0, 1, 2, 3, 4], 5)]


def test_controller_passes_pads_last_group_with_the_zero_controller():
    groups = controller_passes(5, per_pass=2)
    assert groups == [([0, 1], 2), ([2, 3], 2), ([4, 0], 1)]


def test_controller_passes_exact_split_has_no_padding():
    assert controller_passes(4, per_pass=2) == [([0, 1], 2), ([2, 3], 2)]


# ---------------------------------------------------------------- reduction

def test_reduce_pass_one_row_per_body_and_real_controller():
    bodies = pd.DataFrame({"body_set": ["random", "front"], "urdf_idx": [0, 7],
                           "body_id": [3, 4]})
    m = _metrics(N=2, P=3, F=4)
    rows = reduce_pass(m, controller_ids=[5, 6, 0], n_real=2, bodies=bodies)
    df = pd.DataFrame(rows)
    assert len(df) == 4                               # 2 bodies × 2 real controllers
    assert sorted(df["controller_id"].unique()) == [5, 6]
    r = df[(df["body_id"] == 4) & (df["controller_id"] == 6)].iloc[0]
    assert r["body_set"] == "front" and r["urdf_idx"] == 7
    # per_urdf value = mean over forests of 100 + 10*1 + 1*1 + 0.1*f
    assert r["fitness"] == pytest.approx(111.0 + 0.1 * 1.5)
    assert r["progress_m"] == pytest.approx(111.15)
    # SE across the 4 forests of [0, .1, .2, .3]
    assert r["fitness_se"] == pytest.approx(np.std([0, .1, .2, .3], ddof=1) / 2)


def test_reduce_pass_single_forest_has_zero_se():
    bodies = pd.DataFrame({"body_set": ["random"], "urdf_idx": [0], "body_id": [0]})
    rows = reduce_pass(_metrics(1, 1, 1), controller_ids=[0], n_real=1, bodies=bodies)
    assert rows[0]["fitness_se"] == 0.0


# ------------------------------------------------------------------- resume

def test_completed_bodies_reads_back_finished_body_ids(tmp_path):
    csv = tmp_path / "generality.csv"
    assert completed_bodies(csv) == set()
    pd.DataFrame({"body_id": [0, 0, 3], "controller_id": [0, 1, 0],
                  "fitness": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
    assert completed_bodies(csv) == {0, 3}


# ------------------------------------------------------------------ summary

def test_summarize_means_over_bodies_per_generation():
    controllers = pd.DataFrame({"controller_id": [0, 1, 2],
                                "kind": ["zero", "best", "best"],
                                "gen": [-1, 0, 4]})
    rows = []
    for body_id, body_set, base in ((0, "random", 10.0), (1, "random", 20.0),
                                    (2, "front", 50.0)):
        for cid in (0, 1, 2):
            rows.append({"body_id": body_id, "body_set": body_set,
                         "controller_id": cid, "fitness": base + cid,
                         "progress_m": 1.0, "cost_of_transport": 0.2,
                         "crash_rate": 0.5, "velocity": 12.0})
    s = summarize(pd.DataFrame(rows), controllers)
    rnd = s[(s["body_set"] == "random") & (s["kind"] == "best")].sort_values("gen")
    assert list(rnd["gen"]) == [0, 4]
    assert rnd["fitness"].tolist() == pytest.approx([16.0, 17.0])
    assert rnd["n_bodies"].tolist() == [2, 2]
    # SE across the two bodies: std([11, 21], ddof=1)/sqrt(2)
    assert rnd["fitness_se"].iloc[0] == pytest.approx(np.std([11, 21], ddof=1) / np.sqrt(2))
    zero = s[(s["body_set"] == "front") & (s["kind"] == "zero")]
    assert len(zero) == 1 and zero["fitness"].iloc[0] == pytest.approx(50.0)
    assert zero["fitness_se"].iloc[0] == 0.0    # a single body has no spread


# ---------------------------------------------------------------- companions

def test_final_centroid_is_the_last_phase_mean(tmp_path):
    run = _fake_run(tmp_path)
    c = final_centroid(run)
    pop = pd.read_csv(run / "results" / "outer_population.csv")
    last = pop[pop["outer_gen"] == 1]
    np.testing.assert_allclose(c, last[[f"g{i}" for i in range(15)]].mean().to_numpy())


# ------------------------------------------------------------- provenance

def test_resume_conflicts_flag_measurement_differences():
    from WP2_Outer_Loop.controller_generality import resume_conflicts
    base = {"forest": {"x_upper": 300.0}, "forest_seed": 0, "n_forests": 64,
            "random_morphs": 128, "morph_seed": 0, "own_bodies": "front",
            "gens": [0, 1, 2], "checkpoint_md5": "abc", "tag": "x"}
    assert resume_conflicts(base, dict(base)) == []
    other = dict(base, n_forests=32, gens=[0, 2], tag="y")
    conflicts = resume_conflicts(base, other)
    assert any(c.startswith("n_forests") for c in conflicts)
    assert any(c.startswith("gens") for c in conflicts)
    assert not any(c.startswith("tag") for c in conflicts)


def test_resume_conflicts_none_without_existing_provenance():
    from WP2_Outer_Loop.controller_generality import resume_conflicts
    assert resume_conflicts({}, {"n_forests": 64}) == []


# ------------------------------------------------------------------- plots

def _summary_df(body_sets=("random", "front"), gens=(0, 4, 8)):
    rows = []
    for bs in body_sets:
        base = 100.0 if bs == "random" else 150.0
        rows.append({"body_set": bs, "kind": "zero", "gen": -1, "n_bodies": 3,
                     "fitness": base, "fitness_se": 1.0, "progress_m": 90.0,
                     "progress_m_se": 1.0, "cost_of_transport": 0.3,
                     "cost_of_transport_se": 0.01, "crash_rate": 0.5,
                     "crash_rate_se": 0.05, "velocity": 12.0, "velocity_se": 0.1})
        for g in gens:
            rows.append({"body_set": bs, "kind": "best", "gen": g, "n_bodies": 3,
                         "fitness": base - g, "fitness_se": 1.0,
                         "progress_m": 90.0 - g, "progress_m_se": 1.0,
                         "cost_of_transport": 0.3, "cost_of_transport_se": 0.01,
                         "crash_rate": 0.5, "crash_rate_se": 0.05,
                         "velocity": 12.0, "velocity_se": 0.1})
    return pd.DataFrame(rows)


def test_draw_generality_has_a_baseline_line_per_body_set():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.controller_generality import draw_generality
    fig, axes = plt.subplots(1, 3)
    draw_generality(axes, _summary_df(), body_sets=("random", "front"))
    labels = [l.get_label() for l in axes[0].get_lines()]
    assert labels.count("generalist - random morphologies") == 1
    assert labels.count("generalist - front morphologies") == 1
    assert labels.count("hebbian - random morphologies") == 1
    assert labels.count("hebbian - front morphologies") == 1
    # no twin axis: the morphology-diversity companion line is gone
    assert len(fig.axes) == 3
    plt.close(fig)


def test_draw_generality_ignores_absent_body_sets():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.controller_generality import draw_generality
    fig, axes = plt.subplots(1, 3)
    draw_generality(axes, _summary_df(body_sets=("random",)),
                    body_sets=("random", "front"))
    labels = [l.get_label() for l in axes[0].get_lines()]
    assert not any("front" in l for l in labels)
    plt.close(fig)


def test_plot_generality_writes_png(tmp_path):
    from WP2_Outer_Loop.controller_generality import plot_generality
    out = plot_generality(_summary_df(), tmp_path / "g.png",
                          body_sets=("random", "front"), title="t")
    assert out.is_file() and out.stat().st_size > 0


def test_nearest_generations_picks_closest_available():
    from WP2_Outer_Loop.controller_generality import nearest_generations
    avail = [0, 4, 8, 12, 16, 19]
    assert nearest_generations(avail, [0, 5, 10, 19]) == [0, 4, 8, 19]
    # duplicates collapse
    assert nearest_generations(avail, [7, 9]) == [8]


def _long_results(bodies, controllers):
    rows = []
    for _, b in bodies.iterrows():
        for _, c in controllers.iterrows():
            rows.append({"body_id": int(b["body_id"]), "body_set": b["body_set"],
                         "urdf_idx": int(b["urdf_idx"]),
                         "controller_id": int(c["controller_id"]),
                         "fitness": 100.0 - 0.1 * c["gen"] * b["body_id"],
                         "fitness_se": 1.0, "progress_m": 80.0,
                         "progress_m_se": 1.0, "cost_of_transport": 0.3,
                         "cost_of_transport_se": 0.01, "velocity": 12.0,
                         "velocity_se": 0.1, "crash_rate": 0.4,
                         "crash_rate_se": 0.05})
    return pd.DataFrame(rows)


def test_plot_delta_vs_distance_writes_png(tmp_path):
    from WP2_Outer_Loop.controller_generality import plot_delta_vs_distance
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=6, morph_seed=0, own="none")
    controllers, _ = build_controllers(run, [0, 2, 5], np.full(N_GENES, 0.5))
    df = _long_results(bodies, controllers)
    out = plot_delta_vs_distance(df, bodies, controllers, final_centroid(run),
                                 gens=[0, 5], out_path=tmp_path / "d.png")
    assert out.is_file() and out.stat().st_size > 0


# ------------------------------------------------------------ output bundle

def test_write_outputs_produces_summary_and_the_three_figures(tmp_path):
    from WP2_Outer_Loop.controller_generality import write_outputs
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=4, morph_seed=0, own="front")
    controllers, _ = build_controllers(run, [0, 2, 5], np.full(N_GENES, 0.5))
    df = _long_results(bodies, controllers)
    out_dir = tmp_path / "out"
    written = write_outputs(out_dir, df, controllers, bodies, run)
    names = {p.name for p in written}
    assert {"summary.csv", "generality_all.png", "generality_random.png",
            "generality_front.png", "delta_vs_distance.png"} <= names
    s = pd.read_csv(out_dir / "summary.csv")
    assert set(s["body_set"]) == {"random", "front"}


def test_write_outputs_random_only_skips_front_and_overlay(tmp_path):
    from WP2_Outer_Loop.controller_generality import write_outputs
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=4, morph_seed=0, own="none")
    controllers, _ = build_controllers(run, [0, 5], np.full(N_GENES, 0.5))
    df = _long_results(bodies, controllers)
    written = {p.name for p in write_outputs(tmp_path / "out", df, controllers, bodies, run)}
    assert "generality_random.png" in written
    assert "generality_front.png" not in written
    assert "generality_all.png" not in written


def test_plot_only_cli_rebuilds_outputs_from_csvs(tmp_path):
    from WP2_Outer_Loop.controller_generality import main
    from WP2_Outer_Loop.transfer_eval import write_results
    run = _fake_run(tmp_path)
    bodies = body_table(run, random_n=4, morph_seed=0, own="front")
    controllers, _ = build_controllers(run, [0, 2, 5], np.full(N_GENES, 0.5))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_results(_long_results(bodies, controllers), {"n_forests": 2},
                  out_dir / "generality.csv")
    controllers.to_csv(out_dir / "controllers.csv", index=False)
    bodies.to_csv(out_dir / "bodies.csv", index=False)
    rc = main(["--run", str(run), "-o", str(out_dir), "--plot-only"])
    assert rc == 0
    assert (out_dir / "generality_all.png").is_file()
    assert (out_dir / "summary.csv").is_file()


# ------------------------------------------------------------------ workers

def test_cli_exposes_parallel_scene_workers():
    from WP2_Outer_Loop.controller_generality import _build_parser
    ap = _build_parser()
    assert ap.parse_args(["--run", "x"]).workers == 1
    assert ap.parse_args(["--run", "x", "--workers", "3"]).workers == 3
