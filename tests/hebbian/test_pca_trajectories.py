"""Tests for WP2_Outer_Loop.pca_trajectories — joint-PCA morphology
trajectories across outer-loop runs.

Pure pandas/numpy/matplotlib: runnable outside the Genesis docker image with
    PYTHONPATH=src python3 -m pytest tests/hebbian/test_pca_trajectories.py \
        -o addopts="--import-mode=importlib"
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from WP2_Outer_Loop.pca_trajectories import (
    _output_dirname,
    centroid_paths,
    fit_pca,
    load_genomes,
    main,
    plot_pca_trajectories,
    run_labels,
)


# ----------------------------------------------------------------------------
#  Fixture helpers
# ----------------------------------------------------------------------------

def _write_population_csv(
    run_dir: Path, genomes_per_gen: Dict[int, Sequence[Sequence[float]]]
) -> Path:
    """A minimal ``results/outer_population.csv`` in the real schema."""
    n_genes = len(next(iter(genomes_per_gen.values()))[0])
    gene_cols = [f"g{i}" for i in range(n_genes)]
    out = run_dir / "results" / "outer_population.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["outer_gen", "urdf_idx", "urdf_file", "n_score_gens",
             "obj_source", "obj_fitness", "obj_cost_of_transport",
             "fitness", "cost_of_transport", "progress_m", "velocity",
             "crash_rate"] + gene_cols
        )
        for gen, genomes in genomes_per_gen.items():
            for i, genome in enumerate(genomes):
                # Spread objectives so every gen has >=1 non-dominated row.
                fit, cot = 1.0 + i, 1.0 + i
                w.writerow(
                    [gen, i, f"urdf_{i}.urdf", 4, "phase_mean", fit, cot,
                     fit, cot, 50.0, 10.0, 0.1] + list(genome)
                )
    return out


def _population_df(
    genomes_per_gen: Dict[int, Sequence[Sequence[float]]]
) -> pd.DataFrame:
    """In-memory frame in ``load_genomes`` format: outer_gen + gene cols."""
    rows = []
    for gen, genomes in genomes_per_gen.items():
        for genome in genomes:
            rows.append([gen] + list(genome))
    n_genes = len(rows[0]) - 1
    return pd.DataFrame(
        rows, columns=["outer_gen"] + [f"g{i}" for i in range(n_genes)]
    )


# ----------------------------------------------------------------------------
#  fit_pca
# ----------------------------------------------------------------------------

def test_fit_pca_recovers_dominant_direction():
    rng = np.random.default_rng(0)
    direction = np.array([3.0, 4.0, 0.0, 0.0, 0.0]) / 5.0
    t = rng.normal(size=200)
    X = np.outer(t, direction) * 10.0 + rng.normal(size=(200, 5)) * 0.01
    components, explained = fit_pca(X)
    assert components.shape == (2, 5)
    assert abs(components[0] @ direction) > 0.99
    assert explained[0] > explained[1]
    assert explained[0] > 0.99
    assert 0.0 <= explained.sum() <= 1.0 + 1e-9


def test_fit_pca_sign_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    components, _ = fit_pca(X)
    for pc in components:
        assert pc[np.argmax(np.abs(pc))] > 0


# ----------------------------------------------------------------------------
#  load_genomes
# ----------------------------------------------------------------------------

def test_load_genomes_population_orders_by_gen(tmp_path):
    # Written out of order on purpose: 2, 0, 1.
    _write_population_csv(tmp_path, {
        2: [[0.5, 0.5], [0.6, 0.6]],
        0: [[0.1, 0.1], [0.2, 0.2]],
        1: [[0.3, 0.3], [0.4, 0.4]],
    })
    df = load_genomes(tmp_path, source="population")
    assert list(df.columns) == ["outer_gen", "g0", "g1"]
    assert df["outer_gen"].tolist() == [0, 0, 1, 1, 2, 2]
    assert df["g0"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_load_genomes_front_builds_missing_csv(tmp_path):
    _write_population_csv(tmp_path, {
        0: [[0.1, 0.2], [0.3, 0.4]],
        1: [[0.5, 0.6], [0.7, 0.8]],
    })
    assert not (tmp_path / "results" / "pareto_front.csv").is_file()
    df = load_genomes(tmp_path, source="front")
    assert (tmp_path / "results" / "pareto_front.csv").is_file()
    assert list(df.columns) == ["outer_gen", "g0", "g1"]
    assert sorted(df["outer_gen"].unique().tolist()) == [0, 1]
    # Front <= population per gen.
    assert (df["outer_gen"] == 0).sum() <= 2


def test_load_genomes_missing_population_raises(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(FileNotFoundError):
        load_genomes(tmp_path, source="population")


# ----------------------------------------------------------------------------
#  centroid_paths
# ----------------------------------------------------------------------------

def test_centroid_paths_means_per_generation():
    # All variance along g0 -> PC1 is the (centered) g0 axis, PC2 flat.
    run_a = _population_df({0: [[0.0, 0.5], [2.0, 0.5]],
                            1: [[4.0, 0.5], [6.0, 0.5]]})
    run_b = _population_df({0: [[10.0, 0.5], [12.0, 0.5]]})
    paths, explained = centroid_paths([run_a, run_b], ["a", "b"])
    assert list(paths.columns) == ["run", "outer_gen", "pc1", "pc2"]
    assert paths["run"].tolist() == ["a", "a", "b"]
    assert paths["outer_gen"].tolist() == [0, 1, 0]
    g0_mean = (0 + 2 + 4 + 6 + 10 + 12) / 6.0
    np.testing.assert_allclose(
        paths["pc1"].to_numpy(),
        [1.0 - g0_mean, 5.0 - g0_mean, 11.0 - g0_mean], atol=1e-9,
    )
    np.testing.assert_allclose(paths["pc2"].to_numpy(), 0.0, atol=1e-9)
    assert explained[0] > 0.99


def test_centroid_paths_keeps_non_contiguous_gens():
    run = _population_df({0: [[0.0, 0.0]], 2: [[1.0, 1.0]], 5: [[2.0, 2.0]]})
    paths, _ = centroid_paths([run], ["solo"])
    assert paths["outer_gen"].tolist() == [0, 2, 5]


def test_centroid_paths_fit_on_centroids_ignores_within_gen_noise():
    # Within-gen spread is along g1 (large); the per-gen centroid drifts
    # along g0 (small). A row-fit PCA makes PC1 the g1 noise axis; a
    # centroid-fit PCA must recover the g0 drift instead.
    run = _population_df({
        0: [[0.0, -4.5], [0.0, 5.5]],
        1: [[1.0, -4.5], [1.0, 5.5]],
        2: [[2.0, -4.5], [2.0, 5.5]],
    })
    rows_paths, _ = centroid_paths([run], ["solo"], fit_on="rows")
    np.testing.assert_allclose(rows_paths["pc1"].to_numpy(), 0.0, atol=1e-9)

    paths, explained = centroid_paths([run], ["solo"], fit_on="centroids")
    assert list(paths.columns) == ["run", "outer_gen", "pc1", "pc2"]
    np.testing.assert_allclose(
        paths["pc1"].to_numpy(), [-1.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(paths["pc2"].to_numpy(), 0.0, atol=1e-9)
    assert explained[0] > 0.99


def test_centroid_paths_rejects_unknown_fit_on():
    run = _population_df({0: [[0.0, 1.0]], 1: [[1.0, 0.0]]})
    with pytest.raises(ValueError, match="fit_on"):
        centroid_paths([run], ["solo"], fit_on="nonsense")


def test_centroid_paths_rejects_gene_dim_mismatch():
    run_a = _population_df({0: [[0.0, 1.0]]})
    run_b = pd.DataFrame({"outer_gen": [0], "g0": [0.0], "g1": [1.0],
                          "g2": [2.0]})
    with pytest.raises(ValueError, match="gene"):
        centroid_paths([run_a, run_b], ["a", "b"])


# ----------------------------------------------------------------------------
#  run_labels / output dirname
# ----------------------------------------------------------------------------

def test_run_labels_strip_timestamp_prefix(tmp_path):
    d1 = tmp_path / "2026-08-10_21-25-32_morph_only"
    d2 = tmp_path / "2026-08-10_21-57-06_codesign"
    assert run_labels([d1, d2]) == ["morph_only", "codesign"]


def test_run_labels_collision_falls_back_to_parent(tmp_path):
    d1 = tmp_path / "outer_morph_r0" / "2026-08-10_21-25-32_morph_only"
    d2 = tmp_path / "outer_morph_r2" / "2026-08-11_09-00-00_morph_only"
    labels = run_labels([d1, d2])
    assert len(set(labels)) == 2
    assert "outer_morph_r0" in labels[0]
    assert "outer_morph_r2" in labels[1]


def test_run_labels_explicit_labels_win(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    assert run_labels([d1, d2], ["x", "y"]) == ["x", "y"]
    with pytest.raises(ValueError):
        run_labels([d1, d2], ["only_one"])


def test_output_dirname_joins_labels():
    assert _output_dirname(["a", "b"]) == "pca_trajectories_a__b"


def test_output_dirname_caps_length():
    labels = [f"very_long_run_label_number_{i:03d}" for i in range(12)]
    name = _output_dirname(labels)
    assert len(name) <= 150
    assert name == _output_dirname(labels)  # stable


# ----------------------------------------------------------------------------
#  End-to-end plotting
# ----------------------------------------------------------------------------

def _two_synthetic_runs(tmp_path) -> List[Path]:
    rng = np.random.default_rng(7)
    dirs = []
    for name, drift in [("runA", +0.02), ("runB", -0.02)]:
        run_dir = tmp_path / name
        genomes = {
            gen: (0.5 + drift * gen
                  + rng.normal(scale=0.01, size=(4, 15))).clip(0, 1).tolist()
            for gen in range(3)
        }
        _write_population_csv(run_dir, genomes)
        dirs.append(run_dir)
    return dirs


def test_plot_writes_folder_into_each_run(tmp_path):
    dirs = _two_synthetic_runs(tmp_path)
    written = plot_pca_trajectories(dirs, labels=["runA", "runB"])
    expected_dir = "pca_trajectories_runA__runB"
    for run_dir in dirs:
        out = run_dir / "plots" / expected_dir
        for fname in ("pca_population.png", "pca_population_centroid_fit.png",
                      "pca_pareto_front.png",
                      "pca_pareto_front_centroid_fit.png"):
            assert (out / fname).is_file()
            assert (out / fname).stat().st_size > 0
    assert all(Path(p).is_file() for p in written)
    # One folder per run, four figures each.
    assert len(written) == 8


def test_main_cli(tmp_path):
    dirs = _two_synthetic_runs(tmp_path)
    main([str(dirs[0]), str(dirs[1]), "--labels", "runA", "runB"])
    assert (dirs[0] / "plots" / "pca_trajectories_runA__runB"
            / "pca_population.png").is_file()


# ----------------------------------------------------------------------------
#  Condition groups: colour by group, seed-paired distance summary, --out
# ----------------------------------------------------------------------------

def _write_seed(run_dir: Path, seed: int) -> None:
    import yaml
    cfg = run_dir / "reproducibility" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg, "w") as f:
        yaml.safe_dump({"seed": seed, "outer": {}}, f)


def _constant_run(tmp_path, name: str, seed: int, final: float) -> Path:
    """Two gens; every genome of the last gen equals ``final`` on all 15
    genes, so the final centroid is exactly known."""
    run_dir = tmp_path / name
    genomes = {0: [[0.5] * 15] * 3, 1: [[final] * 15] * 3}
    _write_population_csv(run_dir, genomes)
    _write_seed(run_dir, seed)
    return run_dir


def test_load_seed_reads_config_and_tolerates_absence(tmp_path):
    from WP2_Outer_Loop.pca_trajectories import load_seed
    run = _constant_run(tmp_path, "a", 67, 0.6)
    assert load_seed(run) == 67
    (run / "reproducibility" / "config.yaml").unlink()
    assert load_seed(run) is None


def test_final_centroid_distances_classify_seed_pairs(tmp_path):
    from WP2_Outer_Loop.pca_trajectories import (
        RunGroup, final_centroid_distances, load_genomes, summarize_distances)
    a = _constant_run(tmp_path, "cod67", 67, 0.6)
    b = _constant_run(tmp_path, "cod68", 68, 0.7)
    c = _constant_run(tmp_path, "morph67", 67, 0.6)   # same endpoint as cod67
    d = _constant_run(tmp_path, "morph68", 68, 0.9)
    groups = [RunGroup("co-design", "red", [a, b], ["cod67", "cod68"]),
              RunGroup("morph-only", "blue", [c, d], ["morph67", "morph68"])]
    dist = final_centroid_distances(groups)
    assert set(dist.columns) >= {"run_a", "run_b", "group_a", "group_b",
                                 "seed_a", "seed_b", "relation", "distance"}
    assert len(dist) == 6                      # 4 choose 2
    rel = dict(zip(zip(dist.run_a, dist.run_b), dist.relation))
    assert rel[("cod67", "cod68")] == "same condition"
    assert rel[("morph67", "morph68")] == "same condition"
    assert rel[("cod67", "morph67")] == "across conditions, same seed"
    assert rel[("cod67", "morph68")] == "across conditions, different seed"
    d_ = dict(zip(zip(dist.run_a, dist.run_b), dist.distance))
    assert d_[("cod67", "morph67")] == pytest.approx(0.0)
    assert d_[("cod67", "cod68")] == pytest.approx(np.sqrt(15) * 0.1)
    summ = summarize_distances(dist).set_index("relation")
    assert summ.loc["across conditions, same seed", "n"] == 2
    assert summ.loc["same condition", "n"] == 2
    assert summ.loc["across conditions, different seed", "n"] == 2
    assert summ.loc["across conditions, same seed", "mean"] == pytest.approx(
        (0.0 + np.sqrt(15) * 0.2) / 2)


def test_group_mode_writes_to_out_dir_only_with_group_legend(tmp_path):
    a = _constant_run(tmp_path, "cod67", 67, 0.6)
    b = _constant_run(tmp_path, "cod68", 68, 0.7)
    c = _constant_run(tmp_path, "morph67", 67, 0.65)
    d = _constant_run(tmp_path, "morph68", 68, 0.9)
    out = tmp_path / "pca_out"
    main(["--group", "co-design", "red", str(a), str(b),
          "--group", "morph-only", "blue", str(c), str(d),
          "--labels", "cod67", "cod68", "morph67", "morph68",
          "--out", str(out)])
    for fname in ("pca_population.png", "pca_population_centroid_fit.png",
                  "pca_pareto_front.png", "pca_pareto_front_centroid_fit.png",
                  "final_centroid_distances.csv", "distance_summary.csv"):
        assert (out / fname).stat().st_size > 0, fname
    # nothing written into the run folders in --out mode
    assert not any((r / "plots").exists() for r in (a, b, c, d))
    dist = pd.read_csv(out / "final_centroid_distances.csv")
    assert len(dist) == 6


def test_group_legend_lists_groups_not_runs():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pca_trajectories import RunGroup, _draw_trajectories
    paths = pd.DataFrame({
        "run": ["a", "a", "b", "b", "c", "c"],
        "outer_gen": [0, 1, 0, 1, 0, 1],
        "pc1": [0.0, 1.0, 0.0, -1.0, 0.5, 0.5],
        "pc2": [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
    })
    groups = [RunGroup("co-design", "red", [], ["a", "b"]),
              RunGroup("morph-only", "blue", [], ["c"])]
    fig = _draw_trajectories(paths, np.array([0.6, 0.3]), ["a", "b", "c"],
                             "t", groups=groups)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert any(t.startswith("co-design") for t in texts)
    assert any(t.startswith("morph-only") for t in texts)
    assert not any(t.startswith("a ") or t.startswith("b ") for t in texts)


# ----------------------------------------------------------------------------
#  Pairwise-by-seed panels (shared basis, one panel per seed)
# ----------------------------------------------------------------------------

def test_seed_panels_group_runs_by_seed_and_skip_unseeded():
    from WP2_Outer_Loop.pca_trajectories import RunGroup, seed_panels
    groups = [RunGroup("cod", "red", [], ["cod67", "cod68", "cod_noseed"]),
              RunGroup("morph", "blue", [], ["morph67", "morph68"])]
    seeds = {"cod67": 67, "cod68": 68, "cod_noseed": None,
             "morph67": 67, "morph68": 68}
    panels = seed_panels(groups, seeds)
    assert [s for s, _ in panels] == [67, 68]
    assert dict(panels)[67] == ["cod67", "morph67"]
    assert dict(panels)[68] == ["cod68", "morph68"]


def test_draw_by_seed_makes_one_panel_per_seed_with_shared_limits():
    import matplotlib.pyplot as plt
    from WP2_Outer_Loop.pca_trajectories import RunGroup, _draw_by_seed
    paths = pd.DataFrame({
        "run": ["cod67", "cod67", "morph67", "morph67", "cod68", "cod68", "morph68", "morph68"],
        "outer_gen": [0, 1] * 4,
        "pc1": [0.0, 1.0, 0.0, 0.5, 0.0, -1.0, 0.0, -0.5],
        "pc2": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0],
    })
    groups = [RunGroup("co-design", "red", [], ["cod67", "cod68"]),
              RunGroup("morph-only", "blue", [], ["morph67", "morph68"])]
    seeds = {"cod67": 67, "cod68": 68, "morph67": 67, "morph68": 68}
    fig = _draw_by_seed(paths, np.array([0.5, 0.3]), ["cod67", "cod68", "morph67", "morph68"],
                        "t", groups, seeds)
    panels = [ax for ax in fig.axes if ax.get_title()]
    titles = [ax.get_title() for ax in panels]
    xl = [ax.get_xlim() for ax in panels]; yl = [ax.get_ylim() for ax in panels]
    plt.close(fig)
    assert titles == ["seed 67", "seed 68"]
    assert all(np.allclose(l, xl[0]) for l in xl) and all(np.allclose(l, yl[0]) for l in yl)


def test_group_mode_writes_by_seed_figures(tmp_path):
    a = _constant_run(tmp_path, "cod67", 67, 0.6)
    b = _constant_run(tmp_path, "cod68", 68, 0.7)
    c = _constant_run(tmp_path, "morph67", 67, 0.65)
    d = _constant_run(tmp_path, "morph68", 68, 0.9)
    out = tmp_path / "pca_out"
    main(["--group", "co-design", "red", str(a), str(b),
          "--group", "morph-only", "blue", str(c), str(d),
          "--labels", "cod67", "cod68", "morph67", "morph68",
          "--out", str(out)])
    for fname in ("pca_population_by_seed.png", "pca_population_centroid_fit_by_seed.png",
                  "pca_pareto_front_by_seed.png", "pca_pareto_front_centroid_fit_by_seed.png"):
        assert (out / fname).stat().st_size > 0, fname
