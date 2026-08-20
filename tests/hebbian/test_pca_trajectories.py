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
