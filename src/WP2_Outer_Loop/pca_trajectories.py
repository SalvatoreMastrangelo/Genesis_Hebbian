"""
Joint-PCA morphology trajectories across outer-loop runs.
=========================================================

Concatenates the normalized 15D morphology genomes of several runs — every
row of ``results/outer_population.csv``, in appearance order (run, outer
gen, individual) — fits ONE shared PCA on the joint matrix, and plots each
run's per-generation centroid (mean genome) as a path in PC1–PC2 space.
Because both runs are projected into the same basis, the paths show whether
e.g. a morphology-only run and a full co-design run drift toward different
regions of morphology space.

Four figures per invocation:

* ``pca_population.png``   — centroids over ALL evaluated genomes per gen.
* ``pca_pareto_front.png`` — centroids over each gen's Pareto-front members
  only (``results/pareto_front.csv``, backfilled when missing; exam-filtered
  and min-progress-gated like the front plots), with its own PCA fit.
* ``*_centroid_fit.png``   — same two, but the PCA is fit on the per-gen
  centroid genomes instead of all rows: within-gen mutation spread stops
  eating explained variance, so the PCs align with the drift of the mean.

Both figures are written into EVERY included run's ``plots/`` folder, under
a subfolder naming all included runs, e.g.
``<run_dir>/plots/pca_trajectories_morphology_only__codesign/``.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories \
        <run_dir_1> <run_dir_2> [...] [--labels NAME1 NAME2 ...]
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .pareto_fronts import build_pareto_front_csv_safe

_GENE_RE = re.compile(r"^g\d+$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")

# One sequential colormap per run (light -> dark encodes outer generation),
# hues in a fixed colorblind-aware order, never cycled per-generation.
_RUN_CMAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys",
              "YlOrBr", "RdPu")

_MAX_DIRNAME_LEN = 150


# ----------------------------------------------------------------------------
#  Data loading
# ----------------------------------------------------------------------------

def _gene_columns(df: pd.DataFrame) -> List[str]:
    """The ``g<i>`` genome columns, in file order (g0, g1, ... g14)."""
    return [c for c in df.columns if _GENE_RE.match(c)]


def load_genomes(run_dir: Path | str, source: str = "population") -> pd.DataFrame:
    """``outer_gen`` + gene columns for one run, rows sorted by outer gen.

    ``source`` is ``"population"`` (every evaluated genome, from
    ``outer_population.csv``) or ``"front"`` (Pareto-front members only, from
    ``pareto_front.csv`` — rebuilt from the population CSV when possible, so
    a missing or stale front file self-heals like in ``pareto_plots``).
    """
    run_dir = Path(run_dir)
    if source == "population":
        csv_path = run_dir / "results" / "outer_population.csv"
    elif source == "front":
        build_pareto_front_csv_safe(run_dir)
        csv_path = run_dir / "results" / "pareto_front.csv"
    else:
        raise ValueError(f"Unknown genome source: {source!r}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"No {source} genomes: {csv_path} is missing")

    df = pd.read_csv(csv_path)
    genes = _gene_columns(df)
    if not genes:
        raise ValueError(f"No g<i> genome columns in {csv_path}")
    df = df[["outer_gen"] + genes]
    return df.sort_values("outer_gen", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
#  PCA + centroid paths
# ----------------------------------------------------------------------------

def fit_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First two principal components of ``X`` (rows = genomes).

    Returns ``(components, explained)``: a (2, n_genes) row-vector basis and
    the two explained-variance ratios. Plain PCA on mean-centered data — the
    genes already share the [0, 1] scale, so no per-gene standardization
    (z-scoring would inflate near-constant genes into noise directions).
    Component signs are fixed so each PC's largest-magnitude loading is
    positive, making outputs reproducible across BLAS/SVD implementations.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(f"PCA needs a (>=2, >=2) matrix, got {X.shape}")
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total = float((s ** 2).sum())
    explained = (s[:2] ** 2 / total) if total > 0 else np.zeros(2)
    components = vt[:2].copy()
    for pc in components:
        if pc[np.argmax(np.abs(pc))] < 0:
            pc *= -1.0
    return components, explained


def centroid_paths(
    dfs: Sequence[pd.DataFrame], labels: Sequence[str],
    fit_on: str = "rows",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Per-generation centroids of every run in one shared PC1–PC2 basis.

    ``dfs`` are ``load_genomes`` frames, one per run. Each run's genomes are
    averaged per outer generation — persistent NSGA-II elites appear once
    per generation they survive, which deliberately weights centroids toward
    where the population sits.

    ``fit_on`` picks the matrix the shared PCA is fit on:

    * ``"rows"`` — the concatenation of ALL individual genomes. Explained
      variance includes within-generation mutation spread.
    * ``"centroids"`` — the per-(run, gen) centroid genomes themselves,
      suppressing within-gen noise so the PCs align with the drift of the
      population mean.

    Either way the projected points are the same centroids, just in a
    different basis. Returns ``(paths, explained)``: a frame with columns
    ``run, outer_gen, pc1, pc2`` — runs in input order, generations
    ascending — and the two explained-variance ratios of the fit matrix.
    """
    gene_sets = [_gene_columns(df) for df in dfs]
    if any(g != gene_sets[0] for g in gene_sets[1:]):
        raise ValueError(
            "gene columns differ between runs: "
            + "; ".join(f"{lab}: {len(g)}" for lab, g in zip(labels, gene_sets))
        )
    genes = gene_sets[0]

    meta = []  # (run label, outer_gen) per centroid row
    centroids = []
    for label, df in zip(labels, dfs):
        G = df[genes].to_numpy(dtype=float)
        gens = df["outer_gen"].to_numpy()
        for gen in np.unique(gens):  # np.unique sorts ascending
            meta.append((label, gen))
            centroids.append(G[gens == gen].mean(axis=0))
    C = np.vstack(centroids)

    if fit_on == "rows":
        X = np.vstack([df[genes].to_numpy(dtype=float) for df in dfs])
    elif fit_on == "centroids":
        X = C
    else:
        raise ValueError(f"Unknown fit_on: {fit_on!r} "
                         "(expected 'rows' or 'centroids')")
    components, explained = fit_pca(X)
    # Projecting the centroids == averaging the projected rows (linearity),
    # so the "rows" fit reproduces the mean of per-genome projections.
    proj = (C - X.mean(axis=0)) @ components.T

    paths = pd.DataFrame(
        [(label, gen, p[0], p[1]) for (label, gen), p in zip(meta, proj)],
        columns=["run", "outer_gen", "pc1", "pc2"],
    )
    return paths, explained


# ----------------------------------------------------------------------------
#  Run labels / output naming
# ----------------------------------------------------------------------------

def run_labels(
    run_dirs: Sequence[Path | str], labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """One unique label per run for legends and the output folder name.

    Explicit ``labels`` win. Otherwise the run dir's basename with the
    ``YYYY-MM-DD_HH-MM-SS_`` timestamp stripped; colliding names (e.g. r0/r2
    repeats of the same config) get their parent directory name prefixed,
    and any survivors of that get a positional suffix.
    """
    if labels is not None:
        if len(labels) != len(run_dirs):
            raise ValueError(
                f"{len(labels)} labels for {len(run_dirs)} run dirs"
            )
        return list(labels)

    dirs = [Path(d) for d in run_dirs]
    out = [_TIMESTAMP_RE.sub("", d.name) or d.name for d in dirs]
    dupes = {label for label in out if out.count(label) > 1}
    out = [
        f"{d.parent.name}_{label}" if label in dupes and d.parent.name
        else label
        for label, d in zip(out, dirs)
    ]
    dupes = {label for label in out if out.count(label) > 1}
    return [
        f"{label}_{i}" if label in dupes else label
        for i, label in enumerate(out)
    ]


def _output_dirname(labels: Sequence[str]) -> str:
    """``pca_trajectories_<lab1>__<lab2>...``, hash-truncated when too long
    for one path component."""
    name = "pca_trajectories_" + "__".join(labels)
    if len(name) > _MAX_DIRNAME_LEN:
        digest = hashlib.sha1(name.encode()).hexdigest()[:10]
        name = name[:_MAX_DIRNAME_LEN - 11] + "_" + digest
    return name


# ----------------------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------------------

def _draw_trajectories(
    paths: pd.DataFrame, explained: np.ndarray, labels: Sequence[str],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    handles = []
    for k, label in enumerate(labels):
        sub = paths[paths["run"] == label]
        if sub.empty:
            continue
        cmap = plt.get_cmap(_RUN_CMAPS[k % len(_RUN_CMAPS)])
        gens = sub["outer_gen"].to_numpy(dtype=float)
        span = gens.max() - gens.min()
        frac = (gens - gens.min()) / span if span > 0 else np.ones_like(gens)
        # Light -> dark encodes generation; start above 0.35 so the earliest
        # centroids stay visible on the white background.
        colors = cmap(0.35 + 0.6 * frac)

        x, y = sub["pc1"].to_numpy(), sub["pc2"].to_numpy()
        ax.plot(x, y, color=cmap(0.7), lw=1.1, alpha=0.5, zorder=1)
        ax.scatter(x, y, c=colors, s=22, edgecolors="none", zorder=2)
        ax.scatter(x[0], y[0], marker="o", s=90, facecolors="none",
                   edgecolors=cmap(0.95), lw=1.5, zorder=3)
        ax.scatter(x[-1], y[-1], marker="*", s=180, color=cmap(0.95),
                   edgecolors="white", lw=0.5, zorder=3)
        handles.append(Line2D(
            [], [], color=cmap(0.7), marker="o", markersize=6,
            label=f"{label} (gen {int(gens.min())} → {int(gens.max())})",
        ))

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title(f"{title}\n(○ = first gen, ★ = last gen; "
                 f"darker = later generation)")
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    return fig


_SOURCES = (
    # (load_genomes source, filename stem, figure title)
    ("population", "pca_population",
     "Morphology PCA trajectory — per-gen centroid of all genomes"),
    ("front", "pca_pareto_front",
     "Morphology PCA trajectory — per-gen centroid of Pareto front"),
)

_FITS = (
    # (centroid_paths fit_on, filename suffix, title suffix)
    ("rows", "", ""),
    ("centroids", "_centroid_fit", "\n(PCA fit on per-gen centroids)"),
)


def plot_pca_trajectories(
    run_dirs: Sequence[Path | str], labels: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Render both PCA-trajectory figures and copy them into every run.

    Returns the list of written PNG paths. The front figure is skipped (with
    a message) for run sets where no front CSV can be produced.
    """
    dirs = [Path(d) for d in run_dirs]
    names = run_labels(dirs, labels)
    out_dirname = _output_dirname(names)

    written: List[Path] = []
    for source, stem, title in _SOURCES:
        try:
            dfs = [load_genomes(d, source) for d in dirs]
        except FileNotFoundError as exc:
            if source == "population":
                raise
            print(f"[pca] Skipping {stem}*.png: {exc}")
            continue
        for fit_on, fname_suffix, title_suffix in _FITS:
            paths, explained = centroid_paths(dfs, names, fit_on=fit_on)
            fig = _draw_trajectories(
                paths, explained, names, title + title_suffix)
            for run_dir in dirs:
                out = (run_dir / "plots" / out_dirname
                       / f"{stem}{fname_suffix}.png")
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out, dpi=150)
                written.append(out)
            plt.close(fig)

    for path in written:
        print(f"[pca] Wrote {path}")
    return written


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Joint-PCA morphology trajectories across outer-loop runs"
    )
    parser.add_argument("run_dirs", nargs="+",
                        help="Outer-loop run directories to compare")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="One legend/folder label per run dir "
                             "(default: dir names, timestamps stripped)")
    args = parser.parse_args(argv)
    plot_pca_trajectories(args.run_dirs, args.labels)


if __name__ == "__main__":
    main()
