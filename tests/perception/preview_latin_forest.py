"""Preview / visualization for the proposed ``"latin"`` forest mode.

This is a standalone *visualization* helper, not a pytest test (the ``preview_``
prefix keeps pytest from collecting it). It mirrors the design of the planned
``ForestGenerator._sample_latin_forest`` so the layout can be eyeballed before /
alongside the real implementation:

  * the forest rectangle ``[x_lower, x_upper] x [y_lower, y_upper]`` is split into
    columns whose widths interpolate linearly from ``x_spacing_start`` (low x) to
    ``x_spacing_end`` (high x) and are rescaled to tile the length exactly;
  * each column independently splits the FULL width into equal rows, with a
    per-column row count derived from a target y-spacing that interpolates
    ``y_spacing_max`` (first column) -> ``y_spacing_min`` (last column);
  * cells therefore tile the whole rectangle with no gaps / no overflow, and one
    tree is sampled uniformly inside each cell.

Run directly to regenerate the preview PNGs in the repo root::

    python -m tests.perception.preview_latin_forest
    python tests/perception/preview_latin_forest.py --out-dir /tmp
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class LatinForestParams:
    x_lower: float = 0.0
    x_upper: float = 300.0          # 300 m long (forward axis)
    y_lower: float = -50.0
    y_upper: float = 50.0           # 100 m wide (lateral axis)
    x_spacing_start: float = 15.0   # column width at low x  (sparse)
    x_spacing_end: float = 5.0      # column width at high x (dense)
    y_spacing_max: float = 15.0     # row height, first column (sparse)
    y_spacing_min: float = 5.0      # row height, last column  (dense)
    seed: int = 0


@dataclass
class LatinForestLayout:
    trees_x: np.ndarray             # (T,)
    trees_y: np.ndarray             # (T,)
    x_edges: np.ndarray             # (n_cols + 1,) column boundaries
    row_edges: list                 # per-column array of row boundaries
    n_cols: int


def build_latin_forest(p: LatinForestParams) -> LatinForestLayout:
    """Compute the cell grid and one uniform tree per cell."""
    rng = np.random.default_rng(p.seed)
    L = p.x_upper - p.x_lower
    y_width = p.y_upper - p.y_lower

    # Columns: widths interpolate s0 -> s1, rescaled to tile L exactly.
    n_cols = max(1, int(round(2.0 * L / (p.x_spacing_start + p.x_spacing_end))))
    if n_cols == 1:
        widths = np.array([L])
    else:
        widths = np.linspace(p.x_spacing_start, p.x_spacing_end, n_cols)
        widths = widths * (L / widths.sum())
    x_edges = p.x_lower + np.concatenate([[0.0], np.cumsum(widths)])
    x_edges[-1] = p.x_upper  # kill float drift

    # Per-column target y-spacing -> equal rows tiling the full width.
    col_sp = (np.full(n_cols, p.y_spacing_max) if n_cols == 1
              else np.linspace(p.y_spacing_max, p.y_spacing_min, n_cols))

    trees_x, trees_y, row_edges = [], [], []
    for i in range(n_cols):
        xl, xr = x_edges[i], x_edges[i + 1]
        n_rows = max(1, int(round(y_width / col_sp[i])))
        y_e = np.linspace(p.y_lower, p.y_upper, n_rows + 1)
        row_edges.append(y_e)

        trees_x.append(xl + rng.random(n_rows) * (xr - xl))
        trees_y.append(y_e[:-1] + rng.random(n_rows) * np.diff(y_e))

    return LatinForestLayout(
        trees_x=np.concatenate(trees_x),
        trees_y=np.concatenate(trees_y),
        x_edges=x_edges,
        row_edges=row_edges,
        n_cols=n_cols,
    )


def plot_latin_forest(p: LatinForestParams, out_path: str, title: str) -> str:
    """Render the cell grid (dotted gray) + one tree per cell to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = build_latin_forest(p)

    fig, ax = plt.subplots(figsize=(16, 6))
    for i in range(layout.n_cols):
        xl, xr = layout.x_edges[i], layout.x_edges[i + 1]
        for yb in layout.row_edges[i]:                       # row lines
            ax.plot([xl, xr], [yb, yb], color="0.7", ls=":", lw=0.6, zorder=1)
        ax.plot([xr, xr], [p.y_lower, p.y_upper],            # column edge
                color="0.7", ls=":", lw=0.6, zorder=1)
    ax.plot([p.x_lower, p.x_lower], [p.y_lower, p.y_upper],
            color="0.7", ls=":", lw=0.6, zorder=1)

    ax.scatter(layout.trees_x, layout.trees_y, s=6, color="forestgreen",
               zorder=3, label=f"{layout.trees_x.size} trees (1 / cell)")

    ax.set_xlim(p.x_lower - 5, p.x_upper + 5)
    ax.set_ylim(p.y_lower - 5, p.y_upper + 5)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x (m) — forward axis, {p.x_upper - p.x_lower:g} m "
                  f"(col width {p.x_spacing_start:g} -> {p.x_spacing_end:g})")
    ax.set_ylabel(f"y (m) — lateral, {p.y_upper - p.y_lower:g} m wide")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"n_cols={layout.n_cols}  trees={layout.trees_x.size}  ->  {out_path}")
    return out_path


# Variants requested during design review.
VARIANTS = {
    "latin_forest_x15-5_y15-5.png": (
        LatinForestParams(y_spacing_max=15.0, y_spacing_min=5.0),
        "'latin' forest — x-spacing 15->5, y-spacing 15->5 "
        "(cells tile 300 x 100 m, 1 tree/cell)",
    ),
    "latin_forest_x15-5_y3.png": (
        LatinForestParams(y_spacing_max=3.0, y_spacing_min=3.0),
        "'latin' forest — x-spacing 15->5, y-spacing constant 3 "
        "(cells tile 300 x 100 m, 1 tree/cell)",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 'latin' forest preview PNGs.")
    ap.add_argument("--out-dir", default=REPO_ROOT,
                    help="directory to write the preview PNGs (default: repo root)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (params, title) in VARIANTS.items():
        plot_latin_forest(params, os.path.join(args.out_dir, name), title)


if __name__ == "__main__":
    main()
