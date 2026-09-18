"""Top-view figures of the forest generators for thesis section 4.2.

Uses the project's real ForestGenerator (loaded by path, torch on CPU).
"""
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon, Rectangle

REPO = Path("/home/salvatore/Desktop/code/hebbian/Genesis_Hebbian")
OUT = REPO / "tesis/images/04_Simulation_Environment"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "forest", REPO / "src/winged_drone_train/perception/forest.py")
forest = importlib.util.module_from_spec(spec)
sys.modules["forest"] = forest
spec.loader.exec_module(forest)

R = 0.75          # tree radius [m]
SPAN = 1.4        # reference drone wingspan used by the collision test [m] (two halves of 0.7 m)
TREE = "#0f8a5f"  # validated with the dataviz palette checker (with ACCENT)
ACCENT = "#D55E00"
INK = "#222222"
MUTED = "#8a8a8a"
GRID = "#c9c9c9"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#555555",
    "pdf.fonttype": 42,
})
W_IN = 6.0  # figure width [in] (= \textwidth of the thesis, 15 cm)


def gen(mode, seed, **kw):
    torch.manual_seed(seed)
    cfg = forest.ForestConfig(**kw)
    cfg.forest_mode = mode
    g = forest.ForestGenerator(num_envs=1, evaluation=True, unique_forests_eval=True, config=cfg)
    cyl, _ = g.generate()
    xy = cyl[0, :, :2].numpy()
    # drop the padding trees parked outside the corridor
    return xy[np.abs(xy[:, 1]) <= 50.0 + 1e-6]


def draw_trees(ax, xy, color=TREE, zorder=3, radius=R):
    pc = PatchCollection([Circle((x, y), radius) for x, y in xy],
                         facecolor=color, edgecolor="none", zorder=zorder)
    ax.add_collection(pc)


def style_course(ax, x0, x1, start=(-30.0, 0.0), ylab=True):
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-54, 54)
    # lateral limits of the corridor
    for y in (-50, 50):
        ax.plot([x0, x1], [y, y], color=INK, lw=0.9, zorder=2)
    ax.set_yticks([-50, 0, 50])
    if ylab:
        ax.set_ylabel("y [m]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if start is not None:
        ax.plot(*start, marker=">", ms=4.5, color=ACCENT, zorder=5, mec="white", mew=0.4)
        ax.annotate("", xy=(start[0] + 22, start[1]), xytext=(start[0] + 3, start[1]),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.9, mutation_scale=7), zorder=5)
        ax.text(start[0], start[1] + 5.5, "start", color=INK, ha="center", va="bottom", fontsize=7)


# --------------------------------------------------------------------------- #
# 1. Uniform forest
# --------------------------------------------------------------------------- #
def fig_uniform():
    xy = gen("uniform", seed=3, x_lower=0.0, x_upper=300.0, num_trees=975)
    x0, x1 = -40, 310
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 108 / (x1 - x0) + 0.35))
    draw_trees(ax, xy)
    style_course(ax, x0, x1)
    ax.set_xlabel("x [m]")
    ax.set_xticks(np.arange(0, 301, 50))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "forest_uniform.pdf")
    fig.savefig(OUT / "forest_uniform.png", dpi=300)
    plt.close(fig)
    return len(xy)


# --------------------------------------------------------------------------- #
# 2. Growing forest (+ trees per 10 m against the linear profile)
# --------------------------------------------------------------------------- #
def fig_growing():
    L, d0, d1 = 300.0, 1.0, 5.5
    xy = gen("growing", seed=11, x_lower=0.0, x_upper=L, dens_min=d0, dens_max=d1)
    x0, x1 = -40, 310
    h_map = W_IN * 108 / (x1 - x0)
    fig, (ax_h, ax) = plt.subplots(
        2, 1, figsize=(W_IN, h_map + 1.45), sharex=True,
        gridspec_kw=dict(height_ratios=[0.95, h_map], hspace=0.08))
    # top strip: trees per 10 m of course vs expected
    edges = np.arange(0, L + 1e-6, 10.0)
    counts, _ = np.histogram(xy[:, 0], bins=edges)
    ax_h.bar(edges[:-1] + 5.0, counts, width=8.4, color=TREE, alpha=0.55, linewidth=0, zorder=2)
    xs = np.linspace(0, L, 200)
    ax_h.plot(xs, 10.0 * (d0 + (d1 - d0) * xs / L), color=INK, lw=1.0, zorder=3)
    ax_h.set_ylabel("trees per 10 m")
    ax_h.set_ylim(0, 72)
    ax_h.set_yticks([0, 30, 60])
    for s in ("top", "right"):
        ax_h.spines[s].set_visible(False)
    ax_h.text(150, 10.0 * (d0 + (d1 - d0) * 0.5) + 9, r"$10\,\lambda(x)$", color=INK, ha="right", va="bottom", fontsize=8)
    ax_h.tick_params(labelbottom=False)
    draw_trees(ax, xy)
    style_course(ax, x0, x1)
    ax.set_xlabel("x [m]")
    ax.set_xticks(np.arange(0, 301, 50))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.985, bottom=0.115)
    fig.savefig(OUT / "forest_growing.pdf")
    fig.savefig(OUT / "forest_growing.png", dpi=300)
    plt.close(fig)
    return len(xy)


# --------------------------------------------------------------------------- #
# 3. Stratified ("Latin") forest with its cells + random vs stratified zoom
# --------------------------------------------------------------------------- #
def latin_cells(x_lower, L, s0, s1, ysp0, ysp1, y_lower=-50.0, y_upper=50.0):
    """Replicates the cell construction of ForestGenerator._sample_latin_forest."""
    n_cols = max(1, int(round(2.0 * L / (s0 + s1))))
    widths = np.linspace(s0, s1, n_cols)
    widths = widths * (L / widths.sum())
    x_edges = x_lower + np.concatenate([[0.0], np.cumsum(widths)])
    col_sp = np.linspace(ysp0, ysp1, n_cols)
    rows = []
    for i in range(n_cols):
        n_rows = max(1, int(round((y_upper - y_lower) / col_sp[i])))
        rows.append(np.linspace(y_lower, y_upper, n_rows + 1))
    return x_edges, rows


def close_pairs(xy, thr):
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    iu = np.triu_indices(len(xy), k=1)
    m = d[iu] < thr
    return np.stack([iu[0][m], iu[1][m]], axis=1)


def zoom_panel(ax, xy, box, title):
    (bx0, bx1), (by0, by1) = box
    m = (xy[:, 0] > bx0 - 3) & (xy[:, 0] < bx1 + 3) & (xy[:, 1] > by0 - 3) & (xy[:, 1] < by1 + 3)
    pts = xy[m]
    closed = close_pairs(pts, 2 * R + SPAN)
    overl = close_pairs(pts, 2 * R)
    bad = np.zeros(len(pts), bool)
    bad[overl.ravel()] = True
    # gaps narrower than the wingspan: a bar between the two trunks
    if len(closed):
        ax.add_collection(LineCollection([[pts[i], pts[j]] for i, j in closed],
                                         colors=ACCENT, linewidths=1.6, zorder=2, capstyle="round"))
    draw_trees(ax, pts[~bad], color=TREE, zorder=3)
    if bad.any():
        ax.add_collection(PatchCollection([Circle(p, R) for p in pts[bad]], facecolor=ACCENT,
                                          edgecolor=INK, linewidth=0.5, zorder=4))
    ax.set_aspect("equal")
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("x [m]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    inside = (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1) & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
    return int(inside.sum())


def fig_latin():
    L, s0, s1, y0, y1 = 300.0, 8.0, 3.4, 8.0, 3.4
    xy = gen("latin", seed=5, x_lower=0.0, x_upper=L, forest_length=L,
             x_spacing_start=s0, x_spacing_end=s1, y_spacing_max=y0, y_spacing_min=y1)
    x_edges, rows = latin_cells(0.0, L, s0, s1, y0, y1)
    x0, x1 = -40, 310
    h_map = W_IN * 108 / (x1 - x0)
    fig, ax = plt.subplots(figsize=(W_IN, h_map + 0.35))
    segs = [[(xe, -50), (xe, 50)] for xe in x_edges]
    for i, ye in enumerate(rows):
        segs += [[(x_edges[i], y), (x_edges[i + 1], y)] for y in ye[1:-1]]
    ax.add_collection(LineCollection(segs, colors=GRID, linewidths=0.3, zorder=1))
    draw_trees(ax, xy)
    style_course(ax, x0, x1)
    ax.set_xlabel("x [m]")
    ax.set_xticks(np.arange(0, 301, 50))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "forest_latin.pdf")
    fig.savefig(OUT / "forest_latin.png", dpi=300)
    plt.close(fig)
    return len(xy)


def fig_compare():
    # same number of trees, one per 5 m x 5 m cell on average, over 100 m x 100 m
    lat = gen("latin", seed=21, x_lower=0.0, x_upper=100.0, forest_length=100.0,
              x_spacing_start=5.0, x_spacing_end=5.0, y_spacing_max=5.0, y_spacing_min=5.0)
    uni = gen("uniform", seed=22, x_lower=0.0, x_upper=100.0, num_trees=len(lat))
    box = ((25.0, 75.0), (-25.0, 25.0))
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.05), sharey=True)
    n_u = zoom_panel(axes[0], uni, box, "independent positions")
    # cells behind the stratified panel
    gx = np.arange(25.0, 75.0 + 1e-6, 5.0)
    gy = np.arange(-25.0, 25.0 + 1e-6, 5.0)
    axes[1].add_collection(LineCollection(
        [[(g, -25), (g, 25)] for g in gx] + [[(25, g), (75, g)] for g in gy],
        colors=GRID, linewidths=0.4, zorder=1))
    n_l = zoom_panel(axes[1], lat, box, "one tree per cell")
    axes[0].set_ylabel("y [m]")
    # legend: shapes carry the meaning, not colour alone
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc=TREE, mec="none", ms=5, label="tree"),
        Line2D([], [], marker="o", ls="none", mfc=ACCENT, mec=INK, mew=0.5, ms=5, label="trunk overlapping another"),
        Line2D([], [], color=ACCENT, lw=1.8, label="gap narrower than the wingspan"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.215, wspace=0.06)
    fig.savefig(OUT / "forest_random_vs_stratified.pdf")
    fig.savefig(OUT / "forest_random_vs_stratified.png", dpi=300)
    plt.close(fig)
    return n_u, n_l


# --------------------------------------------------------------------------- #
# 4. What the drone perceives: depth rays and the collision rectangle
# --------------------------------------------------------------------------- #
def sector_depths(xy, pos, yaw, n_sec=20, cone=math.radians(80.0), rng=30.0):
    """Same rule as DepthSolver: each angular sector reports the distance to the
    nearest trunk surface among the trees whose angular extent overlaps it."""
    half = cone / 2
    sw = cone / n_sec
    c, s = math.cos(yaw), math.sin(yaw)
    d = xy - pos
    xl = c * d[:, 0] + s * d[:, 1]
    yl = -s * d[:, 0] + c * d[:, 1]
    r = np.hypot(xl, yl) + 1e-9
    th = np.arctan2(yl, xl)
    delta = np.arcsin(np.clip(R / r, 0, 1))
    ok = (np.abs(th) <= half) & (r <= rng)
    il = np.clip(np.floor((th - delta + half) / sw).astype(int), 0, n_sec - 1)
    ir = np.clip(np.floor((th + delta + half) / sw).astype(int), 0, n_sec - 1)
    depth = np.full(n_sec, rng)
    for k in np.where(ok)[0]:
        depth[il[k]:ir[k] + 1] = np.minimum(depth[il[k]:ir[k] + 1], max(r[k] - R, 0.0))
    return depth


def fig_sensing():
    from matplotlib.patches import Wedge
    L, d0, d1 = 300.0, 1.0, 5.5
    xy = gen("growing", seed=11, x_lower=0.0, x_upper=L, dens_min=d0, dens_max=d1)
    yaw = math.radians(6.0)
    n_sec, cone, rng = 20, math.radians(80.0), 30.0
    best = None
    for px in np.arange(60.0, 100.0, 1.0):
        for py in np.arange(-12.0, 12.1, 1.0):
            pos = np.array([px, py])
            if np.linalg.norm(xy - pos, axis=1).min() < 5.0:
                continue
            dep = sector_depths(xy, pos, yaw)
            score = abs(int((dep >= rng - 1e-6).sum()) - 7)
            if best is None or score < best[0]:
                best = (score, pos, dep)
    _, pos, dep = best
    box = ((pos[0] - 9.0, pos[0] + 37.0), (pos[1] - 22.0, pos[1] + 22.0))
    fig, ax = plt.subplots(figsize=(W_IN * 0.74, W_IN * 0.74 * 44 / 46 * 0.93))
    sw = math.degrees(cone) / n_sec
    a0 = math.degrees(yaw) - math.degrees(cone) / 2
    for k in range(n_sec):
        blocked = dep[k] < rng - 1e-6
        ax.add_patch(Wedge(pos, dep[k], a0 + k * sw, a0 + (k + 1) * sw,
                           facecolor=ACCENT if blocked else MUTED, alpha=0.30 if blocked else 0.16,
                           edgecolor="white", linewidth=0.5, zorder=2))
    draw_trees(ax, xy)
    hx, hy = R, SPAN / 2 + R
    c, s = math.cos(yaw), math.sin(yaw)
    rotm = np.array([[c, s], [-s, c]])
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rotm + pos
    ax.add_patch(Polygon(corners, closed=True, facecolor="white", edgecolor=INK, lw=0.9, zorder=6))
    wing = np.array([[0, -SPAN / 2], [0, SPAN / 2]]) @ rotm + pos
    ax.plot(wing[:, 0], wing[:, 1], color=INK, lw=1.4, zorder=7, solid_capstyle="butt")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=MUTED, alpha=0.30, label="free sector, 30 m"),
                        Patch(facecolor=ACCENT, alpha=0.40, label="sector stopped by a trunk"),
                        Patch(facecolor=TREE, label="tree"),
                        Patch(facecolor="white", edgecolor=INK, linewidth=0.9, label="collision rectangle")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.5, handlelength=1.4,
               columnspacing=1.6, bbox_to_anchor=(0.54, -0.005))
    ax.set_aspect("equal")
    ax.set_xlim(*box[0])
    ax.set_ylim(*box[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 0.095, 1, 1))
    fig.savefig(OUT / "forest_sensing.pdf")
    fig.savefig(OUT / "forest_sensing.png", dpi=300)
    plt.close(fig)
    return int((dep < rng - 1e-6).sum()), pos.tolist()


if __name__ == "__main__":
    print("uniform trees:", fig_uniform())
    print("growing trees:", fig_growing())
    print("latin trees:", fig_latin())
    print("compare trees in box (independent, stratified):", fig_compare())
    print("rays that hit a trunk:", fig_sensing())
