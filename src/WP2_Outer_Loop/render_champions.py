"""Render the exam champions of an outer-loop run (still images of the morphology).

Two morphologies per run, taken from the exam-scored rows of
``results/outer_population.csv``:

* **progress champion** — best exam progress over the whole run, i.e. the
  objective-0 record holder of ``pareto_plots._cumulative_champions``.
* **CoT champion** — cheapest exam cost of transport among the rows admitted
  by the run's ``outer.min_progress_m`` gate (``pareto_fronts._admission_mask``,
  the same gate the fronts and hypervolumes use). The ungated CoT record
  holder is a morph that barely flies (a few metres), so it is deliberately
  not the one rendered here.

Both picks are asserted to lie on the gated cumulative front. Each champion's
URDF is rebuilt from its CSV genome (bit-identical to the ``urdfs_gen_*``
``genomes.txt`` entry) and rendered twice — a 3/4 view and a top view (nose
up) — with the camera aimed at the visual bounding box and ONE common scale
for both champions, so wing spans are comparable across images. URDF
materials are honoured (red profiles, grey fuselage, black propeller).

Outputs go to ``<run>/plots/champion_renders/``: ``<stem>_3quarter.png``,
``<stem>_top.png``, the regenerated ``<stem>.urdf`` and ``champions.csv``
(objectives, generation, slot, physical genome).

``pareto_plots.plot_outer_run`` calls :func:`render_run` by default, so a
finished run (``WP2_Outer_Loop.run``) renders its champions along with the
Pareto plots. The renderer reuses a Genesis runtime that is already
initialised (the live run's) and only creates/destroys its own when none is.

Usage (Genesis runtime required — locally via the ``mygenesis`` docker image)::

    python -m WP2_Outer_Loop.render_champions RUN_DIR [RUN_DIR ...] [--work DIR] [--res W H]

``RUN_DIR`` is the timestamped run folder or its synced ``outer_<exp>_rX``
wrapper. Runs whose objectives are not exam-scored are reported and skipped.
"""
from __future__ import annotations

import argparse
import tempfile
import math
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from WP2_Outer_Loop.pareto_fronts import (
    _admission_mask,
    _load_min_progress,
    _nondominated_mask,
)
from WP2_Outer_Loop.pareto_plots import (
    _cumulative_champions,
    _load_objective_specs,
    _objective_source,
    _read_results_csv,
)

FOV_DEG = 35.0          # vertical field of view of the render camera
MARGIN = 1.15           # framing slack around the bounding box
AZIMUTH_DEG = 35.0      # 3/4 view: yaw around +Z (nose is +X)
ELEVATION_DEG = 24.0    # 3/4 view: pitch above the horizontal plane
DEFAULT_RES = (1200, 900)
OUT_SUBDIR = Path("plots") / "champion_renders"


# ----------------------------------------------------------------------------
#  Champion selection (pure pandas — testable without Genesis)
# ----------------------------------------------------------------------------

def select_champions(run_dir: Path) -> Tuple[List[Tuple[str, pd.Series]], float]:
    """``[(label, row), ...]`` for the progress and gated-CoT champions.

    ``label`` is ``progress_champion`` or ``cot_champion_gated<G>m``; ``row``
    is the exam row that set the record. Returns the gate value too.
    Raises ``RuntimeError`` when the run is not exam-scored, has other
    objectives, or a pick is not on the gated cumulative front.
    """
    run_dir = Path(run_dir)
    df = _read_results_csv(run_dir / "results" / "outer_population.csv")
    if df is None:
        raise FileNotFoundError(run_dir / "results" / "outer_population.csv")
    df, source = _objective_source(df)
    if source != "exam":
        raise RuntimeError(f"{run_dir}: objectives are {source!r}, not exam-scored")
    specs = _load_objective_specs(run_dir)
    if len(specs) < 2:
        raise RuntimeError(f"{run_dir}: need two objectives, got {specs}")
    (n0, d0), (n1, d1) = specs[0], specs[1]
    if (n0, d0, n1, d1) != ("progress_m", "maximize", "cost_of_transport", "minimize"):
        raise RuntimeError(f"unexpected objectives {specs}")
    o0, o1 = f"obj_{n0}", f"obj_{n1}"
    if "urdf_file" not in df.columns:
        raise RuntimeError(f"{run_dir}: outer_population.csv has no urdf_file column")

    champs = _cumulative_champions(df, specs).iloc[-1]
    m = ((df["urdf_file"].astype(str) == champs["a_urdf"])
         & (df[o0] == champs["a_obj0"]) & (df[o1] == champs["a_obj1"]))
    if int(m.sum()) != 1:
        raise RuntimeError(f"progress record holder ambiguous ({int(m.sum())} rows)")
    prog_row = df[m].iloc[0]

    min_prog = _load_min_progress(run_dir)
    gated = df[_admission_mask(df, specs, min_prog)]
    if gated.empty:
        raise RuntimeError(f"min_progress_m={min_prog:g} admits no row")
    cot_row = gated.loc[gated[o1].idxmin()]

    pts = np.stack([gated[o0].to_numpy(dtype=float), -gated[o1].to_numpy(dtype=float)], axis=1)
    front = gated[_nondominated_mask(pts)]
    for lab, r in (("progress", prog_row), ("cot", cot_row)):
        on = bool(((front["outer_gen"] == r["outer_gen"])
                   & (front["urdf_idx"] == r["urdf_idx"])).any())
        print(f"[render_champions] {lab} champion: gen {int(r['outer_gen'])} "
              f"{r['urdf_file']} {n0}={r[o0]:.2f} {n1}={r[o1]:.4f} "
              f"on gated front: {on}")
        if not on:
            raise RuntimeError(f"{lab} champion is not on the gated cumulative front")
    print(f"[render_champions] gate min_progress_m={min_prog:g}: "
          f"{len(gated)}/{len(df)} rows admitted, front size {len(front)}")
    return [("progress_champion", prog_row),
            (f"cot_champion_gated{min_prog:g}m", cot_row)], min_prog


def champion_stem(label: str, row: pd.Series) -> str:
    """File stem shared by a champion's PNGs and URDF."""
    return f"{label}_gen{int(row['outer_gen']):02d}_{Path(str(row['urdf_file'])).stem}"


def _distances(sizes: Sequence[np.ndarray], res: Tuple[int, int]) -> Tuple[float, float]:
    """Common camera distances ``(3/4 view, top view)`` that fit every bbox
    size ``(sx, sy, sz)`` in ``sizes`` — one scale for all champions."""
    aspect = res[0] / res[1]
    t = math.tan(math.radians(FOV_DEG / 2))
    el = math.radians(ELEVATION_DEG)
    d34 = d_top = 0.0
    for s in sizes:
        dxy = math.hypot(float(s[0]), float(s[1]))
        # top view: image vertical is x (up=(1,0,0)), horizontal is y
        d_top = max(d_top, MARGIN * max(s[0] / (2 * t), s[1] / (2 * t * aspect)))
        # 3/4 view: xy-diagonal across the width, z plus tilted diagonal down the height
        d34 = max(d34, MARGIN * max(dxy / (2 * t * aspect),
                                    (s[2] * math.cos(el) + dxy * math.sin(el)) / (2 * t)))
    return float(d34), float(d_top)


# ----------------------------------------------------------------------------
#  Rendering (Genesis)
# ----------------------------------------------------------------------------

def _materialize(row: pd.Series, stem: str, work: Path) -> Tuple[Path, List[float]]:
    from drone_making import UrdfMaker
    from morph_evolution.chromosome_drone import Chromosome_Drone

    norm = [float(row[f"g{i}"]) for i in range(15)]
    phys = Chromosome_Drone.to_physical(Chromosome_Drone.snap_genome_norm(norm))
    urdf = Path(UrdfMaker(phys, out_dir=work).create_urdf(filename=stem + ".urdf"))
    return urdf, phys


def _build(urdf: Path, res: Tuple[int, int]):
    """Scene with the URDF and a debug camera in the current Genesis runtime;
    returns ``(scene, camera, bbox_min, bbox_max)`` of the visual geometry."""
    import genesis as gs
    from genesis.utils.misc import tensor_to_array
    from data_processing.top_drone_visualization import TopDroneVisualizer

    scene = TopDroneVisualizer(resolution=res)._build_scene((2, 1, 1), (0, 0, 0))
    ent = scene.add_entity(gs.morphs.URDF(
        file=str(urdf), pos=(0, 0, 0), quat=(1, 0, 0, 0),
        collision=False, merge_fixed_links=True,
        prioritize_urdf_material=True,   # URDF colours win over mesh colours
    ))
    cam = scene.add_camera(res=res, pos=(2, 1, 1), lookat=(0, 0, 0), up=(0, 0, 1),
                           fov=FOV_DEG, near=0.05, far=20.0, debug=True)
    scene.build(n_envs=0)
    verts = np.concatenate([
        tensor_to_array(link.get_vverts()).reshape(-1, 3)
        for link in ent.links if link.n_vverts > 0
    ])
    return scene, cam, verts.min(axis=0), verts.max(axis=0)


def _shoot(cam, path: Path) -> Path:
    import imageio
    from genesis.utils.misc import tensor_to_array

    rgb, *_ = cam.render(rgb=True, depth=False, segmentation=False, normal=False,
                         antialiasing=True, force_render=True)
    arr = tensor_to_array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]      # RGB already; a BGR flip would turn red profiles blue
    imageio.imwrite(path, arr)
    return path


def render_run(run_dir: Path | str, work: Path | str | None = None,
               res: Tuple[int, int] = DEFAULT_RES) -> Path:
    """Render both champions of ``run_dir``; returns the output folder.

    Uses the Genesis runtime that is already initialised when there is one
    (e.g. called from the end of a live run) and otherwise initialises and
    destroys its own. Scenes are always destroyed after use.
    """
    import genesis as gs

    run_dir = Path(run_dir).resolve()
    work = Path(work).resolve() if work else Path(tempfile.mkdtemp(prefix="champion_urdfs_"))
    out = run_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    targets, min_prog = select_champions(run_dir)
    items: List[Dict] = []
    for label, row in targets:
        stem = champion_stem(label, row)
        urdf, phys = _materialize(row, stem, work)
        (out / (stem + ".urdf")).write_bytes(urdf.read_bytes())
        items.append(dict(label=label, row=row, phys=phys, stem=stem, urdf=urdf))

    owns_runtime = not getattr(gs, "_initialized", False)
    if owns_runtime:
        gs.init(logging_level="warning")
    try:
        # pass 1: bounding boxes -> one common scale for every champion
        for it in items:
            scene, _, lo, hi = _build(it["urdf"], res)
            scene.destroy()
            it["lo"], it["hi"] = lo, hi
            print(f"[render_champions] {it['stem']}: bbox x={hi[0]-lo[0]:.3f} "
                  f"y={hi[1]-lo[1]:.3f} z={hi[2]-lo[2]:.3f} m")
        d34, d_top = _distances([it["hi"] - it["lo"] for it in items], res)
        print(f"[render_champions] camera distance: 3/4 view {d34:.2f} m, top view {d_top:.2f} m")

        # pass 2: render, camera aimed at the bbox centre
        az, el = math.radians(AZIMUTH_DEG), math.radians(ELEVATION_DEG)
        records = []
        for it in items:
            scene, cam, lo, hi = _build(it["urdf"], res)
            try:
                c = (lo + hi) / 2
                pos = c + d34 * np.array([math.cos(el) * math.cos(az),
                                          math.cos(el) * math.sin(az), math.sin(el)])
                cam.set_pose(pos=pos, lookat=c, up=(0, 0, 1))
                p34 = _shoot(cam, out / f"{it['stem']}_3quarter.png")
                cam.set_pose(pos=c + np.array([0.0, 0.0, d_top]), lookat=c, up=(1, 0, 0))
                ptop = _shoot(cam, out / f"{it['stem']}_top.png")
            finally:
                scene.destroy()
            r = it["row"]
            print(f"[render_champions] {it['label']} -> {p34.name}, {ptop.name}")
            records.append({
                "label": it["label"], "outer_gen": int(r["outer_gen"]),
                "urdf_idx": int(r["urdf_idx"]), "urdf_file": r["urdf_file"],
                "progress_m": r["obj_progress_m"], "cost_of_transport": r["obj_cost_of_transport"],
                "min_progress_m": min_prog, "png_3quarter": p34.name, "png_top": ptop.name,
                **{f"phys_{i}": v for i, v in enumerate(it["phys"])},
            })
    finally:
        if owns_runtime:
            gs.destroy()
    pd.DataFrame(records).to_csv(out / "champions.csv", index=False)
    print(f"[render_champions] wrote {out / 'champions.csv'}")
    return out


def render_runs(run_dirs: Sequence[Path | str], work: Path | str | None = None,
                res: Tuple[int, int] = DEFAULT_RES) -> Dict[str, str]:
    """Render every run in ``run_dirs`` (wrappers resolved), never stopping
    on one failure. Returns ``{run: "ok" | "failed: <reason>"}`` and prints
    a summary."""
    from WP2_Outer_Loop.pareto_overlay import resolve_run_dir

    status: Dict[str, str] = {}
    for rd in run_dirs:
        key = str(rd)
        try:
            run_dir = resolve_run_dir(rd)
            sub = Path(work) / Path(rd).name if work else None
            print(f"\n[render_champions] === {run_dir} ===")
            render_run(run_dir, sub, res)
            status[key] = "ok"
        except Exception as exc:  # noqa: BLE001 — batch must go on
            status[key] = f"failed: {exc}"
            traceback.print_exc()
    print("\n[render_champions] summary:")
    for k, v in status.items():
        print(f"  {v:>6.6}  {k}" if v == "ok" else f"  FAILED  {k} — {v[8:]}")
    return status


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", metavar="RUN_DIR",
                    help="outer-loop run directory (timestamped folder or its outer_<exp>_rX wrapper)")
    ap.add_argument("--work", default=None,
                    help="where to materialise URDFs (default: a temp dir per run)")
    ap.add_argument("--res", nargs=2, type=int, default=DEFAULT_RES, metavar=("W", "H"))
    a = ap.parse_args(argv)
    status = render_runs(a.run_dirs, a.work, tuple(a.res))
    if any(v != "ok" for v in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
