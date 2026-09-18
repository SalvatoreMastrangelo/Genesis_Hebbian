"""Still renders of the standard drone for thesis section 4.3.

Reuses the champion-render helpers (same camera, colours and framing as the
champion renders of chapter 6). Run inside the Genesis docker image from the
repository root:

    docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/workspace/bind -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
        mygenesis:latest python tesis/images/04_Simulation_Environment/render_standard_drone.py
"""
import math
import tempfile
from pathlib import Path

import numpy as np

import genesis as gs
from drone_making import UrdfMaker
from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
from WP2_Outer_Loop.render_champions import (
    AZIMUTH_DEG, ELEVATION_DEG, DEFAULT_RES, _build, _distances, _shoot,
)

OUT = Path(__file__).resolve().parent
RES = DEFAULT_RES

# joint name -> angle [rad] for the "actuated" pose (within the URDF limits)
ACTUATED = {
    "joint_0_sweep_left_wing": None,    # filled below from the URDF limits
    "joint_0_sweep_right_wing": None,
    "joint_1_twist_left_wing": None,
    "joint_1_twist_right_wing": None,
    "elevator_pitch_joint": None,
    "rudder_yaw_joint": None,
}


def crop_pairs(margin: int = 25) -> None:
    """Crop the white border; the same box for the neutral and the actuated
    image of each view, so that the two keep the same scale and registration."""
    from PIL import Image

    for view in ("3quarter", "top"):
        files = [OUT / f"standard_drone_{tag}_{view}.png" for tag in ("neutral", "actuated")]
        imgs = [Image.open(f).convert("RGB") for f in files]
        boxes = []
        for im in imgs:
            a = np.asarray(im)
            ys, xs = np.where((a < 250).any(axis=2))
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        x0 = max(0, min(b[0] for b in boxes) - margin)
        y0 = max(0, min(b[1] for b in boxes) - margin)
        x1 = min(imgs[0].width, max(b[2] for b in boxes) + margin)
        y1 = min(imgs[0].height, max(b[3] for b in boxes) + margin)
        for f, im in zip(files, imgs):
            im.crop((x0, y0, x1, y1)).save(f)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="standard_drone_"))
    urdf = Path(UrdfMaker(list(STANDARD_MYDRONE_GENOME), out_dir=work).create_urdf(filename="standard.urdf"))
    gs.init(logging_level="warning")
    try:
        scene, _, lo, hi = _build(urdf, RES)
        scene.destroy()
        size = hi - lo
        print(f"bbox x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f} m")
        d34, d_top = _distances([size], RES)
        az, el = math.radians(AZIMUTH_DEG), math.radians(ELEVATION_DEG)

        scene, cam, lo, hi = _build(urdf, RES)
        ent = scene.entities[-1]
        c = (lo + hi) / 2
        pos34 = c + d34 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

        def shoot(tag: str) -> None:
            cam.set_pose(pos=pos34, lookat=c, up=(0, 0, 1))
            _shoot(cam, OUT / f"standard_drone_{tag}_3quarter.png")
            cam.set_pose(pos=c + np.array([0.0, 0.0, d_top]), lookat=c, up=(1, 0, 0))
            _shoot(cam, OUT / f"standard_drone_{tag}_top.png")

        shoot("neutral")

        # actuated pose: both wings swept back to their limit, twisted in opposite senses,
        # elevator and rudder deflected, to show what each joint moves
        names = list(ACTUATED)
        idx, lim = [], []
        for n in names:
            j = ent.get_joint(n)
            di = j.dofs_idx_local
            di = di[0] if isinstance(di, (list, tuple)) else di
            idx.append(int(di))
        lo_l, hi_l = ent.get_dofs_limit(idx)
        lo_l = np.asarray(lo_l.cpu() if hasattr(lo_l, "cpu") else lo_l, dtype=float).reshape(-1)
        hi_l = np.asarray(hi_l.cpu() if hasattr(hi_l, "cpu") else hi_l, dtype=float).reshape(-1)
        for n, a, b in zip(names, lo_l, hi_l):
            print(f"{n}: [{a:+.3f}, {b:+.3f}] rad")
        signs = {
            "joint_0_sweep_left_wing": -1, "joint_0_sweep_right_wing": +1,   # both wings swept back
            "joint_1_twist_left_wing": +1, "joint_1_twist_right_wing": -1,
            "elevator_pitch_joint": +1, "rudder_yaw_joint": +1,
        }
        target = np.array([hi_l[i] if signs[n] > 0 else lo_l[i] for i, n in enumerate(names)], dtype=np.float32)
        ent.set_dofs_position(target, idx, zero_velocity=True)
        try:
            scene.visualizer.update_visual_states()
        except Exception as exc:  # pragma: no cover - API differences
            print("visual update:", exc)
        shoot("actuated")
        scene.destroy()
        crop_pairs()
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
