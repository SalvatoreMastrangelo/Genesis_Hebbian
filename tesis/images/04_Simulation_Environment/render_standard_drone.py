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


# --------------------------------------------------------------------------- #
# Actuation figure: soft lighting, high resolution, neutral pose as a ghost
# --------------------------------------------------------------------------- #
ACT_RES = (2400, 1800)
ACT_AZ, ACT_EL = 30.0, 40.0


def _nice_scene(urdf: Path):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=1),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0], show_world_frame=False, show_link_frame=False,
            background_color=(1.0, 1.0, 1.0), ambient_light=(0.55, 0.55, 0.55),
            shadow=False, plane_reflection=False,
            lights=[
                {"type": "directional", "dir": (-0.5, -0.35, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 2.2},
                {"type": "directional", "dir": (0.8, 0.6, -0.4), "color": (1.0, 1.0, 1.0), "intensity": 1.0},
            ],
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
        show_viewer=False, renderer=gs.renderers.Rasterizer(),
    )
    ent = scene.add_entity(gs.morphs.URDF(file=str(urdf), pos=(0, 0, 0), quat=(1, 0, 0, 0), collision=False,
                                          merge_fixed_links=True, prioritize_urdf_material=True))
    cam = scene.add_camera(res=ACT_RES, pos=(2, 1, 1), lookat=(0, 0, 0), up=(0, 0, 1),
                           fov=14.0, near=0.05, far=30.0, debug=True)
    scene.build(n_envs=0)
    return scene, ent, cam


def _rgb_mask(cam):
    from genesis.utils.misc import tensor_to_array
    rgb, _, seg, _ = cam.render(rgb=True, depth=False, segmentation=True, normal=False,
                                antialiasing=True, force_render=True)
    rgb = tensor_to_array(rgb); seg = tensor_to_array(seg)
    if rgb.ndim == 4:
        rgb, seg = rgb[0], seg[0]
    rgb = rgb[..., :3].astype(np.float32)
    seg = np.asarray(seg).reshape(rgb.shape[0], rgb.shape[1], -1)[..., 0]
    return rgb, seg != seg[0, 0]


def actuation_figure() -> None:
    from PIL import Image, ImageFilter

    work = Path(tempfile.mkdtemp(prefix="standard_drone_act_"))
    urdf = Path(UrdfMaker(list(STANDARD_MYDRONE_GENOME), out_dir=work).create_urdf(filename="standard.urdf"))
    scene, ent, cam = _nice_scene(urdf)
    try:
        from genesis.utils.misc import tensor_to_array
        verts = np.concatenate([tensor_to_array(l.get_vverts()).reshape(-1, 3) for l in ent.links if l.n_vverts > 0])
        c = (verts.min(0) + verts.max(0)) / 2
        az, el = math.radians(ACT_AZ), math.radians(ACT_EL)
        dist = 5.6   # long lens: nearly orthographic, no wide-angle distortion
        pos = c + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        cam.set_pose(pos=pos, lookat=c, up=(0, 0, 1))
        rgb_n, m_n = _rgb_mask(cam)

        names = ["joint_0_sweep_left_wing", "joint_0_sweep_right_wing", "joint_1_twist_left_wing",
                 "joint_1_twist_right_wing", "elevator_pitch_joint", "rudder_yaw_joint"]
        signs = [-1, +1, +1, -1, +1, +1]          # both panels swept back, opposite twist
        idx = []
        for n in names:
            di = ent.get_joint(n).dofs_idx_local
            idx.append(int(di[0] if isinstance(di, (list, tuple)) else di))
        lo_l, hi_l = ent.get_dofs_limit(idx)
        lo_l = np.asarray(lo_l.cpu() if hasattr(lo_l, "cpu") else lo_l, dtype=float).reshape(-1)
        hi_l = np.asarray(hi_l.cpu() if hasattr(hi_l, "cpu") else hi_l, dtype=float).reshape(-1)
        target = np.array([hi_l[i] if sg > 0 else lo_l[i] for i, sg in enumerate(signs)], dtype=np.float32)
        ent.set_dofs_position(target, idx, zero_velocity=True)
        try:
            scene.visualizer.update_visual_states()
        except Exception as exc:  # pragma: no cover
            print("visual update:", exc)
        rgb_a, m_a = _rgb_mask(cam)
    finally:
        scene.destroy()

    def soft(mask):
        im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
        return (np.asarray(im).astype(np.float32) / 255.0)[..., None]

    white = np.full_like(rgb_a, 255.0)
    # neutral pose as a pale ghost: grey silhouette with a slightly darker rim
    grey = rgb_n.mean(axis=2, keepdims=True)
    ghost_fill = 0.16 * np.repeat(grey, 3, axis=2) + 0.84 * 255.0
    a_n = soft(m_n)
    out = white * (1 - a_n) + ghost_fill * a_n
    edge = np.asarray(Image.fromarray((m_n * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
                      .filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)[..., None] / 255.0
    out = out * (1 - 0.55 * np.clip(edge * 2.0, 0, 1)) + 150.0 * 0.55 * np.clip(edge * 2.0, 0, 1)
    a_a = soft(m_a)
    out = out * (1 - a_a) + rgb_a * a_a

    union = m_n | m_a
    ys, xs = np.where(union)
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(out.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(out.shape[1], xs.max() + pad)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)[y0:y1, x0:x1])
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)   # supersampling
    img.save(OUT / "standard_drone_actuation.png")
    print("wrote", OUT / "standard_drone_actuation.png", img.size)


if __name__ == "__main__":
    import sys
    if "--actuation-only" in sys.argv:
        gs.init(logging_level="warning")
        try:
            actuation_figure()
        finally:
            gs.destroy()
    else:
        main()
        gs.init(logging_level="warning")
        try:
            actuation_figure()
        finally:
            gs.destroy()
