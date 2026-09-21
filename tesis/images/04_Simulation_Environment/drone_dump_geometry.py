"""Step 1 of the drone figures of chapter 4 (standard_drone_*.png): dump the visual
geometry of the standard drone, in the world frame, for the neutral and for the
actuated pose. Genesis does the kinematics, so the pose is the one the simulator
would show. Run by make_drone_figures.sh inside the Genesis docker image:

    python drone_dump_geometry.py OUT.npz
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from drone_making import UrdfMaker
from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME

# actuated pose: left panel swept back to its limit, right panel not swept, the two
# twisted in opposite senses, elevator and rudder deflected
# (+1 = upper joint limit, -1 = lower joint limit, 0 = neutral)
POSE = {
    "joint_0_sweep_left_wing": -1, "joint_0_sweep_right_wing": 0,
    "joint_1_twist_left_wing": +1, "joint_1_twist_right_wing": -1,
    "elevator_pitch_joint": +1, "rudder_yaw_joint": +1,
}


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def snapshot(scene, ent):
    """World-frame vertices, faces and colour of every visual geometry."""
    scene.rigid_solver.update_vgeoms()
    parts = []
    for link in ent.links:
        for vg in link.vgeoms:
            pos = np.asarray(tensor_to_array(vg.get_pos()), dtype=np.float64).reshape(3)
            quat = np.asarray(tensor_to_array(vg.get_quat()), dtype=np.float64).reshape(4)
            v = np.asarray(vg.init_vverts, dtype=np.float64).reshape(-1, 3) @ _quat_to_R(quat).T + pos
            f = np.asarray(vg.init_vfaces, dtype=np.int64).reshape(-1, 3)
            col = np.asarray(vg.vmesh.surface.get_texture().color, dtype=float)
            parts.append((link.name, v, f, col))
    # the same vertices the Genesis renders of the other figures are made of
    ref = np.concatenate([tensor_to_array(l.get_vverts()).reshape(-1, 3) for l in ent.links if l.n_vverts > 0])
    assert np.abs(ref - np.concatenate([p[1] for p in parts])).max() < 1e-5
    return parts


def main(out: Path) -> None:
    work = Path(tempfile.mkdtemp(prefix="standard_drone_dump_"))
    urdf = Path(UrdfMaker(list(STANDARD_MYDRONE_GENOME), out_dir=work).create_urdf(filename="standard.urdf"))
    gs.init(logging_level="warning")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            show_viewer=False,
        )
        ent = scene.add_entity(gs.morphs.URDF(file=str(urdf), pos=(0, 0, 0), quat=(1, 0, 0, 0), collision=False,
                                              merge_fixed_links=True, prioritize_urdf_material=True))
        scene.build(n_envs=0)
        poses = {"neutral": snapshot(scene, ent)}

        idx = []
        for n in POSE:
            di = ent.get_joint(n).dofs_idx_local
            idx.append(int(di[0] if isinstance(di, (list, tuple)) else di))
        lo, hi = ent.get_dofs_limit(idx)
        lo = np.asarray(tensor_to_array(lo), dtype=float).reshape(-1)
        hi = np.asarray(tensor_to_array(hi), dtype=float).reshape(-1)
        for n, a, b in zip(POSE, lo, hi):
            print(f"{n}: [{a:+.3f}, {b:+.3f}] rad")
        target = np.array([max(s, 0) * hi[i] - min(s, 0) * lo[i] for i, s in enumerate(POSE.values())],
                          dtype=np.float32)
        ent.set_dofs_position(target, idx, zero_velocity=True)
        poses["actuated"] = snapshot(scene, ent)
        scene.destroy()
    finally:
        gs.destroy()

    data = {}
    for tag, parts in poses.items():
        for i, (link, v, f, col) in enumerate(parts):
            key = f"{tag}|{i}|{link}"
            data[key + "|v"], data[key + "|f"], data[key + "|c"] = v, f, col
    np.savez_compressed(out, **data)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
