"""Bake a URDF into a single 3D object file (GLB / STL / OBJ / PLY).

Walks the kinematic tree, resolves ``package://`` mesh paths relative to the
URDF directory, applies per-link forward kinematics at a chosen joint pose and
concatenates every visual (or collision) geometry into one scene.

Examples
--------
    python -m multi_urdf_utils.urdf_to_mesh path/to/ind_000.urdf -o drone.glb
    python -m multi_urdf_utils.urdf_to_mesh path/to/ind_000.urdf -o drone.stl --merge
    python -m multi_urdf_utils.urdf_to_mesh path/to/ind_000.urdf -o hull.glb --collision
    python -m multi_urdf_utils.urdf_to_mesh path/to/ind_000.urdf -o swept.glb \
        --joint joint_0_sweep_left_wing=0.2 --joint joint_0_sweep_right_wing=-0.2
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, rotation_matrix

DEFAULT_RGBA = (0.6, 0.6, 0.6, 1.0)


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in text.replace(",", " ").split()], dtype=float)


def _origin_matrix(elem: ET.Element | None) -> np.ndarray:
    """<origin xyz=... rpy=.../> -> 4x4. URDF rpy is extrinsic XYZ (Rz@Ry@Rx)."""
    if elem is None:
        return np.eye(4)
    origin = elem.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = _floats(origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _floats(origin.get("rpy"), (0.0, 0.0, 0.0))
    mat = euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")
    mat[:3, 3] = xyz
    return mat


def _joint_motion(joint: ET.Element, value: float) -> np.ndarray:
    """Transform contributed by a joint displaced by `value` from its zero pose."""
    jtype = joint.get("type", "fixed")
    if jtype in ("fixed", "floating", "planar") or value == 0.0:
        return np.eye(4)
    axis_elem = joint.find("axis")
    axis = _floats(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(4)
    axis = axis / norm
    if jtype == "prismatic":
        mat = np.eye(4)
        mat[:3, 3] = axis * value
        return mat
    return rotation_matrix(value, axis)  # revolute / continuous


def _resolve_mesh_path(filename: str, urdf_dir: Path) -> Path:
    """package://meshes/x.stl, file://..., or a plain relative/absolute path."""
    for prefix in ("package://", "model://", "file://"):
        if filename.startswith(prefix):
            filename = filename[len(prefix) :]
            break
    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (urdf_dir, urdf_dir.parent):
        probe = base / candidate
        if probe.exists():
            return probe
        # tolerate a leading package name: package://mydrone/meshes/x.stl
        if len(candidate.parts) > 1:
            probe = base / Path(*candidate.parts[1:])
            if probe.exists():
                return probe
    raise FileNotFoundError(f"cannot resolve mesh {filename!r} relative to {urdf_dir}")


def _geometry_to_mesh(geom: ET.Element, urdf_dir: Path) -> tuple[trimesh.Trimesh, np.ndarray] | None:
    """Return (mesh, extra_transform). Extra transform carries non-uniform mesh scale."""
    mesh_elem = geom.find("mesh")
    if mesh_elem is not None:
        path = _resolve_mesh_path(mesh_elem.get("filename", ""), urdf_dir)
        loaded = trimesh.load(path, force="mesh", process=False)
        scale = _floats(mesh_elem.get("scale"), (1.0, 1.0, 1.0))
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        scale_mat = np.diag([*scale, 1.0])
        return loaded, scale_mat

    box = geom.find("box")
    if box is not None:
        return trimesh.creation.box(extents=_floats(box.get("size"), (1.0, 1.0, 1.0))), np.eye(4)

    cyl = geom.find("cylinder")
    if cyl is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cyl.get("radius", 1.0)),
            height=float(cyl.get("length", 1.0)),
            sections=64,
        )
        return mesh, np.eye(4)

    sph = geom.find("sphere")
    if sph is not None:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(sph.get("radius", 1.0))), np.eye(4)

    return None


def _material_table(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    table: dict[str, tuple[float, float, float, float]] = {}
    for mat in root.findall("material"):
        name = mat.get("name")
        color = mat.find("color")
        if name and color is not None:
            rgba = _floats(color.get("rgba"), DEFAULT_RGBA)
            table[name] = tuple(float(v) for v in rgba[:4])
    return table


def _visual_rgba(elem: ET.Element, table: dict[str, tuple[float, float, float, float]]):
    mat = elem.find("material")
    if mat is None:
        return DEFAULT_RGBA
    color = mat.find("color")
    if color is not None:
        return tuple(float(v) for v in _floats(color.get("rgba"), DEFAULT_RGBA)[:4])
    return table.get(mat.get("name", ""), DEFAULT_RGBA)


def forward_kinematics(root: ET.Element, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
    """World transform of every link, starting from the link that is never a child."""
    joints = root.findall("joint")
    children = {j.find("child").get("link"): j for j in joints if j.find("child") is not None}
    links = [ln.get("name") for ln in root.findall("link")]

    transforms: dict[str, np.ndarray] = {}

    def resolve(link: str, seen: frozenset[str] = frozenset()) -> np.ndarray:
        if link in transforms:
            return transforms[link]
        if link in seen:
            raise ValueError(f"cycle in URDF kinematic tree at link {link!r}")
        joint = children.get(link)
        if joint is None:  # root
            transforms[link] = np.eye(4)
            return transforms[link]
        parent = joint.find("parent").get("link")
        name = joint.get("name", "")
        mat = (
            resolve(parent, seen | {link})
            @ _origin_matrix(joint)
            @ _joint_motion(joint, joint_values.get(name, 0.0))
        )
        transforms[link] = mat
        return mat

    for link in links:
        resolve(link)
    return transforms


def urdf_to_scene(
    urdf_path: str | Path,
    *,
    use_collision: bool = False,
    joint_values: dict[str, float] | None = None,
) -> trimesh.Scene:
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()
    materials = _material_table(root)
    transforms = forward_kinematics(root, joint_values or {})

    tag = "collision" if use_collision else "visual"
    scene = trimesh.Scene()
    for link in root.findall("link"):
        link_name = link.get("name", "link")
        link_T = transforms.get(link_name, np.eye(4))
        for idx, elem in enumerate(link.findall(tag)):
            geom = elem.find("geometry")
            if geom is None:
                continue
            built = _geometry_to_mesh(geom, urdf_dir)
            if built is None:
                continue
            mesh, scale_mat = built
            mesh = mesh.copy()
            mesh.apply_transform(link_T @ _origin_matrix(elem) @ scale_mat)
            rgba = _visual_rgba(elem, materials) if not use_collision else (0.2, 0.6, 0.9, 0.45)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh, face_colors=np.tile(np.asarray(rgba) * 255, (len(mesh.faces), 1))
            )
            scene.add_geometry(mesh, node_name=f"{link_name}_{tag}_{idx}", geom_name=f"{link_name}_{tag}_{idx}")
    if not scene.geometry:
        raise ValueError(f"no <{tag}> geometry found in {urdf_path}")
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .glb/.stl/.obj/.ply")
    ap.add_argument("--collision", action="store_true", help="bake collision geometry instead of visual")
    ap.add_argument("--merge", action="store_true", help="concatenate into one mesh (required for .stl)")
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniform output scale; URDF is metres, so use 1000 for mm-based slicers",
    )
    ap.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=RAD",
        help="displace a joint from its zero pose (repeatable)",
    )
    ap.add_argument(
        "--up",
        choices=("auto", "y", "z"),
        default="auto",
        help="up axis of the output. 'auto' (default) writes Y-up for glTF/GLB, "
        "which the spec mandates, and keeps the URDF's native Z-up otherwise",
    )
    args = ap.parse_args()

    joint_values = {}
    for spec in args.joint:
        name, _, value = spec.partition("=")
        joint_values[name.strip()] = float(value)

    scene = urdf_to_scene(args.urdf, use_collision=args.collision, joint_values=joint_values)
    if args.scale != 1.0:
        scene.apply_scale(args.scale)

    suffix = args.out.suffix.lower()

    # URDF is Z-up; glTF/GLB declares +Y up, so a verbatim export shows up
    # tipped 90 deg about X in every viewer. Rotating -90 deg about X sends
    # URDF +Z to glTF +Y (and URDF +Y to glTF -Z, the usual forward).
    up = args.up
    if up == "auto":
        up = "y" if suffix in (".glb", ".gltf") else "z"
    if up == "y":
        scene.apply_transform(rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0]))

    obj = scene
    if args.merge or suffix in (".stl", ".ply"):
        obj = trimesh.util.concatenate(list(scene.dump()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    obj.export(args.out)

    extents = scene.extents
    unit = "m" if args.scale == 1.0 else f"m*{args.scale:g}"
    print(f"{args.urdf.name} -> {args.out}")
    print(f"  parts   : {len(scene.geometry)}")
    print(f"  up axis : {up.upper()}" + ("  (rotated from URDF Z-up)" if up == "y" else ""))
    print(f"  bbox ({unit}): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f}")
    if isinstance(obj, trimesh.Trimesh):
        print(f"  faces   : {len(obj.faces)}  watertight={obj.is_watertight}")


if __name__ == "__main__":
    main()
