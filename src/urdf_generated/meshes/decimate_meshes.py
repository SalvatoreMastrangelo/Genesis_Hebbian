#!/usr/bin/env python3
"""
Batch decimate STL meshes in this folder.

Requires: trimesh, fast_simplification
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import fast_simplification as fs
import trimesh


def decimate_mesh(path: Path, out_path: Path, target_faces: int, agg: int) -> tuple[int, int]:
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    src_faces = len(mesh.faces)
    if src_faces <= target_faces:
        mesh.export(out_path)
        return src_faces, src_faces

    verts, faces = fs.simplify(mesh.vertices, mesh.faces, target_count=target_faces, agg=agg)
    low = trimesh.Trimesh(verts, faces, process=False)
    low.export(out_path)
    return src_faces, len(low.faces)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decimate STL meshes in a directory.")
    parser.add_argument("--in-dir", type=Path, default=Path(__file__).parent, help="Input directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: in-place).",
    )
    parser.add_argument("--target-faces", type=int, default=1000, help="Target face count.")
    parser.add_argument("--agg", type=int, default=8, help="Aggressiveness (0-8).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir == in_dir and not args.overwrite:
        args.overwrite = True

    stl_files = sorted(in_dir.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in {in_dir}")
        return 1

    for stl in stl_files:
        out_path = out_dir / stl.name
        if out_path.exists() and not args.overwrite:
            print(f"skip {stl.name} (exists)")
            continue
        try:
            src_faces, dst_faces = decimate_mesh(stl, out_path, args.target_faces, args.agg)
            print(f"{stl.name}: {src_faces} -> {dst_faces}")
        except Exception as exc:
            print(f"{stl.name}: ERROR {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
