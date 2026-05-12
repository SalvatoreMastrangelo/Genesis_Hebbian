from __future__ import annotations

"""URDF catalog generation helpers for general-policy workflows."""

from pathlib import Path
import shutil
from typing import List, Optional, Sequence, Set, Tuple
import random

import numpy as np
import genesis as gs

from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
from winged_drone_train.urdf_resolver import _infer_genome_for_missing_urdf, _parse_physical_genome_from_name
from drone_making import UrdfMaker
from morph_evolution.chromosome_drone import Chromosome_Drone


def _write_catalog_txt(catalog_dir: Path, urdfs: List[Path]) -> None:
    """
    Write `catalog.txt` listing all URDF filenames in `catalog_dir`.

    We store only filenames (not absolute paths) so that the catalog is
    portable. `Gen_Env` will resolve them relative to `catalog_dir`.
    """
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_file = catalog_dir / "catalog.txt"

    lines = [p.name for p in urdfs]
    catalog_file.write_text("\n".join(lines))
    print(f"[catalog] Wrote {len(urdfs)} entries to: {catalog_file}")


def _write_genomes_txt(
    catalog_dir: Path,
    urdfs: List[Path],
    genomes: List[Sequence[float]],
) -> None:
    """
    Write `genomes.txt` listing the physical genome of each URDF, one per row.

    Format: ``<urdf_filename>: g0, g1, ..., g14``. Row order matches
    `catalog.txt`, so consumers can either parse the genome columns directly
    or rely on positional alignment with the URDF list.
    """
    catalog_dir.mkdir(parents=True, exist_ok=True)
    genomes_file = catalog_dir / "genomes.txt"

    lines: List[str] = []
    for urdf_path, genome in zip(urdfs, genomes):
        values = ", ".join(f"{float(v):g}" for v in genome)
        lines.append(f"{urdf_path.name}: {values}")
    genomes_file.write_text("\n".join(lines))
    print(f"[catalog] Wrote {len(genomes)} genomes to: {genomes_file}")


def build_catalog(
    catalog_dir: Path,
    n: int,
    seed: int = 0,
    extra_genomes: Optional[Sequence[Sequence[float]]] = None,
    include_standard_mydrone: bool = True,
) -> List[Path]:
    """
    Build a catalog of `n` unique URDFs and write `catalog.txt` in `catalog_dir`.

    Uses:
      - Chromosome_Drone: normalized continuous genome generator in [0, 1]^D
      - Chromosome_Drone.to_physical: mapping genome -> physical parameters
      - UrdfMaker:       physical parameters -> URDF file

    Optionally includes a fixed "reasonable" baseline morphology as the
    first URDF, then samples additional morphologies randomly from the
    continuous design space.

    Parameters
    ----------
    catalog_dir:
        Directory where URDFs and `catalog.txt` will be written.
    n:
        Target number of URDFs to generate.
    seed:
        Random seed for reproducible catalog generation
        (affects both `random` and `numpy.random`).
    """
    catalog_dir = catalog_dir.expanduser().resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)

    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    print(f"[catalog] dir={catalog_dir}  n={n}  seed={seed}")
    random.seed(seed)
    np.random.seed(seed)

    seen: Set[Tuple[float, ...]] = set()
    urdfs: List[Path] = []
    phys_genomes: List[List[float]] = []

    max_attempts = max(n * 25, 200)
    attempts = 0
    dupes = 0

    while len(urdfs) < n and attempts < max_attempts:
        attempts += 1

        if include_standard_mydrone and len(urdfs) == 0:
            # A known "reasonable" baseline morphology expressed directly
            # in physical parameter space.
            phys_genome = list(STANDARD_MYDRONE_GENOME)
        else:
            # Sample a random genome in [0, 1]^D and map to physical space.
            genome_norm = Chromosome_Drone.random_genome()
            phys_genome = Chromosome_Drone.to_physical(genome_norm)

        # Use the physical genome as uniqueness key (URDF geometry).
        key = tuple(float(v) for v in phys_genome)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)

        # Build URDF from physical parameters.
        path_str = UrdfMaker(phys_genome, out_dir=catalog_dir).create_urdf()
        urdf_path = Path(path_str).resolve()
        urdfs.append(urdf_path)
        phys_genomes.append([float(v) for v in phys_genome])

        step = max(1, n // 20)
        if len(urdfs) % step == 0 or len(urdfs) == n:
            print(f"[catalog]   {len(urdfs)}/{n} URDF generated")

    extra_genomes = extra_genomes or []
    for genome in extra_genomes:
        phys_genome = list(genome)
        if len(phys_genome) != 15:
            raise ValueError("Expected a 15-value genome sequence.")
        key = tuple(float(v) for v in phys_genome)
        if key in seen:
            continue
        seen.add(key)
        path_str = UrdfMaker(phys_genome, out_dir=catalog_dir).create_urdf()
        urdf_path = Path(path_str).resolve()
        urdfs.append(urdf_path)
        phys_genomes.append([float(v) for v in phys_genome])

    _write_catalog_txt(catalog_dir, urdfs)
    _write_genomes_txt(catalog_dir, urdfs, phys_genomes)
    print(f"[catalog] Attempts={attempts}  Duplicates={dupes}  Unique={len(urdfs)}")

    if len(urdfs) < n:
        print(f"[catalog] WARNING: generated only {len(urdfs)}/{n} URDFs.")

    return urdfs


def write_single_urdf_catalog(
    catalog_dir: Path,
    urdf_path: Path | str,
    *,
    drone_key: Optional[str] = None,
) -> Path:
    """
    Mirror the multi-URDF catalog layout for a single-URDF run.

    Copies the URDF into `catalog_dir` and writes `catalog.txt` and
    `genomes.txt` so single-URDF runs produce the same catalog artifacts
    as multi-URDF runs. The physical genome is inferred from the filename
    or, for known defaults like the standard mydrone, from
    `STANDARD_MYDRONE_GENOME`. If no genome can be inferred (e.g. lisparrow),
    `genomes.txt` is still written but with a `<unknown>` placeholder.
    """
    catalog_dir = Path(catalog_dir).expanduser().resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)

    src = Path(urdf_path).expanduser().resolve()
    dst = catalog_dir / src.name
    if src != dst:
        shutil.copy2(src, dst)

    # Mirror UrdfMaker._ensure_support_files: copy the aero sidecars and the
    # meshes folder so the catalog is self-contained (matches the layout
    # produced by build_catalog() for multi-URDF runs).
    src_dir = src.parent
    for name in ("aero_parameters.yaml", "actuators.csv"):
        sidecar = src_dir / name
        if sidecar.is_file():
            target = catalog_dir / name
            if not target.exists():
                shutil.copy2(sidecar, target)
    src_meshes = src_dir / "meshes"
    dst_meshes = catalog_dir / "meshes"
    if src_meshes.is_dir() and not dst_meshes.exists():
        shutil.copytree(src_meshes, dst_meshes)

    genome = _parse_physical_genome_from_name(src.name)
    if genome is None:
        genome = _infer_genome_for_missing_urdf(src, drone_key=drone_key)

    _write_catalog_txt(catalog_dir, [dst])
    if genome is None:
        genomes_file = catalog_dir / "genomes.txt"
        genomes_file.write_text(f"{dst.name}: <unknown>")
        print(f"[catalog] Wrote 1 entry (genome unknown) to: {genomes_file}")
    else:
        _write_genomes_txt(catalog_dir, [dst], [list(genome)])
    return catalog_dir


__all__ = ["build_catalog", "write_single_urdf_catalog"]
