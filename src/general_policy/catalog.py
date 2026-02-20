from __future__ import annotations

"""URDF catalog generation helpers for general-policy workflows."""

from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
import random

import numpy as np
import genesis as gs

from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
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


def build_catalog(
    catalog_dir: Path,
    n: int,
    seed: int = 0,
    extra_genomes: Optional[Sequence[Sequence[float]]] = None,
) -> List[Path]:
    """
    Build a catalog of `n` unique URDFs and write `catalog.txt` in `catalog_dir`.

    Uses:
      - Chromosome_Drone: normalized continuous genome generator in [0, 1]^D
      - Chromosome_Drone.to_physical: mapping genome -> physical parameters
      - UrdfMaker:       physical parameters -> URDF file

    A fixed "reasonable" baseline morphology is used for the first URDF,
    then additional morphologies are sampled randomly from the continuous
    design space.

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

    max_attempts = max(n * 25, 200)
    attempts = 0
    dupes = 0

    while len(urdfs) < n and attempts < max_attempts:
        attempts += 1

        if len(urdfs) == 0:
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

    _write_catalog_txt(catalog_dir, urdfs)
    print(f"[catalog] Attempts={attempts}  Duplicates={dupes}  Unique={len(urdfs)}")

    if len(urdfs) < n:
        print(f"[catalog] WARNING: generated only {len(urdfs)}/{n} URDFs.")

    return urdfs


__all__ = ["build_catalog"]
