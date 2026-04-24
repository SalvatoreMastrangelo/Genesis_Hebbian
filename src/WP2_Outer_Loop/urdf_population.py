"""
URDF population management for the NSGA-II outer loop.
=======================================================

Each NSGA-II individual is a normalized 15D ``Chromosome_Drone`` genome in
``[0, 1]^15``. This module:

- Samples the initial outer-loop population (random ``P-1`` + optional
  standard-mydrone seed).
- Materialises a population of normalized genomes into URDFs on disk.
- Writes a ``catalog.txt`` compatible with ``WP2.HebbianCMAES`` so it can
  pick up the URDFs for the inner CMA-ES loop.

Design note: ``morph_evolution.chromosome_drone.Chromosome_Drone`` maps
``[0,1]^15 → physical params``. The standard drone genome, however, is
given in physical space (``winged_drone_train.defaults.STANDARD_MYDRONE_GENOME``);
we round-trip it through ``Chromosome_Drone.from_physical`` to bring it back
into the normalized space so it can participate in SBX + polynomial
variation like any other individual.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def sample_initial_population(
    pop_size: int,
    seed: int,
    seed_standard_drone: bool = True,
) -> np.ndarray:
    """Sample the gen-0 URDF population in normalized genome space.

    Parameters
    ----------
    pop_size : int
        Number of individuals P in the outer-loop population.
    seed : int
        RNG seed for the random samples.
    seed_standard_drone : bool
        If True, slot 0 is the standard-mydrone genome (mapped back into
        normalized space via ``Chromosome_Drone.from_physical``). The
        remaining P-1 slots are sampled uniformly in [0,1]^15.
        If False, all P slots are random.

    Returns
    -------
    np.ndarray, shape (P, 15)
    """
    from morph_evolution.chromosome_drone import Chromosome_Drone

    D = Chromosome_Drone.num_genes()
    rng = np.random.default_rng(seed)

    pop = rng.random((pop_size, D), dtype=np.float64)

    if seed_standard_drone and pop_size >= 1:
        from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
        std_norm = np.asarray(
            Chromosome_Drone.from_physical(list(STANDARD_MYDRONE_GENOME)),
            dtype=np.float64,
        )
        pop[0] = std_norm

    return pop


def materialize_urdfs(
    genomes: Sequence[Sequence[float]],
    out_dir: Path,
) -> List[str]:
    """Convert a batch of normalized genomes to URDF files on disk.

    Each genome is snapped on discrete (NACA) genes, mapped to physical
    parameters, and emitted by ``UrdfMaker``. The order of returned paths
    matches ``genomes``. Filenames are disambiguated by a deterministic
    ``_{index}`` suffix appended to each URDF filename to avoid overwriting
    when two individuals map to the same physical genome.

    Parameters
    ----------
    genomes : sequence of length-15 sequences
        Normalized genomes in [0,1]^15.
    out_dir : Path
        Target directory. Created if missing.

    Returns
    -------
    list[str]
        Absolute URDF paths, one per input genome (same order).
    """
    import genesis as gs
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from drone_making import UrdfMaker

    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for i, genome in enumerate(genomes):
        snapped = Chromosome_Drone.snap_genome_norm(list(genome))
        phys = Chromosome_Drone.to_physical(snapped)
        maker = UrdfMaker(phys, out_dir=out_dir)
        # Filename includes the individual index so two individuals that
        # snap to the same physical genome still get distinct URDFs.
        filename = f"ind_{i:03d}.urdf"
        out_path = Path(maker.create_urdf(filename=filename))
        paths.append(str(out_path.resolve()))

    return paths


def write_catalog_txt(urdf_paths: Sequence[str], out_dir: Path) -> Path:
    """Write a ``catalog.txt`` listing each URDF filename, one per line.

    ``HebbianCMAES`` reads this file and resolves entries relative to its
    parent directory (see ``_load_catalog_paths`` in ``WP2/evolve_cma.py``).
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "catalog.txt"
    names: List[str] = []
    for p in urdf_paths:
        name = Path(p).name
        names.append(name)
    catalog_path.write_text("\n".join(names))
    return catalog_path
