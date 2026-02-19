#!/usr/bin/env python3
"""
chromosome_drone.py

Continuous drone morphology encoding used by the NSGA-II evolution:

- The *normalized genome* lives in [0, 1]^D.
- `Chromosome_Drone.to_physical()` maps this normalized genome to
  physically meaningful parameters for `UrdfMaker`.

The airfoil is represented by three discrete genes (NACA 4-digit):
  * first digit (camber)        -> {0, 1, 2, 3, 4}
  * second digit (camber pos.)  -> {2, 3, 4, 5}
  * last two digits (thickness) -> {08..22}

These discrete genes are stored in normalized space but are snapped
to valid values when mapping to physical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """
    Specification of a single physical parameter.

    Parameters
    ----------
    name : str
        Human-readable parameter name (for debugging / logging).
    min_val : float
        Lower bound in physical units.
    max_val : float
        Upper bound in physical units. If equal to min_val,
        the parameter is effectively fixed.
    """

    name: str
    min_val: float
    max_val: float

    def clip(self, x: float) -> float:
        """Clip a normalized value x ∈ [0, 1] to the physical interval."""
        x_clamped = min(max(x, 0.0), 1.0)
        return self.min_val + (self.max_val - self.min_val) * x_clamped


class Chromosome_Drone:
    """
    Continuous morphology encoding for the drone.

    The genome is a vector of D floats in [0, 1]. Each entry is mapped
    to a physical parameter using `ParamSpec.clip()`.

    Parameters (logical order / indices)
    ------------------------------------
      0: wing_span                [m]
      1: wing_aspect_ratio        [-]
      2: fuselage_length          [m]
      3: fuselage_cg_ratio        [-] (x_cg / fuselage_length)
      4: wing_attach_ratio        [-] (x_attach / fuselage_length)
      5: elevator_span            [m]
      6: elevator_aspect_ratio    [-]
      7: rudder_span              [m]
      8: rudder_aspect_ratio      [-]
      9: dihedral_deg             [deg]
     10: sweep_multiplier         [-]
     11: twist_multiplier         [-]
     12: naca_d1                  [-]  first digit (0..4)
     13: naca_d2                  [-]  second digit (2..5)
     14: naca_last2               [-]  last two digits (08..22)
    """

    # Discrete NACA value sets
    NACA_D1_VALUES = [0, 1, 2, 3, 4]
    NACA_D2_VALUES = [2, 3, 4, 5]
    NACA_LAST2_VALUES = list(range(10, 21))

    NACA_GENE_INDICES = (12, 13, 14)
    NACA_VALUES_BY_INDEX = {
        12: NACA_D1_VALUES,
        13: NACA_D2_VALUES,
        14: NACA_LAST2_VALUES,
    }

    # List of ParamSpec for each gene in the *physical* genome.
    PARAMS: List[ParamSpec] = [
        # 0: wing_span (m) — range molto ridotto per mantenere S e AR stabili
        ParamSpec("wing_span", 0.45, 0.75),

        # 1: wing_aspect_ratio (span / chord) — AR molto diverse = drag molto diverso
        ParamSpec("wing_aspect_ratio", 1.5, 5.0),

        # 2: fuselage_length (m) — se varia troppo cambia il lever arm e lo static margin
        ParamSpec("fus_length", 0.45, 0.75),

        # 3: fuselage CG ratio (x_cg / fus_length) — se varia troppo il drone diventa ingovernabile
        ParamSpec("cg_x_ratio", 0.30, 0.50),

        # 4: wing attach ratio — variazione moderata OK, variazione ampia = instabilità
        ParamSpec("attach_x_ratio", 0.30, 0.50),

        # 5: elevator span (m) — se troppo piccolo manca autorità, se troppo grande lo destabilizza
        ParamSpec("elevator_span", 0.2, 0.6),

        # 6: elevator aspect ratio — OK range stretto
        ParamSpec("elevator_aspect_ratio", 1.5, 4.0),

        # 7: rudder span (m)
        ParamSpec("rudder_span", 0.10, 0.30),

        # 8: rudder aspect ratio
        ParamSpec("rudder_aspect_ratio", 1.5, 4.0),

        # 9: dihedral (deg) — range stretto: >10° o <−10° causa forti instabilità laterali
        ParamSpec("dihedral_deg", -5.0, 5.0),

        # 10: sweep multiplier — questi range enormi creano differenze assurde nei limiti del giunto
        ParamSpec("sweep_multiplier", 1.5, 3.5),

        # 11: twist multiplier
        ParamSpec("twist_multiplier", 1.5, 3.5),

        # 12: naca_d1 (discrete)
        ParamSpec("naca_d1", min(NACA_D1_VALUES), max(NACA_D1_VALUES)),

        # 13: naca_d2 (discrete)
        ParamSpec("naca_d2", min(NACA_D2_VALUES), max(NACA_D2_VALUES)),

        # 14: naca_last2 (discrete)
        ParamSpec("naca_last2", min(NACA_LAST2_VALUES), max(NACA_LAST2_VALUES)),
    ]

    # Cached arrays of min / max (useful for env normalization / logging)
    PHYS_MIN = np.array([p.min_val for p in PARAMS], dtype=np.float32)
    PHYS_MAX = np.array([p.max_val for p in PARAMS], dtype=np.float32)

    @classmethod
    def num_genes(cls) -> int:
        """Return the dimensionality of the genome."""
        return len(cls.PARAMS)

    # ------------------------------------------------------------------ #
    # Mapping between normalized genome and physical parameters          #
    # ------------------------------------------------------------------ #

    @classmethod
    def _discrete_index(cls, x: float, n: int) -> int:
        x_clamped = min(max(float(x), 0.0), 1.0)
        idx = int(np.floor(x_clamped * n))
        return min(max(idx, 0), n - 1)

    @classmethod
    def snap_genome_norm(cls, genome_norm: Sequence[float]) -> List[float]:
        """
        Snap discrete genes to the center of their normalized bins.

        This keeps SBX/mutation continuous while ensuring discrete genes
        stay consistent with the airfoil value sets.
        """
        out = list(genome_norm)
        for idx in cls.NACA_GENE_INDICES:
            if idx >= len(out):
                continue
            values = cls.NACA_VALUES_BY_INDEX[idx]
            bin_idx = cls._discrete_index(out[idx], len(values))
            out[idx] = (bin_idx + 0.5) / len(values)
        return out

    @classmethod
    def apply_discrete_mutation(
        cls, before: Sequence[float], after: Sequence[float]
    ) -> List[float]:
        """
        Force a bin change for discrete genes when mutation is applied.

        If the mutated value stays in the same bin, we move to the next
        higher/lower bin based on the mutation direction.
        """
        out = list(after)
        for idx in cls.NACA_GENE_INDICES:
            if idx >= len(out) or idx >= len(before):
                continue
            values = cls.NACA_VALUES_BY_INDEX[idx]
            n_bins = len(values)
            old_val = float(before[idx])
            new_val = float(after[idx])
            idx_old = cls._discrete_index(old_val, n_bins)
            idx_new = cls._discrete_index(new_val, n_bins)
            if idx_new == idx_old:
                if new_val > old_val and idx_old < n_bins - 1:
                    idx_new = idx_old + 1
                elif new_val < old_val and idx_old > 0:
                    idx_new = idx_old - 1
            out[idx] = (idx_new + 0.5) / n_bins
        return out

    @classmethod
    def to_physical(cls, genome_norm: Sequence[float]) -> List[float]:
        """
        Map a normalized genome in [0, 1]^D to physical parameters.

        Parameters
        ----------
        genome_norm : sequence of float
            Normalized genome, length must be `num_genes()`.

        Returns
        -------
        list[float]
            Physical parameter vector compatible with `UrdfMaker`.
        """
        if len(genome_norm) != cls.num_genes():
            raise ValueError(
                f"Expected genome of length {cls.num_genes()}, "
                f"got {len(genome_norm)}."
            )

        phys: List[float] = []
        for i, (g, spec) in enumerate(zip(genome_norm, cls.PARAMS)):
            if i in cls.NACA_VALUES_BY_INDEX:
                values = cls.NACA_VALUES_BY_INDEX[i]
                idx = cls._discrete_index(float(g), len(values))
                phys.append(float(values[idx]))
            else:
                phys_val = spec.clip(float(g))
                phys.append(phys_val)

        return phys

    @classmethod
    def from_physical(cls, phys: Sequence[float]) -> List[float]:
        """
        Map physical parameters back to a normalized [0, 1]^D genome.

        This is mostly useful for debugging or importing existing designs.

        Parameters
        ----------
        phys : sequence of float
            Physical parameter vector, length must be `num_genes()`.

        Returns
        -------
        list[float]
            Normalized genome in [0, 1]^D.
        """
        if len(phys) != cls.num_genes():
            raise ValueError(
                f"Expected physical vector of length {cls.num_genes()}, "
                f"got {len(phys)}."
            )

        genome: List[float] = []
        for i, (v, spec) in enumerate(zip(phys, cls.PARAMS)):
            if i in cls.NACA_VALUES_BY_INDEX:
                values = cls.NACA_VALUES_BY_INDEX[i]
                try:
                    idx = values.index(int(round(float(v))))
                except ValueError:
                    idx = int(np.argmin([abs(float(v) - vv) for vv in values]))
                genome.append((idx + 0.5) / len(values))
                continue
            if spec.max_val == spec.min_val:
                genome.append(0.5)
                continue

            # Inverse of `clip` (assuming no prior clipping):
            # v = min + (max - min) * x  → x = (v - min) / (max - min)
            x = (float(v) - spec.min_val) / (spec.max_val - spec.min_val)
            genome.append(float(np.clip(x, 0.0, 1.0)))

        return genome

    @classmethod
    def random_genome(cls) -> List[float]:
        """Generate a random normalized genome in [0,1]^D."""
        return np.random.rand(cls.num_genes()).tolist()

    @classmethod
    def genome_min_max(cls) -> Tuple[List[float], List[float]]:
        """Return physical min/max bounds for the genome (for normalization)."""
        return cls.PHYS_MIN.tolist(), cls.PHYS_MAX.tolist()

    @classmethod
    def naca_from_physical(cls, phys: Sequence[float]) -> str | None:
        """Decode a NACA 4-digit code from a physical genome sequence."""
        if len(phys) != cls.num_genes():
            return None
        d1_val = float(phys[cls.NACA_GENE_INDICES[0]])
        d2_val = float(phys[cls.NACA_GENE_INDICES[1]])
        last2_val = float(phys[cls.NACA_GENE_INDICES[2]])

        def nearest(values: List[int], v: float) -> int:
            return values[int(np.argmin([abs(v - vv) for vv in values]))]

        d1 = nearest(cls.NACA_D1_VALUES, d1_val)
        d2 = nearest(cls.NACA_D2_VALUES, d2_val)
        last2 = nearest(cls.NACA_LAST2_VALUES, last2_val)
        return f"{d1}{d2}{last2:02d}"


    # ------------------------------------------------------------------ #
    # Convenience helpers                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_bounds(cls) -> Tuple[List[float], List[float]]:
        """
        Return normalized bounds for each gene (always 0.0–1.0).

        DEAP's SBX / polynomial mutation operators expect per-gene
        lower and upper bounds. In normalized space everything is
        simply [0, 1].
        """
        n = cls.num_genes()
        low = [0.0] * n
        up = [1.0] * n
        return low, up
