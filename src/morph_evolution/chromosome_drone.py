#!/usr/bin/env python3
"""
chromosome_drone.py

Continuous drone morphology encoding used by the NSGA-II evolution:

- The *normalized genome* lives in [0, 1]^D.
- `Chromosome_Drone.to_physical()` maps this normalized genome to
  physically meaningful parameters for `UrdfMaker`.

HARD-CODED / FIXED PARAMETERS
-----------------------------
Two aerodynamic parameters are kept fixed (not explored by the GA):

  * hinge_le_ratio  = 0.25
  * cl_alpha_2d     = 2.0

These are implemented as ParamSpec entries where min_val == max_val,
so they are constants in physical space while the genome dimension
stays the same for compatibility with existing code.
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
     10: hinge_le_ratio           [-]  FIXED at 0.25
     11: sweep_multiplier         [-]
     12: twist_multiplier         [-]
     13: cl_alpha_2d              [-]  FIXED at 2.0
     14: alpha0_2d_deg            [deg]
    """

    # List of ParamSpec for each gene in the *physical* genome.
    # NOTE: hinge_le_ratio and cl_alpha_2d are FIXED by setting min == max.
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
        ParamSpec("dihedral_deg", -0.0, 0.0),

        # 10: hinge_le_ratio (fixed)
        ParamSpec("hinge_le_ratio", 0.25, 0.25),

        # 11: sweep multiplier — questi range enormi creano differenze assurde nei limiti del giunto
        ParamSpec("sweep_multiplier", 1.5, 3.5),

        # 12: twist multiplier
        ParamSpec("twist_multiplier", 1.5, 3.5),

        # 13: cl_alpha_2d (fixed)
        ParamSpec("cl_alpha_2d", 2.0, 2.0),

        # 14: alpha0_2d (deg) — range ristretto per mantenere comportamento simile
        ParamSpec("alpha0_2d_deg", -5.0, 0.0),
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
        for g, spec in zip(genome_norm, cls.PARAMS):
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
        for v, spec in zip(phys, cls.PARAMS):
            if spec.max_val == spec.min_val:
                # Fixed parameter → arbitrary (but consistent) normalized value
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
