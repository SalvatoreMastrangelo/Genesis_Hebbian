"""
Neutral import surface for aerodynamic URDF model parsing.

This module re-exports the canonical model currently implemented in
`genesis.assets.urdf.common.drone_model` to provide a location that is not
drone-name specific.
"""

from genesis.assets.urdf.common.drone_model import (  # noqa: F401
    ActuatorInfo,
    AeroSurface,
    DroneAeroModel,
    DroneModel,
    SurfaceKind,
)
