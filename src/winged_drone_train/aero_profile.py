from __future__ import annotations

import copy
from typing import Final


LISPARROW_SOLVER_ALIASES: Final[frozenset[str]] = frozenset(
    ("lisparrow", "cpp", "morphing")
)


def normalize_solver_kind(solver_kind: str | None) -> str:
    """Return a normalized aero solver kind string."""
    return (solver_kind or "").strip().lower()


def is_lisparrow_profile(solver_kind: str | None) -> bool:
    """Whether the requested solver kind maps to the Lisparrow aero profile."""
    return normalize_solver_kind(solver_kind) in LISPARROW_SOLVER_ALIASES


def resolve_aero_config(solver_kind: str | None) -> dict:
    """
    Resolve the aerodynamic configuration dictionary for a solver kind.

    Notes:
    - This function only selects the *configuration profile*.
    - It does not switch the runtime solver class in Genesis.
    """
    from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroParameters
    from genesis.engine.solvers.drones.lisparrow import LisparrowAeroParameters

    if is_lisparrow_profile(solver_kind):
        return copy.deepcopy(LisparrowAeroParameters.as_dict())
    return copy.deepcopy(SimpleDroneAeroParameters.as_dict())


def configure_runtime_aero_solver(solver_kind: str | None) -> None:
    """
    Optionally swap Genesis runtime AeroSolver class for Lisparrow.

    This mirrors the existing explicit monkey-patch behavior used by
    interactive scripts. Environments that do not call this function keep
    the default Genesis `AeroSolver` export.
    """
    if not is_lisparrow_profile(solver_kind):
        return

    from genesis.engine.solvers.drones.lisparrow import LisparrowAeroSolver
    import genesis.engine.simulator as gs_sim
    import genesis.engine.solvers as gs_solvers

    gs_sim.AeroSolver = LisparrowAeroSolver
    gs_solvers.AeroSolver = LisparrowAeroSolver
