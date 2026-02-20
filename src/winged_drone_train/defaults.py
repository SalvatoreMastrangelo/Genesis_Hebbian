from __future__ import annotations

from pathlib import Path

# Canonical "standard" morphology used across train/eval scripts.
STANDARD_MYDRONE_GENOME: tuple[float, ...] = (
    0.7,
    3.5,
    0.73,
    0.38,
    0.38,
    0.5,
    4.0,
    0.2,
    2.0,
    0.0,
    2.0,
    2.5,
    3.0,
    4.0,
    16.0,
)

STANDARD_MYDRONE_URDF_FILENAME = (
    "[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 2, 2.5, 3, 4, 16].urdf"
)
LISPARROW_URDF_FILENAME = "lisparrow.urdf"

_ROOT_FROM_SRC = Path(__file__).resolve().parents[2]
_LEGACY_ROOT = Path(__file__).resolve().parents[3]


def _candidate_roots() -> tuple[Path, ...]:
    roots = [_LEGACY_ROOT, _ROOT_FROM_SRC]
    out: list[Path] = []
    for root in roots:
        if root not in out:
            out.append(root)
    return tuple(out)


def _resolve_existing(relative_path: Path) -> Path:
    """
    Resolve a project-relative path with backward-compatible root preference.

    Preference order:
    1) legacy layout root (historical scripts),
    2) current repository root.
    """
    for root in _candidate_roots():
        candidate = (root / relative_path).resolve()
        if candidate.exists():
            return candidate
    return (_candidate_roots()[0] / relative_path).resolve()


def default_mydrone_urdf_path() -> Path:
    return _resolve_existing(
        Path("genesis") / "assets" / "urdf" / "mydrone" / STANDARD_MYDRONE_URDF_FILENAME
    )


def default_lisparrow_urdf_path() -> Path:
    return _resolve_existing(
        Path("genesis") / "assets" / "urdf" / "lisparrow" / LISPARROW_URDF_FILENAME
    )


def default_mydrone_urdf_dir() -> Path:
    return default_mydrone_urdf_path().parent


def resolve_default_urdf_for_drone(drone_key: str) -> Path:
    key = (drone_key or "").strip().lower()
    if "lisparrow" in key:
        return default_lisparrow_urdf_path()
    return default_mydrone_urdf_path()
