from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Sequence

from morph_evolution.utils.runtime import create_urdf_with_retry
from winged_drone_train.defaults import (
    STANDARD_MYDRONE_GENOME,
    STANDARD_MYDRONE_URDF_FILENAME,
    default_mydrone_urdf_path,
    resolve_default_urdf_for_drone,
)


def _parse_physical_genome_from_name(name: str) -> list[float] | None:
    stem = Path(name).stem.strip()
    if not stem:
        return None
    try:
        parsed = ast.literal_eval(stem)
    except Exception:
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 15:
        return None
    try:
        return [float(v) for v in parsed]
    except Exception:
        return None


def _infer_genome_for_missing_urdf(target: Path, drone_key: str | None = None) -> Sequence[float] | None:
    if target.name == STANDARD_MYDRONE_URDF_FILENAME:
        return STANDARD_MYDRONE_GENOME

    try:
        if target.resolve() == default_mydrone_urdf_path().resolve():
            return STANDARD_MYDRONE_GENOME
    except Exception:
        pass

    key = (drone_key or "").strip().lower()
    if key and "mydrone" in key:
        return STANDARD_MYDRONE_GENOME

    return _parse_physical_genome_from_name(target.name)


def resolve_or_generate_urdf(
    *,
    urdf_file: str | Path | None = None,
    drone_key: str | None = None,
) -> str:
    """
    Resolve a URDF path and auto-generate it when possible.

    Resolution order:
    1) explicit `urdf_file`,
    2) default URDF for `drone_key`,
    3) standard mydrone URDF.

    If the resolved file is missing, try to infer a 15-value physical genome
    from the filename (or from known defaults) and generate the URDF into its
    parent directory.
    """
    if urdf_file is not None:
        target = Path(urdf_file).expanduser()
    elif drone_key:
        target = resolve_default_urdf_for_drone(drone_key)
    else:
        target = default_mydrone_urdf_path()

    target = target.resolve()
    if target.exists():
        return str(target)

    genome = _infer_genome_for_missing_urdf(target, drone_key=drone_key)
    if genome is None:
        raise FileNotFoundError(
            f"URDF not found: {target}. Automatic generation is only supported "
            "for mydrone defaults or filenames encoding a 15-value genome."
        )

    generated = create_urdf_with_retry(genome, target.parent)
    generated_path = Path(generated).expanduser().resolve()
    if generated_path != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, target)
        return str(target)
    return str(generated_path)
