from __future__ import annotations

from pathlib import Path
from typing import List


def load_catalog_urdfs(catalog_path: Path) -> List[str]:
    """Load URDF paths from catalog.txt (if present) or from *.urdf files."""
    catalog_file = catalog_path / "catalog.txt"
    if catalog_file.exists():
        lines = [s.strip() for s in catalog_file.read_text().splitlines() if s.strip()]
        urdfs: List[str] = []
        for s in lines:
            p = Path(s)
            if not p.is_absolute():
                p = catalog_path / p
            urdfs.append(str(p))
        return urdfs
    return sorted(str(p) for p in catalog_path.glob("*.urdf"))


def split_even(total: int, k: int) -> List[int]:
    """Split total into k integer parts as evenly as possible."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
