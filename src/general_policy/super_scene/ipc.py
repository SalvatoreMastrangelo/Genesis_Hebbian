from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WorkerCommand:
    cmd: str
    payload: Dict[str, Any]


@dataclass
class WorkerReply:
    ok: bool
    payload: Dict[str, Any]
    error: str = ""
