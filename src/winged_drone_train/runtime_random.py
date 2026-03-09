from __future__ import annotations

import random
import secrets

import numpy as np
import torch


def seed_runtime_randomness(context: str) -> int:
    """
    Seed Python, NumPy, and Torch from OS entropy.

    This is intentionally independent from any user-facing URDF seed so that
    runtime stochasticity (forests, noise, stochastic policy sampling) does not
    repeat across runs unless the process state itself is duplicated.
    """
    seed = secrets.randbits(63)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[runtime_seed] context={context} seed={seed}")
    return seed
