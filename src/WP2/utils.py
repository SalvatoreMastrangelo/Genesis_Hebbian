"""
Utility functions for WP2: seed control, genome encoding/decoding,
serialisation helpers, and git info capture.
"""

from __future__ import annotations

import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from WP2.config import HebbianEvolutionConfig, HebbianConfig


# ============================================================================
#  Seed control
# ============================================================================

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#  RNG state save / load (for resuming)
# ============================================================================

def save_rng_state(path: Path) -> None:
    """Save Python/NumPy/Torch RNG states for exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_rng_state(path: Path) -> None:
    """Restore RNG states from a checkpoint."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ============================================================================
#  Genome encoding / decoding
# ============================================================================

def decode_hebbian_genes(
    genome_section: Sequence[float],
    hebb_cfg: HebbianConfig,
    out_features: int = 7,
    in_features: int = 64,
) -> Dict[str, torch.Tensor]:
    """Decode a normalised [0,1] Hebbian genome section into per-weight tensors.

    Layout: ``[A_flat | B_flat | C_flat | D_flat | (lam_flat) | (eta_flat)]``

    - A, B, C, D are always present (4 × n_weights genes).
    - lam is present only when ``evolve_decay=True``; otherwise the scalar
      ``hebb_cfg.decay`` is used as a constant tensor.
    - eta is present only when ``evolve_eta=True``; otherwise ``hebb_cfg.eta``
      is used as a global scalar.

    Parameters
    ----------
    genome_section : sequence of float
        Flat gene values in [0,1], length = ``hebbian_genome_dim()`` from config.
    hebb_cfg : HebbianConfig
    out_features : int
        Last-layer output dimension (num_actions).
    in_features : int
        Last-layer input dimension (hidden_dim).

    Returns
    -------
    dict
        Keys: ``A``, ``B``, ``C``, ``D``, ``lam`` (each shape ``(out, in)``),
        and optionally ``eta`` if ``evolve_eta=True``.
    """
    n_weights = out_features * in_features
    genes = np.asarray(genome_section, dtype=np.float32)

    def _rescale(block: np.ndarray, lo: float, hi: float) -> torch.Tensor:
        return torch.from_numpy(block * (hi - lo) + lo).reshape(out_features, in_features)

    def _constant(value: float) -> torch.Tensor:
        return torch.full((out_features, in_features), value, dtype=torch.float32)

    idx = 0
    A = _rescale(genes[idx:idx + n_weights], *hebb_cfg.A_range); idx += n_weights
    B = _rescale(genes[idx:idx + n_weights], *hebb_cfg.B_range); idx += n_weights
    C = _rescale(genes[idx:idx + n_weights], *hebb_cfg.C_range); idx += n_weights
    D = _rescale(genes[idx:idx + n_weights], *hebb_cfg.D_range); idx += n_weights

    result = {"A": A, "B": B, "C": C, "D": D}

    if hebb_cfg.evolve_decay:
        lam = _rescale(genes[idx:idx + n_weights], *hebb_cfg.decay_range); idx += n_weights
    else:
        lam = _constant(hebb_cfg.decay)
    result["lam"] = lam

    if hebb_cfg.evolve_eta:
        eta = _rescale(genes[idx:idx + n_weights], *hebb_cfg.eta_range)
        result["eta"] = eta

    return result


def encode_hebbian_genes(
    rules: Dict[str, torch.Tensor],
    hebb_cfg: HebbianConfig,
) -> List[float]:
    """Inverse of decode_hebbian_genes: per-weight tensors -> normalised [0,1] genome."""

    def _normalise(tensor: torch.Tensor, lo: float, hi: float) -> np.ndarray:
        return ((tensor.cpu().numpy().flatten() - lo) / (hi - lo)).clip(0, 1)

    parts = [
        _normalise(rules["A"], *hebb_cfg.A_range),
        _normalise(rules["B"], *hebb_cfg.B_range),
        _normalise(rules["C"], *hebb_cfg.C_range),
        _normalise(rules["D"], *hebb_cfg.D_range),
    ]
    if hebb_cfg.evolve_decay and "lam" in rules:
        parts.append(_normalise(rules["lam"], *hebb_cfg.decay_range))
    if hebb_cfg.evolve_eta and "eta" in rules:
        parts.append(_normalise(rules["eta"], *hebb_cfg.eta_range))

    return np.concatenate(parts).tolist()


def create_zero_initialized_genome(
    cfg: HebbianEvolutionConfig,
) -> List[float]:
    """Create a genome with zero Hebbian rules (A=B=C=D=0, others random).

    A, B, C, D are set to 0.5 in [0,1] space, which maps to the midpoint
    of their configured ranges (0.0 for symmetric ranges like [-1, 1]).
    Decay and eta sections (if evolved) are kept as random [0, 1].
    """
    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    genome: List[float] = []

    # A, B, C, D: 0.5 → midpoint of range → zero for symmetric [-1, 1]
    for _ in range(4):
        genome.extend([0.5] * n_weights)

    if cfg.hebbian.evolve_decay:
        genome.extend([random.random() for _ in range(n_weights)])
    if cfg.hebbian.evolve_eta:
        genome.extend([random.random() for _ in range(n_weights)])

    return genome


# ============================================================================
#  Reproducibility artefacts
# ============================================================================

def save_git_info(path: Path) -> None:
    """Save git commit hash and any uncommitted diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff"], stderr=subprocess.DEVNULL
        ).decode()
        with open(path, "w") as f:
            f.write(f"commit: {commit}\n")
            if diff:
                f.write(f"\n--- uncommitted changes ---\n{diff}")
    except Exception as exc:
        with open(path, "w") as f:
            f.write(f"git info unavailable: {exc}\n")


def save_environment_info(path: Path) -> None:
    """Save pip freeze output for environment reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
        with open(path, "w") as f:
            f.write(freeze)
    except Exception as exc:
        with open(path, "w") as f:
            f.write(f"pip freeze unavailable: {exc}\n")
