"""
Utility functions for WP2: seed control, genome encoding/decoding,
serialisation helpers, and git info capture.
"""

from __future__ import annotations

import os
import pickle
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from WP2_old.config import HebbianEvolutionConfig, HebbianConfig


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
    # DEAP uses the Python random module internally — covered by random.seed()


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

    The genome section is laid out as consecutive blocks of
    ``n_weights = out_features * in_features`` values (typically 7 × 64 = 448):
        [A_flat | B_flat | C_flat | D_flat | (lam_flat) | (eta_flat)]

    Block order:
    - A, B, C, D are always present (4 × n_weights genes).
    - lam is present only when ``evolve_decay=True``; otherwise all weights
      use the scalar ``hebb_cfg.decay`` via a constant tensor at the midpoint
      of ``decay_range``.
    - eta is present only when ``evolve_eta=True``; otherwise the global scalar
      ``hebb_cfg.eta`` is used.

    Each block is rescaled from [0,1] to the configured range (e.g. A_range).

    Parameters
    ----------
    genome_section : sequence of float
        Flat gene values in [0,1], length = ``hebbian_genome_dim()`` from config.
    hebb_cfg : HebbianConfig
        Provides range bounds (A_range, B_range, …) and evolve_* flags.
    out_features : int
        Last-layer output dimension (num_actions).  Default 7.
    in_features : int
        Last-layer input dimension (hidden_dim).  Default 64.

    Returns
    -------
    dict
        Keys: ``A``, ``B``, ``C``, ``D``, ``lam`` (each shape ``(out, in)``),
        and optionally ``eta`` if ``evolve_eta=True``.
    """
    n_weights = out_features * in_features  # e.g. 448 when out_features=7, in_features=64
    genes = np.asarray(genome_section, dtype=np.float32)

    def _rescale(block: np.ndarray, lo: float, hi: float) -> torch.Tensor:
        return torch.from_numpy(block * (hi - lo) + lo).reshape(out_features, in_features)

    def _constant_value(value: float) -> torch.Tensor:
        """Create a constant tensor with a specific value."""
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
        lam = _constant_value(hebb_cfg.decay)
    result["lam"] = lam

    if hebb_cfg.evolve_eta:
        eta = _rescale(genes[idx:idx + n_weights], *hebb_cfg.eta_range)
        result["eta"] = eta
        idx += n_weights

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
    """Create a genome with zero Hebbian rules (A, B, C, D = 0, others random).

    The genome has:
    - A, B, C, D blocks set to 0.5 in [0, 1] space (maps to midpoint of ranges,
      which is 0.0 for symmetric ranges like [-1, 1])
    - All other sections (decay, eta, morphology) kept as random [0, 1]

    This is useful for ablation studies starting from no Hebbian plasticity.

    Parameters
    ----------
    cfg : HebbianEvolutionConfig
        Configuration with genome dimensions.

    Returns
    -------
    list of float
        Full genome in [0, 1], with A/B/C/D zeroed and others random.
    """
    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    genome = []

    # A, B, C, D: 0.5 maps to midpoint of each range, which acts as "zero" behavior
    # For symmetric ranges like [-1, 1], 0.5 maps to 0.0
    for _ in range(4):
        genome.extend([0.5] * n_weights)

    # decay (lambda) and eta: keep as random [0, 1] if evolved
    if cfg.hebbian.evolve_decay:
        genome.extend([random.random() for _ in range(n_weights)])
    if cfg.hebbian.evolve_eta:
        genome.extend([random.random() for _ in range(n_weights)])

    # Morphology: keep as random [0, 1]
    if cfg.morphology.evolve:
        morph_dim = cfg.morphology_genome_dim()
        genome.extend([random.random() for _ in range(morph_dim)])

    return genome


def split_genome(
    genome: Sequence[float],
    cfg: HebbianEvolutionConfig,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Split a full genome into (hebbian_section, morphology_section).

    Either part may be None if not enabled.
    """
    hebb_dim = cfg.hebbian_genome_dim()
    morph_dim = cfg.morphology_genome_dim()
    genome = list(genome)

    hebb_part = genome[:hebb_dim] if hebb_dim > 0 else None
    morph_part = genome[hebb_dim:hebb_dim + morph_dim] if morph_dim > 0 else None
    return hebb_part, morph_part


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


# ============================================================================
#  Pareto solution serialisation
# ============================================================================

def save_pareto_front(
    run_dir: Path,
    pareto_individuals: list,
    cfg: HebbianEvolutionConfig,
) -> None:
    """Save each Pareto-optimal individual's rules, morphology, and fitness."""
    pareto_dir = run_dir / "pareto_solutions"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    obj_names = cfg.active_objective_names()

    for i, ind in enumerate(pareto_individuals):
        ind_dir = pareto_dir / f"individual_{i:03d}"
        ind_dir.mkdir(parents=True, exist_ok=True)

        hebb_part, morph_part = split_genome(list(ind), cfg)

        # Fitness
        fitness_dict = {}
        for j, name in enumerate(obj_names):
            fitness_dict[name] = float(ind.fitness.values[j])
        with open(ind_dir / "fitness.yaml", "w") as f:
            yaml.dump(fitness_dict, f, sort_keys=False)

        # Hebbian rules
        if hebb_part is not None:
            rules = decode_hebbian_genes(hebb_part, cfg.hebbian)
            rules_dict = {k: v.cpu().numpy().tolist() for k, v in rules.items()}
            with open(ind_dir / "hebbian_rules.yaml", "w") as f:
                yaml.dump(rules_dict, f, sort_keys=False)

        # Morphology
        if morph_part is not None:
            from morph_evolution.chromosome_drone import Chromosome_Drone
            phys = Chromosome_Drone.to_physical(morph_part)
            naca = Chromosome_Drone.naca_from_physical(phys)
            morph_dict = {
                "genome_normalised": morph_part,
                "genome_physical": phys,
                "naca": naca,
            }
            with open(ind_dir / "morphology.yaml", "w") as f:
                yaml.dump(morph_dict, f, sort_keys=False)

        row = {"individual": i, "genome": list(ind)}
        row.update(fitness_dict)
        summary_rows.append(row)

    # Summary CSV
    if summary_rows:
        import csv
        csv_path = pareto_dir / "summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
