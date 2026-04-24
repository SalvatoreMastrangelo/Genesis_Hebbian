"""
NSGA-II wrapper built on DEAP for the WP2 outer loop.
======================================================

Design
------
- Individuals are 15D lists bounded in [0,1] (same space as
  ``Chromosome_Drone``). Discrete NACA genes are *not* snapped here — they
  are only snapped at URDF materialisation time via
  ``Chromosome_Drone.snap_genome_norm``. This keeps SBX / polynomial
  variation unconstrained in continuous space and simplifies bookkeeping.
- Operators are standard DEAP:
    * ``tools.cxSimulatedBinaryBounded`` (SBX) with bounds [0,1].
    * ``tools.mutPolynomialBounded`` with bounds [0,1].
    * ``tools.selNSGA2`` for (P+Q)→P environmental selection.
    * ``tools.selTournamentDCD`` for binary tournament on crowding distance.
- Objectives are configured per ``ObjectiveSpec`` (``maximize`` → weight
  +1, ``minimize`` → weight -1). We rebuild the DEAP ``creator`` classes
  on each ``make_toolbox`` call keyed by the number of objectives to keep
  the module re-usable across runs with different objective sets.

Fitness values passed in / out are numpy arrays of shape ``(n_obj,)`` per
individual, in the order declared by ``cfg.objectives``.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
from deap import base, creator, tools

from .config import OuterLoopConfig


# ============================================================================
#  DEAP creator (re-created per-run to allow different objective counts)
# ============================================================================

_CREATED_WEIGHTS: Tuple[float, ...] = ()


def _ensure_creator(weights: Sequence[float]) -> None:
    """Create / refresh ``creator.OuterFitness`` and ``creator.OuterIndividual``.

    DEAP's ``creator`` is a module-level registry. Re-registering the same
    class name with different weights would silently keep the first
    definition, so we delete + recreate when weights change.
    """
    global _CREATED_WEIGHTS
    weights_tuple = tuple(float(w) for w in weights)
    if weights_tuple == _CREATED_WEIGHTS and hasattr(creator, "OuterFitness"):
        return
    if hasattr(creator, "OuterFitness"):
        del creator.OuterFitness
    if hasattr(creator, "OuterIndividual"):
        del creator.OuterIndividual
    creator.create("OuterFitness", base.Fitness, weights=weights_tuple)
    creator.create("OuterIndividual", list, fitness=creator.OuterFitness)
    _CREATED_WEIGHTS = weights_tuple


# ============================================================================
#  Toolbox construction
# ============================================================================

def make_toolbox(cfg: OuterLoopConfig, genome_dim: int = 15) -> base.Toolbox:
    """Build a DEAP toolbox for NSGA-II over [0,1]^genome_dim.

    The caller is responsible for seeding ``random`` before variation.
    """
    _ensure_creator(cfg.objective_weights())

    tb = base.Toolbox()

    low = [0.0] * genome_dim
    up = [1.0] * genome_dim

    tb.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=low, up=up, eta=float(cfg.nsga2.sbx_eta),
    )
    tb.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=low, up=up,
        eta=float(cfg.nsga2.pm_eta),
        indpb=float(cfg.nsga2.mutation_prob),
    )
    tb.register("select", tools.selNSGA2)
    # Binary tournament on crowding distance (needs prior assignCrowdingDist).
    tb.register("mating_select", tools.selTournamentDCD)

    return tb


# ============================================================================
#  Population helpers
# ============================================================================

def arrays_to_individuals(
    genomes: np.ndarray,
    objectives: np.ndarray,
) -> List["creator.OuterIndividual"]:
    """Wrap ``(P, D)`` genomes + ``(P, n_obj)`` objective values as DEAP
    individuals with valid ``fitness.values``.

    Requires ``_ensure_creator`` to have been called (``make_toolbox`` does
    this).
    """
    if genomes.shape[0] != objectives.shape[0]:
        raise ValueError(
            f"genomes/objectives pop-size mismatch: "
            f"{genomes.shape[0]} vs {objectives.shape[0]}"
        )
    inds: List[creator.OuterIndividual] = []
    for g, f in zip(genomes, objectives):
        ind = creator.OuterIndividual(list(map(float, g)))
        ind.fitness.values = tuple(float(x) for x in f)
        inds.append(ind)
    return inds


def individuals_to_array(
    population: Sequence["creator.OuterIndividual"],
) -> np.ndarray:
    """Stack a list of individuals into a ``(P, D)`` genome matrix."""
    return np.asarray([list(ind) for ind in population], dtype=np.float64)


# ============================================================================
#  Variation (offspring generation)
# ============================================================================

def make_offspring(
    toolbox: base.Toolbox,
    parents: Sequence["creator.OuterIndividual"],
    crossover_prob: float,
) -> List["creator.OuterIndividual"]:
    """Produce ``len(parents)`` offspring via mating selection + SBX + mutation.

    The parent list must already have crowding distances assigned
    (``tools.emo.assignCrowdingDist``). We:
      1. Binary-tournament by dominance + crowding (``selTournamentDCD``) to
         pick ``len(parents)`` mating candidates.
      2. Walk pairs: with prob ``crossover_prob`` apply SBX, always apply
         polynomial mutation. Clear offspring fitness (needs re-eval).
    """
    k = len(parents)
    if k % 2 != 0:
        raise ValueError(f"selTournamentDCD needs even pop size, got {k}")

    offspring = toolbox.mating_select(parents, k)
    offspring = [creator.OuterIndividual(list(ind)) for ind in offspring]

    for i in range(0, k, 2):
        a, b = offspring[i], offspring[i + 1]
        if random.random() < crossover_prob:
            toolbox.mate(a, b)
            del a.fitness.values
            del b.fitness.values
        toolbox.mutate(a)
        toolbox.mutate(b)
        if a.fitness.valid:
            del a.fitness.values
        if b.fitness.valid:
            del b.fitness.values

    return offspring


def environmental_select(
    toolbox: base.Toolbox,
    parents: Sequence["creator.OuterIndividual"],
    offspring: Sequence["creator.OuterIndividual"],
    k: int,
) -> List["creator.OuterIndividual"]:
    """NSGA-II (P+Q)→k environmental selection."""
    combined = list(parents) + list(offspring)
    return toolbox.select(combined, k)
