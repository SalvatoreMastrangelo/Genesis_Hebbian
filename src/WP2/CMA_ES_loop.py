"""
CMA-ES Evolution Loop for Hebbian Rules
========================================

Evolves Hebbian plasticity rules (ABCD + decay/eta) using CMA-ES algorithm.

Key features:
- Flat genome representation: rules flattened to 1D vector for CMA-ES
- Population-based evolution with custom reward function
- Two strategies for controller management:
  1. Create new HebbianControllerBatch per generation (memory-intensive but clean)
  2. Reuse single batch, reset weights/hidden states, update rules (memory-efficient)
- Supports batched environment evaluation
- Tracks fitness history and evolution statistics

Typical workflow:
    from src.WP2.hebbian import HebbianController, HebbianControllerBatch
    from src.WP2.CMA_ES_loop import CMAESHebbianEvolution

    # Custom reward function (depends on your environment)
    def evaluate_fitness(controllers, env, num_episodes=3):
        '''Returns reward tensor (pop_size,)'''
        # Run episodes, return reward per individual
        pass

    evolution = CMAESHebbianEvolution(
        actor=frozen_actor,
        w_checkpoint=checkpoint_weights,
        reward_fn=evaluate_fitness,
        pop_size=50,
        add_decay=True,
        device="cuda"
    )

    # Evolve for N generations
    for gen in range(num_generations):
        best_ind, best_fit, stats = evolution.step()
        print(f"Gen {gen}: best={best_fit:.3f}, mean={stats['mean']:.3f}")

    # Access final population and rules
    best_controller = evolution.get_best_controller()
    best_rules = evolution.get_best_rules()
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

try:
    from cma import CMAEvolutionStrategy
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    logging.warning("cma package not installed. Install with: pip install cma")

from .checkpoint_loader import get_actor_dimensions
from .hebbian import (
    HebbianController,
    HebbianControllerBatch,
    create_hebbian_rules,
    get_hebbian_genome_dim,
)


@dataclass
class EvolutionStats:
    """Per-generation evolution statistics."""

    generation: int
    pop_size: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    std_fitness: float
    median_fitness: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "pop_size": self.pop_size,
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "median_fitness": self.median_fitness,
        }


class CMAESHebbianEvolution:
    """CMA-ES evolution of Hebbian plasticity rules.

    Parameters
    ----------
    actor : nn.Module
        Frozen actor network (from load_wp1_actor).
    w_checkpoint : Tensor
        Frozen baseline weights from the actor's last layer.
        Shape: (num_actions, hidden_dim).
    reward_fn : Callable[[list[HebbianController], ...], Tensor]
        Fitness function that takes a list of controllers and returns
        reward tensor of shape (pop_size,). Can be asynchronous.
    pop_size : int
        Population size for CMA-ES. Default: 50.
    add_decay : bool
        Whether to evolve decay (lam) coefficients. Default: True.
    add_eta : bool
        Whether to evolve eta (learning rate) per individual. Default: False.
    w_max : float
        Weight clipping bound for all controllers. Default: 3.0.
    use_oja_coefficient : bool
        Whether all controllers use Oja normalization. Default: True.
    reuse_batch : bool
        If True, reuse single HebbianControllerBatch and reset weights/rules each gen.
        If False, create new batch each generation. Default: True (memory-efficient).
    device : str or torch.device
        Device for tensors. Default: "cpu".
    seed : int, optional
        RNG seed for reproducibility. Default: None.
    """

    def __init__(
        self,
        actor: nn.Module,
        w_checkpoint: Tensor,
        reward_fn: Callable,
        pop_size: int = 50,
        add_decay: bool = True,
        add_eta: bool = False,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        reuse_batch: bool = True,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if not HAS_CMA:
            raise RuntimeError(
                "cma package required for CMAESHebbianEvolution. "
                "Install with: pip install cma"
            )

        self.actor = actor
        self.w_checkpoint = w_checkpoint.clone()
        self.reward_fn = reward_fn
        self.pop_size = pop_size
        self.add_decay = add_decay
        self.add_eta = add_eta
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.reuse_batch = reuse_batch
        self.device = torch.device(device)
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Get actor dimensions
        self.hidden_dim, self.num_actions = get_actor_dimensions(actor)
        self.n_weights = self.hidden_dim * self.num_actions

        # Genome dimension
        self.genome_dim = get_hebbian_genome_dim(
            actor, add_decay=add_decay, add_eta=add_eta
        )

        # Initialize CMA-ES
        self._init_cma_es()

        # State tracking
        self.generation = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_rules = None
        self.best_individual = None
        self.best_controller = None

        # Reusable batch (if reuse_batch=True)
        self._batch = None

    def _init_cma_es(self) -> None:
        """Initialize CMA-ES strategy."""
        # Initial mean: random normal, scaled to reasonable rule magnitude
        x0 = np.random.normal(0, 0.05, self.genome_dim)

        # Initial standard deviation
        sigma0 = 0.1

        # CMA-ES options
        opts = {
            "seed": self.seed,
            "popsize": self.pop_size,
            "verbose": 0,  # Suppress CMA-ES logging; we handle it
        }

        self.cmaes = CMAEvolutionStrategy(x0, sigma0, opts)

    def _genome_to_rules(self, genome: np.ndarray) -> Dict[str, Tensor]:
        """Convert flat genome vector to Hebbian rules dict.

        Parameters
        ----------
        genome : np.ndarray
            Flat genome vector of shape (genome_dim,).

        Returns
        -------
        rules : dict[str, Tensor]
            Dictionary with keys: A, B, C, D, lam (+ eta if add_eta=True).
            Each tensor has shape (num_actions, hidden_dim).
        """
        genome = np.asarray(genome, dtype=np.float32)
        if genome.shape != (self.genome_dim,):
            raise ValueError(
                f"Genome shape {genome.shape} doesn't match "
                f"expected {(self.genome_dim,)}"
            )

        idx = 0
        rules = {}

        # Unpack A, B, C, D
        for coeff in ["A", "B", "C", "D"]:
            end = idx + self.n_weights
            rules[coeff] = (
                torch.from_numpy(genome[idx:end])
                .float()
                .reshape(self.num_actions, self.hidden_dim)
                .to(self.device)
            )
            idx = end

        # Unpack decay (lam) if evolved
        if self.add_decay:
            end = idx + self.n_weights
            rules["lam"] = (
                torch.from_numpy(genome[idx:end])
                .float()
                .reshape(self.num_actions, self.hidden_dim)
                .to(self.device)
            )
            # Clamp to [0, 1] for decay coefficient
            rules["lam"] = torch.sigmoid(rules["lam"])
            idx = end

        # Unpack eta if evolved
        if self.add_eta:
            end = idx + self.n_weights
            rules["eta"] = (
                torch.from_numpy(genome[idx:end])
                .float()
                .reshape(self.num_actions, self.hidden_dim)
                .to(self.device)
            )
            # Ensure positive (use softplus or exp)
            rules["eta"] = torch.nn.functional.softplus(rules["eta"])
            idx = end
        else:
            rules["eta"] = 0.01  # Default scalar eta

        return rules

    def _rules_to_genome(self, rules: Dict[str, Tensor]) -> np.ndarray:
        """Convert Hebbian rules dict to flat genome vector.

        Parameters
        ----------
        rules : dict[str, Tensor]
            Dictionary with keys: A, B, C, D, lam (+ eta if add_eta=True).

        Returns
        -------
        genome : np.ndarray
            Flat genome vector of shape (genome_dim,).
        """
        parts = []

        # Pack A, B, C, D
        for coeff in ["A", "B", "C", "D"]:
            tensor = rules[coeff].cpu().detach().flatten().numpy().astype(np.float32)
            parts.append(tensor)

        # Pack decay (lam) if evolved
        if self.add_decay:
            # Inverse sigmoid to store in genome
            lam = rules["lam"]
            lam_clamped = torch.clamp(lam, 1e-6, 1.0 - 1e-6)
            lam_logit = torch.logit(lam_clamped)
            tensor = lam_logit.cpu().detach().flatten().numpy().astype(np.float32)
            parts.append(tensor)

        # Pack eta if evolved
        if self.add_eta:
            # Inverse softplus to store in genome
            eta = rules["eta"]
            eta_genome = torch.log(torch.exp(eta) - 1.0)
            tensor = eta_genome.cpu().detach().flatten().numpy().astype(np.float32)
            parts.append(tensor)

        genome = np.concatenate(parts, axis=0)
        return genome

    def _create_controller(self, genome: np.ndarray) -> HebbianController:
        """Create a single HebbianController from a genome.

        Parameters
        ----------
        genome : np.ndarray
            Flat genome vector.

        Returns
        -------
        controller : HebbianController
        """
        rules = self._genome_to_rules(genome)
        controller = HebbianController(
            actor=self.actor,
            rules=rules,
            w_checkpoint=self.w_checkpoint.to(self.device),
            w_max=self.w_max,
            use_oja_coefficient=self.use_oja_coefficient,
            device=self.device,
        )
        return controller

    def _create_batch(self, genomes: np.ndarray) -> HebbianControllerBatch:
        """Create a HebbianControllerBatch from population genomes.

        Parameters
        ----------
        genomes : np.ndarray
            Population genomes, shape (pop_size, genome_dim).

        Returns
        -------
        batch : HebbianControllerBatch
        """
        controllers = [self._create_controller(genomes[i]) for i in range(self.pop_size)]
        batch = HebbianControllerBatch(controllers, device=self.device)
        return batch

    def _reset_batch_with_rules(
        self, genomes: np.ndarray
    ) -> HebbianControllerBatch:
        """Reset existing batch with new rules from genomes.

        Parameters
        ----------
        genomes : np.ndarray
            Population genomes, shape (pop_size, genome_dim).

        Returns
        -------
        batch : HebbianControllerBatch (reused if possible)
        """
        if self._batch is None or len(self._batch.controllers) != self.pop_size:
            # Create new batch
            self._batch = self._create_batch(genomes)
        else:
            # Reset existing controllers with new rules
            for i in range(self.pop_size):
                rules = self._genome_to_rules(genomes[i])
                ctrl = self._batch.controllers[i]

                # Update rules
                ctrl.A.copy_(rules["A"])
                ctrl.B.copy_(rules["B"])
                ctrl.C.copy_(rules["C"])
                ctrl.D.copy_(rules["D"])
                ctrl.lam.copy_(rules["lam"])
                if isinstance(rules["eta"], Tensor):
                    ctrl.eta.copy_(rules["eta"])
                else:
                    ctrl.eta = rules["eta"]

                # Reset weights and hidden states
                ctrl.reset_weights()

            self._batch.reset_hidden_states()

        return self._batch

    def step(
        self,
        create_new_batch: bool | None = None,
    ) -> Tuple[np.ndarray, float, EvolutionStats]:
        """Run one generation of CMA-ES evolution.

        Parameters
        ----------
        create_new_batch : bool, optional
            Whether to create new batch or reuse/reset. If None, uses self.reuse_batch.
            Default: None.

        Returns
        -------
        best_individual : np.ndarray
            Best genome in this generation.
        best_fitness : float
            Fitness value of best individual.
        stats : EvolutionStats
            Per-generation statistics.
        """
        if create_new_batch is None:
            create_new_batch = not self.reuse_batch

        # Get candidate solutions from CMA-ES
        solutions = self.cmaes.ask()
        solutions = np.asarray(solutions, dtype=np.float32)

        # Create batch and evaluate
        if create_new_batch:
            batch = self._create_batch(solutions)
        else:
            batch = self._reset_batch_with_rules(solutions)

        # Evaluate population using reward function
        # reward_fn should return tensor of shape (pop_size,)
        rewards = self.reward_fn(batch.controllers)
        rewards = np.asarray(rewards, dtype=np.float32).flatten()

        if rewards.shape[0] != self.pop_size:
            raise ValueError(
                f"reward_fn returned {rewards.shape[0]} values, "
                f"expected {self.pop_size}"
            )

        # CMA-ES minimizes, so negate rewards (maximize reward = minimize -reward)
        fitnesses = -rewards

        # Tell CMA-ES the fitness values
        self.cmaes.tell(solutions, fitnesses.tolist())

        # Track best
        best_idx = np.argmin(fitnesses)
        self.best_individual = solutions[best_idx].copy()
        self.best_fitness_history.append(-fitnesses[best_idx])
        self.mean_fitness_history.append(-rewards.mean())
        self.best_rules = self._genome_to_rules(self.best_individual)

        # Create controller for best individual
        self.best_controller = self._create_controller(self.best_individual)

        # Statistics
        stats = EvolutionStats(
            generation=self.generation,
            pop_size=self.pop_size,
            best_fitness=float(-fitnesses[best_idx]),
            worst_fitness=float(-fitnesses.max()),
            mean_fitness=float(-rewards.mean()),
            std_fitness=float(rewards.std()),
            median_fitness=float(np.median(-fitnesses)),
        )

        self.generation += 1

        return self.best_individual, -fitnesses[best_idx], stats

    def ask(self) -> np.ndarray:
        """Request candidate solutions from CMA-ES (manual control).

        Returns
        -------
        solutions : np.ndarray
            Population genomes, shape (pop_size, genome_dim).
        """
        solutions = self.cmaes.ask()
        return np.asarray(solutions, dtype=np.float32)

    def tell(
        self, fitnesses: np.ndarray, create_new_batch: bool | None = None
    ) -> Tuple[np.ndarray, float, EvolutionStats]:
        """Tell CMA-ES the fitness values and advance generation.

        Use this for more control over evaluation (e.g., parallel evaluation).

        Parameters
        ----------
        fitnesses : np.ndarray
            Fitness values of shape (pop_size,). Higher is better (will be negated).
        create_new_batch : bool, optional
            Whether to create controllers after this step. Default: None (True).

        Returns
        -------
        best_individual : np.ndarray
        best_fitness : float
        stats : EvolutionStats
        """
        fitnesses = np.asarray(fitnesses, dtype=np.float32).flatten()

        if fitnesses.shape[0] != self.pop_size:
            raise ValueError(
                f"fitnesses shape {fitnesses.shape[0]}, expected {self.pop_size}"
            )

        # CMA-ES minimizes, so negate
        cmaes_fitnesses = -fitnesses

        # Tell CMA-ES
        solutions = self.cmaes.tell(None, cmaes_fitnesses.tolist())

        # Get last solutions (what we just evaluated)
        # For manual control, you might want to track these separately
        # For now, we'll get new solutions from CMA-ES
        solutions = self.cmaes.ask()
        solutions = np.asarray(solutions, dtype=np.float32)

        # Track best
        best_idx = np.argmin(cmaes_fitnesses)
        self.best_fitness_history.append(float(fitnesses[best_idx]))
        self.mean_fitness_history.append(float(fitnesses.mean()))

        # Get best individual (need to track this separately in manual mode)
        # For now, assume it's in the returned solutions
        self.best_individual = solutions[0].copy()  # Placeholder
        self.best_rules = self._genome_to_rules(self.best_individual)
        self.best_controller = self._create_controller(self.best_individual)

        # Statistics
        stats = EvolutionStats(
            generation=self.generation,
            pop_size=self.pop_size,
            best_fitness=float(fitnesses[best_idx]),
            worst_fitness=float(fitnesses.min()),
            mean_fitness=float(fitnesses.mean()),
            std_fitness=float(fitnesses.std()),
            median_fitness=float(np.median(fitnesses)),
        )

        self.generation += 1

        return self.best_individual, float(fitnesses[best_idx]), stats

    def get_best_controller(self) -> HebbianController:
        """Get best controller found so far.

        Returns
        -------
        controller : HebbianController
        """
        if self.best_controller is None:
            raise RuntimeError("No evolution step has been run yet")
        return self.best_controller

    def get_best_rules(self) -> Dict[str, Tensor]:
        """Get best Hebbian rules found so far.

        Returns
        -------
        rules : dict[str, Tensor]
        """
        if self.best_rules is None:
            raise RuntimeError("No evolution step has been run yet")
        return copy.deepcopy(self.best_rules)

    def get_best_individual(self) -> np.ndarray:
        """Get best genome found so far.

        Returns
        -------
        genome : np.ndarray
            Shape (genome_dim,).
        """
        if self.best_individual is None:
            raise RuntimeError("No evolution step has been run yet")
        return self.best_individual.copy()

    def save_state(self, filepath: str | Path) -> None:
        """Save evolution state (population, CMA-ES state, history).

        Parameters
        ----------
        filepath : str or Path
            Path to save checkpoint.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "generation": self.generation,
            "best_individual": self.best_individual,
            "best_fitness_history": self.best_fitness_history,
            "mean_fitness_history": self.mean_fitness_history,
            "cmaes_state": self.cmaes.__dict__,
            "config": {
                "pop_size": self.pop_size,
                "add_decay": self.add_decay,
                "add_eta": self.add_eta,
                "w_max": self.w_max,
                "use_oja_coefficient": self.use_oja_coefficient,
                "reuse_batch": self.reuse_batch,
            },
        }

        torch.save(checkpoint, filepath)
        logging.info(f"Saved evolution state to {filepath}")

    def load_state(self, filepath: str | Path) -> None:
        """Load evolution state from checkpoint.

        Parameters
        ----------
        filepath : str or Path
            Path to checkpoint file.
        """
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        self.generation = checkpoint["generation"]
        self.best_individual = checkpoint["best_individual"]
        self.best_fitness_history = checkpoint["best_fitness_history"]
        self.mean_fitness_history = checkpoint["mean_fitness_history"]

        # Restore CMA-ES state (approximate)
        # Note: full CMA-ES state restoration is complex; this is a partial restore
        logging.info(
            f"Loaded evolution state from {filepath} "
            f"(generation {self.generation})"
        )

        # Regenerate best rules and controller from best individual
        if self.best_individual is not None:
            self.best_rules = self._genome_to_rules(self.best_individual)
            self.best_controller = self._create_controller(self.best_individual)
