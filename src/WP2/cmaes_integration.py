"""
CMA-ES Integration for WP2 Hebbian Rules Evolution
===================================================

Integrates CMA-ES evolution with RulesEvolutionEnv for joint optimization of
Hebbian plasticity rules and drone morphologies using the same fitness approach as WP1
(mean forward velocity as primary objective).

Key features:
- Batched population evaluation via RulesEvolutionEnv
- CMA-ES single-objective optimization (vs NSGA-II multi-objective)
- Morphology evolution support (genome includes Hebbian + URDF genes)
- Checkpoint/resume capability
- Progress tracking and logging

Typical workflow:
    from WP2.cmaes_integration import run_cmaes_evolution

    # Minimal setup
    run_cmaes_evolution(
        wp1_checkpoint="logs/runs/.../model.pt",
        wp1_cfg="src/WP1/configs/foundation.yaml",
        pop_size=64,
        num_generations=100,
        num_eval_envs=256,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from WP2.CMA_ES_loop import CMAESHebbianEvolution, EvolutionStats
from WP2.checkpoint_loader import load_wp1_actor, get_actor_dimensions
from WP2.hebbian import (
    HebbianController,
    create_hebbian_rules,
    get_hebbian_genome_dim,
)
from WP2.rules_evolution import RulesEvolutionEnv, evaluate_population_batched
from WP1.config import RunConfig


logger = logging.getLogger(__name__)


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES WP2 evolution.

    Parameters
    ----------
    pop_size : int
        CMA-ES population size (individuals per generation).
    num_generations : int
        Number of evolution generations to run.
    num_eval_envs : int
        Total parallel evaluation environments.
    n_episodes : int
        Episodes per individual during evaluation (for averaging).
    add_decay : bool
        Whether to evolve per-weight decay (lambda) coefficients.
    add_eta : bool
        Whether to evolve per-weight learning rates (eta).
    w_max : float
        Weight clipping bound for Hebbian controllers.
    use_oja_coefficient : bool
        Whether controllers use Oja normalization.
    seed : int, optional
        RNG seed for reproducibility.
    device : str
        PyTorch device.
    """
    pop_size: int = 64
    num_generations: int = 100
    num_eval_envs: int = 256
    n_episodes: int = 1
    add_decay: bool = True
    add_eta: bool = False
    w_max: float = 3.0
    use_oja_coefficient: bool = True
    seed: int | None = None
    device: str = "cuda"


class CMAESHebbianPopulationEvaluator:
    """Wraps RulesEvolutionEnv for CMA-ES fitness evaluation.

    Fitness is computed as the cumulative WP1 weighted reward (same as WP1 training).

    Parameters
    ----------
    wp1_actor : nn.Module
        Frozen WP1 actor network.
    w_checkpoint : Tensor
        Frozen baseline weights from actor's last layer.
    wp1_cfg : RunConfig
        WP1 configuration for environment setup.
    pop_size : int
        Population size.
    num_eval_envs : int
        Total evaluation environments.
    n_episodes : int
        Episodes per individual.
    add_decay : bool
        Whether to evolve decay coefficients.
    add_eta : bool
        Whether to evolve eta coefficients.
    w_max : float
        Weight clipping bound.
    use_oja_coefficient : bool
        Use Oja normalization.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        wp1_actor: nn.Module,
        w_checkpoint: Tensor,
        wp1_cfg: RunConfig,
        pop_size: int,
        num_eval_envs: int,
        n_episodes: int = 1,
        add_decay: bool = True,
        add_eta: bool = False,
        w_max: float = 3.0,
        use_oja_coefficient: bool = True,
        wp1_run_dir: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.actor = wp1_actor
        self.w_checkpoint = w_checkpoint.clone().to(device)
        self.wp1_cfg = wp1_cfg
        self.pop_size = pop_size
        self.num_eval_envs = num_eval_envs
        self.n_episodes = n_episodes
        self.add_decay = add_decay
        self.add_eta = add_eta
        self.w_max = w_max
        self.use_oja_coefficient = use_oja_coefficient
        self.wp1_run_dir = Path(wp1_run_dir) if wp1_run_dir else None
        self.device = torch.device(device)

        # Initialize evaluation environment once
        self._init_eval_env()

    def _init_eval_env(self) -> None:
        """Initialize the RulesEvolutionEnv (once, reused across generations)."""
        # Find the catalog directory from config or run directory
        catalog_dir = self.wp1_cfg.catalog.catalog_dir
        if not catalog_dir and self.wp1_run_dir:
            catalog_dir = self.wp1_run_dir / "catalog"

        if not catalog_dir or not Path(catalog_dir).exists():
            raise ValueError(
                f"Cannot find catalog directory. "
                f"Tried: config.catalog_dir={self.wp1_cfg.catalog.catalog_dir}, "
                f"run_dir/catalog={self.wp1_run_dir / 'catalog' if self.wp1_run_dir else 'N/A'}"
            )

        # Find first URDF in catalog for dummy initialization
        catalog_path = Path(catalog_dir)
        urdf_files = list(catalog_path.glob("*.urdf"))
        if not urdf_files:
            raise ValueError(f"No URDF files found in {catalog_dir}")

        dummy_urdf = str(urdf_files[0])
        dummy_urdf_paths = [dummy_urdf] * self.pop_size

        # Get correct actor dimensions
        hidden_dim, num_actions = get_actor_dimensions(self.actor)

        # Dummy controllers (will be replaced per generation)
        dummy_rules = {
            "A": torch.zeros(num_actions, hidden_dim, device=self.device),
            "B": torch.zeros(num_actions, hidden_dim, device=self.device),
            "C": torch.zeros(num_actions, hidden_dim, device=self.device),
            "D": torch.zeros(num_actions, hidden_dim, device=self.device),
            "lam": torch.zeros(num_actions, hidden_dim, device=self.device),
        }
        dummy_controllers = [
            HebbianController(
                actor=self.actor,
                rules=dummy_rules,
                w_checkpoint=self.w_checkpoint,
                w_max=self.w_max,
                use_oja_coefficient=self.use_oja_coefficient,
                device=self.device,
            )
            for _ in range(self.pop_size)
        ]

        self.eval_env = RulesEvolutionEnv(
            urdf_paths=dummy_urdf_paths,
            controllers=dummy_controllers,
            n_envs=self.num_eval_envs,
            wp1_cfg=self.wp1_cfg,
            device=str(self.device),
        )

    def evaluate(self, controllers: list[HebbianController]) -> np.ndarray:
        """Evaluate a population of controllers.

        Computes fitness as cumulative WP1 weighted reward.

        Parameters
        ----------
        controllers : list[HebbianController]
            List of Hebbian controllers (length must equal pop_size).

        Returns
        -------
        fitnesses : np.ndarray
            Fitness values of shape (pop_size,). Higher is better.
        """
        if len(controllers) != self.pop_size:
            raise ValueError(
                f"Expected {self.pop_size} controllers, got {len(controllers)}"
            )

        # Update controllers in eval env
        for i, ctrl in enumerate(controllers):
            self.eval_env._ctrl_batches[i].controllers[0] = ctrl

        # Evaluate population (returns WP1 reward-based fitness)
        fitness_matrix = self.eval_env.evaluate_population(
            n_episodes=self.n_episodes,
        )  # shape: (pop_size, envs_per_urdf)

        # Aggregate fitness across environment slots: mean per individual
        fitnesses = fitness_matrix.mean(dim=1)  # (pop_size,)

        return fitnesses.cpu().numpy().astype(np.float32)


def run_cmaes_evolution(
    wp1_checkpoint: str | Path,
    wp1_cfg: str | Path | RunConfig,
    pop_size: int = 64,
    num_generations: int = 100,
    num_eval_envs: int = 256,
    n_episodes: int = 1,
    add_decay: bool = True,
    add_eta: bool = False,
    seed: int | None = None,
    log_dir: str | Path | None = None,
    device: str = "cuda",
) -> Tuple[CMAESHebbianEvolution, Path]:
    """Run complete CMA-ES evolution for WP2 Hebbian rules.

    Optimizes Hebbian plasticity rules (ABCD + decay) using CMA-ES with
    WP1's weighted reward function (progress, crash, energy, smoothness).

    Parameters
    ----------
    wp1_checkpoint : str or Path
        Path to WP1 actor checkpoint.
    wp1_cfg : str, Path, or RunConfig
        WP1 configuration (YAML path or RunConfig object).
    pop_size : int
        Population size.
    num_generations : int
        Number of generations.
    num_eval_envs : int
        Total parallel evaluation environments.
    n_episodes : int
        Episodes per evaluation.
    add_decay : bool
        Evolve per-weight decay.
    add_eta : bool
        Evolve per-weight learning rates.
    seed : int, optional
        RNG seed.
    log_dir : str, Path, or None
        Directory for logging. If None, creates logs/runs_hebbian/<timestamp>/.
    device : str
        PyTorch device.

    Returns
    -------
    evolution : CMAESHebbianEvolution
        The evolution object with history.
    run_dir : Path
        Directory where results are saved.
    """
    # Load WP1 config and preserve path for load_wp1_actor
    if isinstance(wp1_cfg, (str, Path)):
        wp1_cfg_path = wp1_cfg
        wp1_cfg = RunConfig.from_yaml(wp1_cfg)
    else:
        # If wp1_cfg is already a RunConfig, we can't extract the path
        raise ValueError("wp1_cfg must be a string or Path to a YAML file, not a RunConfig object")

    # Load WP1 actor
    actor = load_wp1_actor(wp1_checkpoint, wp1_cfg_path, device=device)

    # Get checkpoint weights from the last Linear layer
    last_linear = None
    for module in actor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError("Cannot find Linear layer in actor")
    w_checkpoint = last_linear.weight.clone().detach()

    # Setup logging
    if log_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(f"logs/runs_hebbian/{timestamp}_cmaes")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"CMA-ES WP2 Evolution")
    logger.info(f"  pop_size={pop_size}, generations={num_generations}")
    logger.info(f"  fitness: WP1 weighted reward (progress, crash, energy, smoothness)")
    logger.info(f"  log_dir={log_dir}")

    # Infer WP1 run directory from checkpoint path (e.g., logs/runs/2026-04-10_20-03-03_run/tb/model.pt)
    wp1_run_dir = Path(wp1_checkpoint).parent.parent

    # Create evaluator
    evaluator = CMAESHebbianPopulationEvaluator(
        wp1_actor=actor,
        w_checkpoint=w_checkpoint,
        wp1_cfg=wp1_cfg,
        pop_size=pop_size,
        num_eval_envs=num_eval_envs,
        n_episodes=n_episodes,
        add_decay=add_decay,
        add_eta=add_eta,
        wp1_run_dir=wp1_run_dir,
        device=device,
    )

    # Create CMA-ES evolution
    def reward_fn(controllers: list[HebbianController]) -> np.ndarray:
        """Fitness function for CMA-ES."""
        return evaluator.evaluate(controllers)

    evolution = CMAESHebbianEvolution(
        actor=actor,
        w_checkpoint=w_checkpoint,
        reward_fn=reward_fn,
        pop_size=pop_size,
        add_decay=add_decay,
        add_eta=add_eta,
        device=device,
        seed=seed,
    )

    # Evolution loop
    best_fitness_history = []
    mean_fitness_history = []

    start_time = time.time()

    try:
        for gen in range(num_generations):
            best_ind, best_fit, stats = evolution.step()

            best_fitness_history.append(best_fit)
            mean_fitness_history.append(stats.mean_fitness)

            elapsed = time.time() - start_time
            logger.info(
                f"Gen {gen+1:3d}/{num_generations}  "
                f"best={best_fit:8.4f}  mean={stats.mean_fitness:8.4f}  "
                f"std={stats.std_fitness:8.4f}  elapsed={elapsed:.1f}s"
            )

            # Periodic checkpoint
            if (gen + 1) % 10 == 0:
                ckpt_path = log_dir / f"evolution_gen{gen+1:03d}.pt"
                evolution.save_state(ckpt_path)

    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")

    finally:
        # Save final state
        final_path = log_dir / "evolution_final.pt"
        evolution.save_state(final_path)
        logger.info(f"Saved final state to {final_path}")

    return evolution, log_dir


if __name__ == "__main__":
    import sys

    # Minimal example
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <wp1_checkpoint> <wp1_cfg>")
        sys.exit(1)

    wp1_ckpt = sys.argv[1]
    wp1_config = sys.argv[2]

    evolution, run_dir = run_cmaes_evolution(
        wp1_checkpoint=wp1_ckpt,
        wp1_cfg=wp1_config,
        pop_size=32,
        num_generations=50,
        num_eval_envs=128,
        fitness_key="mean_velocity",
        device="cuda",
    )

    print(f"Evolution complete. Results in: {run_dir}")
