"""
CMA-ES Evolution Loop — Usage Examples
=======================================

This file demonstrates how to use CMAESHebbianEvolution with different reward functions
and evaluation strategies.

Example patterns:
1. Simple synchronous evaluation (rollout for each controller)
2. Batched evaluation (multiple controllers in parallel)
3. Multi-episode evaluation with aggregation
4. Integration with Genesis environment
"""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
import torch
from torch import Tensor

from .checkpoint_loader import load_wp1_actor
from .CMA_ES_loop import CMAESHebbianEvolution
from .hebbian import HebbianController, create_hebbian_rules


# =============================================================================
#  Example 1: Simple Reward Function (Synchronous)
# =============================================================================


def reward_fn_simple(controllers: List[HebbianController]) -> np.ndarray:
    """Evaluate each controller independently.

    This is the simplest pattern: run each controller and aggregate rewards.

    Parameters
    ----------
    controllers : List[HebbianController]
        List of controllers to evaluate.

    Returns
    -------
    rewards : np.ndarray
        Reward per controller, shape (pop_size,).
    """
    pop_size = len(controllers)
    rewards = np.zeros(pop_size, dtype=np.float32)

    for idx, ctrl in enumerate(controllers):
        # TODO: Implement your evaluation logic here.
        # Example: rollout in environment, compute fitness metric
        # reward = evaluate_one_controller(ctrl, env=env, num_episodes=3)
        # rewards[idx] = reward
        pass

    return rewards


# =============================================================================
#  Example 2: Batched Evaluation (with HebbianControllerBatch)
# =============================================================================


def reward_fn_batched(
    controllers: List[HebbianController],
    env,  # Your Genesis or RL environment
    num_episodes: int = 3,
) -> np.ndarray:
    """Evaluate controllers using batched forward passes.

    Leverages HebbianControllerBatch for efficient parallel evaluation.

    Parameters
    ----------
    controllers : List[HebbianController]
        Controllers to evaluate.
    env
        Environment with batch_step() method.
        Should support:
        - env.reset(num_envs=pop_size) → obs_batch (pop_size, obs_dim)
        - env.step(actions_batch) → obs, rewards, dones, info
    num_episodes : int
        Number of episodes to run per controller. Default: 3.

    Returns
    -------
    rewards : np.ndarray
        Aggregate reward per controller, shape (pop_size,).
    """
    from .hebbian import HebbianControllerBatch

    pop_size = len(controllers)
    batch = HebbianControllerBatch(controllers)

    rewards_all = np.zeros(pop_size, dtype=np.float32)

    for episode in range(num_episodes):
        # Reset environment and batch
        obs_batch = env.reset(num_envs=pop_size)
        batch.reset_hidden_states()

        episode_rewards = np.zeros(pop_size, dtype=np.float32)

        # Simulate episode
        for step in range(env.max_steps):
            # Batched forward pass
            actions_batch = batch.forward_batch(obs_batch)

            # Environment step
            obs_batch, step_rewards, dones, info = env.step(actions_batch)

            # Accumulate rewards
            episode_rewards += step_rewards

            # Get Hebbian presynaptic/postsynaptic activations for update
            # This depends on your architecture; here's a placeholder:
            # x_batch, y_batch = extract_activations(batch, obs_batch)
            # batch.hebbian_update_batch(x_batch, y_batch)

            # Reset controllers for finished episodes (if env supports mid-episode reset)
            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                batch.reset_hidden_states(list(done_indices))

        rewards_all += episode_rewards

    # Average over episodes
    rewards = rewards_all / num_episodes

    return rewards


# =============================================================================
#  Example 3: Custom Evaluation with Multiple Objectives
# =============================================================================


def reward_fn_multi_objective(
    controllers: List[HebbianController],
    env,
    num_episodes: int = 3,
    objectives: List[Callable] = None,
) -> np.ndarray:
    """Evaluate controllers with multiple objectives (multi-objective evolution).

    For CMA-ES, this returns a single scalar reward (typically weighted sum of objectives).
    For multi-objective optimization, consider using NSGA-II instead.

    Parameters
    ----------
    controllers : List[HebbianController]
    env
        Environment.
    num_episodes : int
        Number of episodes per controller.
    objectives : List[Callable], optional
        List of objective functions, each taking (trajectory) and returning scalar.
        Default: None (use single objective: average return).

    Returns
    -------
    rewards : np.ndarray
        Aggregated reward per controller.
    """
    from .hebbian import HebbianControllerBatch

    if objectives is None:
        # Single objective: cumulative reward
        def single_objective(trajectory):
            return np.sum(trajectory["rewards"])

        objectives = [single_objective]

    pop_size = len(controllers)
    batch = HebbianControllerBatch(controllers)

    rewards_all = np.zeros((pop_size, len(objectives)), dtype=np.float32)

    for episode in range(num_episodes):
        obs_batch = env.reset(num_envs=pop_size)
        batch.reset_hidden_states()

        trajectories = [
            {"observations": [], "rewards": [], "actions": []} for _ in range(pop_size)
        ]

        for step in range(env.max_steps):
            # Batched forward
            actions_batch = batch.forward_batch(obs_batch)

            obs_batch, step_rewards, dones, info = env.step(actions_batch)

            # Collect trajectory data
            for idx in range(pop_size):
                trajectories[idx]["rewards"].append(step_rewards[idx])
                trajectories[idx]["actions"].append(actions_batch[idx])

            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                batch.reset_hidden_states(list(done_indices))

        # Evaluate objectives
        for idx, traj in enumerate(trajectories):
            for obj_idx, objective in enumerate(objectives):
                obj_value = objective(traj)
                rewards_all[idx, obj_idx] += obj_value

    # Average over episodes
    rewards_all /= num_episodes

    # Aggregate objectives into single scalar (simple weighted sum)
    # TODO: Customize aggregation as needed
    weights = np.ones(len(objectives)) / len(objectives)
    rewards = np.dot(rewards_all, weights)

    return rewards


# =============================================================================
#  Example 4: Full Integration with Genesis Environment
# =============================================================================


def create_evolution_loop_for_genesis(
    checkpoint_path: str,
    wp1_config_path: str,
    pop_size: int = 50,
    add_decay: bool = True,
    num_episodes: int = 3,
    device: str = "cuda",
) -> CMAESHebbianEvolution:
    """Create a CMA-ES evolution loop for Genesis environment.

    This is a template for integrating with your actual environment.

    Parameters
    ----------
    checkpoint_path : str
        Path to WP1 actor checkpoint.
    wp1_config_path : str
        Path to WP1 config file.
    pop_size : int
        Population size.
    add_decay : bool
        Whether to evolve decay coefficients.
    num_episodes : int
        Episodes per evaluation.
    device : str
        Computation device.

    Returns
    -------
    evolution : CMAESHebbianEvolution
    """
    # Load frozen actor
    actor = load_wp1_actor(checkpoint_path, wp1_config_path, device=device)
    actor.eval()

    # Extract checkpoint weights from last layer
    last_layer = None
    for module in reversed(list(actor.actor.modules())):
        if isinstance(module, torch.nn.Linear):
            last_layer = module
            break

    if last_layer is None:
        raise RuntimeError("Could not find last Linear layer")

    w_checkpoint = last_layer.weight.data.clone()

    # Define reward function (depends on your environment)
    def reward_fn(controllers):
        # TODO: Replace with actual environment evaluation
        # This is a placeholder that returns random rewards
        pop_size = len(controllers)
        return np.random.normal(0, 1, pop_size)

    # Create evolution loop
    evolution = CMAESHebbianEvolution(
        actor=actor,
        w_checkpoint=w_checkpoint,
        reward_fn=reward_fn,
        pop_size=pop_size,
        add_decay=add_decay,
        add_eta=False,
        w_max=3.0,
        use_oja_coefficient=True,
        reuse_batch=True,  # Memory-efficient: reuse batch per generation
        device=device,
        seed=42,
    )

    return evolution


# =============================================================================
#  Example 5: Running Evolution
# =============================================================================


def run_evolution_example(
    evolution: CMAESHebbianEvolution,
    num_generations: int = 100,
    log_interval: int = 10,
) -> None:
    """Run evolution loop and log progress.

    Parameters
    ----------
    evolution : CMAESHebbianEvolution
        Evolution loop instance.
    num_generations : int
        Number of generations to run.
    log_interval : int
        Logging frequency.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Starting CMA-ES evolution for {num_generations} generations")
    logger.info(f"Population size: {evolution.pop_size}, Genome dim: {evolution.genome_dim}")

    for gen in range(num_generations):
        # Run one generation
        best_genome, best_fitness, stats = evolution.step(create_new_batch=False)

        # Log progress
        if gen % log_interval == 0:
            logger.info(
                f"Gen {gen:3d}: best={stats.best_fitness:8.3f} "
                f"mean={stats.mean_fitness:8.3f} std={stats.std_fitness:8.3f}"
            )

    # Final results
    logger.info(f"\nEvolution complete!")
    logger.info(f"Best fitness: {evolution.best_fitness_history[-1]:.3f}")
    logger.info(f"Best rules shape: {evolution.best_rules['A'].shape}")

    # Access best controller for deployment
    best_ctrl = evolution.get_best_controller()
    print(f"Best controller loaded: {best_ctrl}")


# =============================================================================
#  Example 6: Manual Control (ask/tell pattern)
# =============================================================================


def run_evolution_manual(
    evolution: CMAESHebbianEvolution,
    num_generations: int = 100,
) -> None:
    """Run evolution with manual ask/tell pattern.

    Useful for asynchronous or distributed evaluation.

    Parameters
    ----------
    evolution : CMAESHebbianEvolution
    num_generations : int
        Generations to run.
    """
    for gen in range(num_generations):
        # Ask CMA-ES for candidate solutions
        genomes = evolution.ask()  # Shape: (pop_size, genome_dim)

        # Evaluate in parallel (your code here)
        fitnesses = evaluate_population_parallel(genomes)  # Your implementation

        # Tell CMA-ES the results and advance generation
        best_genome, best_fitness, stats = evolution.tell(fitnesses)

        print(f"Gen {gen}: best={best_fitness:.3f}, mean={fitnesses.mean():.3f}")


def evaluate_population_parallel(genomes: np.ndarray) -> np.ndarray:
    """Placeholder for parallel evaluation.

    Parameters
    ----------
    genomes : np.ndarray
        Population genomes, shape (pop_size, genome_dim).

    Returns
    -------
    fitnesses : np.ndarray
        Fitness per individual, shape (pop_size,).
    """
    # TODO: Implement parallel evaluation
    # Example: use Ray, multiprocessing, or distributed system
    pop_size = genomes.shape[0]
    return np.random.normal(0, 1, pop_size)
