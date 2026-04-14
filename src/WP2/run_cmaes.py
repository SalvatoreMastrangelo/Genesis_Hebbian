#!/usr/bin/env python
"""
WP2 CMA-ES Evolution Entry Point
=================================

Command-line interface for running CMA-ES evolution of Hebbian plasticity rules.

Typical usage:
    python -m WP2.run_cmaes \\
        --wp1-checkpoint logs/runs/<run>/model.pt \\
        --wp1-cfg src/WP1/configs/foundation.yaml \\
        --pop-size 64 \\
        --num-generations 100

The evolution optimizes Hebbian rules (ABCD + decay) using CMA-ES with
mean forward velocity as the fitness function (same primary objective as WP1).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Set Genesis parallelization before importing
os.environ["GS_PARA_LEVEL"] = "4"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import genesis as gs

from WP2.cmaes_integration import run_cmaes_evolution
from WP1.config import RunConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for WP2 CMA-ES evolution."""
    parser = argparse.ArgumentParser(
        description="WP2 CMA-ES evolution of Hebbian plasticity rules"
    )

    # WP1 artifacts
    parser.add_argument(
        "--wp1-checkpoint",
        type=str,
        required=True,
        help="Path to WP1 actor checkpoint (e.g., logs/runs/.../model.pt)",
    )
    parser.add_argument(
        "--wp1-cfg",
        type=str,
        required=True,
        help="Path to WP1 config YAML (e.g., src/WP1/configs/foundation.yaml)",
    )

    # Evolution parameters
    parser.add_argument(
        "--pop-size",
        type=int,
        default=64,
        help="CMA-ES population size (default: 64)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=100,
        help="Number of evolution generations (default: 100)",
    )
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=256,
        help="Total parallel evaluation environments (default: 256)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Episodes per individual during evaluation (default: 1)",
    )


    # Hebbian rule configuration
    parser.add_argument(
        "--add-decay",
        action="store_true",
        default=True,
        help="Evolve per-weight decay (lambda) coefficients (default: True)",
    )
    parser.add_argument(
        "--no-decay",
        dest="add_decay",
        action="store_false",
        help="Disable per-weight decay evolution",
    )
    parser.add_argument(
        "--add-eta",
        action="store_true",
        default=False,
        help="Evolve per-weight learning rates (eta) (default: False)",
    )

    # Output and reproducibility
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logging (default: logs/runs_hebbian/<timestamp>/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="PyTorch device (default: cuda)",
    )

    args = parser.parse_args()

    # Validate paths
    wp1_ckpt_path = Path(args.wp1_checkpoint)
    if not wp1_ckpt_path.exists():
        logger.error(f"WP1 checkpoint not found: {wp1_ckpt_path}")
        sys.exit(1)

    wp1_cfg_path = Path(args.wp1_cfg)
    if not wp1_cfg_path.exists():
        logger.error(f"WP1 config not found: {wp1_cfg_path}")
        sys.exit(1)

    # Initialize Genesis
    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)
        logger.info("Genesis initialized")

    # Configure PyTorch
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Run evolution
    logger.info("Starting WP2 CMA-ES Evolution")
    logger.info(f"  WP1 checkpoint: {wp1_ckpt_path}")
    logger.info(f"  WP1 config: {wp1_cfg_path}")
    logger.info(f"  Population size: {args.pop_size}")
    logger.info(f"  Generations: {args.num_generations}")
    logger.info(f"  Eval envs: {args.num_eval_envs}")
    logger.info(f"  Fitness: WP1 weighted reward (progress, crash, energy, smoothness)")
    logger.info(f"  Add decay: {args.add_decay}")
    logger.info(f"  Add eta: {args.add_eta}")

    try:
        evolution, run_dir = run_cmaes_evolution(
            wp1_checkpoint=wp1_ckpt_path,
            wp1_cfg=wp1_cfg_path,
            pop_size=args.pop_size,
            num_generations=args.num_generations,
            num_eval_envs=args.num_eval_envs,
            n_episodes=args.n_episodes,
            add_decay=args.add_decay,
            add_eta=args.add_eta,
            seed=args.seed,
            log_dir=args.log_dir,
            device=args.device,
        )
        logger.info(f"Evolution complete. Results saved to: {run_dir}")
    except Exception as e:
        logger.error(f"Evolution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
