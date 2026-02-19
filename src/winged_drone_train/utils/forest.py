"""Forest generation utilities for obstacle–rich drone environments.

This module focuses purely on sampling tree / obstacle positions.  It does not
interact with Genesis directly, so it can be reused in different simulators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ForestConfig:
    """Configuration for a rectangular forest area.

    Attributes:
        x_lower, x_upper: Bounds of the forest along the forward axis (meters).
        y_lower, y_upper: Lateral bounds (meters).
        tree_radius: Radius of each cylindrical tree trunk (meters).
        tree_height: Height of each tree (meters).
        dens_min, dens_max: Linear density profile parameters (trees / meter)
            used when ``growing_forest=True``.  Density increases linearly from
            ``dens_min`` at ``x_lower`` to ``dens_max`` at ``x_upper``.
        num_trees: Number of trees for the uniform sampling mode.
    """

    x_lower: float = 0.0
    x_upper: float = 250.0
    y_lower: float = -50.0
    y_upper: float = 50.0
    tree_radius: float = 0.75
    tree_height: float = 50.0
    dens_min: float = 0.0
    dens_max: float = 4.0
    num_trees: int = 100


class ForestGenerator:
    """Generate one or several random forests of cylindrical obstacles.

    The generator is deliberately independent from any particular environment.
    It only knows how many forests to create and on which device to place the
    resulting tensors.

    Typical usage::

        cfg = ForestConfig(x_lower=0.0, x_upper=160.0, y_lower=-50.0, y_upper=50.0)
        gen = ForestGenerator(num_envs=64, evaluation=False, unique_forests_eval=False,
                              config=cfg, growing_forest=True, device=device)
        cylinders, total_forests = gen.generate()
        forest_ids = gen.sample_forest_ids()

    """

    def __init__(
        self,
        num_envs: int,
        evaluation: bool,
        *,
        unique_forests_eval: bool = False,
        config: Optional[ForestConfig] = None,
        growing_forest: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = int(num_envs)
        self.evaluation = bool(evaluation)
        self.unique_forests_eval = bool(unique_forests_eval)
        self.growing_forest = bool(growing_forest)
        self.device = torch.device(device)

        self.config = config if config is not None else ForestConfig()

        # Determine how many distinct forests we want to keep in memory.
        if self.evaluation:
            # During evaluation we usually want repeatable scenarios.  If
            # ``unique_forests_eval`` is True we assign more forests so that
            # each environment (or each pair) can have its own layout.
            self.total_forests = (self.num_envs * 1) if self.unique_forests_eval else 1
        else:
            # During training we typically want a variety of layouts; a simple
            # heuristic is to keep two forests per environment.
            self.total_forests = max(1, self.num_envs * 2)

        # Will be allocated in :meth:`generate`.
        self.cylinders: Optional[torch.Tensor] = None  # (F, T, 3)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    @property
    def bounds_xy(self) -> Tuple[float, float, float, float]:
        c = self.config
        return c.x_lower, c.x_upper, c.y_lower, c.y_upper

    def _sample_uniform_forest(self, F: int) -> torch.Tensor:
        """Sample a forest with uniform tree density in the XY rectangle.

        Returns:
            Tensor of shape ``(F, num_trees, 3)`` with (x, y, z) centers.
        """
        c = self.config
        num_trees = int(c.num_trees)
        device = self.device

        xs = torch.rand((F, num_trees), device=device) * (c.x_upper - c.x_lower) + c.x_lower
        ys = torch.rand((F, num_trees), device=device) * (c.y_upper - c.y_lower) + c.y_lower
        zs = torch.full((F, num_trees), 0.5 * c.tree_height, device=device)

        cylinders = torch.stack((xs, ys, zs), dim=-1)  # (F, num_trees, 3)
        return cylinders.float()

    def _sample_growing_forest(self, F: int) -> torch.Tensor:
        """Sample forests where tree density increases along the X axis.

        The linear density profile :math:`\\lambda(x)` goes from ``dens_min`` at
        ``x_lower`` to ``dens_max`` at ``x_upper``.  The expected total number
        of trees is approximated by trapezoidal integration and rounded up.
        """
        c = self.config
        device = self.device

        width_x = c.x_upper - c.x_lower
        if width_x <= 0.0:
            raise ValueError("ForestConfig.x_upper must be greater than x_lower")

        # Approximate expected number of trees under linear density profile.
        expected_N = 0.5 * (c.dens_min + c.dens_max) * width_x
        num_trees = max(1, int(math.ceil(expected_N)))

        # x positions according to PDF proportional to (x - x_lower).
        u = torch.rand((F, num_trees), device=device)
        xs = c.x_lower + width_x * torch.sqrt(u)

        # y positions uniform in [y_lower, y_upper].
        ys = torch.rand((F, num_trees), device=device) * (c.y_upper - c.y_lower) + c.y_lower

        zs = torch.full((F, num_trees), 0.5 * c.tree_height, device=device)

        cylinders = torch.stack((xs, ys, zs), dim=-1)  # (F, num_trees, 3)
        return cylinders.float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> tuple[torch.Tensor, int]:
        """Generate forest layouts according to the configured mode.

        Returns:
            cylinders: Tensor of shape ``(F, T, 3)`` containing tree centers
                for each forest.
            total_forests: Number of distinct forests generated (same as
                ``self.total_forests``).
        """
        F = self.total_forests
        if self.growing_forest:
            cylinders = self._sample_growing_forest(F)
        else:
            cylinders = self._sample_uniform_forest(F)

        self.cylinders = cylinders
        return cylinders, self.total_forests

    def sample_forest_ids(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample an active forest index for each environment.

        Args:
            rng: Optional :class:`torch.Generator` for reproducibility.

        Returns:
            Tensor of shape ``(num_envs,)`` with integer indices in
            ``[0, total_forests)``.
        """
        if self.cylinders is None:
            raise RuntimeError("Forests have not been generated yet. Call generate() first.")

        return torch.randint(
            low=0,
            high=self.total_forests,
            size=(self.num_envs,),
            generator=rng,
            device=self.device,
        )


# Convenience wrapper -------------------------------------------------------
def generate_forests(
    num_envs: int,
    evaluation: bool,
    *,
    unique_forests_eval: bool = False,
    growing_forest: bool = True,
    env_cfg: Optional[dict] = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int, ForestGenerator]:
    """Helper to build a :class:`ForestGenerator` from a simple config dict.

    This is a thin compatibility layer meant to keep ``env.py`` simple.  It
    converts a plain configuration dictionary into a :class:`ForestConfig`,
    constructs a generator, and calls :meth:`ForestGenerator.generate`.

    Args:
        num_envs: Number of environments used by the simulator.
        evaluation: If ``True``, use evaluation mode (affects how many forests
            are generated).
        unique_forests_eval: If ``True`` and ``evaluation`` is ``True``, try to
            give each environment a distinct forest.
        growing_forest: If ``True``, enable the growing–density mode.
        env_cfg: Optional dictionary with raw parameters.  The following keys
            are recognized (all optional): ``x_lower``, ``x_upper``, ``y_lower``,
            ``y_upper``, ``tree_radius``, ``tree_height``, ``dens_min``,
            ``dens_max``, ``num_trees``, ``num_trees_eval``.
        device: Torch device for the resulting tensors.

    Returns:
        cylinders: Forest layouts tensor of shape ``(F, T, 3)``.
        total_forests: Number of distinct forests.
        generator: The :class:`ForestGenerator` instance that produced them.
    """
    cfg_dict = {} if env_cfg is None else dict(env_cfg)

    # Select number of trees depending on train / eval mode.
    num_trees_key = "num_trees_eval" if evaluation else "num_trees"
    num_trees = cfg_dict.get(num_trees_key, None)

    cfg = ForestConfig(
        x_lower=float(cfg_dict.get("x_lower", 0.0)),
        x_upper=float(cfg_dict.get("x_upper", 500.0 if evaluation else 150.0)),
        y_lower=float(cfg_dict.get("y_lower", -50.0)),
        y_upper=float(cfg_dict.get("y_upper", 50.0)),
        tree_radius=float(cfg_dict.get("tree_radius", 0.75)),
        tree_height=float(cfg_dict.get("tree_height", 50.0)),
        dens_min=float(cfg_dict.get("dens_min", 0.0)),
        dens_max=float(cfg_dict.get("dens_max", 4.0 if not evaluation else 5.0)),
        num_trees=int(num_trees) if num_trees is not None else ForestConfig.num_trees,
    )

    gen = ForestGenerator(
        num_envs=num_envs,
        evaluation=evaluation,
        unique_forests_eval=unique_forests_eval,
        config=cfg,
        growing_forest=growing_forest,
        device=device,
    )

    cylinders, total_forests = gen.generate()
    return cylinders, total_forests, gen
