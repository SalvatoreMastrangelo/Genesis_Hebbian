# utils/depth.py
# -----------------------------------------------------------------------------
# Ultra-fast batched depth sensing for Genesis drone envs (GPU with gstaichi).
#
# Rays: S fan-shaped, horizontal in body frame (+X forward, +Y left).
# Intersections with:
#   - vertical cylinders (tree trunks, radius = tree_radius)
#   - lateral walls at y = [y_lower, y_upper]
# FOV shrinks with roll: half_eff = half_nom * cos(roll).
#
# Implementation notes:
#   - No kernel arguments (gstaichi compatibility). Kernels read/write class fields.
#   - Single kernel over (B, S); inner loop over trees (T).
#   - Inputs are Taichi fields filled from torch via .from_torch().
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Optional

import torch
import gstaichi as ti


@ti.data_oriented
class DepthSolver:
    """Batched depth sensing with Taichi kernels."""

    def __init__(
        self,
        num_sectors: int,
        cone_angle_deg: float,
        *,
        max_distance: float = 30.0,
        short_range: float = 0.0,
        tree_radius: float = 1.0,
        y_lower: float = -50.0,
        y_upper: float = 50.0,
        torch_device: torch.device | str = "cuda",
    ) -> None:
        if num_sectors <= 0:
            raise ValueError("num_sectors must be positive")

        self.S = int(num_sectors)
        self.cone_angle_deg = float(cone_angle_deg)
        self._torch_device = torch.device(torch_device)

        # Scalar params (kernel-friendly 0D fields)
        self.max_distance = ti.field(dtype=ti.f32, shape=())
        self.short_range = ti.field(dtype=ti.f32, shape=())
        self.tree_radius = ti.field(dtype=ti.f32, shape=())
        self.y_lower = ti.field(dtype=ti.f32, shape=())
        self.y_upper = ti.field(dtype=ti.f32, shape=())
        self.half_cone_nom = ti.field(dtype=ti.f32, shape=())
        self.T_f = ti.field(dtype=ti.i32, shape=())  # active tree count for kernels

        self.max_distance[None] = float(max_distance)
        self.short_range[None] = float(short_range)
        self.tree_radius[None] = float(tree_radius)
        self.y_lower[None] = float(y_lower)
        self.y_upper[None] = float(y_upper)
        self.half_cone_nom[None] = 0.5 * math.radians(self.cone_angle_deg)
        self.T_f[None] = 0

        # Precomputed body-frame ray directions (cos θ, sin θ)
        self.rays_body = ti.Vector.field(2, dtype=ti.f32, shape=(self.S,))
        self._init_ray_dirs()

        # Input fields (allocated per (B, T))
        self.pos_x_f = None   # (B,)
        self.pos_y_f = None   # (B,)
        self.roll_f = None    # (B,)
        self.yaw_f = None     # (B,)
        self.cyl_f = None     # (B, max(T,1)) of vec2

        # Output field (B, S), allocated per B
        self._B_alloc = 0
        self.depth = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_depth(
        self,
        base_pos: torch.Tensor,       # (B, 3)
        base_euler: torch.Tensor,     # (B, 3) roll, pitch, yaw
        cyl_xy_b: Optional[torch.Tensor] = None,  # (B, T, 2) or None
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        if base_pos.ndim != 2 or base_pos.shape[1] != 3:
            raise ValueError("base_pos must be (B, 3)")
        if base_euler.ndim != 2 or base_euler.shape[1] != 3:
            raise ValueError("base_euler must be (B, 3)")

        B = int(base_pos.shape[0])
        T = 0 if cyl_xy_b is None else int(cyl_xy_b.shape[1])

        self._ensure_buffers(B)
        self._ensure_input_buffers(B, T)

        # Upload inputs (one device sync per step)
        pos_x = base_pos[:, 0].contiguous().to(self._torch_device, dtype=torch.float32)
        pos_y = base_pos[:, 1].contiguous().to(self._torch_device, dtype=torch.float32)
        roll  = base_euler[:, 0].contiguous().to(self._torch_device, dtype=torch.float32)
        yaw   = base_euler[:, 2].contiguous().to(self._torch_device, dtype=torch.float32)

        self.pos_x_f.from_torch(pos_x)
        self.pos_y_f.from_torch(pos_y)
        self.roll_f.from_torch(roll)
        self.yaw_f.from_torch(yaw)

        if T > 0:
            if cyl_xy_b.ndim != 3 or cyl_xy_b.shape[0] != B or cyl_xy_b.shape[2] != 2:
                raise ValueError("cyl_xy_b must be (B, T, 2)")
            self.cyl_f.from_torch(
                cyl_xy_b.contiguous().to(self._torch_device, dtype=torch.float32)
            )
        self.T_f[None] = int(T)

        # Clear and compute
        self._k_clear()
        self._kernel_depth()

        depth = self.depth.to_torch(device=str(self._torch_device))
        if noise_std > 0.0:
            depth = torch.clamp(
                depth + noise_std * torch.randn_like(depth, device=depth.device),
                min=0.0,
                max=float(self.max_distance[None]),
            )
        return depth

    # ------------------------------------------------------------------ #
    # Setup / housekeeping                                               #
    # ------------------------------------------------------------------ #
    def _init_ray_dirs(self) -> None:
        """rays_body[s] = (cos θ, sin θ), θ ∈ [-cone/2, +cone/2]."""
        half = 0.5 * math.radians(self.cone_angle_deg)
        if self.S == 1:
            self.rays_body[0] = ti.Vector([1.0, 0.0])
            return
        for s in range(self.S):
            t = -half + (2.0 * half) * (s / (self.S - 1))
            self.rays_body[s] = ti.Vector([math.cos(t), math.sin(t)])

    def _ensure_buffers(self, B: int) -> None:
        """Allocate output depth field for batch size B."""
        if B <= 0:
            raise ValueError("batch B must be > 0")
        if self.depth is not None and self._B_alloc == B:
            return
        self.depth = ti.field(dtype=ti.f32, shape=(B, self.S))
        self._B_alloc = B

    def _ensure_input_buffers(self, B: int, T: int) -> None:
        """Allocate input fields for current (B, T)."""
        def need(field, shape):
            return (field is None) or (tuple(field.shape) != tuple(shape))

        if need(self.pos_x_f, (B,)):
            self.pos_x_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.pos_y_f, (B,)):
            self.pos_y_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.roll_f, (B,)):
            self.roll_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.yaw_f, (B,)):
            self.yaw_f = ti.field(dtype=ti.f32, shape=(B,))

        TT = max(T, 1)  # keep a valid shape when no trees
        if (self.cyl_f is None) or (self.cyl_f.shape[0] != B) or (self.cyl_f.shape[1] != TT):
            self.cyl_f = ti.Vector.field(2, dtype=ti.f32, shape=(B, TT))

    # ------------------------------------------------------------------ #
    # Kernels (no arguments; iterate over field shapes)                  #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _k_clear(self):
        max_d = self.max_distance[None]
        for b, s in self.depth:
            self.depth[b, s] = max_d

    @ti.kernel
    def _kernel_depth(self):
        max_d = self.max_distance[None]
        r_tree = self.tree_radius[None]
        yL = self.y_lower[None]
        yU = self.y_upper[None]
        halfNom = self.half_cone_nom[None]
        T = self.T_f[None]

        # Walls: initialize depth with nearest lateral wall
        for b, s in self.depth:
            dx_b = self.rays_body[s][0]
            dy_b = self.rays_body[s][1]

            cy = ti.cos(self.yaw_f[b])
            sy = ti.sin(self.yaw_f[b])
            dx_w = cy * dx_b - sy * dy_b
            dy_w = sy * dx_b + cy * dy_b

            if dy_w > 0.999:
                dy_w = 0.999
            if dy_w < -0.999:
                dy_w = -0.999

            py = self.pos_y_f[b]
            tL = max_d
            if dy_w < 0.0:
                tL = (yL - py) / dy_w
            tR = max_d
            if dy_w > 0.0:
                tR = (yU - py) / dy_w

            d_wall = ti.min(ti.max(tL, 0.0), ti.max(tR, 0.0))
            if d_wall > max_d:
                d_wall = max_d

            self.depth[b, s] = d_wall


        # Trees: reduce with nearest cylinder surface
        for b, s in self.depth:
            croll = ti.cos(self.roll_f[b])
            if croll < 1e-3:
                croll = 1e-3
            halfEff = halfNom * croll
            sw = 2.0 * halfEff / ti.cast(self.S, ti.f32)

            cy = ti.cos(self.yaw_f[b])
            sy = ti.sin(self.yaw_f[b])

            best = self.depth[b, s]

            for t in range(T):
                wx = self.cyl_f[b, t][0]
                wy = self.cyl_f[b, t][1]

                dx = wx - self.pos_x_f[b]
                dy = wy - self.pos_y_f[b]
                x_loc = cy * dx + sy * dy
                y_loc = -sy * dx + cy * dy

                r = ti.sqrt(x_loc * x_loc + y_loc * y_loc) + 1e-9
                r_surf = r - r_tree
                if r_surf < 0.0:
                    r_surf = 0.0

                theta = ti.atan2(y_loc, x_loc)
                ratio = r_tree / r
                if ratio > 1.0:
                    ratio = 1.0
                delta = ti.asin(ratio)

                in_cone = (-halfEff <= theta) and (theta <= halfEff) and (r <= max_d)
                near = (r <= self.short_range[None]) and (x_loc >= 0.0)
                if not (in_cone or near):
                    continue

                theta_l = theta - delta
                theta_r = theta + delta
                idx_l = ti.i32(ti.floor((theta_l + halfEff) / sw))
                idx_r = ti.i32(ti.floor((theta_r + halfEff) / sw))
                if idx_l < 0:
                    idx_l = 0
                if idx_r < 0:
                    idx_r = 0
                if idx_l > self.S - 1:
                    idx_l = self.S - 1
                if idx_r > self.S - 1:
                    idx_r = self.S - 1

                if (idx_l <= s) and (s <= idx_r):
                    if r_surf < best:
                        best = r_surf

            self.depth[b, s] = best
