"""
Regression tests for the DepthSolver stale-field trap (2026-08-10).

gstaichi kernels bind the fields they reference at FIRST compilation and keep
reading those exact fields forever — reassigning ``self.cyl_f`` to a new field
(as ``_ensure_input_buffers`` does when the per-forest tree count T changes)
leaves the compiled kernel on the old data. During the outer-loop exam the
forest override changed T 250→825, so every exam was flown on the stale
pre-override forest: the policy dodged phantoms while collision used the real
trees ("blind exam" — 60–72 m mean-free-path ceiling in batch_2/batch_3).

Contract after the fix:
* ``DepthSolver`` (taichi backend) RAISES if compute_depth sees a different
  (B, T) after its kernels were compiled — never silently reads stale trees;
* ``DepthSolver.needs_rebuild(B, T)`` tells callers a fresh instance is needed;
* ``WingedDroneEnv._depth_solver_for`` transparently swaps in a fresh solver
  when the current cylinder tensor no longer matches (fresh instance ⇒ fresh
  kernel bindings), so depth always tracks the live forest;
* the torch backend is shape-dynamic and needs none of this.

Needs the mygenesis container (gstaichi + CUDA).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import gstaichi as ti

from winged_drone_train.perception.depth import DepthSolver


DEVICE = "cuda"


@pytest.fixture(scope="module", autouse=True)
def _ti_runtime():
    ti.init(arch=ti.gpu)
    yield


def _solver(backend: str) -> DepthSolver:
    return DepthSolver(
        num_sectors=8,
        cone_angle_deg=90.0,
        max_distance=30.0,
        short_range=0.0,
        tree_radius=0.5,
        y_lower=-50.0,
        y_upper=50.0,
        torch_device=DEVICE,
        backend=backend,
    )


def _pose(B: int = 1):
    """Drone at the origin, level, facing +x."""
    return (
        torch.zeros((B, 3), device=DEVICE),
        torch.zeros((B, 3), device=DEVICE),
    )


def _trees(*xy) -> torch.Tensor:
    """(1, T, 2) cylinder tensor from (x, y) pairs."""
    return torch.tensor([list(map(list, xy))], device=DEVICE, dtype=torch.float32)


# --------------------------------------------------------------- torch backend

def test_torch_backend_tracks_tree_count_changes():
    solver = _solver("torch")
    pos, euler = _pose()

    d1 = solver.compute_depth(pos, euler, _trees((5.0, 0.0))).clone()
    # nearest surface dead ahead: 5.0 - tree_radius
    assert abs(float(d1.min()) - 4.5) < 0.05

    d2 = solver.compute_depth(
        pos, euler, _trees((5.0, 0.0), (3.0, 0.0), (20.0, 5.0))
    ).clone()
    assert abs(float(d2.min()) - 2.5) < 0.05


# -------------------------------------------------------------- taichi backend

def test_taichi_solver_refuses_tree_count_change_after_compile():
    # The trap this guards: at HEAD the second call silently reused the
    # kernel bound to the T=1 field and returned depth for the OLD tree.
    solver = _solver("taichi")
    pos, euler = _pose()

    d1 = solver.compute_depth(pos, euler, _trees((5.0, 0.0))).clone()
    assert abs(float(d1.min()) - 4.5) < 0.05

    with pytest.raises(RuntimeError, match="fresh DepthSolver"):
        solver.compute_depth(pos, euler, _trees((5.0, 0.0), (3.0, 0.0)))


def test_taichi_solver_same_shape_recompute_is_fine():
    solver = _solver("taichi")
    pos, euler = _pose()

    solver.compute_depth(pos, euler, _trees((5.0, 0.0)))
    d = solver.compute_depth(pos, euler, _trees((2.0, 0.0))).clone()
    # same T, new data: must reflect the moved tree, not the first upload
    assert abs(float(d.min()) - 1.5) < 0.05
    assert not solver.needs_rebuild(1, 1)


def test_needs_rebuild_reports_shape_binding():
    solver = _solver("taichi")
    pos, euler = _pose()

    assert not solver.needs_rebuild(1, 1)      # nothing compiled yet
    solver.compute_depth(pos, euler, _trees((5.0, 0.0)))
    assert not solver.needs_rebuild(1, 1)      # same shapes: fine
    assert solver.needs_rebuild(1, 2)          # T changed: rebuild needed
    assert solver.needs_rebuild(2, 1)          # B changed: rebuild needed

    torch_solver = _solver("torch")
    torch_solver.compute_depth(pos, euler, _trees((5.0, 0.0)))
    assert not torch_solver.needs_rebuild(1, 99)   # torch path is dynamic


# ------------------------------------------------------- env-level transparency

class _DepthEnvStub:
    """Duck-typed WingedDroneEnv carrying just the depth-solver plumbing."""

    def __init__(self, backend: str):
        self._backend = backend
        self.num_envs = 1
        self.depth_solver = self._make_depth_solver()
        self.n_rebuilds = 0

    def _make_depth_solver(self):
        if hasattr(self, "depth_solver"):
            self.n_rebuilds += 1
        return _solver(self._backend)


def test_env_swaps_in_fresh_solver_when_tree_count_changes():
    from winged_drone_train.env import WingedDroneEnv

    stub = _DepthEnvStub("taichi")
    pos, euler = _pose()
    first = stub.depth_solver

    cyl1 = _trees((5.0, 0.0))
    solver = WingedDroneEnv._depth_solver_for(stub, cyl1)
    assert solver is first                      # nothing compiled yet
    d1 = solver.compute_depth(pos, euler, cyl1).clone()
    assert abs(float(d1.min()) - 4.5) < 0.05

    # exam-style override: tree count grows — a fresh solver must be swapped
    # in, and its output must match the shape-dynamic torch reference
    cyl2 = _trees((5.0, 0.0), (3.0, 0.0), (20.0, -4.0))
    solver2 = WingedDroneEnv._depth_solver_for(stub, cyl2)
    assert solver2 is not first
    assert stub.n_rebuilds == 1
    d2 = solver2.compute_depth(pos, euler, cyl2).clone()
    ref = _solver("torch").compute_depth(pos, euler, cyl2)
    assert torch.allclose(d2, ref, atol=1e-3)

    # restore path: back to the original tree count — swaps again, still live
    solver3 = WingedDroneEnv._depth_solver_for(stub, cyl1)
    assert solver3 is not solver2
    d3 = solver3.compute_depth(pos, euler, cyl1).clone()
    assert torch.allclose(d3, d1, atol=1e-3)


def test_env_keeps_solver_when_shapes_stable():
    from winged_drone_train.env import WingedDroneEnv

    stub = _DepthEnvStub("taichi")
    pos, euler = _pose()
    cyl = _trees((5.0, 0.0))

    s1 = WingedDroneEnv._depth_solver_for(stub, cyl)
    s1.compute_depth(pos, euler, cyl)
    s2 = WingedDroneEnv._depth_solver_for(stub, cyl)
    assert s2 is s1
    assert stub.n_rebuilds == 0
