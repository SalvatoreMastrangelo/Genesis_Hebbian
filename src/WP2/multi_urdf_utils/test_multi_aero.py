"""
Test: verify that N independent SimpleDroneAeroSolver instances can coexist
in a single Genesis scene without Taichi field collisions or cross-contamination.

Run this BEFORE the full benchmark to validate the multi-solver approach.

Usage:
    python -m WP2.multi_urdf_utils.test_multi_aero --urdf-a <path> --urdf-b <path>

If no URDFs are provided, generates two random ones from the catalog.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["GS_PARA_LEVEL"] = "4"

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def test_multi_aero_independence(urdf_a: str, urdf_b: str, num_envs: int = 4):
    """Verify that two aero solver instances produce independent forces.

    Steps:
    1. Build one scene with two URDF entities.
    2. Create two independent SimpleDroneAeroSolver instances.
    3. Set different throttles (1.0 vs 0.0).
    4. Step and verify positions diverge.
    """
    import torch
    import genesis as gs
    import re

    device = "cuda:0"

    if not gs._initialized:
        gs.init(logging_level="warning", backend=gs.gpu)

    from genesis.assets.urdf.aero_model import DroneAeroModel
    from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroSolver
    from winged_drone_train.aero_profile import resolve_aero_config
    from WP2.multi_urdf_utils.multi_drone_env import _resolve_servo_joint_names
    from morph_evolution.chromosome_drone import Chromosome_Drone

    print(f"[test] URDF A: {urdf_a}")
    print(f"[test] URDF B: {urdf_b}")
    print(f"[test] n_envs: {num_envs}")

    # Build scene with 2 entities
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.04, substeps=4),
        vis_options=gs.options.VisOptions(enable_rendering=False),
        rigid_options=gs.options.RigidOptions(
            dt=0.04,
            constraint_solver=gs.constraint_solver.CG,
            enable_collision=False,
            enable_joint_limit=True,
        ),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )

    init_pos = [0.0, 0.0, 15.0]

    def _load_aero_config(urdf_path):
        aero_cfg = resolve_aero_config("simple")
        yaml_path = Path(urdf_path).parent / "aero_parameters.yaml"
        if yaml_path.is_file():
            import yaml
            with open(yaml_path) as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                aero_cfg = loaded
        return aero_cfg

    model_a = DroneAeroModel(urdf_a, config_override=_load_aero_config(urdf_a))
    model_b = DroneAeroModel(urdf_b, config_override=_load_aero_config(urdf_b))

    servo_names_a = _resolve_servo_joint_names(urdf_a)
    servo_names_b = _resolve_servo_joint_names(urdf_b)

    entity_a = scene.add_entity(gs.morphs.URDF(
        file=urdf_a, pos=init_pos, collision=False, merge_fixed_links=True,
        links_to_keep=model_a.required_links(servo_names_a),
    ))
    entity_b = scene.add_entity(gs.morphs.URDF(
        file=urdf_b, pos=init_pos, collision=False, merge_fixed_links=True,
        links_to_keep=model_b.required_links(servo_names_b),
    ))

    scene.build(n_envs=num_envs)
    print(f"[test] Scene built: {entity_a.n_links} links (A), {entity_b.n_links} links (B)")

    # Create 2 independent aero solvers and register with simulator
    # so they get called during scene.step() substeps automatically
    solver_a = SimpleDroneAeroSolver(scene, scene.sim)
    solver_a.add_target(entity_a, drone_model=model_a)

    # Check NACA override for A
    match_a = re.search(r"\[([^\]]+)\]\.urdf$", urdf_a)
    if match_a:
        try:
            vals = [float(x) for x in match_a.group(1).split(",")]
            naca = Chromosome_Drone.naca_from_physical(vals)
            if naca and hasattr(solver_a, "apply_naca_wing_override"):
                solver_a.apply_naca_wing_override(naca)
        except Exception:
            pass

    # Build and register solver A with the simulator
    solver_a.build()
    scene.sim._active_solvers.append(solver_a)

    solver_b = SimpleDroneAeroSolver(scene, scene.sim)
    solver_b.add_target(entity_b, drone_model=model_b)

    match_b = re.search(r"\[([^\]]+)\]\.urdf$", urdf_b)
    if match_b:
        try:
            vals = [float(x) for x in match_b.group(1).split(",")]
            naca = Chromosome_Drone.naca_from_physical(vals)
            if naca and hasattr(solver_b, "apply_naca_wing_override"):
                solver_b.apply_naca_wing_override(naca)
        except Exception:
            pass

    # Build and register solver B with the simulator
    solver_b.build()
    scene.sim._active_solvers.append(solver_b)

    print(f"[test] Solver A span: {solver_a.tip_to_tip:.3f} m")
    print(f"[test] Solver B span: {solver_b.tip_to_tip:.3f} m")

    # Record initial positions
    pos_a_init = entity_a.get_pos().clone()
    pos_b_init = entity_b.get_pos().clone()
    print(f"[test] Initial pos A: {pos_a_init[0].cpu().numpy()}")
    print(f"[test] Initial pos B: {pos_b_init[0].cpu().numpy()}")

    # Set different throttles: A=full, B=zero
    thr_a = torch.ones(num_envs, device=device)
    thr_b = torch.zeros(num_envs, device=device)

    print("[test] Running 50 steps: A throttle=1.0, B throttle=0.0 ...")

    for step in range(50):
        solver_a.set_throttle(thr_a)
        solver_b.set_throttle(thr_b)
        scene.step()

    pos_a_final = entity_a.get_pos()
    pos_b_final = entity_b.get_pos()

    dz_a = (pos_a_final[:, 2] - pos_a_init[:, 2]).mean().item()
    dz_b = (pos_b_final[:, 2] - pos_b_init[:, 2]).mean().item()

    print(f"[test] Final pos A: {pos_a_final[0].cpu().numpy()}")
    print(f"[test] Final pos B: {pos_b_final[0].cpu().numpy()}")
    print(f"[test] Delta-Z A (thrust=1.0): {dz_a:+.4f} m")
    print(f"[test] Delta-Z B (thrust=0.0): {dz_b:+.4f} m")

    # Verify independence
    passed = True

    # Positions must differ — solvers are independent
    if torch.allclose(pos_a_final, pos_b_final, atol=0.01):
        print("[FAIL] Drone A and B have identical final positions — solvers may be coupled")
        passed = False

    # Trajectories must differ meaningfully (different throttle → different outcome)
    pos_diff = (pos_a_final - pos_b_final).norm(dim=1).mean().item()
    print(f"[test] Mean position difference: {pos_diff:.4f} m")
    if pos_diff < 0.1:
        print("[FAIL] Position difference too small — throttle may not be affecting drones independently")
        passed = False

    # Check no NaN
    if torch.isnan(pos_a_final).any() or torch.isnan(pos_b_final).any():
        print("[FAIL] NaN detected in final positions")
        passed = False

    if passed:
        print("\n[PASS] Multi-aero solver independence verified!")
    else:
        print("\n[FAIL] Multi-aero solver test FAILED — see above for details")

    gs.destroy()
    return passed


def main():
    parser = argparse.ArgumentParser(description="Test multi-aero solver independence")
    parser.add_argument("--urdf-a", type=str, default=None)
    parser.add_argument("--urdf-b", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()

    urdf_a = args.urdf_a
    urdf_b = args.urdf_b

    # Generate URDFs if not provided
    if urdf_a is None or urdf_b is None:
        import genesis as gs
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)

        from general_policy.catalog import build_catalog
        catalog_dir = Path("logs/.cache/wp2_5_test_urdfs")
        paths = build_catalog(catalog_dir, n=2, seed=42, include_standard_mydrone=True)
        gs.destroy()

        urdf_a = str(paths[0])
        urdf_b = str(paths[1]) if len(paths) > 1 else str(paths[0])

    success = test_multi_aero_independence(urdf_a, urdf_b, args.num_envs)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
