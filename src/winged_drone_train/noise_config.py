from __future__ import annotations

from typing import Any, Dict

from winged_drone_train.env import WingedDroneEnv


def configure_solver_noise(env: WingedDroneEnv, env_cfg: Dict[str, Any]) -> None:
    """
    Configure aerodynamic noise and aerodynamic parameter randomization.

    This helper is shared between training and evaluation so both paths use
    the same noise magnitudes.
    """
    aero_solver = getattr(env, "aero_solver", None)
    if aero_solver is None:
        return

    aero_noise_enabled = bool(env_cfg.get("aero_noise", False))
    if hasattr(aero_solver, "_enable_noise"):
        aero_solver._enable_noise = aero_noise_enabled

    sigma0 = float(env_cfg.get("aero_noise_sigma0", 0.0))
    if aero_noise_enabled:
        sigma_mag = sigma0
        sigma_dir = sigma0
        sigma_param = float(env_cfg.get("noise_sigma_param", 0.0))
    else:
        sigma_mag = 0.0
        sigma_dir = 0.0
        sigma_param = 0.0

    if hasattr(env, "set_noise_settings"):
        env.set_noise_settings(
            aero_sigma_mag=sigma_mag,
            aero_sigma_dir=sigma_dir,
            aero_sigma_param=sigma_param,
            enable_aero_param_noise=aero_noise_enabled,
        )
    else:
        if hasattr(aero_solver, "noise_sigma_mag"):
            aero_solver.noise_sigma_mag = sigma_mag
        if hasattr(aero_solver, "noise_sigma_dir"):
            aero_solver.noise_sigma_dir = sigma_dir
        if hasattr(aero_solver, "noise_sigma_param"):
            aero_solver.noise_sigma_param = sigma_param

    if env_cfg.get("debug", False):
        print(
            f"[configure_solver_noise] Aero noise enabled: {aero_noise_enabled}, "
            f"sigma_mag: {getattr(aero_solver, 'noise_sigma_mag', 'N/A')}, "
            f"sigma_dir: {getattr(aero_solver, 'noise_sigma_dir', 'N/A')}, "
            f"sigma_param: {getattr(aero_solver, 'noise_sigma_param', 'N/A')}"
        )
