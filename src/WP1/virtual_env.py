"""
VirtualMultiSceneEnv — aggregate S parallel Genesis scenes into one env.
=========================================================================

Presents the standard RSL-RL environment interface to ``OnPolicyRunner``
while distributing physics simulation across S worker subprocesses, each
owning one compiled Genesis scene.

Architecture
------------
::

    OnPolicyRunner
        │ step(actions)            ← receives (S*E, num_actions) from runner
        │ reset()
        ▼
    VirtualMultiSceneEnv
        ├── [Fan-out actions to all S workers simultaneously]
        │       in_q[0] ← actions[0:E]
        │       in_q[1] ← actions[E:2E]
        │       ...
        ├── [Wait for all workers to complete their physics step]
        │       obs[0], rew[0], done[0] ← out_q[0]
        │       obs[1], rew[1], done[1] ← out_q[1]
        │       ...
        └── [Concatenate along env dimension → (S*N*E, *)]
                return obs (S*N*E, num_obs), rew (S*N*E,), done (S*N*E,), extras

All S workers execute their ``env.step()`` concurrently.  The main process
blocks until **all** results are collected — this is the hard synchronisation
point between PPO iterations.

Compilation phase
-----------------
Workers are spawned at ``__init__`` time and compile their scenes in
parallel.  The coordinator blocks until every worker sends ``"READY"``.
This means the first ``VirtualMultiSceneEnv.__init__()`` call may take
several minutes, but all subsequent iterations are fast.

URDF assignment
---------------
If a URDF catalog is provided (``catalog_path`` argument), S*N URDFs are
assigned round-robin from the catalog.  Scene *i* gets URDFs at positions
``[i*N … (i+1)*N-1] % n_urdfs``.  When no catalog is used, all scenes share
the default morphing-drone URDF (N=1 only; N>1 requires a catalog).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch


class VirtualMultiSceneEnv:
    """RSL-RL–compatible env aggregating S parallel scene workers.

    Parameters
    ----------
    cfg : RunConfig
        Full training configuration.  ``cfg.multi_scene`` controls S, E, and
        ``cpu_threads_per_worker``.
    catalog_path : Path, optional
        Directory containing ``catalog.txt`` and per-URDF sub-folders.
        When provided, URDFs are assigned round-robin across workers.
    """

    def __init__(self, cfg, catalog_path: Optional[Path] = None, num_gpus: int = 1) -> None:
        from WP1.scene_worker_process import worker_main
        from winged_drone_train.defaults import default_mydrone_urdf_path

        ms = cfg.multi_scene
        S = ms.S
        N = ms.N   # URDFs per scene (N=1 → WingedDroneEnv, N>1 → MultiDroneEnv)
        E = ms.E
        device = cfg.training.device

        # Multi-GPU: assign each scene worker to a GPU (round-robin)
        self._num_gpus = num_gpus

        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = cfg.to_legacy_cfgs()

        # Ensure catalog_path is absolute (important for worker subprocesses with different CWD)
        if catalog_path is not None and not catalog_path.is_absolute():
            catalog_path = catalog_path.resolve()

        # ------------------------------------------------------------------ #
        # URDF assignment: S scenes × N URDFs each                           #
        # ------------------------------------------------------------------ #
        # Total URDFs needed = S * N; assign round-robin from catalog
        urdf_paths_flat = _assign_urdfs(S * N, catalog_path, default_mydrone_urdf_path)
        print(f"[VirtualMultiSceneEnv] URDF assignment: total {len(urdf_paths_flat)} URDFs assigned")
        for i, p in enumerate(urdf_paths_flat):
            print(f"  [{i}] {p}")

        # Group into S lists of N paths each
        urdf_paths_per_scene: List[List[str]] = [
            urdf_paths_flat[i * N : (i + 1) * N] for i in range(S)
        ]
        print(f"[VirtualMultiSceneEnv] Scene -> URDF mapping:")
        for scene_idx, paths in enumerate(urdf_paths_per_scene):
            print(f"  Scene {scene_idx}: {len(paths)} URDFs")

        # ------------------------------------------------------------------ #
        # Benchmark instrumentation: record VRAM and wall-clock before spawn  #
        # ------------------------------------------------------------------ #
        try:
            _vram_before_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            _vram_before_mb = 0.0
        _compile_wall_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Spawn workers                                                        #
        # ------------------------------------------------------------------ #
        src_dir = str(Path(__file__).resolve().parent.parent)  # …/src

        ctx = mp.get_context("spawn")
        self._in_qs: List[mp.Queue] = []
        self._out_qs: List[mp.Queue] = []
        self._procs: List[mp.Process] = []

        total_envs = S * N * E
        print(f"[VirtualMultiSceneEnv] **Spawning {S} scene workers** "
              f"(N={N} URDFs, E={E} envs each, total {total_envs} envs) ...")

        for i in range(S):
            in_q: mp.Queue = ctx.Queue()
            out_q: mp.Queue = ctx.Queue()
            # Assign each scene to a GPU (round-robin if num_gpus > 1)
            gpu_id = i % num_gpus
            kwargs: Dict[str, Any] = dict(
                scene_idx=i,
                urdf_paths=urdf_paths_per_scene[i],
                E=E,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                device=device,
                src_dir=src_dir,
                in_q=in_q,
                out_q=out_q,
                cpu_threads_per_worker=ms.cpu_threads_per_worker,
                gpu_id=gpu_id,
                num_gpus=num_gpus,
            )
            # Pass full cfg only when N>1 (MultiDroneEnv needs it)
            if N > 1:
                kwargs["wp1_cfg"] = cfg
                print(f"[VirtualMultiSceneEnv] Scene {i}: will use MultiDroneEnv (N>1)")
            p = ctx.Process(target=worker_main, kwargs=kwargs, daemon=True)
            p.start()
            gpu_info = f" (GPU {gpu_id})" if num_gpus > 1 else ""
            print(f"[VirtualMultiSceneEnv] Scene {i} process started (PID {p.pid}){gpu_info}")
            self._in_qs.append(in_q)
            self._out_qs.append(out_q)
            self._procs.append(p)

        # ------------------------------------------------------------------ #
        # Wait for all workers to finish compilation                          #
        # ------------------------------------------------------------------ #
        print(f"[VirtualMultiSceneEnv] Waiting for {S} scenes to compile ...")
        meta: Optional[Dict] = None
        self._worker_ready_metas: List[Dict] = []
        for i, out_q in enumerate(self._out_qs):
            msg = out_q.get(timeout=900)  # 15-minute timeout for slow nodes
            if not isinstance(msg, dict) or msg.get("status") != "READY":
                raise RuntimeError(
                    f"Worker {i} sent unexpected ready message: {msg}"
                )
            self._worker_ready_metas.append(msg)
            if meta is None:
                meta = msg  # use first worker's metadata
            print(f"[VirtualMultiSceneEnv]  scene {i} ready")

        # Record wall-clock compile time and VRAM delta
        self._compile_wall_time_s: float = time.perf_counter() - _compile_wall_start
        try:
            _vram_after_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            _vram_after_mb = _vram_before_mb
        self._vram_allocated_after_compile_mb: float = _vram_after_mb
        self._vram_delta_compile_mb: float = _vram_after_mb - _vram_before_mb

        print(f"[VirtualMultiSceneEnv] All {S} scenes compiled and ready.")

        # ------------------------------------------------------------------ #
        # Expose env properties expected by RSL-RL OnPolicyRunner             #
        # ------------------------------------------------------------------ #
        assert meta is not None
        self.num_envs: int = S * N * E    # total = scenes × URDFs × envs
        self.num_obs: int = meta["num_obs"]
        self.num_privileged_obs: Optional[int] = meta["num_privileged_obs"]
        self.num_actions: int = meta["num_actions"]
        self.max_episode_length: int = meta["max_episode_length"]
        self.device = torch.device(device)

        # extras dict — updated on every step/reset, read by runner
        self.extras: Dict[str, Any] = {"observations": {}}

        # Cached observations (initialized on first reset/step)
        self._obs: Optional[torch.Tensor] = None

        # Episode length tracking (for RSL-RL runner)
        self.episode_length_buf: torch.Tensor = torch.zeros(
            total_envs, dtype=torch.long, device=torch.device(device)
        )

        # Internal bookkeeping
        self._S = S
        self._N = N
        self._E = E
        self._device = device

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset all S scenes and return concatenated initial observations."""
        for i, in_q in enumerate(self._in_qs):
            in_q.put(("RESET",))

        obs_list: List[torch.Tensor] = []
        priv_obs_list: List[Optional[torch.Tensor]] = []

        for i, out_q in enumerate(self._out_qs):
            r = out_q.get()
            # Check if worker encountered OOM
            if r.get("status") == "OOM" or r.get("error") == "OOM":
                raise RuntimeError(f"Worker {i} OOM: out of memory during reset")
            if "error" in r and "OOM" in r.get("error", ""):
                raise RuntimeError(f"Worker {i} OOM: {r.get('error')}")
            obs_list.append(r["obs"])
            priv_obs_list.append(r.get("priv_obs"))

        obs = torch.cat(obs_list, dim=0).to(self.device)

        self.extras = {"observations": {}}
        if priv_obs_list[0] is not None:
            priv_obs = torch.cat(
                [p for p in priv_obs_list if p is not None], dim=0
            ).to(self.device)
            self.extras["observations"]["critic"] = priv_obs

        self._obs = obs
        self.episode_length_buf.zero_()
        return obs, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Fan-out actions, wait for all workers, concatenate results.

        Parameters
        ----------
        actions : (S*E, num_actions)
            Actions for all environments from the RSL-RL runner.

        Returns
        -------
        obs : (S*E, num_obs)
        rew : (S*E,)
        done : (S*E,)  bool
        extras : dict
        """
        NE = self._N * self._E   # envs per worker = N URDFs × E envs each

        # Fan-out — all workers receive their slice simultaneously
        for i, in_q in enumerate(self._in_qs):
            a_slice = actions[i * NE : (i + 1) * NE].cpu()
            in_q.put(("STEP", a_slice))

        # Collect — blocks until every worker responds
        obs_list: List[torch.Tensor] = []
        rew_list: List[torch.Tensor] = []
        done_list: List[torch.Tensor] = []
        priv_obs_list: List[Optional[torch.Tensor]] = []
        time_outs_list: List[Optional[torch.Tensor]] = []
        episode_dicts: List[Dict[str, float]] = []

        for i, out_q in enumerate(self._out_qs):
            r = out_q.get()
            # Check if worker encountered OOM
            if r.get("status") == "OOM" or r.get("error") == "OOM":
                raise RuntimeError(f"Worker {i} OOM: out of memory during step")
            if "error" in r and "OOM" in r.get("error", ""):
                raise RuntimeError(f"Worker {i} OOM: {r.get('error')}")
            obs_list.append(r["obs"])
            rew_list.append(r["rew"])
            done_list.append(r["done"])
            priv_obs_list.append(r.get("priv_obs"))
            time_outs_list.append(r.get("time_outs"))
            episode_dicts.append(r.get("episode", {}))

        dev = self.device
        obs = torch.cat(obs_list, dim=0).to(dev)
        rew = torch.cat(rew_list, dim=0).to(dev)
        done = torch.cat(done_list, dim=0).to(dev)

        self._obs = obs

        # Update episode length buffer and reset for done environments
        self.episode_length_buf += 1
        self.episode_length_buf[done] = 0

        self.extras = {"observations": {}}

        if priv_obs_list[0] is not None:
            priv_obs = torch.cat(
                [p for p in priv_obs_list if p is not None], dim=0
            ).to(dev)
            self.extras["observations"]["critic"] = priv_obs

        if time_outs_list[0] is not None:
            self.extras["time_outs"] = torch.cat(
                [t for t in time_outs_list if t is not None], dim=0
            ).to(dev)

        # Average episode metrics across scenes (scalars only)
        if any(episode_dicts):
            merged: Dict[str, List[float]] = {}
            for ep in episode_dicts:
                for k, v in ep.items():
                    merged.setdefault(k, []).append(v)
            self.extras["episode"] = {
                k: sum(v) / len(v) for k, v in merged.items()
            }

        return obs, rew, done, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Return cached observations and extras.

        If no observations exist yet (before first reset/step), triggers reset.
        """
        if self._obs is None:
            # First call to get_observations before any reset/step — auto-reset
            return self.reset()
        return self._obs, self.extras

    # ------------------------------------------------------------------ #
    # Graceful shutdown                                                     #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Send STOP to all workers and join their processes."""
        for in_q in getattr(self, "_in_qs", []):
            try:
                in_q.put(("STOP",))
            except Exception:
                pass
        for p in getattr(self, "_procs", []):
            try:
                p.join(timeout=15)
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _assign_urdfs(
    S: int,
    catalog_path: Optional[Path],
    default_urdf_fn,
) -> List[str]:
    """Return a list of S URDF file paths, one per scene.

    If a catalog directory is provided the URDFs listed in
    ``catalog.txt`` are assigned round-robin.  Otherwise all scenes use
    the default morphing-drone URDF.
    """
    if catalog_path is not None:
        # Ensure catalog_path is absolute (important for worker subprocesses with different CWD)
        if not catalog_path.is_absolute():
            catalog_path = catalog_path.resolve()

        if catalog_path.is_dir():
            catalog_txt = catalog_path / "catalog.txt"
            if catalog_txt.exists():
                lines = [l.strip() for l in catalog_txt.read_text().splitlines() if l.strip()]
                if lines:
                    # Lines may be bare filenames or full paths; resolve relative to catalog_path
                    resolved = []
                    for line in lines:
                        p = Path(line)
                        if not p.is_absolute():
                            p = catalog_path / p
                        # Ensure each URDF path is absolute
                        p = p.resolve()
                        resolved.append(str(p))
                    return [resolved[i % len(resolved)] for i in range(S)]
            # Fall back: glob for URDF files
            urdf_files = sorted(catalog_path.rglob("*.urdf"))
            if urdf_files:
                # Resolve all paths to absolute and round-robin assign to S scenes
                resolved_paths = [str(urdf_file.resolve()) for urdf_file in urdf_files]
                return [resolved_paths[i % len(resolved_paths)] for i in range(S)]

    # Single-morphology: all scenes use the same URDF
    default_urdf = str(Path(default_urdf_fn()).resolve())
    return [default_urdf] * S
