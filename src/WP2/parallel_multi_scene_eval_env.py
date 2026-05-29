"""
ParallelMultiSceneEvalEnv — N worker-process parallel wrapper for WP2 evaluation.

Architecture
------------
N worker processes each own a MultiSceneEvalEnv shard of D/N URDFs and run
their Genesis scenes concurrently, exploiting GPU headroom instead of the
default sequential dispatch.

Communication is through mp.Queue pairs (cmd_q / ack_q per worker).  Data
volumes are small (D=64, E≤8 → < 200 KB per step), so queue serialisation
overhead is negligible compared to the Genesis step time.

Workers are spawned (not forked) so each process gets its own CUDA context
and Taichi runtime.  Genesis scenes are built inside workers after spawning;
they cannot be serialised or shared.

DroneStateProxy
---------------
evaluate.py accesses env.drones[d].{nan_envs, reset_buf, last_reward_total,
base_pos, power, base_lin_vel, commands, pre_collision, pre_wall_crash,
pre_angle_limit} after every step.  Workers pack those fields into a flat
float32 tensor (_N_STATE channels per (drone, env) slot) alongside
obs/rew/done.  DroneStateProxy wraps one row and exposes the same attribute
API, returning tensors on the main-process device (GPU).
"""

from __future__ import annotations

import os
import queue as _queue_mod
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp


# Default per-worker init timeout (seconds). Cold Taichi-cache builds of many
# Genesis scenes across many workers can take 10-30 min; the previous 600s
# default was too short for production-scale runs (e.g. D=40 across 10
# workers). Override at runtime via env var WP2_WORKER_INIT_TIMEOUT.
_DEFAULT_INIT_TIMEOUT_S = 1800.0

# ── drone-state buffer layout ─────────────────────────────────────────────────
_N_STATE = 14
_S_POS   = slice(0, 3)   # base_pos xyz
_S_VEL   = slice(3, 6)   # base_lin_vel xyz
_S_CMD0  = 6             # commands[:, 0]
_S_REW   = 7             # last_reward_total
_S_POW   = 8             # power
_S_NAN   = 9             # nan_envs
_S_RST   = 10            # reset_buf
_S_COLL  = 11            # pre_collision
_S_WALL  = 12            # pre_wall_crash
_S_ANG   = 13            # pre_angle_limit


def _collect_state_gpu(env) -> torch.Tensor:
    """Pack all drone state into one GPU tensor, then do a single bulk D2H copy.

    Filling on-device avoids ~10 separate D2H CUDA synchronizations per drone.
    With 32 drones per worker that drops ~320 individual copies down to one.
    """
    Ds  = env.D
    E   = env.E
    dev = env.device
    buf = torch.zeros(Ds, E, _N_STATE, device=dev)
    for d, sub in enumerate(env.drones):
        row = buf[d]
        row[:, _S_POS]  = sub.base_pos[:, :3].float()
        row[:, _S_VEL]  = sub.base_lin_vel[:, :3].float()
        row[:, _S_CMD0] = sub.commands[:, 0].float()
        row[:, _S_REW]  = sub.last_reward_total.float()
        # `self.power` is only assigned inside WingedDroneEnv.step(); it does
        # not exist after a fresh reset (no step yet). Leave the slot at 0 in
        # that case — the rollout never reads power before its first step.
        power = getattr(sub, "power", None)
        if power is not None:
            row[:, _S_POW] = power.float()
        row[:, _S_NAN]  = sub.nan_envs.float()
        row[:, _S_RST]  = sub.reset_buf.float()
        for i, attr in enumerate(("pre_collision", "pre_wall_crash", "pre_angle_limit")):
            flag = getattr(sub, attr, None)
            row[:, _S_COLL + i] = flag.float() if flag is not None else 0.0
    return buf.cpu()  # one D2H transfer for the whole shard


def _collect_reward_comp_gpu(env, n_comp: int) -> Optional[torch.Tensor]:
    """Pack per-component reward tensors across the shard; one bulk D2H copy.

    Returns None when ``n_comp == 0`` (i.e. no active reward components).
    """
    if n_comp <= 0:
        return None
    Ds  = env.D
    E   = env.E
    dev = env.device
    buf = torch.zeros(Ds, E, n_comp, device=dev)
    for d, sub in enumerate(env.drones):
        lrc = getattr(sub, "last_reward_components", None)
        if lrc is not None and lrc.shape[-1] >= n_comp:
            buf[d] = lrc[:, :n_comp].float()
    return buf.cpu()


# ── proxy ─────────────────────────────────────────────────────────────────────

class DroneStateProxy:
    """WingedDroneEnv-compatible view backed by GPU buffer rows.

    Per-step scalars (pos, vel, commands, rewards, flags) come from the
    ``(E, _N_STATE)`` slice of ``_state_buf``.  Optional per-component
    rewards come from a parallel ``(E, n_comp)`` slice of ``_comp_buf``.
    Static metadata (``nominal_mass``, ``reward_names``, ``reward_scales``)
    is captured once at worker init and stored as plain attributes.
    """

    def __init__(
        self,
        row: torch.Tensor,
        nominal_mass: float = 0.0,
        reward_names: Optional[List[str]] = None,
        reward_scales: Optional[dict] = None,
        comp_row: Optional[torch.Tensor] = None,
    ) -> None:
        self._r = row             # (E, _N_STATE) on main-process device
        self._comp = comp_row     # (E, n_comp) on main-process device, or None
        self.nominal_mass = float(nominal_mass)
        self.reward_names = list(reward_names) if reward_names else []
        self.reward_scales = dict(reward_scales) if reward_scales else {}

    @property
    def base_pos(self):          return self._r[:, _S_POS]
    @property
    def base_lin_vel(self):      return self._r[:, _S_VEL]
    @property
    def commands(self):          return self._r[:, _S_CMD0: _S_CMD0 + 1]  # (E,1)→[:,0] works
    @property
    def last_reward_total(self): return self._r[:, _S_REW]
    @property
    def power(self):             return self._r[:, _S_POW]
    @property
    def nan_envs(self):          return self._r[:, _S_NAN].bool()
    @property
    def reset_buf(self):         return self._r[:, _S_RST].bool()
    @property
    def pre_collision(self):     return self._r[:, _S_COLL].bool()
    @property
    def pre_wall_crash(self):    return self._r[:, _S_WALL].bool()
    @property
    def pre_angle_limit(self):   return self._r[:, _S_ANG].bool()
    @property
    def last_reward_components(self):
        return self._comp


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_main(
    worker_id: int,
    urdf_shard: List[str],
    env_kwargs: dict,
    extra_sys_paths: List[str],
    cmd_q: "mp.Queue[tuple]",
    ack_q: "mp.Queue[tuple]",
    gpu_id: Optional[int] = None,
) -> None:
    # Pin this worker to one physical GPU BEFORE importing Genesis. After
    # masking, the assigned GPU appears to torch/Genesis as cuda:0, which is
    # what env_kwargs["device"] already says — so internal code that uses
    # ``cuda:0`` continues to work, while different workers actually land on
    # different physical GPUs.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Restore sys.path so project imports work in the spawned process.
    for p in extra_sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"[worker {worker_id}] entered (urdfs={len(urdf_shard)}, "
          f"gpu={gpu_id if gpu_id is not None else 'default'})", flush=True)
    try:
        import genesis as gs
        from WP2.multi_scene_eval_env import MultiSceneEvalEnv

        print(f"[worker {worker_id}] gs.init() starting", flush=True)
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)
        print(f"[worker {worker_id}] gs.init() done; building "
              f"{len(urdf_shard)} sub-envs", flush=True)

        env = MultiSceneEvalEnv(urdf_paths=urdf_shard, **env_kwargs)
        print(f"[worker {worker_id}] sub-envs built; sending ready", flush=True)

        # Static metadata captured once and shipped in the "ready" ack so the
        # main-process DroneStateProxy instances can expose `nominal_mass`,
        # `reward_names`, `reward_scales` (consumed by _rollout_episode_multi_urdf).
        nominal_masses = [float(getattr(sub, "nominal_mass", 0.0)) for sub in env.drones]
        head = env.drones[0]
        reward_names: List[str] = list(getattr(head, "reward_names", []) or [])
        reward_scales: dict = {
            k: float(v) for k, v in (getattr(head, "reward_scales", {}) or {}).items()
        }
        n_comp_full = len(reward_names)
        ack_q.put((
            "ready", env.num_obs, env.num_actions, float(env.dt),
            nominal_masses, reward_names, reward_scales, n_comp_full,
        ))
    except Exception as exc:
        ack_q.put(("error", str(exc)))
        return

    while True:
        cmd, payload = cmd_q.get()
        try:
            if cmd == "shutdown":
                ack_q.put(("ok",))
                break

            elif cmd == "reset":
                seed: Optional[int] = payload
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                obs, _ = env.reset()
                ack_q.put((
                    "reset_result", obs.cpu(),
                    _collect_state_gpu(env),
                    _collect_reward_comp_gpu(env, n_comp_full),
                ))

            elif cmd == "step":
                obs, rew, done, _ = env.step(payload.to(env.device))
                ack_q.put((
                    "step_result", obs.cpu(), rew.cpu(), done.cpu(),
                    _collect_state_gpu(env),
                    _collect_reward_comp_gpu(env, n_comp_full),
                ))

            elif cmd == "refresh_forests":
                seed = payload
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                env.refresh_forests()
                ack_q.put(("ok",))

            elif cmd == "set_forest_ids":
                # CPU tensor crossed the mp.Queue; restore it to the worker's
                # GPU before WingedDroneEnv indexes into it with GPU env_ids.
                val = payload
                if isinstance(val, torch.Tensor):
                    val = val.to(env.device)
                env._fixed_forest_ids = val
                ack_q.put(("ok",))

            elif cmd == "set_dens_min":
                env.set_dens_min(float(payload))
                ack_q.put(("ok",))

            elif cmd == "set_speed_grid":
                val = payload
                if isinstance(val, torch.Tensor):
                    val = val.to(env.device)
                env._eval_speed_grid = val
                ack_q.put(("ok",))

            elif cmd == "set_crn_enabled":
                env.set_crn_enabled(bool(payload))
                ack_q.put(("ok",))

            else:
                ack_q.put(("error", f"unknown command {cmd!r}"))

        except Exception as exc:
            ack_q.put(("error", f"worker {worker_id} cmd={cmd!r}: {exc}"))


# ── URDF/worker/GPU assignment ────────────────────────────────────────────────

def _plan_assignment(
    D: int, N: int, G: int
) -> Tuple[List[List[int]], List[int]]:
    """Round-robin URDFs and workers across G GPUs.

    Returns
    -------
    worker_urdfs : List[List[int]]
        For each worker w, the global URDF indices (in the original
        ``urdf_paths`` list) that worker owns. Length ``N``.
    worker_gpu : List[int]
        For each worker w, the physical GPU id it is pinned to. Length ``N``.

    Algorithm
    ---------
    1. Assign URDF i to GPU ``i % G`` (round-robin across GPUs).
    2. Assign worker w to GPU ``w % G`` (round-robin across GPUs).
    3. Within each GPU, distribute that GPU's URDFs contiguously across
       that GPU's workers. With imbalanced counts, earlier workers on the
       GPU pick up the extras.
    """
    gpu_urdfs: List[List[int]] = [[] for _ in range(G)]
    for i in range(D):
        gpu_urdfs[i % G].append(i)
    gpu_workers: List[List[int]] = [[] for _ in range(G)]
    for w in range(N):
        gpu_workers[w % G].append(w)

    worker_urdfs: List[List[int]] = [[] for _ in range(N)]
    worker_gpu: List[int] = [0] * N
    for g in range(G):
        urdfs_g = gpu_urdfs[g]
        workers_g = gpu_workers[g]
        if not workers_g:
            # Shouldn't happen: caller clamps G <= N.
            continue
        K = len(workers_g)
        base, rem = divmod(len(urdfs_g), K)
        idx = 0
        for k, w in enumerate(workers_g):
            size = base + (1 if k < rem else 0)
            worker_urdfs[w] = urdfs_g[idx: idx + size]
            worker_gpu[w] = g
            idx += size
    return worker_urdfs, worker_gpu


# ── main class ────────────────────────────────────────────────────────────────

class ParallelMultiSceneEvalEnv:
    """
    Wrap D URDFs across N worker processes, each owning a MultiSceneEvalEnv
    shard of D URDFs sharded across G GPUs. Drop-in replacement for
    MultiSceneEvalEnv.

    Parameters
    ----------
    num_workers : int
        Number of worker processes. Clamped to ``len(urdf_paths)``.
        With the GPU at 40-50% per scene, N=2 is typically enough to
        saturate one GPU without excess memory pressure.
    num_gpus : Optional[int]
        Number of physical GPUs to spread workers across. ``None`` or ``0``
        auto-detects via ``torch.cuda.device_count()``. Clamped to
        ``min(num_gpus, num_workers)``. With G GPUs, URDFs are round-robin
        assigned to GPUs (URDF ``i`` → GPU ``i % G``) and workers are
        round-robin assigned to GPUs (worker ``w`` → GPU ``w % G``). Each
        GPU runs roughly ``num_eval_envs / G`` total env slots. Workers
        pin to their GPU via ``CUDA_VISIBLE_DEVICES`` before importing
        Genesis.
    """

    def __init__(
        self,
        urdf_paths: List[str],
        num_envs_per_drone: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        device: str,
        num_workers: int = 2,
        num_gpus: Optional[int] = None,
    ) -> None:
        D = len(urdf_paths)
        N = max(1, min(num_workers, D))
        E = int(num_envs_per_drone)

        # GPU count: auto-detect from torch when None/0, then clamp to N.
        if num_gpus is None or num_gpus <= 0:
            detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
            G = max(1, detected)
        else:
            G = int(num_gpus)
        G = max(1, min(G, N))

        worker_urdfs, worker_gpu = _plan_assignment(D, N, G)
        shard_sizes = [len(s) for s in worker_urdfs]
        shards: List[List[str]] = [
            [urdf_paths[i] for i in worker_urdfs[w]] for w in range(N)
        ]

        env_kwargs = dict(
            num_envs_per_drone=E,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            device=device,
        )

        print(f"[ParallelMultiSceneEvalEnv] launching {N} workers across "
              f"{G} GPU(s) (D={D}, E={E})", flush=True)
        for w in range(N):
            print(f"[ParallelMultiSceneEvalEnv]   worker {w} → GPU "
                  f"{worker_gpu[w]} | URDFs {worker_urdfs[w]}", flush=True)

        ctx = mp.get_context("spawn")
        self._cmd_qs = [ctx.Queue() for _ in range(N)]
        self._ack_qs = [ctx.Queue() for _ in range(N)]

        self._procs: List[mp.Process] = []
        for w in range(N):
            p = ctx.Process(
                target=_worker_main,
                args=(w, shards[w], env_kwargs, sys.path[:],
                      self._cmd_qs[w], self._ack_qs[w], worker_gpu[w]),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            print(f"[ParallelMultiSceneEvalEnv]   worker {w} pid={p.pid} "
                  f"(gpu={worker_gpu[w]}) started", flush=True)

        print(f"[ParallelMultiSceneEvalEnv] spawned {N} workers "
              f"(shard_sizes: {shard_sizes})", flush=True)

        # Collect dims + static metadata. Genesis init + sub-env construction
        # can be slow on cold Taichi caches (10s of minutes for many concurrent
        # workers + many URDFs per worker); on any failure we MUST kill every
        # spawned worker before raising, otherwise orphans hold GPU memory
        # and starve a subsequent retry.
        init_timeout = float(os.environ.get(
            "WP2_WORKER_INIT_TIMEOUT", _DEFAULT_INIT_TIMEOUT_S))
        print(f"[ParallelMultiSceneEvalEnv] waiting for worker readies "
              f"(timeout={init_timeout:.0f}s/worker; "
              f"override via WP2_WORKER_INIT_TIMEOUT)...", flush=True)
        num_obs = num_actions = dt = None
        reward_names: List[str] = []
        reward_scales: dict = {}
        n_comp_full = 0
        nominal_masses_all: List[float] = [0.0] * D
        ready_count = 0
        for w in range(N):
            try:
                msg = self._ack_qs[w].get(timeout=init_timeout)
            except _queue_mod.Empty:
                self._force_shutdown()
                raise RuntimeError(
                    f"Worker {w} did not reply 'ready' within "
                    f"{init_timeout:.0f}s. Status: {ready_count}/{N} workers "
                    f"replied successfully before the timeout fired. This "
                    f"worker holds {len(worker_urdfs[w])} URDF(s) × E={E} "
                    f"= {E * len(worker_urdfs[w])} env slots on GPU "
                    f"{worker_gpu[w]}. Likely causes: cold Taichi cache + "
                    f"many concurrent Genesis scenes, CPU oversubscription "
                    f"(check --cpus-per-gpu vs num_eval_workers), or GPU OOM. "
                    f"Mitigations: raise WP2_WORKER_INIT_TIMEOUT (currently "
                    f"{init_timeout:.0f}s), reduce num_eval_workers, or "
                    f"reduce num_urdfs."
                ) from None
            if msg[0] == "error":
                self._force_shutdown()
                raise RuntimeError(f"Worker {w} init failed: {msg[1]}")
            _, _no, _na, _dt, _nominal, _names, _scales, _ncomp = msg
            if num_obs is None:
                num_obs, num_actions, dt = _no, _na, _dt
                reward_names = list(_names)
                reward_scales = dict(_scales)
                n_comp_full = int(_ncomp)
            elif num_obs != _no or num_actions != _na:
                self._force_shutdown()
                raise RuntimeError(
                    f"Worker {w} obs/action dim mismatch: "
                    f"({_no},{_na}) vs ({num_obs},{num_actions})"
                )
            for k, d in enumerate(worker_urdfs[w]):
                nominal_masses_all[d] = float(_nominal[k])
            ready_count += 1
            print(f"[ParallelMultiSceneEvalEnv]   worker {w} ready "
                  f"({ready_count}/{N})", flush=True)

        self.D           = D
        self.E           = E
        self.N           = N
        self.G           = G
        self.num_obs     = num_obs
        self.num_actions = num_actions
        self.dt          = dt
        self.device      = device
        self._command_cfg             = command_cfg
        self._worker_urdfs            = worker_urdfs
        self._worker_gpu              = worker_gpu
        # Pre-build long index tensors on the main device for scatter writes.
        self._idx_tensors = [
            torch.as_tensor(worker_urdfs[w], dtype=torch.long, device=device)
            for w in range(N)
        ]
        self._fixed_forest_ids_buf:  Optional[torch.Tensor] = None
        self._eval_speed_grid_buf:   Optional[torch.Tensor] = None
        self._crn_enabled_buf:       bool = False

        # GPU buffers for rollout consumption.
        self._obs_buf   = torch.zeros(D, E, num_obs,   device=device)
        # State proxies live on device so arithmetic in evaluate.py stays on GPU.
        self._state_buf = torch.zeros(D, E, _N_STATE,  device=device)
        # Reward-component buffer (None when no active reward terms).
        self._comp_buf: Optional[torch.Tensor] = (
            torch.zeros(D, E, n_comp_full, device=device) if n_comp_full > 0 else None
        )
        self.drones = [
            DroneStateProxy(
                self._state_buf[d],
                nominal_mass=nominal_masses_all[d],
                reward_names=reward_names,
                reward_scales=reward_scales,
                comp_row=(self._comp_buf[d] if self._comp_buf is not None else None),
            )
            for d in range(D)
        ]

        print(f"[ParallelMultiSceneEvalEnv] ready  D={D}  E={E}  N={N}  "
              f"n_comp={n_comp_full}", flush=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _send_all(self, cmd: str, payload=None) -> None:
        """Broadcast a command to all workers and collect acks."""
        for q in self._cmd_qs:
            q.put((cmd, payload))
        for w, q in enumerate(self._ack_qs):
            msg = q.get()
            if msg[0] == "error":
                raise RuntimeError(f"Worker {w} error on {cmd!r}: {msg[1]}")

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def reset(self) -> Tuple[torch.Tensor, dict]:
        seed = random.randint(0, 2**31 - 1)
        for w in range(self.N):
            self._cmd_qs[w].put(("reset", seed))

        for w in range(self.N):
            msg = self._ack_qs[w].get()
            if msg[0] == "error":
                raise RuntimeError(f"Worker {w} reset error: {msg[1]}")
            _, obs_s, state_s, comp_s = msg
            idx_t = self._idx_tensors[w]
            # Worker returns CPU tensors → scatter to main-device buffers via
            # index_copy_. URDF rows are non-contiguous under round-robin GPU
            # sharding, so plain slice .copy_ no longer works.
            self._obs_buf.index_copy_(0, idx_t, obs_s.to(self.device))
            self._state_buf.index_copy_(0, idx_t, state_s.to(self.device))
            if comp_s is not None and self._comp_buf is not None:
                self._comp_buf.index_copy_(0, idx_t, comp_s.to(self.device))

        return self._obs_buf.clone(), {}

    @torch.no_grad()
    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.D, self.E, self.num_actions):
            raise ValueError(
                f"expected actions ({self.D},{self.E},{self.num_actions}), "
                f"got {tuple(actions.shape)}"
            )
        actions_cpu = actions.cpu()

        # Dispatch all workers simultaneously before blocking on any result.
        # actions_cpu[urdfs] uses advanced indexing → contiguous (ds, E, A) view.
        for w in range(self.N):
            urdfs = self._worker_urdfs[w]
            self._cmd_qs[w].put(("step", actions_cpu[urdfs].clone()))

        rew  = torch.zeros(self.D, self.E, device=self.device)
        done = torch.zeros(self.D, self.E, dtype=torch.bool, device=self.device)

        for w in range(self.N):
            msg = self._ack_qs[w].get()
            if msg[0] == "error":
                raise RuntimeError(f"Worker {w} step error: {msg[1]}")
            _, obs_s, rew_s, done_s, state_s, comp_s = msg
            idx_t = self._idx_tensors[w]
            self._obs_buf.index_copy_(0, idx_t, obs_s.to(self.device))
            rew.index_copy_(0, idx_t, rew_s.to(self.device))
            done.index_copy_(0, idx_t, done_s.bool().to(self.device))
            self._state_buf.index_copy_(0, idx_t, state_s.to(self.device))
            if comp_s is not None and self._comp_buf is not None:
                self._comp_buf.index_copy_(0, idx_t, comp_s.to(self.device))

        return self._obs_buf, rew, done, {}

    def refresh_forests(self) -> None:
        self._send_all("refresh_forests", random.randint(0, 2**31 - 1))

    def set_dens_min(self, value: float) -> None:
        self._send_all("set_dens_min", float(value))

    # ── forest / speed properties ─────────────────────────────────────────────

    @property
    def _fixed_forest_ids(self) -> Optional[torch.Tensor]:
        return self._fixed_forest_ids_buf

    @_fixed_forest_ids.setter
    def _fixed_forest_ids(self, value: Optional[torch.Tensor]) -> None:
        self._fixed_forest_ids_buf = value
        self._send_all("set_forest_ids", value.cpu() if value is not None else None)

    @property
    def _eval_speed_grid(self) -> Optional[torch.Tensor]:
        return self._eval_speed_grid_buf

    @_eval_speed_grid.setter
    def _eval_speed_grid(self, value: Optional[torch.Tensor]) -> None:
        self._eval_speed_grid_buf = value
        self._send_all("set_speed_grid", value.cpu() if value is not None else None)

    def set_crn_enabled(self, enabled: bool) -> None:
        self._crn_enabled = bool(enabled)

    @property
    def _crn_enabled(self) -> bool:
        return self._crn_enabled_buf

    @_crn_enabled.setter
    def _crn_enabled(self, value: bool) -> None:
        self._crn_enabled_buf = bool(value)
        self._send_all("set_crn_enabled", bool(value))

    # ── compatibility shims (single-URDF path only, never called here) ────────

    @property
    def cylinders_array(self):  return None
    @property
    def forest_ids(self):       return None
    @property
    def cylinders_xy(self):     return None
    @property
    def command_cfg(self) -> dict: return self._command_cfg

    def _compute_rewards_drone(self, drone_idx: int, ds) -> None:
        return

    # ── cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        for q in self._cmd_qs:
            try:
                q.put(("shutdown", None))
            except Exception:
                pass
        for p, q in zip(self._procs, self._ack_qs):
            try:
                q.get(timeout=10)
            except Exception:
                pass
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def _force_shutdown(self) -> None:
        """Hard-terminate all worker processes immediately.

        Called when ``__init__`` fails partway through (e.g. ready timeout):
        a graceful ``shutdown()`` would queue a 'shutdown' command and wait
        for an ack, but stuck workers may never reach the cmd loop. We must
        free the GPU before any retry — otherwise orphans keep holding VRAM
        and the next attempt starves on the same machine.
        """
        procs = getattr(self, "_procs", None)
        if not procs:
            return
        # Send SIGTERM to every still-alive worker, then SIGKILL stragglers.
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=3)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass
        # Drain queues so torch.multiprocessing doesn't warn on GC.
        for q in getattr(self, "_cmd_qs", []) + getattr(self, "_ack_qs", []):
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass
        print(f"[ParallelMultiSceneEvalEnv] force-shutdown: terminated "
              f"{len(procs)} worker(s)", flush=True)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
