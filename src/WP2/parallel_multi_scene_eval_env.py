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

import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

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
        row[:, _S_POW]  = sub.power.float()
        row[:, _S_NAN]  = sub.nan_envs.float()
        row[:, _S_RST]  = sub.reset_buf.float()
        for i, attr in enumerate(("pre_collision", "pre_wall_crash", "pre_angle_limit")):
            flag = getattr(sub, attr, None)
            row[:, _S_COLL + i] = flag.float() if flag is not None else 0.0
    return buf.cpu()  # one D2H transfer for the whole shard


# ── proxy ─────────────────────────────────────────────────────────────────────

class DroneStateProxy:
    """WingedDroneEnv-compatible view backed by a (E, _N_STATE) GPU tensor."""

    def __init__(self, row: torch.Tensor) -> None:
        self._r = row  # (E, _N_STATE) on main-process device

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


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_main(
    worker_id: int,
    urdf_shard: List[str],
    env_kwargs: dict,
    extra_sys_paths: List[str],
    cmd_q: "mp.Queue[tuple]",
    ack_q: "mp.Queue[tuple]",
) -> None:
    # Restore sys.path so project imports work in the spawned process.
    for p in extra_sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import genesis as gs
        from WP2.multi_scene_eval_env import MultiSceneEvalEnv

        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)

        env = MultiSceneEvalEnv(urdf_paths=urdf_shard, **env_kwargs)
        ack_q.put(("ready", env.num_obs, env.num_actions, float(env.dt)))
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
                ack_q.put(("reset_result", obs.cpu(), _collect_state_gpu(env)))

            elif cmd == "step":
                obs, rew, done, _ = env.step(payload.to(env.device))
                ack_q.put(("step_result", obs.cpu(), rew.cpu(), done.cpu(), _collect_state_gpu(env)))

            elif cmd == "refresh_forests":
                seed = payload
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                env.refresh_forests()
                ack_q.put(("ok",))

            elif cmd == "set_forest_ids":
                env._fixed_forest_ids = payload
                ack_q.put(("ok",))

            elif cmd == "set_dens_min":
                env.set_dens_min(float(payload))
                ack_q.put(("ok",))

            elif cmd == "set_speed_grid":
                env._eval_speed_grid = payload
                ack_q.put(("ok",))

            else:
                ack_q.put(("error", f"unknown command {cmd!r}"))

        except Exception as exc:
            ack_q.put(("error", f"worker {worker_id} cmd={cmd!r}: {exc}"))


# ── main class ────────────────────────────────────────────────────────────────

class ParallelMultiSceneEvalEnv:
    """
    Wrap D URDFs across N worker processes, each owning a MultiSceneEvalEnv
    shard of D/N URDFs.  Drop-in replacement for MultiSceneEvalEnv.

    Parameters
    ----------
    num_workers : int
        Number of worker processes.  Clamped to ``len(urdf_paths)``.
        With the GPU at 40-50% per scene, N=2 is typically enough to
        saturate the GPU without excess memory pressure.
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
    ) -> None:
        D = len(urdf_paths)
        N = max(1, min(num_workers, D))
        E = int(num_envs_per_drone)

        base, rem   = divmod(D, N)
        shard_sizes = [base + (1 if i < rem else 0) for i in range(N)]
        d_offsets   = [sum(shard_sizes[:i]) for i in range(N)]

        shards: List[List[str]] = []
        start = 0
        for s in shard_sizes:
            shards.append(list(urdf_paths[start: start + s]))
            start += s

        env_kwargs = dict(
            num_envs_per_drone=E,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            device=device,
        )

        ctx = mp.get_context("spawn")
        self._cmd_qs = [ctx.Queue() for _ in range(N)]
        self._ack_qs = [ctx.Queue() for _ in range(N)]

        self._procs: List[mp.Process] = []
        for w in range(N):
            p = ctx.Process(
                target=_worker_main,
                args=(w, shards[w], env_kwargs, sys.path[:],
                      self._cmd_qs[w], self._ack_qs[w]),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        print(f"[ParallelMultiSceneEvalEnv] spawned {N} workers "
              f"(shards: {shard_sizes}), waiting for Genesis init…")

        # Collect dims — Genesis init can be slow, allow generous timeout.
        num_obs = num_actions = dt = None
        for w in range(N):
            msg = self._ack_qs[w].get(timeout=600)
            if msg[0] == "error":
                raise RuntimeError(f"Worker {w} init failed: {msg[1]}")
            _, _no, _na, _dt = msg
            if num_obs is None:
                num_obs, num_actions, dt = _no, _na, _dt
            elif num_obs != _no or num_actions != _na:
                raise RuntimeError(
                    f"Worker {w} obs/action dim mismatch: "
                    f"({_no},{_na}) vs ({num_obs},{num_actions})"
                )

        self.D           = D
        self.E           = E
        self.N           = N
        self.num_obs     = num_obs
        self.num_actions = num_actions
        self.dt          = dt
        self.device      = device
        self._command_cfg             = command_cfg
        self._d_offsets               = d_offsets
        self._shard_sizes             = shard_sizes
        self._fixed_forest_ids_buf:  Optional[torch.Tensor] = None
        self._eval_speed_grid_buf:   Optional[torch.Tensor] = None

        # GPU buffers for rollout consumption.
        self._obs_buf   = torch.zeros(D, E, num_obs,   device=device)
        # State proxies live on device so arithmetic in evaluate.py stays on GPU.
        self._state_buf = torch.zeros(D, E, _N_STATE,  device=device)
        self.drones     = [DroneStateProxy(self._state_buf[d]) for d in range(D)]

        print(f"[ParallelMultiSceneEvalEnv] ready  D={D}  E={E}  N={N}")

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
            _, obs_s, state_s = msg
            d0, ds = self._d_offsets[w], self._shard_sizes[w]
            self._obs_buf[d0: d0 + ds].copy_(obs_s)       # cross-device H2D
            self._state_buf[d0: d0 + ds].copy_(state_s)

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
        for w in range(self.N):
            d0, ds = self._d_offsets[w], self._shard_sizes[w]
            self._cmd_qs[w].put(("step", actions_cpu[d0: d0 + ds].clone()))

        rew  = torch.zeros(self.D, self.E, device=self.device)
        done = torch.zeros(self.D, self.E, dtype=torch.bool, device=self.device)

        for w in range(self.N):
            msg = self._ack_qs[w].get()
            if msg[0] == "error":
                raise RuntimeError(f"Worker {w} step error: {msg[1]}")
            _, obs_s, rew_s, done_s, state_s = msg
            d0, ds = self._d_offsets[w], self._shard_sizes[w]
            self._obs_buf[d0: d0 + ds].copy_(obs_s)
            rew[d0: d0 + ds].copy_(rew_s)
            done[d0: d0 + ds].copy_(done_s.bool())
            self._state_buf[d0: d0 + ds].copy_(state_s)

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

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
