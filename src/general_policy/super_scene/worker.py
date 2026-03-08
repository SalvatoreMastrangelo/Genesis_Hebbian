from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

import torch

from .ipc import WorkerCommand, WorkerReply


def _safe_reply(conn, ok: bool, payload: Dict[str, Any] | None = None, error: str = "") -> None:
    try:
        conn.send(WorkerReply(ok=ok, payload=payload or {}, error=error))
    except Exception:
        pass


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


def _bind_process_to_device(device: str) -> str:
    """
    Bind the worker process to a single CUDA device before Genesis init.

    If the worker is assigned `cuda:N`, we mask visibility to only that GPU and
    then use `cuda:0` locally inside the child process.
    """
    dev = str(device).strip().lower()
    if not dev.startswith("cuda"):
        return device

    try:
        _, idx_str = dev.split(":", 1)
        gpu_idx = int(idx_str)
    except Exception:
        gpu_idx = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return "cuda:0"


def _make_shared_buffers(
    *,
    num_envs: int,
    num_obs: int,
    num_privileged_obs: int,
    num_actions: int,
) -> Dict[str, torch.Tensor]:
    return {
        "actions": torch.zeros((num_envs, num_actions), dtype=torch.float32).share_memory_(),
        "obs": torch.zeros((num_envs, num_obs), dtype=torch.float32).share_memory_(),
        "critic_obs": torch.zeros((num_envs, num_privileged_obs), dtype=torch.float32).share_memory_(),
        "rew": torch.zeros((num_envs,), dtype=torch.float32).share_memory_(),
        "done": torch.zeros((num_envs,), dtype=torch.int64).share_memory_(),
        "time_outs": torch.zeros((num_envs,), dtype=torch.float32).share_memory_(),
    }


def worker_main(
    conn,
    *,
    urdf_list: List[str],
    num_envs: int,
    env_cfg: Dict,
    obs_cfg: Dict,
    reward_cfg: Dict,
    command_cfg: Dict,
    device: str,
    show_viewer: bool,
    use_shared_memory: bool = True,
    mps_active_thread_percentage: Optional[int] = None,
) -> None:
    """
    Worker process: owns one Gen_Env shard and performs env stepping commands.
    """
    env = None
    shared_buffers: Optional[Dict[str, torch.Tensor]] = None
    try:
        # Ensure headless rendering path in workers by default.
        os.environ.setdefault("GS_HEADLESS_NO_GL", "1")
        os.environ.setdefault("GS_PARA_LEVEL", "2")
        if mps_active_thread_percentage is not None and int(mps_active_thread_percentage) > 0:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_active_thread_percentage))
        local_device = _bind_process_to_device(device)

        import genesis as gs
        from general_policy.env_gen import Gen_Env

        gs.init(logging_level="error", backend=gs.gpu)

        env = Gen_Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_list=urdf_list,
            max_scenes=None,
            show_viewer=show_viewer,
            eval=False,
            device=local_device,
        )

        obs, info = env.reset()
        critic = info.get("observations", {}).get("critic")
        if critic is None:
            critic = env.privileged_obs_buf

        if use_shared_memory:
            shared_buffers = _make_shared_buffers(
                num_envs=int(env.num_envs),
                num_obs=int(env.num_obs),
                num_privileged_obs=int(env.num_privileged_obs),
                num_actions=int(env.num_actions),
            )
            shared_buffers["obs"].copy_(_to_cpu(obs))
            shared_buffers["critic_obs"].copy_(_to_cpu(critic))

        _safe_reply(
            conn,
            ok=True,
            payload={
                "event": "ready",
                "meta": {
                    "num_envs": int(env.num_envs),
                    "num_obs": int(env.num_obs),
                    "num_privileged_obs": int(env.num_privileged_obs),
                    "num_actions": int(env.num_actions),
                    "max_episode_length": int(env.max_episode_length),
                    "dt": float(env.dt),
                },
                "obs": _to_cpu(obs) if not use_shared_memory else None,
                "critic_obs": _to_cpu(critic) if not use_shared_memory else None,
                "shared_buffers": shared_buffers if use_shared_memory else None,
            },
        )

        while True:
            msg = conn.recv()
            if isinstance(msg, WorkerCommand):
                cmd = msg.cmd
                payload = msg.payload
            elif isinstance(msg, dict):
                cmd = str(msg.get("cmd", ""))
                payload = dict(msg.get("payload", {}))
            else:
                _safe_reply(conn, ok=False, error="Invalid command format")
                continue

            if cmd == "close":
                _safe_reply(conn, ok=True, payload={"event": "closed"})
                break

            if cmd == "reset":
                obs, info = env.reset()
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                if use_shared_memory and shared_buffers is not None:
                    shared_buffers["obs"].copy_(_to_cpu(obs))
                    shared_buffers["critic_obs"].copy_(_to_cpu(critic))
                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "reset",
                        "obs": _to_cpu(obs) if not use_shared_memory else None,
                        "critic_obs": _to_cpu(critic) if not use_shared_memory else None,
                    },
                )
                continue

            if cmd == "step":
                if use_shared_memory and shared_buffers is not None:
                    actions = shared_buffers["actions"]
                else:
                    actions = payload.get("actions")
                if not torch.is_tensor(actions):
                    _safe_reply(conn, ok=False, error="`actions` must be a tensor")
                    continue
                obs, rew, done, info = env.step(actions.to(env.device))
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                time_outs = info.get("time_outs")
                if time_outs is None:
                    time_outs = torch.zeros_like(done, dtype=torch.float32)
                episode = info.get("episode") if isinstance(info, dict) else None

                if use_shared_memory and shared_buffers is not None:
                    shared_buffers["obs"].copy_(_to_cpu(obs))
                    shared_buffers["critic_obs"].copy_(_to_cpu(critic))
                    shared_buffers["rew"].copy_(_to_cpu(rew))
                    shared_buffers["done"].copy_(_to_cpu(done).long())
                    shared_buffers["time_outs"].copy_(_to_cpu(time_outs).float())

                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "step",
                        "obs": _to_cpu(obs) if not use_shared_memory else None,
                        "critic_obs": _to_cpu(critic) if not use_shared_memory else None,
                        "rew": _to_cpu(rew) if not use_shared_memory else None,
                        "done": _to_cpu(done) if not use_shared_memory else None,
                        "time_outs": _to_cpu(time_outs) if not use_shared_memory else None,
                        "episode": episode if isinstance(episode, dict) else None,
                    },
                )
                continue

            _safe_reply(conn, ok=False, error=f"Unknown command: {cmd}")

    except Exception as exc:
        tb = traceback.format_exc()
        _safe_reply(conn, ok=False, error=f"Worker crash: {exc}\n{tb}")
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            import genesis as gs

            gs.destroy()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
