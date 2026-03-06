from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

import torch

from .ipc import WorkerCommand, WorkerReply


def _safe_reply(conn, ok: bool, payload: Dict[str, Any] | None = None, error: str = "") -> None:
    try:
        conn.send(WorkerReply(ok=ok, payload=payload or {}, error=error))
    except Exception:
        pass


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


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
) -> None:
    """
    Worker process: owns one Gen_Env shard and performs env stepping commands.
    """
    env = None
    try:
        # Ensure headless rendering path in workers by default.
        os.environ.setdefault("GS_HEADLESS_NO_GL", "1")
        os.environ.setdefault("GS_PARA_LEVEL", "2")

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
            device=device,
        )

        obs, info = env.reset()
        critic = info.get("observations", {}).get("critic")
        if critic is None:
            critic = env.privileged_obs_buf

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
                "obs": _to_cpu(obs),
                "critic_obs": _to_cpu(critic),
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
                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "reset",
                        "obs": _to_cpu(obs),
                        "critic_obs": _to_cpu(critic),
                    },
                )
                continue

            if cmd == "step":
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

                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "step",
                        "obs": _to_cpu(obs),
                        "critic_obs": _to_cpu(critic),
                        "rew": _to_cpu(rew),
                        "done": _to_cpu(done),
                        "time_outs": _to_cpu(time_outs),
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
