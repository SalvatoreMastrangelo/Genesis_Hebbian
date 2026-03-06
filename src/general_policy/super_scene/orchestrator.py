from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .ipc import WorkerCommand, WorkerReply
from .worker import worker_main


@dataclass
class WorkerHandle:
    process: mp.Process
    conn: any
    num_envs: int
    sl: slice


class LogicalSuperSceneOrchestrator:
    """
    Multi-process env orchestrator.

    Each worker owns one shard (one Gen_Env with a subset of URDFs). The orchestrator
    exposes a single global-batch API suitable for PPO collection.
    """

    def __init__(
        self,
        *,
        shards: Sequence[Sequence[str]],
        shard_env_counts: Sequence[int],
        env_cfg: Dict,
        obs_cfg: Dict,
        reward_cfg: Dict,
        command_cfg: Dict,
        device: str,
        show_viewer: bool = False,
    ) -> None:
        if len(shards) == 0:
            raise RuntimeError("No shards provided")
        if len(shards) != len(shard_env_counts):
            raise RuntimeError("shards and shard_env_counts must have same length")

        self.device = torch.device(device)
        self._workers: List[WorkerHandle] = []
        self._worker_slices: List[slice] = []

        ctx = mp.get_context("spawn")

        start = 0
        metas: List[Dict] = []
        initial_obs_cpu: List[torch.Tensor] = []
        initial_critic_cpu: List[torch.Tensor] = []

        for i, (urdfs_i, n_env_i) in enumerate(zip(shards, shard_env_counts)):
            if n_env_i <= 0:
                raise RuntimeError(f"Invalid shard env count at worker {i}: {n_env_i}")

            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=worker_main,
                kwargs={
                    "conn": child_conn,
                    "urdf_list": list(urdfs_i),
                    "num_envs": int(n_env_i),
                    "env_cfg": dict(env_cfg),
                    "obs_cfg": dict(obs_cfg),
                    "reward_cfg": dict(reward_cfg),
                    "command_cfg": dict(command_cfg),
                    "device": str(device),
                    "show_viewer": bool(show_viewer and i == 0),
                },
                daemon=True,
            )
            p.start()

            reply = self._recv_reply(parent_conn)
            if not reply.ok:
                raise RuntimeError(reply.error or f"Worker {i} failed to start")
            if reply.payload.get("event") != "ready":
                raise RuntimeError(f"Worker {i} returned unexpected startup event")

            meta = dict(reply.payload["meta"])
            metas.append(meta)
            initial_obs_cpu.append(reply.payload["obs"])
            initial_critic_cpu.append(reply.payload["critic_obs"])

            stop = start + int(meta["num_envs"])
            sl = slice(start, stop)
            self._worker_slices.append(sl)
            self._workers.append(WorkerHandle(process=p, conn=parent_conn, num_envs=int(meta["num_envs"]), sl=sl))
            start = stop

        self.num_envs = start
        self.num_obs = int(metas[0]["num_obs"])
        self.num_privileged_obs = int(metas[0]["num_privileged_obs"])
        self.num_actions = int(metas[0]["num_actions"])
        self.max_episode_length = int(metas[0]["max_episode_length"])
        self.dt = float(metas[0]["dt"])

        for m in metas[1:]:
            if int(m["num_obs"]) != self.num_obs:
                raise RuntimeError("Incompatible shard obs dimensions")
            if int(m["num_privileged_obs"]) != self.num_privileged_obs:
                raise RuntimeError("Incompatible shard privileged obs dimensions")
            if int(m["num_actions"]) != self.num_actions:
                raise RuntimeError("Incompatible shard action dimensions")

        self._obs = torch.cat(initial_obs_cpu, dim=0).to(self.device)
        self._critic = torch.cat(initial_critic_cpu, dim=0).to(self.device)

    def _recv_reply(self, conn) -> WorkerReply:
        msg = conn.recv()
        if isinstance(msg, WorkerReply):
            return msg
        if isinstance(msg, dict):
            return WorkerReply(ok=bool(msg.get("ok", False)), payload=dict(msg.get("payload", {})), error=str(msg.get("error", "")))
        return WorkerReply(ok=False, payload={}, error="Invalid worker reply format")

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for w in self._workers:
            if not w.process.is_alive():
                raise RuntimeError(
                    f"Worker process died before reset send (pid={w.process.pid}, exitcode={w.process.exitcode})."
                )
            try:
                w.conn.send(WorkerCommand(cmd="reset", payload={}))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"Broken pipe sending reset to worker (pid={w.process.pid}, exitcode={w.process.exitcode})."
                ) from exc

        obs_chunks: List[torch.Tensor] = []
        critic_chunks: List[torch.Tensor] = []
        for w in self._workers:
            rep = self._recv_reply(w.conn)
            if not rep.ok:
                raise RuntimeError(rep.error)
            obs_chunks.append(rep.payload["obs"])
            critic_chunks.append(rep.payload["critic_obs"])

        self._obs = torch.cat(obs_chunks, dim=0).to(self.device)
        self._critic = torch.cat(critic_chunks, dim=0).to(self.device)
        return self._obs, self._critic

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}"
            )

        # Send in parallel
        actions_cpu = actions.detach().to("cpu")
        for w in self._workers:
            if not w.process.is_alive():
                raise RuntimeError(
                    f"Worker process died before step send (pid={w.process.pid}, exitcode={w.process.exitcode})."
                )
            try:
                w.conn.send(WorkerCommand(cmd="step", payload={"actions": actions_cpu[w.sl].contiguous()}))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"Broken pipe sending step to worker (pid={w.process.pid}, exitcode={w.process.exitcode})."
                ) from exc

        obs_chunks: List[torch.Tensor] = []
        critic_chunks: List[torch.Tensor] = []
        rew_chunks: List[torch.Tensor] = []
        done_chunks: List[torch.Tensor] = []
        timeout_chunks: List[torch.Tensor] = []
        episodes: List[Dict] = []

        for w in self._workers:
            rep = self._recv_reply(w.conn)
            if not rep.ok:
                raise RuntimeError(rep.error)
            payload = rep.payload
            obs_chunks.append(payload["obs"])
            critic_chunks.append(payload["critic_obs"])
            rew_chunks.append(payload["rew"])
            done_chunks.append(payload["done"])
            timeout_chunks.append(payload["time_outs"])
            ep = payload.get("episode")
            if isinstance(ep, dict):
                episodes.append(ep)

        obs = torch.cat(obs_chunks, dim=0).to(self.device)
        critic = torch.cat(critic_chunks, dim=0).to(self.device)
        rew = torch.cat(rew_chunks, dim=0).to(self.device).float()
        done = torch.cat(done_chunks, dim=0).to(self.device)
        if done.dtype != torch.long and done.dtype != torch.int64:
            done = done.long()
        time_outs = torch.cat(timeout_chunks, dim=0).to(self.device).float()

        extras = {
            "observations": {"critic": critic},
            "time_outs": time_outs,
        }
        if episodes:
            agg: Dict[str, float] = {}
            c = 0
            for e in episodes:
                c += 1
                for k, v in e.items():
                    try:
                        agg[k] = agg.get(k, 0.0) + float(v)
                    except Exception:
                        pass
            if c > 0:
                for k in list(agg.keys()):
                    agg[k] /= float(c)
                extras["episode"] = agg

        self._obs = obs
        self._critic = critic

        return obs, critic, rew, done, extras

    def close(self) -> None:
        for w in self._workers:
            try:
                w.conn.send(WorkerCommand(cmd="close", payload={}))
            except Exception:
                pass

        for w in self._workers:
            try:
                _ = self._recv_reply(w.conn)
            except Exception:
                pass

        for w in self._workers:
            try:
                w.conn.close()
            except Exception:
                pass

        for w in self._workers:
            try:
                w.process.join(timeout=5.0)
            except Exception:
                pass
            try:
                if w.process.is_alive():
                    w.process.terminate()
            except Exception:
                pass
