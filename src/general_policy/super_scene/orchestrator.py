from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .ipc import WorkerCommand, WorkerReply
from .worker import worker_main


@dataclass
class WorkerHandle:
    process: mp.Process
    conn: any
    num_envs: int
    sl: slice
    shm: Optional[Dict[str, torch.Tensor]] = None
    worker_idx: int = -1
    worker_device: str = ""


class LogicalSuperSceneOrchestrator:
    """
    Multi-process env orchestrator.

    Each worker owns one shard (one Gen_Env with a subset of URDFs). The orchestrator
    exposes a single global-batch API suitable for PPO collection.
    """

    @staticmethod
    def _gpu_usage_snapshot() -> List[str]:
        """
        Return one or more human-readable lines describing GPU compute processes.

        We rely on ``nvidia-smi`` because the orchestrator runs in the main process and
        needs a device-wide view, not just PyTorch allocator stats.
        """
        if shutil.which("nvidia-smi") is None:
            return ["[logical-super-scene][gpu-usage] nvidia-smi not available"]

        try:
            gpu_query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            proc_query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            suffix = f": {stderr}" if stderr else ""
            return [f"[logical-super-scene][gpu-usage] nvidia-smi query failed{suffix}"]

        gpu_rows: List[Dict[str, object]] = []
        for raw_line in gpu_query.stdout.splitlines():
            parts = [item.strip() for item in raw_line.split(",")]
            if len(parts) < 5:
                continue
            try:
                total_mb = int(parts[3])
                used_mb = int(parts[4])
            except ValueError:
                continue
            gpu_rows.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "processes": [],
                }
            )

        gpu_by_uuid = {str(row["uuid"]): row for row in gpu_rows}
        for raw_line in proc_query.stdout.splitlines():
            parts = [item.strip() for item in raw_line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid = parts[0]
            row = gpu_by_uuid.get(gpu_uuid)
            if row is None:
                continue
            try:
                used_mb = int(parts[3])
            except ValueError:
                used_mb = -1
            row["processes"].append(
                {
                    "pid": parts[1],
                    "name": parts[2],
                    "used_mb": used_mb,
                }
            )

        if not gpu_rows:
            return ["[logical-super-scene][gpu-usage] no GPU rows returned by nvidia-smi"]

        lines: List[str] = []
        for row in gpu_rows:
            total_mb = int(row["total_mb"])
            used_mb = int(row["used_mb"])
            pct = (100.0 * used_mb / total_mb) if total_mb > 0 else 0.0
            processes = row["processes"]
            if processes:
                proc_desc = "; ".join(
                    f"pid={proc['pid']} name={proc['name']} mem={proc['used_mb']}MiB"
                    for proc in processes
                )
            else:
                proc_desc = "no compute processes"
            lines.append(
                "[logical-super-scene][gpu-usage] "
                f"gpu={row['index']} name={row['name']} mem={used_mb}/{total_mb}MiB ({pct:.1f}%) "
                f"procs=[{proc_desc}]"
            )
        return lines

    def __init__(
        self,
        *,
        shards: Sequence[Sequence[str]],
        shard_env_counts: Sequence[int],
        worker_devices: Sequence[str],
        env_cfg: Dict,
        obs_cfg: Dict,
        reward_cfg: Dict,
        command_cfg: Dict,
        device: str,
        show_viewer: bool = False,
        use_shared_memory: bool = True,
        mps_active_thread_percentage: int = 0,
    ) -> None:
        if len(shards) == 0:
            raise RuntimeError("No shards provided")
        if len(shards) != len(shard_env_counts):
            raise RuntimeError("shards and shard_env_counts must have same length")
        if len(shards) != len(worker_devices):
            raise RuntimeError("shards and worker_devices must have same length")

        self.device = torch.device(device)
        self._device_is_cuda = self.device.type == "cuda"
        self._workers: List[WorkerHandle] = []
        self._worker_slices: List[slice] = []
        self.use_shared_memory = bool(use_shared_memory)

        ctx = mp.get_context("spawn")

        start = 0
        metas: List[Dict] = []
        initial_obs_cpu: List[torch.Tensor] = []
        initial_critic_cpu: List[torch.Tensor] = []
        startup_t0 = time.perf_counter()

        workers_per_device: Dict[str, int] = {}
        for dev in worker_devices:
            workers_per_device[dev] = workers_per_device.get(dev, 0) + 1

        total_cpu = max(1, int(os.cpu_count() or 1))
        cpu_threads_per_worker = max(1, total_cpu // max(1, len(worker_devices)) // 2)

        pending_workers: Dict[any, Tuple[int, str, Sequence[str], mp.Process]] = {}

        for i, (urdfs_i, n_env_i, worker_device) in enumerate(zip(shards, shard_env_counts, worker_devices)):
            if n_env_i <= 0:
                raise RuntimeError(f"Invalid shard env count at worker {i}: {n_env_i}")

            if mps_active_thread_percentage > 0:
                per_worker_pct = max(1, min(100, int(mps_active_thread_percentage)))
            else:
                per_worker_pct = max(
                    1,
                    min(100, int(100 // max(1, workers_per_device.get(worker_device, 1)))),
                )

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
                    "device": str(worker_device),
                    "show_viewer": bool(show_viewer and i == 0),
                    "use_shared_memory": self.use_shared_memory,
                    "mps_active_thread_percentage": per_worker_pct,
                    "cpu_threads": cpu_threads_per_worker,
                },
                daemon=True,
            )

            launch_t0 = time.perf_counter()
            p.start()
            print(
                "[logical-super-scene] "
                f"launched worker={i} pid={p.pid} device={worker_device} "
                f"target_envs={int(n_env_i)} urdfs={len(urdfs_i)} "
                f"mps_thread_pct={per_worker_pct} cpu_threads={cpu_threads_per_worker} "
                f"launch_s={time.perf_counter() - launch_t0:.3f}"
            )
            pending_workers[parent_conn] = (i, worker_device, urdfs_i, p)

        print(
            "[logical-super-scene] "
            f"all {len(pending_workers)} workers launched in {time.perf_counter() - startup_t0:.3f}s; "
            "waiting for ready replies"
        )

        # Allow 40 minutes per worker for Genesis scene compilation + reset.
        # Workers typically take ~1500s; this is a safety net against silent deadlocks.
        _STARTUP_TIMEOUT_S = float(os.getenv("LOGICAL_SUPER_SCENE_STARTUP_TIMEOUT", "2400"))

        ready_wait_t0 = time.perf_counter()
        for i, worker_device, urdfs_i, parent_conn, p in pending_workers:
            reply_wait_t0 = time.perf_counter()
            reply = self._recv_reply(parent_conn, worker_idx=i, process=p, timeout=_STARTUP_TIMEOUT_S)
            if not reply.ok:
                raise RuntimeError(reply.error or f"Worker {i} failed to start")
            if reply.payload.get("event") != "ready":
                raise RuntimeError(f"Worker {i} returned unexpected startup event")

                event = str(reply.payload.get("event", ""))
                if event == "init_progress":
                    local_completed = int(reply.payload.get("local_completed_scenes", 0))
                    prev_completed = worker_scene_counts.get(i, 0)
                    if local_completed > prev_completed:
                        completed_scenes += local_completed - prev_completed
                        worker_scene_counts[i] = local_completed
                    print(
                        "[logical-super-scene] "
                        f"init progress scenes={completed_scenes}/{total_scenes} "
                        f"worker={i} local={local_completed}/{int(reply.payload.get('local_total_scenes', len(urdfs_i)))} "
                        f"last_scene_s={float(reply.payload.get('scene_init_s', 0.0)):.3f} "
                        f"last_build_s={float(reply.payload.get('scene_build_s', 0.0)):.3f} "
                        f"urdf='{str(reply.payload.get('urdf', ''))}'"
                    )
                    while (
                        total_scenes > 0
                        and next_progress_report_pct <= 100
                        and completed_scenes * 100 >= total_scenes * next_progress_report_pct
                    ):
                        print(
                            "[logical-super-scene] "
                            f"GPU process snapshot at {next_progress_report_pct}% env initialization"
                        )
                        for line in self._gpu_usage_snapshot():
                            print(line)
                        next_progress_report_pct += 10
                    continue

            shm = reply.payload.get("shared_buffers")
            # Worker may have fallen back to pipe-based transfer if share_memory_() failed.
            worker_used_shm = isinstance(shm, dict) and bool(meta.get("use_shared_memory", True))
            if self.use_shared_memory and not worker_used_shm:
                print(
                    f"[logical-super-scene] WARNING: worker={i} fell back to pipe transfer "
                    "(shared memory unavailable); continuing without shared memory for this worker."
                )

            if worker_used_shm:
                initial_obs_cpu.append(shm["obs"])
                initial_critic_cpu.append(shm["critic_obs"])
            else:
                initial_obs_cpu.append(reply.payload["obs"])
                initial_critic_cpu.append(reply.payload["critic_obs"])

            stop = start + int(meta["num_envs"])
            sl = slice(start, stop)
            self._worker_slices.append(sl)
            self._workers.append(
                WorkerHandle(
                    process=p,
                    conn=parent_conn,
                    num_envs=int(meta["num_envs"]),
                    sl=sl,
                    shm=shm if worker_used_shm else None,
                    worker_idx=i,
                    worker_device=worker_device,
                )
                start = stop

                print(
                    "[logical-super-scene] "
                    f"ready worker={i} pid={p.pid} device={worker_device} envs={int(meta['num_envs'])} "
                    f"urdfs={len(urdfs_i)} ready_wait_s={time.perf_counter() - reply_wait_t0:.3f} "
                    f"worker_ready_s={float(meta.get('worker_ready_s', 0.0)):.3f} "
                    f"gs_init_s={float(meta.get('gs_init_s', 0.0)):.3f} "
                    f"gen_env_build_s={float(meta.get('gen_env_build_s', 0.0)):.3f} "
                    f"initial_reset_s={float(meta.get('initial_reset_s', 0.0)):.3f} "
                    f"shm_export_s={float(meta.get('shm_export_s', 0.0)):.3f} "
                    f"scene_init_total_s={float(meta.get('scene_init_total_s', 0.0)):.3f} "
                    f"scene_build_total_s={float(meta.get('scene_build_total_s', 0.0)):.3f} "
                    f"slowest_scene_s={float(meta.get('slowest_scene_s', 0.0)):.3f} "
                    f"slowest_scene_urdf='{str(meta.get('slowest_scene_urdf', ''))}'"
                )

        print(
            "[logical-super-scene] "
            f"all workers ready in {time.perf_counter() - ready_wait_t0:.3f}s "
            f"(startup_total_s={time.perf_counter() - startup_t0:.3f})"
        )

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

        self._obs = torch.empty((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self._critic = torch.empty(
            (self.num_envs, self.num_privileged_obs),
            device=self.device,
            dtype=torch.float32,
        )
        self._rew = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._done = torch.empty((self.num_envs,), device=self.device, dtype=torch.long)
        self._time_outs = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._actions_host = self._make_host_buffer((self.num_envs, self.num_actions), dtype=torch.float32)

        for w, obs_cpu, critic_cpu in zip(self._workers, initial_obs_cpu, initial_critic_cpu):
            self._copy_chunk_into_global_buffers(
                w=w,
                obs_src=obs_cpu,
                critic_src=critic_cpu,
            )

    def _recv_reply(
        self,
        conn,
        worker_idx: Optional[int] = None,
        process: Optional[mp.Process] = None,
        timeout: Optional[float] = None,
    ) -> WorkerReply:
        prefix = f"worker={worker_idx}" if worker_idx is not None else "worker=<unknown>"
        if timeout is not None:
            ready = conn.poll(timeout)
            if not ready:
                details = ""
                if process is not None:
                    alive = process.is_alive()
                    details = f" pid={process.pid} alive={alive} exitcode={process.exitcode}"
                raise RuntimeError(
                    f"Timeout ({timeout:.0f}s) waiting for reply from {prefix}.{details}"
                )
        try:
            msg = conn.recv()
        except EOFError as exc:
            details = ""
            if process is not None:
                details = f" pid={process.pid} exitcode={process.exitcode}"
            raise RuntimeError(f"EOF while waiting reply from {prefix}.{details}") from exc
        if isinstance(msg, WorkerReply):
            return msg
        if isinstance(msg, dict):
            return WorkerReply(
                ok=bool(msg.get("ok", False)),
                payload=dict(msg.get("payload", {})),
                error=str(msg.get("error", "")),
            )
        return WorkerReply(ok=False, payload={}, error="Invalid worker reply format")

    def _assert_worker_alive(self, w: WorkerHandle, phase: str) -> None:
        if not w.process.is_alive():
            raise RuntimeError(
                f"Worker process died before {phase} (pid={w.process.pid}, exitcode={w.process.exitcode})."
            )

    def _make_host_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if self._device_is_cuda:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        return torch.empty(shape, dtype=dtype, device="cpu")

    def _copy_to_device_slice(self, dst: torch.Tensor, sl: slice, src: torch.Tensor) -> None:
        non_blocking = self._device_is_cuda and bool(getattr(src, "is_pinned", lambda: False)())
        dst[sl].copy_(src, non_blocking=non_blocking)

    def _copy_chunk_into_global_buffers(
        self,
        *,
        w: WorkerHandle,
        obs_src: torch.Tensor,
        critic_src: torch.Tensor,
        rew_src: Optional[torch.Tensor] = None,
        done_src: Optional[torch.Tensor] = None,
        time_outs_src: Optional[torch.Tensor] = None,
    ) -> None:
        self._copy_to_device_slice(self._obs, w.sl, obs_src)
        self._copy_to_device_slice(self._critic, w.sl, critic_src)
        if rew_src is not None:
            self._copy_to_device_slice(self._rew, w.sl, rew_src.float())
        if done_src is not None:
            done_view = done_src if done_src.dtype in (torch.long, torch.int64) else done_src.long()
            self._copy_to_device_slice(self._done, w.sl, done_view)
        if time_outs_src is not None:
            self._copy_to_device_slice(self._time_outs, w.sl, time_outs_src.float())

    def _gather_replies(
        self,
        *,
        event_name: str,
    ) -> List[Tuple[WorkerHandle, WorkerReply]]:
        pending = {w.conn: w for w in self._workers}
        replies: List[Tuple[WorkerHandle, WorkerReply]] = []
        while pending:
            for conn in wait(list(pending.keys())):
                w = pending.pop(conn)
                rep = self._recv_reply(w.conn, worker_idx=w.worker_idx, process=w.process)
                if not rep.ok:
                    raise RuntimeError(rep.error)
                event = rep.payload.get("event")
                if event_name and event not in (None, event_name):
                    raise RuntimeError(
                        f"Worker {w.worker_idx} returned unexpected event '{event}' during '{event_name}'."
                    )
                replies.append((w, rep))
        return replies

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for w in self._workers:
            self._assert_worker_alive(w, "reset send")
            try:
                w.conn.send(WorkerCommand(cmd="reset", payload={}))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"Broken pipe sending reset to worker (pid={w.process.pid}, exitcode={w.process.exitcode})."
                ) from exc

        for w, rep in self._gather_replies(event_name="reset"):
            if w.shm is not None:
                obs_src = w.shm["obs"]
                critic_src = w.shm["critic_obs"]
            else:
                obs_src = rep.payload["obs"]
                critic_src = rep.payload["critic_obs"]
            self._copy_chunk_into_global_buffers(
                w=w,
                obs_src=obs_src,
                critic_src=critic_src,
            )

        return self._obs, self._critic

    def _stage_actions_to_host(self, actions: torch.Tensor) -> torch.Tensor:
        if self._device_is_cuda:
            self._actions_host.copy_(actions.detach(), non_blocking=True)
            torch.cuda.current_stream(device=self.device).synchronize()
            return self._actions_host
        self._actions_host.copy_(actions.detach().to("cpu"))
        return self._actions_host

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}"
            )

        if actions.device != self.device:
            actions = actions.to(self.device)
        actions_host = self._stage_actions_to_host(actions)

        for w in self._workers:
            self._assert_worker_alive(w, "step send")
            if w.shm is not None:
                w.shm["actions"].copy_(actions_host[w.sl])
                payload = {}
            else:
                payload = {"actions": actions_host[w.sl].contiguous()}
            try:
                w.conn.send(WorkerCommand(cmd="step", payload=payload))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"Broken pipe sending step to worker (pid={w.process.pid}, exitcode={w.process.exitcode})."
                ) from exc

        episodes: List[Tuple[Dict, int]] = []
        for w, rep in self._gather_replies(event_name="step"):
            if w.shm is not None:
                obs_src = w.shm["obs"]
                critic_src = w.shm["critic_obs"]
                rew_src = w.shm["rew"]
                done_src = w.shm["done"]
                time_outs_src = w.shm["time_outs"]
                done_local = done_src
            else:
                payload = rep.payload
                obs_src = payload["obs"]
                critic_src = payload["critic_obs"]
                rew_src = payload["rew"]
                done_src = payload["done"]
                time_outs_src = payload["time_outs"]
                done_local = done_src

            self._copy_chunk_into_global_buffers(
                w=w,
                obs_src=obs_src,
                critic_src=critic_src,
                rew_src=rew_src,
                done_src=done_src,
                time_outs_src=time_outs_src,
            )

            ep = rep.payload.get("episode")
            if isinstance(ep, dict):
                try:
                    ep_count = int(done_local.sum().item())
                except Exception:
                    ep_count = 1
                episodes.append((ep, max(ep_count, 1)))

        extras = {
            "observations": {"critic": self._critic},
            "time_outs": self._time_outs,
        }
        if episodes:
            agg: Dict[str, float] = {}
            total_weight = 0
            for e, weight in episodes:
                total_weight += weight
                for k, v in e.items():
                    try:
                        agg[k] = agg.get(k, 0.0) + float(v) * weight
                    except Exception:
                        pass
            if total_weight > 0:
                for k in list(agg.keys()):
                    agg[k] /= float(total_weight)
                extras["episode"] = agg

        return self._obs, self._critic, self._rew, self._done, extras

    def close(self) -> None:
        for w in self._workers:
            try:
                w.conn.send(WorkerCommand(cmd="close", payload={}))
            except Exception:
                pass

        for w in self._workers:
            try:
                _ = self._recv_reply(w.conn, worker_idx=w.worker_idx, process=w.process)
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
