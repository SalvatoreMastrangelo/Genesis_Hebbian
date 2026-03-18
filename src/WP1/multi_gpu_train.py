"""
Multi-GPU foundation training via per-GPU worker processes (shared memory).
===========================================================================

Architecture
------------
The URDF catalog is split evenly across N GPUs.  Each GPU runs a
``Gen_Env`` in a dedicated subprocess (one ``gs.init()`` per process).

The *coordinator* (this process) owns the policy and the RSL-RL
``OnPolicyRunner``.  It presents a ``MultiGPUEnv`` proxy to the runner.

IPC design (shared-memory backed)
----------------------------------
- ``multiprocessing.Pipe`` carries only small control/metadata messages
  (commands, episode dicts, errors).  No tensors travel over the Pipe.
- All tensor data (obs, critic, rew, done, time_outs, actions) is passed
  through ``torch.share_memory_()`` CPU buffers created once at startup.

Data flow per step
------------------
1. Coordinator writes ``actions_cpu[sl]`` into each worker's shared
   ``actions`` buffer via ``copy_()``.
2. Coordinator sends a tiny ``_Cmd("step")`` to all workers via Pipe.
3. Each worker reads actions from its shared buffer, runs ``env.step()``,
   and writes all result tensors back into its shared buffers.
4. Each worker sends a tiny ``_Reply`` (episode dict only) via Pipe.
5. Coordinator reads results directly from shared buffers with per-slice
   ``copy_()`` into pre-allocated GPU tensors — zero intermediate
   allocation, no Queue serialisation, no redundant CPU copies.
6. PPO storage is filled, ``alg.update()`` runs on the coordinator GPU.

Usage
-----
Called automatically from ``WP1.train`` when ``--multi-gpu N`` is given::

    python -m WP1.train --cfg src/WP1/experiments/example.yaml --multi-gpu 2

Or programmatically::

    from WP1.multi_gpu_train import train_multi_gpu
    train_multi_gpu(cfg, num_gpus=2)
"""
from __future__ import annotations

import builtins
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from WP1.config import RunConfig


# ---------------------------------------------------------------------------
# IPC types  (only metadata travels over the Pipe — no tensors)
# ---------------------------------------------------------------------------

@dataclass
class _Cmd:
    """Command sent from coordinator to worker via Pipe."""
    cmd: str   # "reset" | "step" | "stop"


@dataclass
class _Reply:
    """Reply sent from worker to coordinator via Pipe."""
    ok: bool
    event: str = ""
    episode: Optional[Dict] = None
    error: str = ""
    meta: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Shared-buffer factory
# ---------------------------------------------------------------------------

def _make_shared_buffers(
    *,
    num_envs: int,
    num_obs: int,
    num_privileged_obs: int,
    num_actions: int,
) -> Dict[str, torch.Tensor]:
    """Allocate CPU shared-memory tensors that both processes can read/write.

    ``device="cpu"`` is mandatory: after ``gs.init(backend=gs.gpu)`` Genesis
    may set the default device to CUDA, causing bare ``torch.zeros(...)`` to
    create GPU tensors.  GPU tensors use CUDA IPC (``_share_cuda_()``) which
    is not supported across independent processes and raises
    ``CUDA driver error: invalid argument``.
    """
    cpu = torch.device("cpu")
    return {
        "actions":           torch.zeros((num_envs, num_actions),         dtype=torch.float32, device=cpu).share_memory_(),
        "obs":               torch.zeros((num_envs, num_obs),             dtype=torch.float32, device=cpu).share_memory_(),
        "critic_obs":        torch.zeros((num_envs, num_privileged_obs),  dtype=torch.float32, device=cpu).share_memory_(),
        "rew":               torch.zeros((num_envs,),                     dtype=torch.float32, device=cpu).share_memory_(),
        "done":              torch.zeros((num_envs,),                     dtype=torch.int64,   device=cpu).share_memory_(),
        "time_outs":         torch.zeros((num_envs,),                     dtype=torch.float32, device=cpu).share_memory_(),
        "episode_length_buf":torch.zeros((num_envs,),                     dtype=torch.int64,   device=cpu).share_memory_(),
    }


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_process(
    rank: int,
    gpu_id: int,
    urdf_list: List[str],
    num_envs: int,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    conn,  # child end of multiprocessing.Pipe
) -> None:
    """
    Runs in a subprocess.  Owns one GPU, builds Gen_Env, then serves
    reset / step / stop commands from the coordinator.

    Tensor data flows through shared memory buffers — the Pipe carries only
    small control signals and episode dicts.
    """

    def _reply(ok: bool, event: str = "", episode=None, error: str = "", meta=None):
        try:
            conn.send(_Reply(ok=ok, event=event, episode=episode, error=error, meta=meta))
        except Exception:
            pass

    env = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["GS_PARA_LEVEL"] = "4"
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("GS_HEADLESS_NO_GL", "1")

        import genesis as gs
        from winged_drone_train.train import configure_solver_noise
        from general_policy.env_gen import Gen_Env

        gs.init(logging_level="error", backend=gs.gpu)

        device = "cuda:0"  # CUDA_VISIBLE_DEVICES isolates to single GPU

        env = Gen_Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_list=urdf_list,
            device=device,
        )
        configure_solver_noise(env, env_cfg)

        # Allocate shared buffers and do initial reset
        shm = _make_shared_buffers(
            num_envs=int(env.num_envs),
            num_obs=int(env.num_obs),
            num_privileged_obs=int(env.num_privileged_obs),
            num_actions=int(env.num_actions),
        )

        obs, info = env.reset()
        critic = info.get("observations", {}).get("critic")
        if critic is None:
            critic = env.privileged_obs_buf

        shm["obs"].copy_(_to_cpu(obs))
        shm["critic_obs"].copy_(_to_cpu(critic))
        shm["episode_length_buf"].copy_(_to_cpu(env.episode_length_buf))

        # Send ready signal + shared buffers in a single message (mirrors base
        # code pattern — avoids race conditions with two separate sends).
        conn.send({
            "ok": True,
            "event": "ready",
            "meta": {
                "num_envs":          int(env.num_envs),
                "num_obs":           int(env.num_obs),
                "num_privileged_obs":int(env.num_privileged_obs),
                "num_actions":       int(env.num_actions),
                "max_episode_length":int(env.max_episode_length),
                "dt":                float(env.dt),
            },
            "shm": shm,
        })

        # Main command loop
        while True:
            msg = conn.recv()
            if not isinstance(msg, _Cmd):
                _reply(ok=False, error=f"Unexpected message type: {type(msg)}")
                continue

            if msg.cmd == "stop":
                break

            elif msg.cmd == "reset":
                obs, info = env.reset()
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                shm["obs"].copy_(_to_cpu(obs))
                shm["critic_obs"].copy_(_to_cpu(critic))
                shm["episode_length_buf"].copy_(_to_cpu(env.episode_length_buf))
                _reply(ok=True, event="reset")

            elif msg.cmd == "step":
                # Actions were already written into shm["actions"] by the coordinator
                actions = shm["actions"].to(device)
                obs, rew, done, info = env.step(actions)
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                time_outs = info.get("time_outs")
                if time_outs is None:
                    time_outs = torch.zeros_like(done, dtype=torch.float32)
                episode = info.get("episode") if isinstance(info, dict) else None

                shm["obs"].copy_(_to_cpu(obs))
                shm["critic_obs"].copy_(_to_cpu(critic))
                shm["rew"].copy_(_to_cpu(rew))
                shm["done"].copy_(_to_cpu(done).long())
                shm["time_outs"].copy_(_to_cpu(time_outs).float())
                shm["episode_length_buf"].copy_(_to_cpu(env.episode_length_buf))

                # Clear GPU cache to prevent memory accumulation over thousands of steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Only the small episode dict travels over the Pipe
                _reply(ok=True, event="step", episode=episode if isinstance(episode, dict) else None)

            else:
                _reply(ok=False, error=f"Unknown command: {msg.cmd}")

    except Exception as exc:
        tb = traceback.format_exc()
        try:
            conn.send({"ok": False, "event": "error", "error": f"Worker crash: {exc}\n{tb}"})
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Coordinator-side worker handle
# ---------------------------------------------------------------------------

class _WorkerHandle:
    """Coordinator-side handle: Pipe connection + shared buffer references."""

    def __init__(self, conn, meta: dict, shm: Dict[str, torch.Tensor]):
        self.conn = conn
        self.num_envs: int          = meta["num_envs"]
        self.num_obs: int           = meta["num_obs"]
        self.num_actions: int       = meta["num_actions"]
        self.num_privileged_obs: int= meta["num_privileged_obs"]
        self.max_episode_length: int= meta["max_episode_length"]
        self.dt: float              = meta["dt"]
        self.shm = shm

    def recv(self) -> _Reply:
        msg = self.conn.recv()
        if not isinstance(msg, _Reply):
            return _Reply(ok=False, error=f"Unexpected message: {msg}")
        return msg


# ---------------------------------------------------------------------------
# MultiGPUEnv proxy (shared-memory backed)
# ---------------------------------------------------------------------------

class MultiGPUEnv:
    """
    Proxy VecEnv backed by shared-memory tensors from GPU workers.

    Tensor data (obs, rew, done, critic, time_outs) flows through shared CPU
    memory: workers write into pre-allocated shared buffers, and the
    coordinator copies directly from those buffers into pre-allocated GPU
    tensors via per-slice ``copy_()`` — no Queue serialisation, no temporary
    allocations, no redundant CPU copies.
    """

    def __init__(self, handles: List[_WorkerHandle], device: str) -> None:
        self._handles = handles
        self.device = torch.device(device)

        self.num_envs           = sum(h.num_envs for h in handles)
        self.num_actions        = handles[0].num_actions
        self.num_obs            = handles[0].num_obs
        self.num_privileged_obs = handles[0].num_privileged_obs
        self.max_episode_length = handles[0].max_episode_length
        self.dt                 = handles[0].dt

        # Contiguous index slices per worker
        self._slices: List[slice] = []
        offset = 0
        for h in handles:
            self._slices.append(slice(offset, offset + h.num_envs))
            offset += h.num_envs

        # Pre-allocated GPU buffers — workers write into shared CPU buffers,
        # coordinator copies into these via per-slice copy_()
        B = self.num_envs
        self.obs_buf            = torch.zeros(B, self.num_obs,            device=self.device)
        self.privileged_obs_buf = torch.zeros(B, self.num_privileged_obs, device=self.device)
        self.rew_buf            = torch.zeros(B,                          device=self.device)
        self.reset_buf          = torch.ones( B, dtype=torch.int64,       device=self.device)
        self.episode_length_buf = torch.zeros(B, dtype=torch.int64,       device=self.device)
        self.extras: Dict = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs":    torch.zeros(B, device=self.device),
        }

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        for h in self._handles:
            h.conn.send(_Cmd("reset"))
        for h in self._handles:
            rep = h.recv()
            if not rep.ok:
                raise RuntimeError(f"Worker error on reset: {rep.error}")
        self._copy_obs_critic()
        self.reset_buf.fill_(1)
        self.extras["time_outs"].zero_()
        if "episode" in self.extras:
            del self.extras["episode"]
        return self.obs_buf, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        # Write actions into each worker's shared buffer (no Pipe overhead)
        actions_cpu = actions.cpu()
        for h, sl in zip(self._handles, self._slices):
            h.shm["actions"].copy_(actions_cpu[sl])

        # Fan out step command — workers read actions from shared memory
        for h in self._handles:
            h.conn.send(_Cmd("step"))

        # Collect tiny replies (episode dicts only — no tensors in Pipe)
        episodes: List[Tuple[dict, int]] = []
        for h in self._handles:
            rep = h.recv()
            if not rep.ok:
                raise RuntimeError(f"Worker error on step: {rep.error}")
            if rep.episode:
                n = max(int(h.shm["done"].sum().item()), 1)
                episodes.append((rep.episode, n))

        # Bulk-copy all result tensors from shared memory → GPU (zero alloc)
        self._copy_step_results()

        if episodes:
            merged: dict = {}
            total = 0
            for ep_dict, w in episodes:
                total += w
                for k, v in ep_dict.items():
                    try:
                        merged[k] = merged.get(k, 0.0) + float(v) * w
                    except Exception:
                        pass
            if total > 0:
                self.extras["episode"] = {k: v / total for k, v in merged.items()}
        elif "episode" in self.extras:
            del self.extras["episode"]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        return self.obs_buf, dict(self.extras)

    def close(self) -> None:
        for h in self._handles:
            try:
                h.conn.send(_Cmd("stop"))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal: direct shared-memory → GPU slice copies (no temporaries)
    # ------------------------------------------------------------------

    def _copy_obs_critic(self) -> None:
        for h, sl in zip(self._handles, self._slices):
            self.obs_buf[sl].copy_(h.shm["obs"])
            self.privileged_obs_buf[sl].copy_(h.shm["critic_obs"])
            self.episode_length_buf[sl].copy_(h.shm["episode_length_buf"])
        self.extras["observations"]["critic"] = self.privileged_obs_buf

    def _copy_step_results(self) -> None:
        for h, sl in zip(self._handles, self._slices):
            self.obs_buf[sl].copy_(h.shm["obs"])
            self.privileged_obs_buf[sl].copy_(h.shm["critic_obs"])
            self.rew_buf[sl].copy_(h.shm["rew"])
            self.reset_buf[sl].copy_(h.shm["done"])
            self.extras["time_outs"][sl].copy_(h.shm["time_outs"])
            self.episode_length_buf[sl].copy_(h.shm["episode_length_buf"])
        self.extras["observations"]["critic"] = self.privileged_obs_buf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_multi_gpu(
    cfg: RunConfig,
    num_gpus: int,
    vis: bool = False,
    resume: bool = False,
) -> None:
    """
    Train a foundation policy with physics distributed across ``num_gpus`` GPUs.

    Uses shared memory for zero-copy tensor passing between workers and
    coordinator — eliminates the per-step CPU bottleneck from Queue transfers.

    Parameters
    ----------
    cfg : RunConfig
        Full training configuration.  Must have ``catalog.n_urdf > 0`` or
        ``catalog.catalog_dir`` pointing to an existing URDF catalog.
    num_gpus : int
        Number of GPUs to use.  Must be <= ``torch.cuda.device_count()``.
    vis : bool
        Ignored (viewer not supported in multi-GPU mode).
    resume : bool
        If ``True``, resume from the latest matching run folder.
    """
    from rsl_rl.runners import OnPolicyRunner
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh
    from winged_drone_train.rl.logging import RLTrainingLogger
    from winged_drone_train.train import _configure_cache_root
    from general_policy.catalog import build_catalog
    from WP1.run_manager import RunManager
    from WP1.csv_logger import CSVLogger
    from WP1.plotting import plot_run
    from WP1.eval_videos import _generate_eval_videos
    from WP1.train import _enrich_csv_with_tensorboard_rewards

    builtins.ActorCriticTanh = ActorCriticTanh

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("GS_HEADLESS_NO_GL", "1")

    _configure_cache_root()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Validate GPU count
    # ------------------------------------------------------------------
    n_available = torch.cuda.device_count()
    if n_available < num_gpus:
        raise RuntimeError(
            f"[MultiGPU] Requested {num_gpus} GPUs but only {n_available} "
            "available (check CUDA_VISIBLE_DEVICES)."
        )

    # ------------------------------------------------------------------
    # Run folder and config
    # ------------------------------------------------------------------
    run = RunManager(cfg, resume=resume)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.to_legacy_cfgs()
    device = cfg.training.device

    # ------------------------------------------------------------------
    # Build / locate URDF catalog
    # ------------------------------------------------------------------
    catalog_path: Optional[Path] = None
    if cfg.catalog.catalog_dir is not None:
        catalog_path = Path(cfg.catalog.catalog_dir)
    if cfg.catalog.n_urdf is not None and cfg.catalog.n_urdf > 0:
        if catalog_path is None:
            catalog_path = run.run_dir / "catalog"
        print(f"[MultiGPU] Building catalog: n={cfg.catalog.n_urdf}, seed={cfg.catalog.urdf_seed}")
        build_catalog(catalog_path, n=cfg.catalog.n_urdf, seed=cfg.catalog.urdf_seed)
    elif catalog_path is not None and catalog_path.is_dir():
        pass
    else:
        raise RuntimeError(
            "[MultiGPU] Multi-GPU training requires a URDF catalog. "
            "Set catalog.n_urdf > 0 or provide catalog.catalog_dir."
        )

    run.save_catalog(catalog_path)

    catalog_file = catalog_path / "catalog.txt"
    if catalog_file.exists():
        urdf_list = []
        for s in catalog_file.read_text().splitlines():
            s = s.strip()
            if not s:
                continue
            p = Path(s)
            if not p.is_absolute():
                p = catalog_path / p
            urdf_list.append(str(p))
    else:
        urdf_list = sorted(str(p) for p in catalog_path.glob("*.urdf"))

    if len(urdf_list) < num_gpus:
        raise RuntimeError(
            f"[MultiGPU] Need at least {num_gpus} URDFs (one per GPU), "
            f"but catalog has only {len(urdf_list)}."
        )

    # ------------------------------------------------------------------
    # Split URDFs and num_envs across GPUs
    # ------------------------------------------------------------------
    K = len(urdf_list)
    base_u, rem_u = divmod(K, num_gpus)
    urdf_shards: List[List[str]] = []
    offset = 0
    for i in range(num_gpus):
        sz = base_u + (1 if i < rem_u else 0)
        urdf_shards.append(urdf_list[offset : offset + sz])
        offset += sz

    base_e, rem_e = divmod(cfg.training.num_envs, num_gpus)
    env_counts = [base_e + (1 if i < rem_e else 0) for i in range(num_gpus)]

    print(
        f"[MultiGPU] GPUs={num_gpus}  "
        f"URDFs/GPU={[len(s) for s in urdf_shards]}  "
        f"envs/GPU={env_counts}"
    )

    # ------------------------------------------------------------------
    # Spawn one worker process per GPU
    # ------------------------------------------------------------------
    ctx = mp.get_context("spawn")
    pending: List[Tuple[int, mp.Process, object]] = []

    for rank in range(num_gpus):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker_process,
            args=(
                rank,
                rank,               # physical GPU index
                urdf_shards[rank],
                env_counts[rank],
                env_cfg,
                obs_cfg,
                reward_cfg,
                command_cfg,
                child_conn,
            ),
            daemon=True,
        )
        p.start()
        pending.append((rank, p, parent_conn))
        print(f"[MultiGPU] Spawned worker {rank} (GPU {rank})")

    # Collect ready messages + shared buffers
    print(f"[MultiGPU] Waiting for all {num_gpus} workers to initialise…")
    handles: List[_WorkerHandle] = []
    processes: List[mp.Process] = []

    for rank, p, parent_conn in pending:
        # Single message: dict with "ok", "event", "meta", and "shm" keys
        msg = parent_conn.recv()
        if not isinstance(msg, dict) or not msg.get("ok") or msg.get("event") != "ready":
            raise RuntimeError(
                f"[MultiGPU] Worker {rank} failed to start: "
                f"{msg.get('error', msg) if isinstance(msg, dict) else msg}"
            )
        shm = msg.get("shm")
        if not isinstance(shm, dict):
            raise RuntimeError(f"[MultiGPU] Worker {rank} did not include shared buffers")

        handle = _WorkerHandle(conn=parent_conn, meta=msg["meta"], shm=shm)
        handles.append(handle)
        processes.append(p)
        print(
            f"[MultiGPU] Worker {rank} ready — "
            f"{handle.num_envs} envs, {len(urdf_shards[rank])} URDFs"
        )

    # ------------------------------------------------------------------
    # Build proxy env and RSL-RL runner
    # ------------------------------------------------------------------
    env = MultiGPUEnv(handles, device=device)

    # Seed initial GPU buffers from the shared memory initial reset
    env._copy_obs_critic()

    runner = OnPolicyRunner(env, train_cfg, str(run.log_dir), device=device)

    try:
        runner.alg.policy = torch.compile(runner.alg.policy, mode="reduce-overhead")
        _orig_save = runner.save

        def _save_unwrapped(path, infos=None):
            policy = runner.alg.policy
            unwrapped = getattr(policy, "_orig_mod", policy)
            runner.alg.policy = unwrapped
            _orig_save(path, infos)
            runner.alg.policy = policy

        runner.save = _save_unwrapped
        print("[MultiGPU] torch.compile enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"[MultiGPU] torch.compile skipped: {e}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    rl_logger = RLTrainingLogger(runner=runner, log_dir=run.log_dir)
    rl_logger.attach()
    csv_logger = CSVLogger(run.eval_dir / "training_log.csv")

    _iter: Dict[str, int] = {"i": 0}

    def _on_iter_end() -> None:
        it = _iter["i"]
        try:
            mean_rew = mean_ep_len = None
            rb = getattr(runner, "rewbuffer", None)
            if rb and len(rb):
                mean_rew = sum(rb) / len(rb)
            lb = getattr(runner, "lenbuffer", None)
            if lb and len(lb):
                mean_ep_len = sum(lb) / len(lb)
            csv_logger.log(it, env.extras, mean_reward=mean_rew, mean_episode_length=mean_ep_len)
        except Exception as exc:
            print(f"[MultiGPU] CSV log error at iter {it}: {exc}")
        _iter["i"] = it + 1

    alg = getattr(runner, "alg", None)
    if alg is not None:
        _orig_update = alg.update

        def _patched_update(*args, **kwargs):
            result = _orig_update(*args, **kwargs)
            _on_iter_end()
            return result

        alg.update = _patched_update

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    try:
        runner.learn(
            num_learning_iterations=cfg.training.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        rl_logger.close()
        csv_logger.close()
        env.close()
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    # ------------------------------------------------------------------
    # Post-training artefacts
    # ------------------------------------------------------------------
    try:
        _enrich_csv_with_tensorboard_rewards(
            csv_path=run.eval_dir / "training_log.csv",
            tb_dir=run.log_dir,
        )
    except Exception as exc:
        print(f"[MultiGPU] TensorBoard reward extraction skipped: {exc}")

    try:
        _generate_eval_videos(run.run_dir)
    except Exception as exc:
        print(f"[MultiGPU] Video generation skipped: {exc}")

    print("[MultiGPU] Generating plots…")
    plot_run(run.run_dir)
    print(f"[MultiGPU] Done. Results in: {run.run_dir}")
