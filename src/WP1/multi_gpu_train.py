"""
Multi-GPU foundation training via per-GPU worker processes.
===========================================================

Architecture
------------
The URDF catalog is split evenly across N GPUs.  Each GPU runs a
``Gen_Env`` in a dedicated subprocess (one ``gs.init()`` per process,
which is the only safe way to use Genesis across multiple GPUs).

The *coordinator* (this process) owns the policy and the RSL-RL
``OnPolicyRunner``.  It presents a ``MultiGPUEnv`` proxy to the runner —
a standard VecEnv whose ``step()`` fans actions out to the workers and
collects physics results back.  The policy, PPO algorithm, logging, and
checkpointing all live in the coordinator.

Data flow per step
------------------
1. Coordinator calls ``runner.alg.act(obs, priv_obs)``  →  actions.
2. ``MultiGPUEnv.step(actions)`` splits actions and puts them on each
   worker's command queue (tensors sent on CPU to avoid CUDA IPC).
3. Each worker runs ``env.step(actions)`` on its GPU and puts the result
   on its result queue.
4. ``MultiGPUEnv`` collects and merges results, returns to the runner.
5. PPO storage is filled, ``alg.update()`` runs on the coordinator's GPU.

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
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from WP1.config import RunConfig


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
    cmd_q: "mp.Queue[dict]",
    result_q: "mp.Queue[dict]",
) -> None:
    """
    Runs in a subprocess.  Owns one GPU, builds Gen_Env, then serves
    'reset' / 'step' / 'stop' commands from the coordinator.

    All tensors are transferred on CPU to avoid cross-process CUDA IPC
    complexities.
    """
    # ------------------------------------------------------------------ #
    # 1. Environment isolation                                           #
    # ------------------------------------------------------------------ #
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["GS_PARA_LEVEL"] = "4"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("GS_HEADLESS_NO_GL", "1")

    import genesis as gs  # imported after CUDA_VISIBLE_DEVICES is set
    gs.init(logging_level="error", backend=gs.gpu)

    from winged_drone_train.train import configure_solver_noise
    from general_policy.env_gen import Gen_Env

    device = "cuda:0"  # single visible GPU in this process

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

    # ------------------------------------------------------------------ #
    # 2. Signal ready and send metadata to coordinator                   #
    # ------------------------------------------------------------------ #
    result_q.put({
        "type": "ready",
        "num_envs": env.num_envs,
        "num_obs": env.num_obs,
        "num_actions": env.num_actions,
        "num_privileged_obs": env.num_privileged_obs,
        "dt": env.dt,
        "max_episode_length": env.max_episode_length,
    })

    # ------------------------------------------------------------------ #
    # 3. Serve requests                                                  #
    # ------------------------------------------------------------------ #
    while True:
        cmd = cmd_q.get()

        if cmd["type"] == "reset":
            obs, info = env.reset()
            result_q.put({
                "type": "reset_result",
                "obs": obs.cpu(),
                "critic": _get_critic(info, env),
                "episode_length_buf": env.episode_length_buf.cpu(),
            })

        elif cmd["type"] == "step":
            actions = cmd["actions"].to(device)
            obs, rew, done, info = env.step(actions)
            result_q.put({
                "type": "step_result",
                "obs": obs.cpu(),
                "rew": rew.cpu(),
                "done": done.cpu(),
                "critic": _get_critic(info, env),
                "time_outs": info.get(
                    "time_outs", torch.zeros(env.num_envs)
                ).cpu(),
                "episode": info.get("episode"),
                "episode_length_buf": env.episode_length_buf.cpu(),
            })

        elif cmd["type"] == "stop":
            break

    try:
        gs.destroy()
    except Exception:
        pass


def _get_critic(info: dict, env) -> Optional[torch.Tensor]:
    """Extract critic obs from info, falling back to env buffer."""
    critic = info.get("observations", {}).get("critic")
    if critic is not None:
        return critic.cpu()
    return env.privileged_obs_buf.cpu()


# ---------------------------------------------------------------------------
# Coordinator-side proxy VecEnv
# ---------------------------------------------------------------------------

class _WorkerProxy:
    """Handle on one worker: queues + cached metadata."""

    def __init__(self, cmd_q: "mp.Queue", result_q: "mp.Queue", meta: dict):
        self.cmd_q = cmd_q
        self.result_q = result_q
        self.num_envs: int = meta["num_envs"]
        self.num_obs: int = meta["num_obs"]
        self.num_actions: int = meta["num_actions"]
        self.num_privileged_obs: int = meta["num_privileged_obs"]
        self.max_episode_length: int = meta["max_episode_length"]
        self.dt: float = meta["dt"]


class MultiGPUEnv:
    """
    Proxy VecEnv that fans step/reset calls across GPU worker processes.

    Presented to RSL-RL's ``OnPolicyRunner`` as a standard VecEnv — the
    runner does not need to know that physics runs on remote processes.
    """

    def __init__(self, proxies: List[_WorkerProxy], device: str) -> None:
        self._proxies = proxies
        self.device = torch.device(device)

        # Aggregate metadata
        self.num_envs = sum(p.num_envs for p in proxies)
        self.num_actions = proxies[0].num_actions
        self.num_obs = proxies[0].num_obs
        self.num_privileged_obs = proxies[0].num_privileged_obs
        self.max_episode_length = proxies[0].max_episode_length
        self.dt = proxies[0].dt

        # Slices: each proxy owns a contiguous block of env indices
        self._slices: List[slice] = []
        offset = 0
        for p in proxies:
            self._slices.append(slice(offset, offset + p.num_envs))
            offset += p.num_envs

        B = self.num_envs
        self.obs_buf = torch.zeros(B, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            B, self.num_privileged_obs, device=self.device
        )
        self.rew_buf = torch.zeros(B, device=self.device)
        self.reset_buf = torch.ones(B, dtype=torch.int64, device=self.device)
        self.episode_length_buf = torch.zeros(
            B, dtype=torch.int64, device=self.device
        )
        self.extras: Dict = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs": torch.zeros(B, device=self.device),
        }

    # ------------------------------------------------------------------ #
    # VecEnv API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        for p in self._proxies:
            p.cmd_q.put({"type": "reset"})
        results = [p.result_q.get() for p in self._proxies]
        self._merge_reset(results)
        self.reset_buf.fill_(1)
        return self.obs_buf, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        actions_cpu = actions.cpu()
        for p, sl in zip(self._proxies, self._slices):
            p.cmd_q.put({"type": "step", "actions": actions_cpu[sl]})
        results = [p.result_q.get() for p in self._proxies]
        self._merge_step(results)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        return self.obs_buf, dict(self.extras)

    def close(self) -> None:
        for p in self._proxies:
            try:
                p.cmd_q.put({"type": "stop"})
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _merge_reset(self, results: List[dict]) -> None:
        self.extras["time_outs"].zero_()
        if "episode" in self.extras:
            del self.extras["episode"]
        for res, sl in zip(results, self._slices):
            self.obs_buf[sl] = res["obs"].to(self.device)
            if res["critic"] is not None:
                self.privileged_obs_buf[sl] = res["critic"].to(self.device)
            self.episode_length_buf[sl] = res["episode_length_buf"].to(self.device)

    def _merge_step(self, results: List[dict]) -> None:
        self.extras["time_outs"].zero_()
        if "episode" in self.extras:
            del self.extras["episode"]

        episodes: List[Tuple[dict, int]] = []
        for res, sl in zip(results, self._slices):
            self.obs_buf[sl] = res["obs"].to(self.device)
            self.rew_buf[sl] = res["rew"].to(self.device)
            self.reset_buf[sl] = res["done"].to(self.device)
            if res["critic"] is not None:
                self.privileged_obs_buf[sl] = res["critic"].to(self.device)
            self.extras["time_outs"][sl] = res["time_outs"].to(self.device)
            self.episode_length_buf[sl] = res["episode_length_buf"].to(
                self.device
            )
            ep = res.get("episode")
            if ep:
                n = max(int(res["done"].sum().item()), 1)
                episodes.append((ep, n))

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
                self.extras["episode"] = {
                    k: v / total for k, v in merged.items()
                }


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

    Parameters
    ----------
    cfg : RunConfig
        Full training configuration.  Must have ``catalog.n_urdf > 0`` or
        ``catalog.catalog_dir`` pointing to an existing URDF catalog.
    num_gpus : int
        Number of GPUs to use.  Must be <= ``torch.cuda.device_count()``.
        The URDF catalog and ``num_envs`` are split evenly across GPUs.
    vis : bool
        Ignored (viewer not supported in multi-GPU mode).
    resume : bool
        If ``True``, resume from the latest matching run folder.
    """
    from rsl_rl.runners import OnPolicyRunner
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh
    from winged_drone_train.rl.logging import RLTrainingLogger
    from winged_drone_train.train import _configure_cache_root
    from winged_drone_train.runtime_random import seed_runtime_randomness
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

    # ------------------------------------------------------------------ #
    # Validate GPU count                                                 #
    # ------------------------------------------------------------------ #
    n_available = torch.cuda.device_count()
    if n_available < num_gpus:
        raise RuntimeError(
            f"[MultiGPU] Requested {num_gpus} GPUs but only {n_available} "
            "available (check CUDA_VISIBLE_DEVICES)."
        )

    # ------------------------------------------------------------------ #
    # Run folder and config                                              #
    # ------------------------------------------------------------------ #
    run = RunManager(cfg, resume=resume)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.to_legacy_cfgs()
    device = cfg.training.device  # coordinator device, e.g. "cuda:0"

    # ------------------------------------------------------------------ #
    # Build / locate URDF catalog                                        #
    # ------------------------------------------------------------------ #
    catalog_path: Optional[Path] = None
    if cfg.catalog.catalog_dir is not None:
        catalog_path = Path(cfg.catalog.catalog_dir)
    if cfg.catalog.n_urdf is not None and cfg.catalog.n_urdf > 0:
        if catalog_path is None:
            catalog_path = run.run_dir / "catalog"
        print(
            f"[MultiGPU] Building catalog: "
            f"n={cfg.catalog.n_urdf}, seed={cfg.catalog.urdf_seed}"
        )
        build_catalog(
            catalog_path, n=cfg.catalog.n_urdf, seed=cfg.catalog.urdf_seed
        )
    elif catalog_path is not None and catalog_path.is_dir():
        pass  # existing catalog directory
    else:
        raise RuntimeError(
            "[MultiGPU] Multi-GPU training requires a URDF catalog. "
            "Set catalog.n_urdf > 0 or provide catalog.catalog_dir."
        )

    run.save_catalog(catalog_path)

    # Load full URDF list
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

    # ------------------------------------------------------------------ #
    # Split URDFs and num_envs across GPUs                              #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Spawn one worker process per GPU (parallel initialization)         #
    # ------------------------------------------------------------------ #
    ctx = mp.get_context("spawn")
    proxies: List[_WorkerProxy] = []
    processes: List[mp.Process] = []

    # Start all worker processes
    worker_queues: List[Tuple[int, "mp.Queue", "mp.Queue"]] = []
    for rank in range(num_gpus):
        cmd_q: mp.Queue = ctx.Queue()
        result_q: mp.Queue = ctx.Queue()

        p = ctx.Process(
            target=_worker_process,
            args=(
                rank,
                rank,                  # physical GPU index
                urdf_shards[rank],
                env_counts[rank],
                env_cfg,
                obs_cfg,
                reward_cfg,
                command_cfg,
                cmd_q,
                result_q,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)
        worker_queues.append((rank, cmd_q, result_q))
        print(f"[MultiGPU] Spawned worker {rank} (GPU {rank})")

    # Collect ready messages from all workers (initialize in parallel)
    print(f"[MultiGPU] Waiting for all {num_gpus} workers to initialise…")
    for rank, cmd_q, result_q in worker_queues:
        meta = result_q.get(timeout=6000)  # genesis scene build can be slow
        if meta.get("type") != "ready":
            raise RuntimeError(
                f"[MultiGPU] Worker {rank} sent unexpected message: {meta}"
            )

        proxy = _WorkerProxy(cmd_q, result_q, meta)
        proxies.append(proxy)
        print(
            f"[MultiGPU] Worker {rank} ready — "
            f"{proxy.num_envs} envs, {len(urdf_shards[rank])} URDFs"
        )

    # ------------------------------------------------------------------ #
    # Build proxy env and RSL-RL runner                                  #
    # ------------------------------------------------------------------ #
    env = MultiGPUEnv(proxies, device=device)

    runner = OnPolicyRunner(env, train_cfg, str(run.log_dir), device=device)

    # Optionally compile policy for faster forward passes
    try:
        runner.alg.policy = torch.compile(
            runner.alg.policy, mode="reduce-overhead"
        )
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

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #
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
            csv_logger.log(
                it, env.extras, mean_reward=mean_rew, mean_episode_length=mean_ep_len
            )
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

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Post-training artefacts                                            #
    # ------------------------------------------------------------------ #
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
