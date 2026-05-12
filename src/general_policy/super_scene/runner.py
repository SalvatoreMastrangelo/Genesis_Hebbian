from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from rsl_rl.runners import OnPolicyRunner

from winged_drone_train.train import _maybe_load_init_checkpoint
from winged_drone_train.rl.logging import RLTrainingLogger

from .catalog import chunk_list, load_catalog_urdfs, split_even
from .orchestrator import LogicalSuperSceneOrchestrator


class LogicalSuperSceneVecEnv:
    """
    VecEnv adapter over LogicalSuperSceneOrchestrator.

    This exposes the same API expected by RSL-RL OnPolicyRunner, so PPO rollout,
    learning, logging and checkpointing are exactly the standard ones.
    """

    def __init__(self, orchestrator: LogicalSuperSceneOrchestrator, device: str) -> None:
        self.orchestrator = orchestrator
        self.device = torch.device(device)

        self.num_envs = int(orchestrator.num_envs)
        self.num_obs = int(orchestrator.num_obs)
        self.num_privileged_obs = int(orchestrator.num_privileged_obs)
        self.num_actions = int(orchestrator.num_actions)
        self.max_episode_length = int(orchestrator.max_episode_length)
        self.dt = float(orchestrator.dt)
        self.step_dt = self.dt

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float32
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.extras: Dict = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs": torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32),
        }

        # Orchestrator already collected an initial reset from each worker at startup.
        # Reusing that state avoids an immediate all-workers reset burst.
        self.obs_buf = orchestrator._obs.to(self.device)  # populated in orchestrator.__init__
        self.privileged_obs_buf = orchestrator._critic.to(self.device)
        self.extras["observations"]["critic"] = self.privileged_obs_buf

    def reset(self):
        obs, critic = self.orchestrator.reset()
        self.obs_buf = obs.to(self.device)
        self.privileged_obs_buf = critic.to(self.device)
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        self.extras = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs": torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32),
        }
        return self.obs_buf, self.extras

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf, {}

    def step(self, actions: torch.Tensor):
        obs, critic, rew, done, extras = self.orchestrator.step(actions.to(self.device))
        self.obs_buf = obs.to(self.device)
        self.privileged_obs_buf = critic.to(self.device)
        self.rew_buf = rew.to(self.device)
        self.reset_buf = done.to(self.device)
        self.episode_length_buf += 1
        done_mask = self.reset_buf > 0
        self.episode_length_buf[done_mask] = 0

        out_extras = dict(extras) if isinstance(extras, dict) else {}
        obs_dict = dict(out_extras.get("observations", {}))
        obs_dict["critic"] = self.privileged_obs_buf
        out_extras["observations"] = obs_dict
        if "time_outs" not in out_extras:
            out_extras["time_outs"] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.extras = out_extras
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
        self.orchestrator.close()


def _validate_shards(shards: Sequence[Sequence[str]], shard_env_counts: Sequence[int], num_envs_total: int) -> None:
    for i, (urdfs_i, b_i) in enumerate(zip(shards, shard_env_counts), start=1):
        if b_i <= 0:
            raise RuntimeError(f"Shard #{i} got 0 envs from total num_envs={num_envs_total}.")
        if len(urdfs_i) > b_i:
            raise RuntimeError(
                f"Shard #{i}: envs={b_i} < urdfs={len(urdfs_i)}. "
                "Increase --num-envs or reduce --urdf-shard-size."
            )


def _choose_worker_count(n_shards: int, num_workers: int) -> int:
    """
    Auto-mode (num_workers <= 0): honor urdf_shard_size — one worker per shard.
    Explicit-mode (num_workers > 0): clamp the requested count to [1, n_shards]
    and rebalance URDFs across that many workers.
    """
    if n_shards <= 0:
        return 0
    if num_workers > 0:
        return max(1, min(int(num_workers), n_shards))
    return n_shards


def _rebalance_shards_for_workers(all_urdfs: Sequence[str], num_workers: int) -> List[List[str]]:
    counts = split_even(len(all_urdfs), num_workers)
    shards: List[List[str]] = []
    start = 0
    for count in counts:
        end = start + count
        shards.append(list(all_urdfs[start:end]))
        start = end
    return [shard for shard in shards if shard]


def run_logical_super_scene_training(
    *,
    experiment_name: str,
    catalog_path: Path,
    train_cfg: Dict,
    env_cfg: Dict,
    obs_cfg: Dict,
    reward_cfg: Dict,
    command_cfg: Dict,
    log_dir: Path,
    num_envs_total: int,
    max_iterations: int,
    urdf_shard_size: int,
    num_workers: int,
    collection_gpus: int,
    device: str,
    init_policy_path: str | None,
    vis: bool,
    eval_dir: Optional[Path] = None,
) -> None:
    """
    Multi-URDF rollout with standard RSL-RL learning path.

    The only difference vs train.py is the environment backend for rollout:
    this uses a logical super-scene (multiple shards, multiple URDFs).
    PPO collection/update/logging/checkpointing remains OnPolicyRunner.learn().
    """
    all_urdfs = load_catalog_urdfs(catalog_path)
    if not all_urdfs:
        raise RuntimeError(f"Empty URDF catalog at: {catalog_path}")
    if urdf_shard_size <= 0:
        raise RuntimeError("logical-super-scene mode requires --urdf-shard-size > 0")

    initial_shards: List[List[str]] = chunk_list(all_urdfs, urdf_shard_size)
    n_shards = len(initial_shards)
    if n_shards <= 0:
        raise RuntimeError("No shards generated from URDF catalog")

    if collection_gpus <= 0:
        raise RuntimeError(f"Invalid collection_gpus={collection_gpus}. Must be > 0.")

    if torch.cuda.is_available():
        visible_cuda = torch.cuda.device_count()
        collection_gpus = min(int(collection_gpus), int(visible_cuda))
    else:
        collection_gpus = 0

    requested_workers = int(num_workers)
    num_workers = _choose_worker_count(n_shards, requested_workers)
    if num_workers == n_shards:
        shards = initial_shards
    else:
        shards = _rebalance_shards_for_workers(all_urdfs, num_workers)
        n_shards = len(shards)

    if collection_gpus == 0:
        worker_devices = ["cpu"] * n_shards
    else:
        worker_devices = [f"cuda:{i % collection_gpus}" for i in range(n_shards)]

    if collection_gpus > 0 and n_shards > collection_gpus:
        per_gpu = [0] * collection_gpus
        for i in range(n_shards):
            per_gpu[i % collection_gpus] += 1
        print(
            "[logical-super-scene] "
            f"{n_shards} workers round-robined across {collection_gpus} GPU(s): "
            f"workers_per_gpu={per_gpu} (each worker holds its own Genesis context — "
            "VRAM scales with workers/gpu)."
        )

    shard_env_counts = split_even(num_envs_total, n_shards)
    _validate_shards(shards, shard_env_counts, num_envs_total)

    print(
        "[logical-super-scene] "
        f"exp={experiment_name} urdfs={len(all_urdfs)} shards={n_shards} shard_size={urdf_shard_size} "
        f"requested_workers={requested_workers} effective_workers={num_workers} "
        f"num_envs_total={num_envs_total} per_shard_envs={shard_env_counts} "
        f"collection_gpus={collection_gpus} learner_device={device}"
    )

    orchestrator = LogicalSuperSceneOrchestrator(
        shards=shards,
        shard_env_counts=shard_env_counts,
        worker_devices=worker_devices,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=device,
        show_viewer=vis,
        use_shared_memory=bool(int(os.getenv("LOGICAL_SUPER_SCENE_SHM", "1"))),
        mps_active_thread_percentage=int(os.getenv("LOGICAL_SUPER_SCENE_MPS_THREAD_PERCENT", "0")),
    )
    env = LogicalSuperSceneVecEnv(orchestrator=orchestrator, device=device)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    _maybe_load_init_checkpoint(runner, init_policy_path, tag="logical-super-scene")
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir, max_iterations=max_iterations)
    rl_logger.attach()
    rl_logger.log_resource_usage(step=0, include_cuda=torch.cuda.is_available())

    # --- CSV logger (per-iteration metrics) ---
    csv_logger = None
    if eval_dir is not None:
        from WP1.csv_logger import CSVLogger
        csv_logger = CSVLogger(eval_dir / "training_log.csv")

        _iter_counter = {"i": 0}
        alg = getattr(runner, 'alg', None)
        if alg is not None:
            _orig_alg_update = alg.update

            def _patched_update(*args, **kwargs):
                """Wrapped PPO update that appends CSV logging."""
                result = _orig_alg_update(*args, **kwargs)
                it = _iter_counter["i"]
                try:
                    extras = env.extras if hasattr(env, 'extras') else {}
                    ppo_metrics = {}
                    rewbuffer = getattr(runner, 'rewbuffer', None)
                    if rewbuffer is not None and len(rewbuffer) > 0:
                        ppo_metrics["mean_reward"] = sum(rewbuffer) / len(rewbuffer)
                    lenbuffer = getattr(runner, 'lenbuffer', None)
                    if lenbuffer is not None and len(lenbuffer) > 0:
                        ppo_metrics["mean_episode_length"] = sum(lenbuffer) / len(lenbuffer)
                    csv_logger.log(it, extras, ppo_metrics=ppo_metrics)
                except Exception as e:
                    print(f"[lss runner] CSV log error at iter {it}: {e}")
                _iter_counter["i"] = it + 1
                return result

            alg.update = _patched_update

    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        try:
            if csv_logger is not None:
                csv_logger.close()
        except Exception:
            pass
        try:
            rl_logger.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
