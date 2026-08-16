"""
WP2.dist_eval — multi-node sharded population evaluation for the outer loop.
============================================================================

Shards the multi-URDF population evaluation across K Slurm tasks (one per
node) by URDF, at *rollout* granularity. Every rank — rank 0 included —
owns a contiguous slice of the URDF catalog and runs the full, unchanged
``evaluate_population_multi_urdf`` on it locally (its own Genesis runtime,
its own frozen actor + all P Hebbian controllers, its own intra-node
``ParallelMultiSceneEvalEnv`` workers). Cross-node synchronization happens
only per generation:

* rank 0 broadcasts a command (build / refresh / overrides / evaluate /
  teardown / shutdown) with the CMA genomes and a seed — ~230 KB;
* every rank rolls out independently (the 30-minute part, zero cross-node
  traffic);
* rank 0 gathers each shard's metric dict (~a few MB) and merges it into
  the exact monolithic format via ``merge_shard_metrics``.

Rank 0 keeps everything global and cheap: CMA-ES, NSGA-II, the validation
env, the exam baseline, CSVs and plots. Worker ranks never create run dirs.

Activation
----------
``detect_dist_env()`` returns ``(rank, world_size)`` from
``WP2_DIST_RANK``/``WP2_DIST_WORLD_SIZE`` (explicit override, used by the
local tests) or from Slurm's ``SLURM_PROCID``/``SLURM_NTASKS`` when the job
was submitted with more than one task (``--nodes=K`` with the launcher's
``--ntasks-per-node=1`` header). Single-task jobs and local runs return
``None`` and the entire module stays inert — the legacy single-process code
path is untouched.

Transport is ``torch.distributed`` with the gloo backend over TCP
(``MASTER_ADDR``/``MASTER_PORT``, exported by ``train.slurm``). Collective
timeout defaults to 4 h (cold Taichi cache builds can take >1 h) — override
with ``WP2_DIST_TIMEOUT_S``.

Seeding semantics
-----------------
* ``refresh_forests`` seeds every rank IDENTICALLY (same as the intra-node
  workers today) so all URDFs across all nodes fly the same forest sets —
  the cross-URDF fairness the NSGA-II ranking relies on.
* ``evaluate`` seeds each rank with ``seed + rank * 9973`` so stochastic
  actor sampling and DR draws are decorrelated across shards.
"""

from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Per-URDF matrices concatenated along the URDF axis during the merge.
_PER_URDF_KEYS: Tuple[str, ...] = (
    "per_urdf_reward", "per_urdf_progress", "per_urdf_velocity",
    "per_urdf_crash", "per_urdf_cot",
    "per_urdf_t", "per_urdf_v_dev", "per_urdf_cot_mean",
)
_PER_SLOT_KEYS: Tuple[str, ...] = (
    "per_slot_reward", "per_slot_progress", "per_slot_velocity",
    "per_slot_crash", "per_slot_cot",
)

# Rank-offset prime for evaluate-time RNG decorrelation across shards.
_RANK_SEED_STRIDE = 9973


# ============================================================================
#  Detection + assignment
# ============================================================================

def detect_dist_env() -> Optional[Tuple[int, int]]:
    """Return ``(rank, world_size)`` when running distributed, else ``None``.

    Precedence: explicit ``WP2_DIST_RANK``/``WP2_DIST_WORLD_SIZE`` env vars
    (local testing), then Slurm's ``SLURM_PROCID``/``SLURM_NTASKS``. A world
    size of 1 (the production single-node submission, every local run) maps
    to ``None`` so the caller takes the unchanged single-process path.
    """
    ws = os.environ.get("WP2_DIST_WORLD_SIZE")
    if ws is not None:
        world = int(ws)
        if world <= 1:
            return None
        return int(os.environ.get("WP2_DIST_RANK", "0")), world

    ntasks = os.environ.get("SLURM_NTASKS")
    if ntasks is not None and int(ntasks) > 1:
        return int(os.environ.get("SLURM_PROCID", "0")), int(ntasks)
    return None


def shard_urdf_indices(n_urdfs: int, world_size: int) -> List[List[int]]:
    """Contiguous URDF index shards, one per rank, in rank order.

    Uneven splits give the earlier ranks one extra URDF (``np.array_split``
    semantics). Placement carries no algorithmic meaning — objectives are
    gathered globally before any NSGA-II selection — so contiguous chunks
    keep the merge a plain concatenation.
    """
    if world_size > n_urdfs:
        raise ValueError(
            f"world_size ({world_size}) > num_urdfs ({n_urdfs}): every rank "
            f"needs at least one URDF. Reduce --nodes or raise "
            f"catalog.num_urdfs."
        )
    return [
        [int(i) for i in chunk]
        for chunk in np.array_split(np.arange(n_urdfs), world_size)
    ]


# ============================================================================
#  Exact cross-shard metric merge
# ============================================================================

def merge_shard_metrics(
    shard_metrics: Sequence[Dict[str, np.ndarray]],
    fitness_aggregator: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Merge per-shard metric dicts into the exact monolithic format.

    Each shard dict is a full ``evaluate_population_multi_urdf`` result for
    that shard's URDF slice. Per-URDF and per-slot matrices concatenate
    along the URDF axis in rank order; every per-individual reduction is
    *recomputed* from the concatenated sufficient statistics using the same
    formulas as ``_rollout_episode_multi_urdf`` (mean over D of the F-mean
    matrices; ratios normalized by the GLOBAL mean time), so the result
    matches a single-process run over the full catalog up to float
    associativity.

    Known accepted deviation: with ``catalog.num_episodes > 1`` and the
    ``median`` aggregator, this computes the median of episode-averaged
    per-slot rewards (median-of-means) where the monolithic path averages
    per-episode medians. Production runs use ``num_episodes = 1``, where the
    two coincide.
    """
    if not shard_metrics:
        raise ValueError("merge_shard_metrics: no shard metrics")

    merged: Dict[str, np.ndarray] = {}
    for key in _PER_URDF_KEYS:
        merged[key] = np.concatenate(
            [np.asarray(m[key], dtype=np.float64) for m in shard_metrics],
            axis=0,
        )
    for key in _PER_SLOT_KEYS:
        merged[key] = np.concatenate(
            [np.asarray(m[key], dtype=np.float64) for m in shard_metrics],
            axis=0,
        )

    d_sizes = [np.asarray(m["per_urdf_reward"]).shape[0] for m in shard_metrics]
    d_total = int(sum(d_sizes))

    # Global per-individual reductions from the concatenated per-URDF stats.
    t_arr = merged["per_urdf_t"].mean(axis=0)          # (P,)
    dx_arr = merged["per_urdf_progress"].mean(axis=0)  # (P,)
    v_dev_raw = merged["per_urdf_v_dev"].mean(axis=0)  # (P,) UNnormalized
    merged["progresses"] = dx_arr
    merged["velocities"] = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)
    merged["v_deviations"] = np.where(t_arr > 1e-6, v_dev_raw / t_arr, 0.0)
    merged["crash_flags"] = merged["per_urdf_crash"].mean(axis=0)
    merged["cots"] = merged["per_urdf_cot_mean"].mean(axis=0)

    if fitness_aggregator == "median":
        d, p, f = merged["per_slot_reward"].shape
        samples = merged["per_slot_reward"].transpose(1, 0, 2).reshape(p, d * f)
        merged["reward_sums"] = np.median(samples, axis=1)
    else:
        merged["reward_sums"] = merged["per_urdf_reward"].mean(axis=0)

    # Reward components: (P, C) shard means, weighted by shard URDF count.
    comp_shards = [
        np.asarray(m.get("reward_components"), dtype=np.float64)
        if m.get("reward_components") is not None else None
        for m in shard_metrics
    ]
    first_comp = next(
        (c for c in comp_shards if c is not None and c.size), None
    )
    if first_comp is not None:
        acc = np.zeros_like(first_comp)
        for ds, comp in zip(d_sizes, comp_shards):
            if comp is not None and comp.size:
                acc += ds * comp
        merged["reward_components"] = acc / d_total
    else:
        p = merged["per_urdf_reward"].shape[1]
        merged["reward_components"] = np.zeros((p, 0), dtype=np.float32)

    merged["reward_names"] = next(
        (list(m.get("reward_names", [])) for m in shard_metrics
         if m.get("reward_names")),
        [],
    )

    return merged["reward_sums"], merged


# ============================================================================
#  Transport
# ============================================================================

class DistContext:
    """Thin wrapper around a ``torch.distributed`` gloo process group.

    Rank 0 drives; all ranks participate in every collective. The timeout
    must cover the slowest single operation between two collectives — env
    rebuilds on a cold Taichi cache can exceed an hour, hence the 4 h
    default (``WP2_DIST_TIMEOUT_S`` to override). A rank that dies leaves
    the others blocked until this timeout, which then kills the job instead
    of burning the full Slurm allocation.
    """

    def __init__(self, rank: int, world_size: int,
                 timeout_s: Optional[float] = None) -> None:
        import torch.distributed as dist

        self._dist = dist
        self.rank = int(rank)
        self.world_size = int(world_size)
        if timeout_s is None:
            timeout_s = float(os.environ.get("WP2_DIST_TIMEOUT_S", 14400))
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(seconds=float(timeout_s)),
        )
        print(f"[dist_eval] rank {self.rank}/{self.world_size} joined gloo "
              f"group (timeout {timeout_s:.0f}s)", flush=True)

    def broadcast_cmd(self, obj=None):
        """Rank 0 sends ``obj``; every rank returns the broadcast object."""
        buf = [obj]
        self._dist.broadcast_object_list(buf, src=0)
        return buf[0]

    def gather(self, obj) -> Optional[list]:
        """Gather one object per rank to rank 0 (rank order); None elsewhere."""
        out = [None] * self.world_size if self.rank == 0 else None
        self._dist.gather_object(obj, out, dst=0)
        return out

    def close(self) -> None:
        try:
            if self._dist.is_initialized():
                self._dist.destroy_process_group()
        except Exception:
            pass


# ============================================================================
#  Per-rank shard session
# ============================================================================

def _seed_all(seed: int) -> None:
    import random as _random

    import torch

    _random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def _wait_for_paths(paths: Sequence[str], timeout_s: float = 120.0) -> None:
    """Block until all files exist on this node (shared-FS visibility lag
    after rank 0 materializes a new URDF population)."""
    deadline = time.monotonic() + timeout_s
    missing = [p for p in paths if not Path(p).is_file()]
    while missing and time.monotonic() < deadline:
        time.sleep(1.0)
        missing = [p for p in missing if not Path(p).is_file()]
    if missing:
        raise RuntimeError(
            f"URDFs not visible on this node after {timeout_s:.0f}s: "
            f"{missing[:3]}{'…' if len(missing) > 3 else ''}"
        )


class ShardEvalSession:
    """One rank's shard: its env, its frozen actor, its rollouts.

    Used identically by rank 0 (driven in-process by the coordinator) and by
    worker ranks (driven by ``shard_worker_main``'s command loop) — one
    implementation, two transports.
    """

    def __init__(self, cfg, wp1_cfg, model_and_layer, rank: int) -> None:
        self.cfg = cfg
        self.wp1_cfg = wp1_cfg
        self.model_and_layer = model_and_layer
        self.rank = int(rank)
        self.env = None
        self.shard_paths: Optional[List[str]] = None

    @staticmethod
    def rank_seed(seed: int, rank: int) -> int:
        """Evaluate-time RNG seed for a rank: decorrelates stochastic actor
        sampling and DR draws across shards while keeping runs reproducible."""
        return int(seed) + int(rank) * _RANK_SEED_STRIDE

    def build(self, urdf_paths: Sequence[str], envs_per_drone: int) -> None:
        import genesis as gs

        from WP2.evaluate import _build_multi_urdf_env

        _wait_for_paths(urdf_paths)
        n_workers = int(getattr(self.cfg.evaluation, "num_eval_workers", 1))
        if not gs._initialized and n_workers <= 1:
            gs.init(logging_level="error", backend=gs.gpu)
        print(f"[dist_eval] rank {self.rank}: building shard env "
              f"({len(urdf_paths)} URDFs × {envs_per_drone} envs)", flush=True)
        self.env = _build_multi_urdf_env(
            urdf_paths=list(urdf_paths),
            cfg=self.cfg,
            wp1_cfg=self.wp1_cfg,
            device=self.cfg.device,
            num_envs_per_drone=int(envs_per_drone),
            num_workers=n_workers,
            num_gpus=int(getattr(self.cfg.evaluation, "num_gpus", 0)) or None,
        )
        self.shard_paths = list(urdf_paths)
        print(f"[dist_eval] rank {self.rank}: shard env ready", flush=True)

    def teardown(self) -> None:
        if self.env is None:
            return
        shutdown = getattr(self.env, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[dist_eval] rank {self.rank}: env shutdown failed: "
                      f"{exc}", flush=True)
        try:
            import genesis as gs
            if gs._initialized:
                gs.destroy()
        except Exception as exc:
            print(f"[dist_eval] rank {self.rank}: gs.destroy failed: {exc}",
                  flush=True)
        self.env = None
        self.shard_paths = None

    def refresh_forests(self, seed: int) -> None:
        # IDENTICAL seed on every rank — all URDFs across all nodes fly the
        # same forest sets (the cross-URDF fairness NSGA-II relies on), same
        # as the intra-node workers before this feature.
        _seed_all(int(seed))
        self.env.refresh_forests()

    def set_dens_min(self, value: float) -> None:
        self.env.set_dens_min(float(value))

    def apply_forest_overrides(self, overrides: Dict) -> Dict:
        return self.env.apply_forest_overrides(dict(overrides))

    def evaluate(self, solutions, cfg_override, seed: int,
                 verbose: bool = False) -> Dict[str, np.ndarray]:
        from WP2.evaluate import evaluate_population_multi_urdf

        _seed_all(self.rank_seed(seed, self.rank))
        cfg = cfg_override if cfg_override is not None else self.cfg
        _fit, metrics = evaluate_population_multi_urdf(
            [np.asarray(s, dtype=np.float64) for s in solutions],
            cfg,
            self.model_and_layer,
            self.wp1_cfg,
            urdf_paths=self.shard_paths,
            existing_env=(self.env, self.shard_paths),
            verbose=verbose,
        )
        return metrics


# ============================================================================
#  Rank-0 coordinator + env handle
# ============================================================================

class DistributedEvalCoordinator:
    """Rank-0 driver: broadcast a command, run it on the local shard, gather
    every rank's result, surface any shard error as one RuntimeError."""

    def __init__(self, ctx: DistContext, session: ShardEvalSession, cfg) -> None:
        import random as _random

        self.ctx = ctx
        self.session = session
        self.cfg = cfg
        # Seeds come from the process-global RNG (seeded by seed_everything
        # in run.py) so a rerun with the same cfg.seed replays the same
        # forest/eval seed sequence — parity with the single-process path.
        self._rng = _random

    def _run(self, cmd: Dict, local_fn) -> list:
        self.ctx.broadcast_cmd(cmd)
        try:
            local = ("ok", local_fn())
        except Exception as exc:
            local = ("error",
                     f"rank 0: {cmd['op']}: {type(exc).__name__}: {exc}")
        results = self.ctx.gather(local)
        errors = [r[1] for r in results if r[0] == "error"]
        if errors:
            raise RuntimeError(
                f"distributed op {cmd['op']!r} failed on "
                f"{len(errors)}/{self.ctx.world_size} rank(s): "
                + " | ".join(errors)
            )
        return [r[1] for r in results]

    def build_env(self, urdf_paths: Sequence[str],
                  envs_per_drone: int) -> "DistEnvHandle":
        shards_idx = shard_urdf_indices(len(urdf_paths), self.ctx.world_size)
        shards = [[urdf_paths[i] for i in s] for s in shards_idx]
        print(f"[dist_eval] sharding {len(urdf_paths)} URDFs across "
              f"{self.ctx.world_size} rank(s): "
              f"{[len(s) for s in shards]}", flush=True)
        self._run(
            {"op": "build_env", "shards": shards,
             "envs_per_drone": int(envs_per_drone)},
            lambda: self.session.build(shards[0], int(envs_per_drone)),
        )
        return DistEnvHandle(self, len(urdf_paths), int(envs_per_drone))

    def evaluate_population(self, solutions, cfg_override=None,
                            verbose: bool = False):
        from WP2.evaluate import _resolve_fitness_aggregator

        seed = self._rng.randint(0, 2 ** 31 - 1)
        sols = [np.asarray(s, dtype=np.float64) for s in solutions]
        results = self._run(
            {"op": "evaluate", "solutions": sols,
             "cfg_override": cfg_override, "seed": seed},
            lambda: self.session.evaluate(sols, cfg_override, seed, verbose),
        )
        agg = _resolve_fitness_aggregator(
            cfg_override if cfg_override is not None else self.cfg
        )
        return merge_shard_metrics(results, agg)

    def refresh_forests(self) -> None:
        seed = self._rng.randint(0, 2 ** 31 - 1)
        self._run({"op": "refresh_forests", "seed": seed},
                  lambda: self.session.refresh_forests(seed))

    def set_dens_min(self, value: float) -> None:
        self._run({"op": "set_dens_min", "value": float(value)},
                  lambda: self.session.set_dens_min(value))

    def apply_forest_overrides(self, overrides: Dict) -> Dict:
        results = self._run(
            {"op": "apply_forest_overrides", "overrides": dict(overrides)},
            lambda: self.session.apply_forest_overrides(overrides),
        )
        return results[0]

    def teardown_env(self) -> None:
        self._run({"op": "teardown_env"}, self.session.teardown)

    def shutdown(self) -> None:
        """End the worker processes and leave the process group. Safe to call
        once at the very end of the run (also on failure paths)."""
        try:
            self.ctx.broadcast_cmd({"op": "shutdown"})
        except Exception:
            pass
        try:
            self.session.teardown()
        except Exception:
            pass
        self.ctx.close()


class DistEnvHandle:
    """Duck-typed stand-in for the eval env on rank 0.

    Exposes exactly the surface the run loop and ``nsga_cma`` touch on
    ``self._env`` — ``E``/``D``, ``refresh_forests``, ``set_dens_min``,
    ``apply_forest_overrides``, ``shutdown`` — and broadcasts each call to
    every shard. Rollouts never go through this object; they go through
    ``DistributedEvalCoordinator.evaluate_population``.
    """

    def __init__(self, coordinator: DistributedEvalCoordinator,
                 d_total: int, envs_per_drone: int) -> None:
        self._coord = coordinator
        self.D = int(d_total)
        self.E = int(envs_per_drone)

    def refresh_forests(self) -> None:
        self._coord.refresh_forests()

    def set_dens_min(self, value: float) -> None:
        self._coord.set_dens_min(value)

    def apply_forest_overrides(self, overrides: Dict) -> Dict:
        return self._coord.apply_forest_overrides(overrides)

    def shutdown(self) -> None:
        self._coord.teardown_env()


# ============================================================================
#  Worker entry point (ranks 1..K-1)
# ============================================================================

def _default_session_factory(cfg, rank: int) -> ShardEvalSession:
    """Load the frozen actor + WP1 config for a worker rank (mirrors the head
    of ``HebbianCMAES.__init__`` — but with NO run directory, NO CSVs)."""
    from WP1.config import RunConfig
    from WP2.frozen_actor import load_frozen_actor

    model, last_layer, _, _ = load_frozen_actor(
        cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device
    )
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    model_and_layer = (
        model, last_layer, cfg.hebbian.num_actions, cfg.hebbian.hidden_dim,
    )
    return ShardEvalSession(cfg, wp1_cfg, model_and_layer, rank)


def shard_worker_main(cfg, rank: int, world_size: int,
                      session_factory=None) -> None:
    """Command loop for a worker rank. Blocks on rank-0 broadcasts, executes
    each op on the local shard, gathers ``("ok", payload)`` or
    ``("error", msg)`` back. Exceptions never leave the loop — a failed op is
    reported and the loop keeps serving (the coordinator decides what a
    failure means). ``shutdown`` ends the loop.
    """
    ctx = DistContext(rank, world_size)
    session = None
    try:
        factory = session_factory or _default_session_factory
        session = factory(cfg, rank)
        print(f"[dist_eval] rank {rank}: worker ready", flush=True)
        while True:
            cmd = ctx.broadcast_cmd()
            op = cmd.get("op")
            if op == "shutdown":
                print(f"[dist_eval] rank {rank}: shutdown received", flush=True)
                break
            try:
                if op == "build_env":
                    session.build(cmd["shards"][rank], cmd["envs_per_drone"])
                    result = ("ok", None)
                elif op == "teardown_env":
                    session.teardown()
                    result = ("ok", None)
                elif op == "refresh_forests":
                    session.refresh_forests(cmd["seed"])
                    result = ("ok", None)
                elif op == "set_dens_min":
                    session.set_dens_min(cmd["value"])
                    result = ("ok", None)
                elif op == "apply_forest_overrides":
                    prev = session.apply_forest_overrides(cmd["overrides"])
                    result = ("ok", prev)
                elif op == "evaluate":
                    metrics = session.evaluate(
                        cmd["solutions"], cmd["cfg_override"], cmd["seed"],
                        verbose=False,
                    )
                    result = ("ok", metrics)
                else:
                    result = ("error", f"rank {rank}: unknown op {op!r}")
            except Exception as exc:
                result = ("error",
                          f"rank {rank}: {op}: {type(exc).__name__}: {exc}")
            ctx.gather(result)
    finally:
        if session is not None:
            try:
                session.teardown()
            except Exception:
                pass
        ctx.close()
