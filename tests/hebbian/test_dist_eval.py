"""
Tests for WP2.dist_eval — multi-node sharded evaluation for the outer loop.

Pure-python (no Genesis): rank/world detection, URDF shard assignment, exact
cross-shard metric merging against a monolithic oracle transcribed from
``WP2.evaluate._rollout_episode_multi_urdf``'s reduction formulas, the gloo
command protocol over localhost with fake sessions, the ``_evaluate_population``
dispatch in ``HebbianCMAES``, and the multi-node Slurm launcher script.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2.dist_eval import (
    detect_dist_env,
    merge_shard_metrics,
    shard_urdf_indices,
)


# ============================================================================
#  Task 1 — merge identities the sufficient statistics rely on
# ============================================================================

def test_new_per_urdf_matrices_formulas():
    """per_urdf_t / per_urdf_v_dev / per_urdf_cot_mean must be F-means of the
    raw accumulators (v_dev raw, NOT normalized; cot with 1e-6 clamp + zeroing)
    so that global per-individual reductions equal D-means of per-URDF rows."""
    torch.manual_seed(0)
    D, P, F = 3, 4, 5
    t = torch.rand(D, P * F) * 60
    vdev = torch.rand(D, P * F) * 10
    energy = torch.rand(D, P * F) * 100
    dx = torch.rand(D, P * F) * 50
    mg = torch.rand(D, 1) * 9.81
    cot_slot = torch.where(
        dx > 1e-6, energy / (mg * dx.clamp(min=1e-6)), torch.zeros_like(dx)
    )
    per_urdf = lambda m: m.view(D, P, F).float().mean(dim=2)
    per_ind = lambda m: m.view(D, P, F).float().mean(dim=(0, 2))
    assert torch.allclose(per_urdf(t).mean(0), per_ind(t), atol=1e-5)
    assert torch.allclose(per_urdf(vdev).mean(0), per_ind(vdev), atol=1e-5)
    assert torch.allclose(per_urdf(cot_slot).mean(0), per_ind(cot_slot), atol=1e-4)


def test_rollout_returns_new_matrices_keys():
    """The metric-dict contract: evaluate.py must ship the three sufficient-
    statistic matrices next to the existing per_urdf_* keys. Checked on the
    accumulator dict of evaluate_population_multi_urdf (source-level, no env)."""
    import inspect
    from WP2 import evaluate as ev

    src = inspect.getsource(ev.evaluate_population_multi_urdf)
    for key in ("per_urdf_t", "per_urdf_v_dev", "per_urdf_cot_mean"):
        assert key in src, f"{key} missing from evaluate_population_multi_urdf acc"
    rollout_src = inspect.getsource(ev._rollout_episode_multi_urdf)
    for key in ("per_urdf_t", "per_urdf_v_dev", "per_urdf_cot_mean"):
        assert key in rollout_src, f"{key} missing from _rollout_episode_multi_urdf"


# ============================================================================
#  Task 2 — shard assignment + detection
# ============================================================================

def test_shard_urdf_indices_contiguous_cover():
    shards = shard_urdf_indices(64, 4)
    assert len(shards) == 4
    assert [len(s) for s in shards] == [16, 16, 16, 16]
    flat = [i for s in shards for i in s]
    assert flat == list(range(64))  # contiguous, ordered, complete


def test_shard_urdf_indices_uneven():
    shards = shard_urdf_indices(7, 3)
    assert [len(s) for s in shards] == [3, 2, 2]
    assert [i for s in shards for i in s] == list(range(7))


def test_shard_urdf_indices_world_too_large_raises():
    with pytest.raises(ValueError):
        shard_urdf_indices(3, 4)


def _clean_env(monkeypatch):
    for k in ("WP2_DIST_WORLD_SIZE", "WP2_DIST_RANK", "SLURM_NTASKS",
              "SLURM_PROCID"):
        monkeypatch.delenv(k, raising=False)


def test_detect_dist_env_default_none(monkeypatch):
    _clean_env(monkeypatch)
    assert detect_dist_env() is None


def test_detect_dist_env_slurm_single_task_none(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert detect_dist_env() is None


def test_detect_dist_env_slurm_multi(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "2")
    assert detect_dist_env() == (2, 4)


def test_detect_dist_env_explicit_wins(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("WP2_DIST_WORLD_SIZE", "2")
    monkeypatch.setenv("WP2_DIST_RANK", "1")
    assert detect_dist_env() == (1, 2)


# ============================================================================
#  Task 2 — exact merge vs monolithic oracle
# ============================================================================

def _oracle_metrics(raw, D, P, F, aggregator="mean"):
    """Monolithic reduction transcribed from _rollout_episode_multi_urdf
    (lines ~877-990 of WP2/evaluate.py) in float64 numpy."""
    t, dx, energy, vdev, reward, crash, comp, mg = (
        raw["t"], raw["dx"], raw["energy"], raw["vdev"], raw["reward"],
        raw["crash"], raw["comp"], raw["mg"],
    )
    per_ind = lambda m: m.reshape(D, P, F).mean(axis=(0, 2))
    per_urdf = lambda m: m.reshape(D, P, F).mean(axis=2)
    if aggregator == "median":
        samples = reward.reshape(D, P, F).transpose(1, 0, 2).reshape(P, D * F)
        reward_arr = np.median(samples, axis=1)
    else:
        reward_arr = per_ind(reward)
    t_arr = per_ind(t)
    dx_arr = per_ind(dx)
    v_dev_arr = per_ind(vdev)
    crash_arr = per_ind(crash)
    v_arr = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)
    v_dev_out = np.where(t_arr > 1e-6, v_dev_arr / t_arr, 0.0)
    cot_slot = np.where(
        dx > 1e-6, energy / (mg * np.clip(dx, 1e-6, None)), 0.0
    )
    cot_arr = per_ind(cot_slot)
    pu_t = per_urdf(t)
    pu_dx = per_urdf(dx)
    pu_reward = per_urdf(reward)
    pu_crash = per_urdf(crash)
    pu_velocity = np.where(pu_t > 1e-6, pu_dx / pu_t, 0.0)
    pu_energy = per_urdf(energy)
    pu_cot = pu_energy / (mg * np.clip(pu_dx, 1e-2, None))
    n_comp = comp.shape[-1]
    comp_per_ind = (
        comp.reshape(D, P, F, n_comp).mean(axis=(0, 2))
        if n_comp else np.zeros((P, 0))
    )
    per_slot = lambda m: m.reshape(D, P, F)
    ps_t = per_slot(t)
    ps_dx = per_slot(dx)
    return {
        "reward_sums": reward_arr,
        "reward_components": comp_per_ind,
        "reward_names": [f"c{i}" for i in range(n_comp)],
        "progresses": dx_arr,
        "velocities": v_arr,
        "crash_flags": crash_arr,
        "cots": cot_arr,
        "v_deviations": v_dev_out,
        "per_urdf_reward": pu_reward,
        "per_urdf_progress": pu_dx,
        "per_urdf_velocity": pu_velocity,
        "per_urdf_crash": pu_crash,
        "per_urdf_cot": pu_cot,
        "per_urdf_t": pu_t,
        "per_urdf_v_dev": per_urdf(vdev),
        "per_urdf_cot_mean": per_urdf(cot_slot),
        "per_slot_reward": per_slot(reward),
        "per_slot_progress": ps_dx,
        "per_slot_velocity": np.where(
            ps_t > 1e-6, ps_dx / np.maximum(ps_t, 1e-12), 0.0
        ),
        "per_slot_crash": per_slot(crash),
        "per_slot_cot": per_slot(energy / (mg * np.clip(dx, 1e-2, None))),
    }


def _make_raw(D, P, F, n_comp, seed):
    rng = np.random.default_rng(seed)
    E = P * F
    raw = {
        "t": rng.uniform(0.1, 60.0, (D, E)),
        "dx": rng.uniform(0.0, 100.0, (D, E)),
        "energy": rng.uniform(1.0, 500.0, (D, E)),
        "vdev": rng.uniform(0.0, 30.0, (D, E)),
        "reward": rng.normal(50.0, 20.0, (D, E)),
        "crash": rng.integers(0, 2, (D, E)).astype(np.float64),
        "comp": rng.normal(0.0, 1.0, (D, E, n_comp)),
        "mg": rng.uniform(3.0, 9.0, (D, 1)),
    }
    # A few crashed-at-spawn slots (dx == 0) to exercise the CoT branches.
    raw["dx"][:, :2] = 0.0
    return raw


def _shard_raw(raw, idx):
    out = {k: v[idx] for k, v in raw.items() if k != "comp"}
    out["comp"] = raw["comp"][idx]
    return out


@pytest.mark.parametrize("aggregator", ["mean", "median"])
@pytest.mark.parametrize("D,K,n_comp", [(6, 2, 3), (7, 3, 0), (64, 4, 2)])
def test_merge_matches_monolithic_oracle(aggregator, D, K, n_comp):
    P, F = 4, 3
    raw = _make_raw(D, P, F, n_comp, seed=D * 10 + K)
    expected = _oracle_metrics(raw, D, P, F, aggregator=aggregator)

    shards = shard_urdf_indices(D, K)
    shard_dicts = []
    for s in shards:
        m = _oracle_metrics(_shard_raw(raw, np.asarray(s)), len(s), P, F,
                            aggregator=aggregator)
        shard_dicts.append(m)

    fitnesses, merged = merge_shard_metrics(shard_dicts, aggregator)

    np.testing.assert_allclose(fitnesses, expected["reward_sums"], rtol=1e-9)
    for key, val in expected.items():
        if key == "reward_names":
            assert merged[key] == val
            continue
        np.testing.assert_allclose(
            merged[key], val, rtol=1e-9, atol=1e-12, err_msg=f"key={key}"
        )


def test_merge_preserves_shard_row_order():
    """per_urdf rows must land in global URDF order (rank-0 shard first)."""
    D, P, F, K = 4, 2, 2, 2
    raw = _make_raw(D, P, F, 0, seed=1)
    # Make per-URDF rewards identifiable per URDF index.
    for d in range(D):
        raw["reward"][d, :] = float(d)
    shards = shard_urdf_indices(D, K)
    shard_dicts = [
        _oracle_metrics(_shard_raw(raw, np.asarray(s)), len(s), P, F)
        for s in shards
    ]
    _, merged = merge_shard_metrics(shard_dicts, "mean")
    np.testing.assert_allclose(
        merged["per_urdf_reward"][:, 0], np.arange(D, dtype=float)
    )


# ============================================================================
#  Task 3 — gloo command protocol over localhost (world_size=2, FakeSession)
# ============================================================================

from WP2.dist_eval import (  # noqa: E402  (grouped with the protocol tests)
    _PER_SLOT_KEYS,
    _PER_URDF_KEYS,
    _RANK_SEED_STRIDE,
    DistContext,
    DistEnvHandle,
    DistributedEvalCoordinator,
    ShardEvalSession,
    shard_worker_main,
)


class FakeSession:
    """ShardEvalSession stand-in: records commands, encodes what it saw into
    the metric matrices so the coordinator-side merge can be asserted on."""

    F = 2

    def __init__(self, rank):
        self.rank = rank
        self.shard_paths = None
        self.envs_per_drone = None
        self.last_refresh_seed = float("nan")
        self.dens_min = None
        self.torn_down = False

    def build(self, urdf_paths, envs_per_drone):
        self.shard_paths = list(urdf_paths)
        self.envs_per_drone = int(envs_per_drone)

    def teardown(self):
        self.torn_down = True

    def refresh_forests(self, seed):
        self.last_refresh_seed = float(seed)

    def set_dens_min(self, value):
        self.dens_min = float(value)

    def apply_forest_overrides(self, overrides):
        return {"prev_rank": self.rank}

    def evaluate(self, solutions, cfg_override, seed, verbose=False):
        if cfg_override == "BOOM" and self.rank == 1:
            raise RuntimeError("synthetic shard failure")
        n = len(self.shard_paths)
        P = len(solutions)
        # urdf_<global_idx>.urdf → per_urdf_reward encodes global order.
        idx = np.array(
            [float(Path(p).stem.split("_")[-1]) for p in self.shard_paths]
        )
        m = {}
        for key in _PER_URDF_KEYS:
            m[key] = np.zeros((n, P))
        for key in _PER_SLOT_KEYS:
            m[key] = np.zeros((n, P, self.F))
        m["per_urdf_reward"] = np.tile(idx[:, None], (1, P))
        # Side-channels for protocol assertions:
        m["per_urdf_t"] = np.full((n, P), float(seed))          # base eval seed
        m["per_urdf_v_dev"] = np.full((n, P), self.last_refresh_seed)
        m["reward_components"] = np.zeros((P, 0), dtype=np.float32)
        m["reward_names"] = []
        return m


def _protocol_worker(rank, world, port):
    """Child process for the protocol test (module-level for spawn pickling)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WP2_DIST_TIMEOUT_S"] = "120"
    shard_worker_main(
        None, rank, world, session_factory=lambda cfg, r: FakeSession(r)
    )


def test_gloo_protocol_end_to_end(tmp_path):
    import multiprocessing as mp
    import socket

    from WP2_Outer_Loop.config import OuterNSGA2Config

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WP2_DIST_TIMEOUT_S"] = "120"

    ctx_mp = mp.get_context("spawn")
    child = ctx_mp.Process(target=_protocol_worker, args=(1, 2, port))
    child.start()

    ctx = None
    try:
        ctx = DistContext(rank=0, world_size=2, timeout_s=120)
        cfg = OuterNSGA2Config()
        session = FakeSession(0)
        coord = DistributedEvalCoordinator(ctx, session, cfg)

        # build: 5 URDFs over 2 ranks → shards of 3 (rank 0) and 2 (rank 1).
        paths = [str(tmp_path / f"urdf_{i}.urdf") for i in range(5)]
        handle = coord.build_env(paths, envs_per_drone=8)
        assert isinstance(handle, DistEnvHandle)
        assert handle.D == 5 and handle.E == 8
        assert session.shard_paths == paths[:3]

        # refresh: same seed must reach both ranks (asserted via evaluate).
        handle.refresh_forests()

        fit, metrics = coord.evaluate_population([np.zeros(4)] * 3)
        # Routing + rank order: global URDF indices 0..4 in order.
        np.testing.assert_allclose(
            metrics["per_urdf_reward"][:, 0], np.arange(5, dtype=float)
        )
        # Same base evaluate seed on both ranks (per-rank offset lives inside
        # the real ShardEvalSession, not in the protocol).
        assert len(set(metrics["per_urdf_t"][:, 0].tolist())) == 1
        # Same refresh seed on both ranks.
        refresh_seeds = set(metrics["per_urdf_v_dev"][:, 0].tolist())
        assert len(refresh_seeds) == 1
        assert not np.isnan(next(iter(refresh_seeds)))

        # Overrides: rank-0 local prev comes back.
        prev = handle.apply_forest_overrides({"dens_min": 3.0})
        assert prev == {"prev_rank": 0}

        handle.set_dens_min(2.5)
        assert session.dens_min == 2.5

        # A shard failure surfaces as RuntimeError naming the rank; the
        # protocol stays in lockstep afterwards.
        with pytest.raises(RuntimeError, match="rank 1"):
            coord.evaluate_population([np.zeros(4)], cfg_override="BOOM")

        fit2, _ = coord.evaluate_population([np.zeros(4)] * 2)
        assert fit2.shape == (2,)

        handle.shutdown()          # teardown_env
        assert session.torn_down
        coord.shutdown()

        child.join(timeout=60)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        if ctx is not None:
            ctx.close()


def test_rank_seed_offset():
    assert ShardEvalSession.rank_seed(7, 0) == 7
    assert ShardEvalSession.rank_seed(7, 3) == 7 + 3 * _RANK_SEED_STRIDE


# ============================================================================
#  Task 4 — HebbianCMAES dispatch wiring
# ============================================================================

def _bare_cmaes():
    """HebbianCMAES shell with hand-set attributes (no checkpoint load)."""
    from WP2.evolve_cma import HebbianCMAES

    obj = object.__new__(HebbianCMAES)
    obj._dist = None
    obj.cfg = "CFG"
    obj._model_and_layer = "MAL"
    obj._wp1_cfg = "WP1"
    obj._urdf_paths = ["a.urdf", "b.urdf"]
    obj._env = "ENV"
    obj._env_urdf_path = ["a.urdf", "b.urdf"]
    return obj


def test_evaluate_population_single_process_exact_legacy_args(monkeypatch):
    """With no coordinator attached, _evaluate_population must forward the
    EXACT legacy argument tuple to evaluate_population_multi_urdf — patched
    at WP2.evaluate, the historical patch point the exam tests rely on."""
    from WP2 import evaluate as evaluate_mod

    obj = _bare_cmaes()
    seen = {}

    def fake_eval(solutions, cfg, mal, wp1, urdf_paths, existing_env,
                  verbose):
        seen.update(
            solutions=solutions, cfg=cfg, mal=mal, wp1=wp1,
            urdf_paths=urdf_paths, existing_env=existing_env, verbose=verbose,
        )
        return "FIT", "MET"

    monkeypatch.setattr(
        evaluate_mod, "evaluate_population_multi_urdf", fake_eval
    )
    out = obj._evaluate_population(["sol"], verbose=True)
    assert out == ("FIT", "MET")
    assert seen["cfg"] is obj.cfg
    assert seen["mal"] is obj._model_and_layer
    assert seen["wp1"] is obj._wp1_cfg
    assert seen["urdf_paths"] is obj._urdf_paths
    assert seen["existing_env"] == (obj._env, obj._env_urdf_path)
    assert seen["verbose"] is True


def test_evaluate_population_cfg_override_replaces_cfg(monkeypatch):
    from WP2 import evaluate as evaluate_mod

    obj = _bare_cmaes()
    seen = {}

    def fake_eval(solutions, cfg, *a, **kw):
        seen["cfg"] = cfg
        return "F", "M"

    monkeypatch.setattr(
        evaluate_mod, "evaluate_population_multi_urdf", fake_eval
    )
    obj._evaluate_population(["sol"], cfg_override="REF_CFG")
    assert seen["cfg"] == "REF_CFG"


def test_evaluate_population_none_env_forwards_none(monkeypatch):
    from WP2 import evaluate as evaluate_mod

    obj = _bare_cmaes()
    obj._env = None
    seen = {}

    def fake_eval(solutions, cfg, mal, wp1, urdf_paths, existing_env,
                  verbose):
        seen["existing_env"] = existing_env
        return "F", "M"

    monkeypatch.setattr(
        evaluate_mod, "evaluate_population_multi_urdf", fake_eval
    )
    obj._evaluate_population(["sol"])
    assert seen["existing_env"] is None


def test_evaluate_population_distributed_routes_to_coordinator(monkeypatch):
    obj = _bare_cmaes()
    calls = {}

    class StubCoord:
        def evaluate_population(self, solutions, cfg_override=None,
                                verbose=False):
            calls.update(solutions=solutions, cfg_override=cfg_override,
                         verbose=verbose)
            return "DFIT", "DMET"

    obj._dist = StubCoord()

    def explode(*a, **kw):
        raise AssertionError("legacy evaluator must not run in dist mode")

    from WP2 import evaluate as evaluate_mod
    monkeypatch.setattr(
        evaluate_mod, "evaluate_population_multi_urdf", explode
    )
    out = obj._evaluate_population(["sol"], cfg_override="REF", verbose=True)
    assert out == ("DFIT", "DMET")
    assert calls == {"solutions": ["sol"], "cfg_override": "REF",
                     "verbose": True}


def test_attach_distributed_sets_coordinator():
    obj = _bare_cmaes()
    obj.attach_distributed("COORD")
    assert obj._dist == "COORD"


def test_call_sites_route_through_evaluate_population():
    """Drift guard: the run loop, the reference-actor path, UH re-eval and
    the exam must all use _evaluate_population (never a direct multi-URDF
    call that would bypass distribution)."""
    import inspect

    from WP2.evolve_cma import HebbianCMAES
    from WP2_Outer_Loop.nsga_cma import NSGA2MorphCMAES

    assert "self._evaluate_population(" in inspect.getsource(HebbianCMAES.run)
    ref_src = inspect.getsource(HebbianCMAES._evaluate_reference_actor)
    assert "self._evaluate_population(" in ref_src
    assert "self._evaluate_population(" in inspect.getsource(
        HebbianCMAES._uh_reevaluate
    )
    exam_src = inspect.getsource(NSGA2MorphCMAES._run_exam)
    assert "self._evaluate_population(" in exam_src
    assert "evaluate_population_multi_urdf(" not in exam_src


def test_run_entry_dispatches_worker_ranks():
    """Drift guard: WP2_Outer_Loop.run must detect the rank and route worker
    ranks into shard_worker_main before any run-dir creation."""
    run_src = (
        Path(__file__).parent.parent.parent
        / "src" / "WP2_Outer_Loop" / "run.py"
    ).read_text()
    assert "detect_dist_env" in run_src
    assert "shard_worker_main" in run_src
    assert "attach_distributed" in run_src


# ============================================================================
#  Task 5 — multi-node Slurm launcher
# ============================================================================

_SLURM_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "src" / "WP2_Outer_Loop" / "slurm_jobs" / "train.slurm"
)


def test_train_slurm_parses():
    res = subprocess.run(
        ["bash", "-n", str(_SLURM_SCRIPT)], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_train_slurm_multi_node_header():
    """Per-node resource directives so `--nodes=K` alone scales the job:
    --ntasks=1 would cap the job at ONE task regardless of --nodes, and a
    job-total --gpus=1 would starve multi-node runs."""
    src = _SLURM_SCRIPT.read_text()
    assert "--ntasks-per-node=1" in src
    assert "--ntasks=1" not in src
    assert "--gpus-per-node=1" in src
    # No job-total GPU spec left in the header ("--gpus=1" etc.).
    for line in src.splitlines():
        if line.startswith("#SBATCH"):
            assert "--gpus=" not in line, line


def test_train_slurm_exports_rendezvous():
    src = _SLURM_SCRIPT.read_text()
    assert "MASTER_ADDR" in src
    assert "MASTER_PORT" in src
    assert "--kill-on-bad-exit=1" in src


def test_train_slurm_stage_is_rank_guarded():
    """Only task 0 stages the shared URDF meshes; other ranks wait on the
    sentinel — otherwise K ranks race on the same cp into shared scratch."""
    src = _SLURM_SCRIPT.read_text()
    assert "SLURM_PROCID" in src
    assert ".stage_done" in src


def test_dist_env_handle_delegates():
    class RecordingCoord:
        def __init__(self):
            self.calls = []

        def refresh_forests(self):
            self.calls.append("refresh")

        def set_dens_min(self, v):
            self.calls.append(("dens", v))

        def apply_forest_overrides(self, ov):
            self.calls.append(("ov", dict(ov)))
            return {"prev": 1}

        def teardown_env(self):
            self.calls.append("teardown")

    rc = RecordingCoord()
    h = DistEnvHandle(rc, d_total=64, envs_per_drone=3840)
    assert h.D == 64 and h.E == 3840
    assert hasattr(h, "set_dens_min")  # run-loop hasattr() gate
    h.refresh_forests()
    h.set_dens_min(1.5)
    assert h.apply_forest_overrides({"x_upper": 300.0}) == {"prev": 1}
    h.shutdown()
    assert rc.calls == [
        "refresh", ("dens", 1.5), ("ov", {"x_upper": 300.0}), "teardown",
    ]
