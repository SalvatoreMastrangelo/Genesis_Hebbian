"""
End-to-end test: WP2 outer loop sharded across 2 local ranks vs 1 process.
==========================================================================

Runs the REAL ``WP2_Outer_Loop.run`` entry point — real Genesis scenes, the
committed batch_0 WP1 checkpoint, CMA-ES, NSGA-II refresh, phase-end exam
with forest overrides, validation, baseline — twice:

* distributed: two subprocesses (WP2_DIST_RANK 0/1) rendezvousing over gloo
  on localhost, sharding the 4-URDF population 2+2;
* control: the same config single-process (the legacy path).

Needs the Genesis GPU container (mygenesis:latest); skipped elsewhere.
Run explicitly:

    docker run --rm --gpus all -v "$PWD":/workspace/bind \
      -e PYTHONPATH=/workspace/bind/src -w /workspace/bind mygenesis:latest \
      python -m pytest tests/hebbian/test_dist_e2e.py -v
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "src"
EXP = SRC / "WP2_Outer_Loop" / "experiments" / "batch_0" / "full_run_4_64_64"
CKPT = EXP / "checkpoint.pt"
WP1_CFG = EXP / "wp1_config.yaml"

try:
    import genesis  # noqa: F401
    _GENESIS = True
except Exception:
    _GENESIS = False

pytestmark = pytest.mark.skipif(
    not (_GENESIS and CKPT.is_file()),
    reason="needs the Genesis GPU container + committed batch_0 checkpoint",
)

# Tiny but complete outer config: N=4 URDFs, H=4 CMA individuals, F=2
# forests/(URDF, individual) → 32 slots. num_generations=6 → pycma maxiter
# stops after gens 0..5, refreshes fire at gens 2 and 4 → 2 exam-scored
# phases (the second on the REBUILT env) + the phase-mean final flush.
# Short 60 m course so episodes terminate in a few seconds of sim time;
# exam overrides ON to exercise the distributed forest-override RPC.
_E2E_YAML = """
exp_name: dist_e2e
checkpoint_path: ""
checkpoint_config_path: ""

outer:
  n_elites: 2
  score_top_frac: 0.5
  score_window: 0
  rescore: true
  rescore_top_frac: 0.25
  exam_forest:
    override_forest: true
    x_upper: 80.0
    dens_min: 0.5
    dens_max: 3.5
  exam_baseline: true
  min_progress_m: 0.0
  objectives:
    - name: progress_m
      direction: maximize
    - name: cost_of_transport
      direction: minimize

hebbian:
  enabled: true
  eta: 0.005
  decay: 0.05
  initialize_rules_to_zero: true
  w_max: 5000.0

evolution:
  num_generations: 6
  population_size: 0

cmaes:
  algorithm: "cmaes"
  sigma0: 0.075
  population_size: 4

catalog:
  path: ""
  num_urdfs: 4
  num_episodes: 1
  refresh_urdfs_every: 2
  mutate: false

validation:
  enable: true
  n_val_envs: 8
  validation_catalog: ""
  period: 2

evaluation:
  num_eval_envs: 32
  num_eval_workers: 1
  num_gpus: 1
  vmin: 10.0
  vmax: 20.0
  stochastic: true
  run_baseline: true
  baseline_every: 2
  refresh_forests_per_generation: true
  crn: false

forest:
  mode: "growing"
  x_lower: 0
  x_upper: 60
  dens_min: 0.0
  dens_max: 3.0

seed: 3
device: "cuda:0"
"""


def _write_cfg(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "e2e.yaml"
    cfg_path.write_text(_E2E_YAML)
    return cfg_path


def _run_cmd(cfg_path: Path, base_dir: Path) -> list:
    return [
        sys.executable, "-m", "WP2_Outer_Loop.run",
        "--cfg", str(cfg_path),
        "--cfg.checkpoint_path", str(CKPT),
        "--cfg.checkpoint_config_path", str(WP1_CFG),
        "--cfg.base_dir", str(base_dir),
    ]


def _base_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("WP2_DIST_WORLD_SIZE", "WP2_DIST_RANK",
              "SLURM_NTASKS", "SLURM_PROCID"):
        env.pop(k, None)
    return env


def _tail(label: str, proc_out: str, n: int = 40) -> str:
    lines = proc_out.strip().splitlines()[-n:]
    return f"\n----- {label} (last {n} lines) -----\n" + "\n".join(lines)


def _assert_run_outputs(base_dir: Path):
    """Common assertions on a finished run dir; returns (run_dir, pop_df)."""
    import pandas as pd

    runs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    assert len(runs) == 1, f"expected exactly 1 run dir, got {runs}"
    run = runs[0]

    pop = pd.read_csv(run / "results" / "outer_population.csv")
    # Every outer generation must carry all 4 URDFs.
    for og, grp in pop.groupby("outer_gen"):
        assert len(grp) == 4, f"outer_gen {og} has {len(grp)} rows"
    # The two mid-run phases end with the env alive → exam-scored.
    exam_gens = pop[pop["obj_source"] == "exam"]["outer_gen"].nunique()
    assert exam_gens >= 2, f"expected ≥2 exam-scored outer gens, got {exam_gens}"
    # Objectives finite everywhere.
    for col in ("obj_progress_m", "obj_cost_of_transport"):
        assert np.isfinite(pop[col]).all(), f"non-finite {col}"

    summary = pd.read_csv(run / "results" / "cma_summary.csv")
    # pycma maxiter = num_generations stops the loop after gens 0..5.
    assert summary["generation"].nunique() == 6

    # Baseline (reference actor on the population env → distributed path when
    # sharded) and held-out validation both produced rows.
    baseline = pd.read_csv(run / "results" / "baseline_summary.csv")
    assert len(baseline) >= 2
    validation = pd.read_csv(run / "results" / "validation_summary.csv")
    assert len(validation) >= 2
    # Exam baseline (the Pareto star) has one row per exam'd phase.
    exam_base = pd.read_csv(run / "results" / "outer_exam_baseline.csv")
    assert len(exam_base) >= 2
    return run, pop


def test_dist_two_ranks_end_to_end(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    base_dir = tmp_path / "runs_dist"
    base_dir.mkdir()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    procs = []
    for rank in (0, 1):
        env = _base_env()
        env.update({
            "WP2_DIST_WORLD_SIZE": "2",
            "WP2_DIST_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WP2_DIST_TIMEOUT_S": "900",
        })
        procs.append(subprocess.Popen(
            _run_cmd(cfg_path, base_dir),
            cwd=REPO, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ))

    outs = []
    for rank, p in enumerate(procs):
        try:
            out, _ = p.communicate(timeout=1200)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            out, _ = p.communicate()
            pytest.fail(f"rank {rank} timed out" + _tail(f"rank {rank}", out))
        outs.append(out)

    for rank, (p, out) in enumerate(zip(procs, outs)):
        assert p.returncode == 0, (
            f"rank {rank} exited {p.returncode}" + _tail(f"rank {rank}", out)
        )

    assert "Distributed coordinator: rank 0/2" in outs[0]
    assert "Worker rank 1 done." in outs[1]
    # Worker ranks never create run dirs — exactly rank 0's is present.
    run, pop = _assert_run_outputs(base_dir)
    # Sharding actually happened: 2+2 URDFs.
    assert "sharding 4 URDFs across 2 rank(s): [2, 2]" in outs[0]


def test_single_process_control(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    base_dir = tmp_path / "runs_single"
    base_dir.mkdir()

    proc = subprocess.Popen(
        _run_cmd(cfg_path, base_dir),
        cwd=REPO, env=_base_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        out, _ = proc.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail("single-process control timed out" + _tail("control", out))

    assert proc.returncode == 0, (
        f"control exited {proc.returncode}" + _tail("control", out)
    )
    # No distributed machinery on the legacy path.
    assert "Distributed coordinator" not in out
    _assert_run_outputs(base_dir)
