"""
Tests for the offline exam-baseline reconstruction (the Pareto star).

Pure-python: covers container-path rebasing, the zero-plasticity reference
genome, the run-grouping signature, and the CSV contract with
``pareto_plots`` — no Genesis required, so the actual rollout is not
exercised here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2_Outer_Loop.config import OuterNSGA2Config
from WP2_Outer_Loop.exam_baseline_rerun import (
    WHOLE_RUN_GEN,
    _rebase_container_path,
    _validation_is_standard_drone,
    eval_signature,
    exam_overrides,
    reference_checkpoint,
    reference_genome,
    write_exam_baseline_csv,
)
from WP2_Outer_Loop.pareto_plots import (
    _baseline_is_constant,
    _load_exam_baseline,
    _load_exam_baseline_series,
)


# ----------------------------------------------------------------------------
#  Container-path rebasing
# ----------------------------------------------------------------------------

def test_rebase_maps_container_path_onto_local_checkout(tmp_path):
    # Cluster configs record paths under the container's bind mount; the file
    # only exists here, under a differently-rooted checkout.
    target = tmp_path / "src" / "WP2_Outer_Loop" / "experiments" / "e" / "ckpt.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    recorded = "/workspace/bind/src/WP2_Outer_Loop/experiments/e/ckpt.pt"
    assert _rebase_container_path(recorded, tmp_path) == str(target)


def test_rebase_leaves_existing_path_untouched(tmp_path):
    real = tmp_path / "already_here.pt"
    real.write_bytes(b"x")
    assert _rebase_container_path(str(real), tmp_path) == str(real)


def test_rebase_returns_input_when_no_local_counterpart(tmp_path):
    # Rebasing must not invent a path: a missing file has to stay missing so
    # the caller's existence check reports the real problem.
    recorded = "/workspace/bind/src/WP2_Outer_Loop/experiments/gone/ckpt.pt"
    assert _rebase_container_path(recorded, tmp_path) == recorded


def test_rebase_ignores_paths_without_a_src_segment(tmp_path):
    assert _rebase_container_path("/data/elsewhere/ckpt.pt", tmp_path) == \
        "/data/elsewhere/ckpt.pt"


# ----------------------------------------------------------------------------
#  Reference actor / genome
# ----------------------------------------------------------------------------

def test_reference_genome_is_all_mid_range_without_evolved_decay():
    # ABCD = 0 in a symmetric [-r, r] range is the 0.5 gene → no plasticity.
    cfg = OuterNSGA2Config()
    cfg.hebbian.evolve_decay = False
    genome = reference_genome(cfg)
    assert genome.shape == (cfg.hebbian_genome_dim(),)
    assert np.all(genome == 0.5)


def test_reference_genome_zeroes_the_evolved_decay_block():
    # Decay maps [0, r] → gene 0 is zero decay, so the block is 0.0, not 0.5;
    # otherwise the reference would drift off the checkpoint mid-episode.
    cfg = OuterNSGA2Config()
    cfg.hebbian.evolve_decay = True
    genome = reference_genome(cfg)
    n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    start = 4 * cfg.hebbian.abcd_block_size()
    assert np.all(genome[:start] == 0.5)
    assert np.all(genome[start: start + n_weights] == 0.0)


def test_reference_checkpoint_prefers_the_dedicated_baseline_actor():
    cfg = OuterNSGA2Config()
    cfg.checkpoint_path = "main.pt"
    cfg.checkpoint_config_path = "main.yaml"
    cfg.baseline_checkpoint_path = "base.pt"
    cfg.baseline_checkpoint_config_path = "base.yaml"
    assert reference_checkpoint(cfg) == ("base.pt", "base.yaml")


def test_reference_checkpoint_falls_back_to_the_frozen_actor():
    cfg = OuterNSGA2Config()
    cfg.checkpoint_path = "main.pt"
    cfg.checkpoint_config_path = "main.yaml"
    assert reference_checkpoint(cfg) == ("main.pt", "main.yaml")


# ----------------------------------------------------------------------------
#  Exam overrides / validation gate
# ----------------------------------------------------------------------------

def test_exam_overrides_empty_when_the_exam_flew_nominal_forests():
    cfg = OuterNSGA2Config()
    cfg.outer.exam_forest.override_forest = False
    cfg.outer.exam_forest.dens_min = 3.0
    assert exam_overrides(cfg) == {}


def test_exam_overrides_carry_the_non_null_fields():
    cfg = OuterNSGA2Config()
    cfg.outer.exam_forest.override_forest = True
    cfg.outer.exam_forest.dens_min = 3.0
    cfg.outer.exam_forest.x_upper = 300.0
    assert exam_overrides(cfg) == {"dens_min": 3.0, "x_upper": 300.0}


@pytest.mark.parametrize("enable,catalog,expected", [
    (True, "", True),
    (True, "none", True),
    (True, "my_catalog.txt", False),   # a custom drone would be mislabelled
    (False, "", False),                # no validation env → no standard drone
])
def test_validation_standard_drone_gate(enable, catalog, expected):
    cfg = OuterNSGA2Config()
    cfg.validation.enable = enable
    cfg.validation.validation_catalog = catalog
    assert _validation_is_standard_drone(cfg) is expected


# ----------------------------------------------------------------------------
#  Grouping signature: which runs may share one measurement
# ----------------------------------------------------------------------------

def _sig_cfg():
    cfg = OuterNSGA2Config()
    cfg.checkpoint_path = ""       # no file → hashed as "" on both sides
    cfg.checkpoint_config_path = ""
    cfg.outer.exam_forest.override_forest = True
    cfg.outer.exam_forest.dens_min = 3.0
    cfg.outer.exam_forest.x_upper = 300.0
    return cfg


def test_signature_ignores_sizing_and_seed():
    # These change the sample drawn, not the distribution measured, so runs
    # differing only here would have produced the same reference point.
    a, b = _sig_cfg(), _sig_cfg()
    b.seed = a.seed + 1
    b.evaluation.num_eval_envs = a.evaluation.num_eval_envs * 2
    b.evaluation.num_eval_workers = 8
    b.cmaes.population_size = 2
    b.evolution.num_generations = 999
    assert eval_signature(a) == eval_signature(b)


def test_signature_separates_different_exam_forests():
    a, b = _sig_cfg(), _sig_cfg()
    b.outer.exam_forest.dens_min = 4.0
    assert eval_signature(a) != eval_signature(b)


def test_signature_separates_nominal_from_overridden_exam():
    a, b = _sig_cfg(), _sig_cfg()
    b.outer.exam_forest.override_forest = False
    assert eval_signature(a) != eval_signature(b)


def test_signature_separates_different_speed_bands():
    a, b = _sig_cfg(), _sig_cfg()
    b.evaluation.vmax = a.evaluation.vmax + 5.0
    assert eval_signature(a) != eval_signature(b)


# ----------------------------------------------------------------------------
#  CSV contract with pareto_plots
# ----------------------------------------------------------------------------

_MEANS = {
    "fitness": 340.4, "velocity": 13.81, "progress": 224.5,
    "crash_rate": 0.968, "cot": 0.2883, "v_deviation": 0.791,
}
_PASSES = [dict(_MEANS, progress=224.0), dict(_MEANS, progress=225.0)]


def _run_dir_with_star(tmp_path):
    """Run dir whose config passes the plot's standard-drone gate, carrying a
    reconstructed baseline CSV."""
    run_dir = tmp_path / "run"
    (run_dir / "reproducibility").mkdir(parents=True)
    (run_dir / "reproducibility" / "config.yaml").write_text(
        "validation:\n  enable: true\n  validation_catalog: ''\n"
    )
    write_exam_baseline_csv(
        run_dir, _MEANS, _PASSES, n_forests=16384, warmup=1,
        overrides={"dens_min": 3.0, "x_upper": 300.0},
    )
    return run_dir


def test_written_csv_is_read_back_as_the_star(tmp_path):
    # The whole point of the file: pareto_plots must find the reference point
    # without knowing it was reconstructed offline.
    run_dir = _run_dir_with_star(tmp_path)
    star = _load_exam_baseline(run_dir, "progress_m", "cost_of_transport")
    assert star == pytest.approx((224.5, 0.2883))


def test_written_csv_maps_the_velocity_deviation_alias(tmp_path):
    run_dir = _run_dir_with_star(tmp_path)
    star = _load_exam_baseline(run_dir, "progress_m", "velocity_deviation")
    assert star == pytest.approx((224.5, 0.791))


def test_written_row_reads_as_a_constant_baseline(tmp_path):
    # One whole-run measurement belongs on the champion panels as a horizontal
    # line, not as a per-phase curve.
    run_dir = _run_dir_with_star(tmp_path)
    base = _load_exam_baseline_series(run_dir, ["progress_m"])
    assert base is not None
    assert list(base.index) == [WHOLE_RUN_GEN]
    assert _baseline_is_constant(base)


def test_per_pass_csv_records_every_measured_pass(tmp_path):
    # Provenance: the spread across passes is the only visible error bar on
    # the star, so the passes must survive the aggregation.
    run_dir = _run_dir_with_star(tmp_path)
    lines = (run_dir / "results" / "outer_exam_baseline_passes.csv") \
        .read_text().strip().splitlines()
    assert len(lines) == 1 + len(_PASSES)
    assert lines[0].startswith("pass,n_forests,")


def test_written_csv_records_its_provenance(tmp_path):
    run_dir = _run_dir_with_star(tmp_path)
    text = (run_dir / "results" / "outer_exam_baseline.csv").read_text()
    header, row = text.strip().splitlines()[:2]
    assert "source" in header and "exam_baseline_rerun" in row
    assert "16384" in row
