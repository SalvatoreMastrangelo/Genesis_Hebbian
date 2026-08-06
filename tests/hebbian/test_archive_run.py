"""
Tests for the post-run archive step (``slurm_jobs/archive_run.sh``).

Regression cover for the 2026-08-05 cluster failure: ``train.slurm`` archived
the scratch run directory to ``$HOME`` with a plain ``cp -a``, which dragged
along the Taichi/gstaichi kernel caches (~50% of every run directory, measured
373-694 MB across the batch_3 runs). That exhausted the home quota, and because
``train.slurm`` runs under ``set -euo pipefail`` the failing ``cp`` aborted the
whole job -- Slurm recorded FAILED 1:0 for runs whose science had completed
successfully (jobs 3092665, 3092828).

Pure shell-out tests: no Genesis, no cluster.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
ARCHIVE_SH = REPO_ROOT / "src" / "WP2_Outer_Loop" / "slurm_jobs" / "archive_run.sh"
TRAIN_SLURM = REPO_ROOT / "src" / "WP2_Outer_Loop" / "slurm_jobs" / "train.slurm"


def _make_run_dir(root: Path) -> Path:
    """Build a fake run directory mirroring the real scratch layout."""
    run = root / "outer_exam_6_64_64_r2"

    # Disposable caches -- these live at BOTH depths on the cluster.
    (run / ".cache" / "gstaichi").mkdir(parents=True)
    (run / ".cache" / "gstaichi" / "kernel.tic").write_text("x" * 512)
    (run / "logs" / ".cache" / "gstaichi" / "ticache").mkdir(parents=True)
    (run / "logs" / ".cache" / "gstaichi" / "ticache" / "T0b3e.tic").write_text("y" * 512)
    (run / "logs" / ".cache" / "mesa_shader_cache").mkdir(parents=True)
    (run / "logs" / ".cache" / "mesa_shader_cache" / "index").write_text("z")

    # Real artifacts that MUST survive.
    (run / "logs").mkdir(exist_ok=True)
    (run / "logs" / "outer_generations.csv").write_text("outer_gen,obj_source\n0,exam\n")
    (run / "2026-07-23_23-36-52_exam_6_64_64" / "plots").mkdir(parents=True)
    (run / "2026-07-23_23-36-52_exam_6_64_64" / "plots" / "fitness.png").write_text("png")
    (run / "2026-07-23_23-36-52_exam_6_64_64" / "urdfs" / "meshes").mkdir(parents=True)
    (run / "2026-07-23_23-36-52_exam_6_64_64" / "urdfs" / "meshes" / "wing.obj").write_text("obj")
    (run / "urdf_generated" / "meshes").mkdir(parents=True)
    (run / "urdf_generated" / "meshes" / "body.obj").write_text("obj")
    return run


def _run_archive(src: Path, dest_parent: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(ARCHIVE_SH), str(src), str(dest_parent)],
        capture_output=True,
        text=True,
    )


# ----------------------------------------------------------------------------
#  The archive script exists and is wired up
# ----------------------------------------------------------------------------

def test_archive_script_exists():
    assert ARCHIVE_SH.is_file(), f"missing {ARCHIVE_SH}"


def test_train_slurm_uses_archive_script_and_not_bare_cp():
    body = TRAIN_SLURM.read_text()
    assert "archive_run.sh" in body, "train.slurm must delegate archiving"
    assert 'cp -a "${RUN_DIR}"' not in body, (
        "train.slurm still uses the quota-blowing bare `cp -a` of the run dir"
    )


def test_archive_failure_cannot_mark_the_job_failed():
    """The science is done by the time we archive; a full quota must not
    turn a completed run into Slurm FAILED."""
    # Join backslash continuations so a multi-line invocation is checked as the
    # single logical command it is.
    body = TRAIN_SLURM.read_text().replace("\\\n", " ")
    # The archive invocation must be guarded against `set -e`.
    call_lines = [
        ln for ln in body.splitlines()
        if "archive_run.sh" in ln and not ln.strip().startswith("#")
    ]
    assert call_lines, "no archive_run.sh invocation found"
    guarded = any("||" in ln for ln in call_lines)
    assert guarded, (
        "archive_run.sh call is unguarded under `set -e`; a failed archive "
        "would abort the job and report FAILED for a successful run"
    )


# ----------------------------------------------------------------------------
#  Cache exclusion -- the actual bug
# ----------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not available")
def test_archive_excludes_cache_at_every_depth(tmp_path):
    src = _make_run_dir(tmp_path / "scratch")
    dest_parent = tmp_path / "home"
    dest_parent.mkdir()

    proc = _run_archive(src, dest_parent)
    assert proc.returncode == 0, f"archive failed:\n{proc.stdout}\n{proc.stderr}"

    archived = dest_parent / src.name
    assert archived.is_dir(), "run directory was not archived"

    leaked = sorted(p for p in archived.rglob(".cache"))
    assert not leaked, f"cache directories leaked into the archive: {leaked}"

    leaked_files = sorted(p for p in archived.rglob("*.tic"))
    assert not leaked_files, f"taichi cache files leaked: {leaked_files}"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not available")
def test_archive_preserves_real_artifacts(tmp_path):
    src = _make_run_dir(tmp_path / "scratch")
    dest_parent = tmp_path / "home"
    dest_parent.mkdir()

    proc = _run_archive(src, dest_parent)
    assert proc.returncode == 0, f"archive failed:\n{proc.stdout}\n{proc.stderr}"

    archived = dest_parent / src.name
    expected = [
        Path("logs/outer_generations.csv"),
        Path("2026-07-23_23-36-52_exam_6_64_64/plots/fitness.png"),
        Path("2026-07-23_23-36-52_exam_6_64_64/urdfs/meshes/wing.obj"),
        Path("urdf_generated/meshes/body.obj"),
    ]
    for rel in expected:
        assert (archived / rel).is_file(), f"artifact dropped from archive: {rel}"

    # Content must be intact, not just present.
    assert (archived / "logs" / "outer_generations.csv").read_text() == (
        "outer_gen,obj_source\n0,exam\n"
    )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not available")
def test_archive_is_idempotent(tmp_path):
    """Re-archiving an already-archived run must succeed (manual re-runs after
    a quota clear are the recovery path for jobs 3092665 / 3092828)."""
    src = _make_run_dir(tmp_path / "scratch")
    dest_parent = tmp_path / "home"
    dest_parent.mkdir()

    assert _run_archive(src, dest_parent).returncode == 0
    second = _run_archive(src, dest_parent)
    assert second.returncode == 0, f"second archive failed:\n{second.stderr}"

    archived = dest_parent / src.name
    assert (archived / "logs" / "outer_generations.csv").is_file()
    assert not list(archived.rglob(".cache"))


# ----------------------------------------------------------------------------
#  Argument handling
# ----------------------------------------------------------------------------

def test_archive_rejects_missing_source(tmp_path):
    dest_parent = tmp_path / "home"
    dest_parent.mkdir()
    proc = _run_archive(tmp_path / "does_not_exist", dest_parent)
    assert proc.returncode != 0, "missing source must be reported as an error"


def test_archive_requires_both_arguments():
    proc = subprocess.run(
        ["bash", str(ARCHIVE_SH)], capture_output=True, text=True
    )
    assert proc.returncode != 0, "missing arguments must be reported as an error"
