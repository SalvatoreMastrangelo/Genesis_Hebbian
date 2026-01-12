#!/bin/bash -l
#SBATCH --job-name=genesis_build
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# INPUT (HOME): keep critical inputs on /home (backed up & replicated on SCITAS).
# ------------------------------------------------------------
TAR_HOME="$HOME/tar/mygenesis.tar"
if [ ! -f "$TAR_HOME" ]; then
  echo "[ERROR] Input tar not found: $TAR_HOME"
  exit 1
fi

# ------------------------------------------------------------
# SCRATCH workspace: fast temporary storage (no backup, auto-cleanup policy).
# ------------------------------------------------------------
SCR_BASE="/scratch/$USER"
BUILD_DIR="$SCR_BASE/build/${SLURM_JOB_ID}"
SIF_DIR="$SCR_BASE/sif"
mkdir -p "$BUILD_DIR" "$SIF_DIR"

# ------------------------------------------------------------
# Apptainer cache/tmp on scratch to avoid filling $HOME.
# ------------------------------------------------------------
export APPTAINER_CACHEDIR="$SCR_BASE/apptainer-cache"
export APPTAINER_TMPDIR="$SCR_BASE/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ------------------------------------------------------------
# Stage the tar into scratch (local fast I/O during build).
# ------------------------------------------------------------
echo "[INFO] Staging Docker archive tar to scratch..."
cp -v "$TAR_HOME" "$BUILD_DIR/"

TAR_SCR="$BUILD_DIR/mygenesis.tar"
OUT_SIF="$SIF_DIR/mygenesis.sif"

# ------------------------------------------------------------
# Build from a Docker archive tar:
# IMPORTANT: use docker-archive: URI scheme (not a plain path).
# ------------------------------------------------------------
SOURCE_URI="docker-archive:$TAR_SCR"

echo "[INFO] Source URI: $SOURCE_URI"
echo "[INFO] Output SIF: $OUT_SIF"

# ------------------------------------------------------------
# Fakeroot depends on host configuration.
# Try with --fakeroot first; if it fails, retry without it.
# ------------------------------------------------------------
echo "[INFO] Trying build with --fakeroot..."
set +e
apptainer build --fakeroot "$OUT_SIF" "$SOURCE_URI"
rc=$?
set -e

if [ $rc -ne 0 ]; then
  echo "[WARN] --fakeroot build failed (exit code $rc). Retrying without --fakeroot..."
  apptainer build "$OUT_SIF" "$SOURCE_URI"
fi

echo "[OK] Build completed: $OUT_SIF"
