#!/bin/bash -l
#SBATCH --job-name=wp1_single
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# Paths (adjust to your cluster layout)
# ------------------------------------------------------------
IMG="/scratch/$USER/sif/mygenesis.sif"
SRC="$HOME/binding/Genesis-dev"

# ------------------------------------------------------------
# Run params (override via sbatch --export=...)
# ------------------------------------------------------------
CFG_FILE=${CFG_FILE:-configs/default.yaml}
RUN_TAG=${RUN_TAG:-wp1_single_${SLURM_JOB_ID}}
CLI_OVERRIDES=${CLI_OVERRIDES:-}

# ------------------------------------------------------------
# Output layout
# ------------------------------------------------------------
OUT_BASE="$HOME/binding/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/wp1_training/${RUN_TAG}}"
CACHE_DIR="${RUN_DIR}/.cache"

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# Safety checks
if [ ! -f "$IMG" ]; then
  echo "[ERROR] SIF not found: $IMG"
  exit 1
fi
if [ ! -d "$SRC" ]; then
  echo "[ERROR] Bind source directory not found: $SRC"
  exit 1
fi

# ------------------------------------------------------------
# Bind mounts
# ------------------------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:ro,${RUN_DIR}:/workspace/out:rw"

# ------------------------------------------------------------
# Thread tuning
# ------------------------------------------------------------
CPUS_ASSIGNED="${SLURM_CPUS_PER_GPU:-8}"
export OMP_NUM_THREADS="${CPUS_ASSIGNED}"
echo "[RUNTIME] OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
srun apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}
  export XDG_CACHE_HOME=/workspace/out/.cache
  export TI_CACHE_DIR=/workspace/out/.cache/taichi
  export TAICHI_CACHE_DIR=/workspace/out/.cache/taichi
  export GSTAICHI_CACHE_DIR=/workspace/out/.cache/gstaichi
  export MPLCONFIGDIR=/workspace/out/.cache/mpl

  mkdir -p /workspace/out/.cache
  cd /workspace/out

  echo \"[RUNTIME] Node: \$(hostname)\"
  echo \"[RUNTIME] CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-<unset>}\"
  nvidia-smi -L || true

  # Copy meshes and aero files needed by the environment
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    mkdir -p /workspace/out/urdf_generated/meshes
    cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/
  fi
  for fname in aero_parameters.yaml actuators.csv; do
    if [ -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      mkdir -p /workspace/out/urdf_generated
      cp -a \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
    fi
  done

  python -m WP1.train \
    --cfg /workspace/bind/src/WP1/${CFG_FILE} \
    ${CLI_OVERRIDES}
"
