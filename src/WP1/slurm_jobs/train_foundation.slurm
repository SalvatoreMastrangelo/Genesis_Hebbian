#!/bin/bash -l
#SBATCH --job-name=wp1_foundation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err
#SBATCH --mail-user=salvatore.mastrangelo@epfl.ch
#SBATCH --mail-type=END

set -euo pipefail

# ------------------------------------------------------------
# Paths (adjust to your cluster layout)
# ------------------------------------------------------------
IMG="/scratch/izar/$USER/sif/mygenesis.sif"
SRC="$HOME/Genesis-dev"

# ------------------------------------------------------------
# Run params (override via sbatch --export=...)
# ------------------------------------------------------------
CFG_FILE=${CFG_FILE:-configs/foundation.yaml}
RUN_TAG=${RUN_TAG:-wp1_foundation_${SLURM_JOB_ID}}
CLI_OVERRIDES=${CLI_OVERRIDES:-}

# Multi-GPU: set MULTI_GPU=N to split the URDF catalog across N GPUs.
# MULTI_GPU must be <= the number of --gpus requested above.
# Example: sbatch --gpus=2 --export=ALL,MULTI_GPU=2,CFG_FILE=... train_foundation.slurm
MULTI_GPU=${MULTI_GPU:-1}

# CUDA MPS controls (for multi-GPU concurrency)
ENABLE_CUDA_MPS=${ENABLE_CUDA_MPS:-0}
MPS_THREAD_PCT=${MPS_THREAD_PCT:-0}

# ------------------------------------------------------------
# Output layout
# ------------------------------------------------------------
OUT_BASE="/scratch/izar/$USER/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/wp1_training/${RUN_TAG}}"
CACHE_DIR="${RUN_DIR}/.cache"

mkdir -p "$RUN_DIR" "$CACHE_DIR"
mkdir -p "${RUN_DIR}/.mps/pipe" "${RUN_DIR}/.mps/log"

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
CPUS_ASSIGNED=""
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  CPUS_ASSIGNED="${SLURM_CPUS_PER_TASK}"
elif [ -n "${SLURM_CPUS_PER_GPU:-}" ]; then
  GPUS_ASSIGNED_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}}}"
  GPUS_ASSIGNED="$(printf '%s' "${GPUS_ASSIGNED_RAW}" | tr -cd '0-9')"
  if [ -z "${GPUS_ASSIGNED}" ]; then
    GPUS_ASSIGNED=1
  fi
  CPUS_ASSIGNED=$(( SLURM_CPUS_PER_GPU * GPUS_ASSIGNED ))
else
  CPUS_ASSIGNED=1
fi
export OMP_NUM_THREADS="${CPUS_ASSIGNED}"
echo "[RUNTIME] cpus_assigned=${CPUS_ASSIGNED} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ------------------------------------------------------------
# Optional: CUDA MPS daemon
# ------------------------------------------------------------
MPS_STARTED=0
if [ "${ENABLE_CUDA_MPS}" -ne 0 ]; then
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    export CUDA_MPS_PIPE_DIRECTORY="${RUN_DIR}/.mps/pipe"
    export CUDA_MPS_LOG_DIRECTORY="${RUN_DIR}/.mps/log"
    rm -f "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control" || true
    nvidia-cuda-mps-control -d
    MPS_STARTED=1
    echo "[MPS] started (pipe=${CUDA_MPS_PIPE_DIRECTORY})"
  else
    echo "[WARN] nvidia-cuda-mps-control not found; running without MPS."
  fi
fi

cleanup_mps() {
  if [ "${MPS_STARTED}" -eq 1 ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    echo "[MPS] stopped"
  fi
}
trap cleanup_mps EXIT

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

  if [ \"${ENABLE_CUDA_MPS}\" -ne 0 ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/workspace/out/.mps/pipe
    export CUDA_MPS_LOG_DIRECTORY=/workspace/out/.mps/log
    export LOGICAL_SUPER_SCENE_MPS_THREAD_PERCENT=${MPS_THREAD_PCT}
  fi

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
    cp -r \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/
  fi
  for fname in aero_parameters.yaml actuators.csv; do
    if [ -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      mkdir -p /workspace/out/urdf_generated
      cp -r \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
    fi
  done

  if [ "${MULTI_GPU:-1}" -gt 1 ]; then
    echo \"[RUNTIME] Multi-GPU mode: distributing across ${MULTI_GPU} GPUs\"
    python -m WP1.train --cfg /workspace/bind/src/WP1/${CFG_FILE} --multi-gpu ${MULTI_GPU} ${CLI_OVERRIDES}
  else
    python -m WP1.train --cfg /workspace/bind/src/WP1/${CFG_FILE} ${CLI_OVERRIDES}
  fi
"

# ------------------------------------------------------------
# Copy run output to home for long-term storage
# ------------------------------------------------------------
echo "[POST] Copying run to $HOME/genesis_runs/ ..."
mkdir -p "$HOME/genesis_runs/wp1_training"
cp -a "${RUN_DIR}" "$HOME/genesis_runs/wp1_training/"
echo "[POST] Done: $HOME/genesis_runs/wp1_training/$(basename ${RUN_DIR})"
