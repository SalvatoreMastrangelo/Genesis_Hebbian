#!/bin/bash -l
#SBATCH --job-name=wp1_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=70:00:00
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
CFG_FILE=${CFG_FILE:-configs/default.yaml}
RUN_TAG=${RUN_TAG:-wp1_train_${SLURM_JOB_ID}}
CLI_OVERRIDES=${CLI_OVERRIDES:-}

# Resume: set RESUME=1 to restart from the latest matching run folder.
RESUME=${RESUME:-0}
RESUME_FLAG=""
[ "${RESUME}" -ne 0 ] && RESUME_FLAG="--resume"

# Logical Super-Scene (sharded multi-URDF) toggle.
# Set LSS=1 and (optionally) URDF_SHARD_SIZE / LSS_NUM_WORKERS / LSS_GPUS to
# enable sharded training. Requires a catalog in the YAML (catalog.n_urdf>0).
# Example:
#   sbatch --gpus=2 --export=ALL,LSS=1,URDF_SHARD_SIZE=32,LSS_GPUS=2,... train.slurm
LSS=${LSS:-0}
URDF_SHARD_SIZE=${URDF_SHARD_SIZE:-0}
LSS_NUM_WORKERS=${LSS_NUM_WORKERS:-0}
LSS_GPUS=${LSS_GPUS:-1}
LSS_FLAGS=""
if [ "${LSS}" -ne 0 ]; then
  LSS_FLAGS="--cfg.lss.enabled true --cfg.lss.urdf_shard_size ${URDF_SHARD_SIZE} --cfg.lss.num_workers ${LSS_NUM_WORKERS} --cfg.lss.collection_gpus ${LSS_GPUS}"
fi

# ------------------------------------------------------------
# Output layout
# ------------------------------------------------------------
OUT_BASE="/scratch/izar/$USER/genesis_runs"
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
OPENBLAS_THREADS=$(( CPUS_ASSIGNED / 4 ))
if [ "$OPENBLAS_THREADS" -lt 2 ]; then
  OPENBLAS_THREADS=2
fi
if [ "$OPENBLAS_THREADS" -gt 8 ]; then
  OPENBLAS_THREADS=8
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_THREADS}"
echo "[RUNTIME] cpus_assigned=${CPUS_ASSIGNED} OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"

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
    cp -r \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/
  fi
  for fname in aero_parameters.yaml actuators.csv; do
    if [ -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      mkdir -p /workspace/out/urdf_generated
      cp -r \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
    fi
  done

  python -m WP1.train --cfg /workspace/bind/src/WP1/${CFG_FILE} ${RESUME_FLAG} ${LSS_FLAGS} ${CLI_OVERRIDES}
"

# ------------------------------------------------------------
# Copy run output to home for long-term storage
# ------------------------------------------------------------
echo "[POST] Copying run to $HOME/genesis_runs/ ..."
mkdir -p "$HOME/genesis_runs/wp1_training"
cp -a "${RUN_DIR}" "$HOME/genesis_runs/wp1_training/"
echo "[POST] Done: $HOME/genesis_runs/wp1_training/$(basename ${RUN_DIR})"
