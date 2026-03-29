#!/bin/bash -l
#SBATCH --job-name=sne_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-7
#SBATCH --gpus=2
#SBATCH --mem=180G
#SBATCH --cpus-per-gpu=20
#SBATCH --time=72:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j_%a.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j_%a.err
#SBATCH --mail-user=salvatore.mastrangelo@epfl.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================
# MODIFY THESE THREE ARRAYS TO DEFINE THE BENCHMARK GRID
# Leave empty () to use the values from the YAML config file.
# ============================================================
S_VALUES=()
N_VALUES=()
E_VALUES=()
# ============================================================

# For multi-GPU runs: set --gpus=N above and GPU_COUNT=N here
GPU_COUNT=${GPU_COUNT:-1}

# Config file path (relative to repo root)
CFG_FILE=${CFG_FILE:-src/WP1/benchmark/configs/sne_benchmark.yaml}

# Number of array jobs (must match --array upper bound + 1)
NUM_JOBS=8

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
IMG="/scratch/izar/smastran/sif/mygenesis.sif"
SRC="$HOME/Genesis-dev"

# ------------------------------------------------------------
# Output layout (mirrors train_foundation.slurm)
# ------------------------------------------------------------
OUT_BASE="/scratch/izar/smastran/genesis_runs"
RUN_TAG="sne_benchmark_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${OUT_BASE}/sne_benchmark/${RUN_TAG}"
CACHE_DIR="${RUN_DIR}/.cache"

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
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
# Thread tuning (mirrors train_foundation.slurm)
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
# OpenBLAS threads: use conservative fraction to avoid pthread_create failures
# IZAR has low nproc limits (4096), so cap per-process threads
OPENBLAS_THREADS=$(( CPUS_ASSIGNED / 4 ))
if [ "$OPENBLAS_THREADS" -lt 2 ]; then
  OPENBLAS_THREADS=2
fi
if [ "$OPENBLAS_THREADS" -gt 8 ]; then
  OPENBLAS_THREADS=8
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_THREADS}"

# ------------------------------------------------------------
# Print job info
# ------------------------------------------------------------
echo "============================================================"
echo "SLURM job: $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Config: $CFG_FILE"
echo "S_VALUES: ${S_VALUES[*]}"
echo "N_VALUES: ${N_VALUES[*]}"
echo "E_VALUES: ${E_VALUES[*]}"
echo "GPU_COUNT: $GPU_COUNT"
echo "RUN_DIR: $RUN_DIR"
echo "[RUNTIME] cpus_assigned=${CPUS_ASSIGNED} OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "============================================================"

mkdir -p "$HOME/slurm_logs"

# ------------------------------------------------------------
# Build optional grid override args (only passed when non-empty,
# so the YAML values are used when the arrays above are left as ())
# ------------------------------------------------------------
GRID_ARGS=""
[ ${#S_VALUES[@]} -gt 0 ] && GRID_ARGS="${GRID_ARGS} --S-values ${S_VALUES[*]}"
[ ${#N_VALUES[@]} -gt 0 ] && GRID_ARGS="${GRID_ARGS} --N-values ${N_VALUES[*]}"
[ ${#E_VALUES[@]} -gt 0 ] && GRID_ARGS="${GRID_ARGS} --E-values ${E_VALUES[*]}"

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
      cp \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
    fi
  done

  cd /workspace/out

  python -m WP1.benchmark.grid_search \
    --cfg /workspace/bind/${CFG_FILE} \
    --job-idx ${SLURM_ARRAY_TASK_ID} \
    --num-jobs ${NUM_JOBS} \
    --num-gpus ${GPU_COUNT} \
    ${GRID_ARGS}
"

# ------------------------------------------------------------
# Copy run output to home for long-term storage
# ------------------------------------------------------------
echo "[POST] Copying results to $HOME/genesis_runs/sne_benchmark/ ..."
mkdir -p "$HOME/genesis_runs/sne_benchmark"
cp -a "${RUN_DIR}" "$HOME/genesis_runs/sne_benchmark/"
echo "[POST] Done: $HOME/genesis_runs/sne_benchmark/${RUN_TAG}"
