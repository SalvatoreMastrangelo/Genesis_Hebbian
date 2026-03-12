#!/bin/bash -l
#SBATCH --job-name=gen_train
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=26:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err
#SBATCH --exclude=kl001

set -euo pipefail

# ------------------------------------------------------------
# Paths on SCITAS (based on your current layout)
# ------------------------------------------------------------
# Container built from your tar (build job): stored on scratch
IMG="/scratch/$USER/sif/mygenesis.sif"

# Repo / code to bind: you cloned Genesis branch dev into $HOME/binding/Genesis-dev
SRC="$HOME/binding/Genesis-dev"

# ------------------------------------------------------------
# Run params (can be overridden via sbatch --export=...)
# ------------------------------------------------------------
EXP_NAME=${EXP_NAME:-foundation-mixture}
RUN_TAG=${RUN_TAG:-${EXP_NAME}_${SLURM_JOB_ID}}
NUM_ENVS=${NUM_ENVS:-65536} #32768  16384
MAX_ITERS=${MAX_ITERS:-4000}
N_URDF=${N_URDF:-200}
URDF_SEED=${URDF_SEED:-10}
LOGICAL_SUPER_SCENE=${LOGICAL_SUPER_SCENE:-1}
URDF_SHARD_SIZE=${URDF_SHARD_SIZE:-50}
NUM_WORKERS=${NUM_WORKERS:-0}
COLLECTION_GPUS=${COLLECTION_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
VIS=${VIS:-0}
LOGICAL_SUPER_SCENE_SHM=${LOGICAL_SUPER_SCENE_SHM:-0}

# CUDA MPS controls (single-GPU multiprocess concurrency)
ENABLE_CUDA_MPS=${ENABLE_CUDA_MPS:-0}
MPS_THREAD_PCT=${MPS_THREAD_PCT:-0}  # 0 => auto split in orchestrator

# ------------------------------------------------------------
# Organized output layout (single bind target)
# ------------------------------------------------------------
OUT_BASE="$HOME/binding/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/training_general/${RUN_TAG}}"

LOG_DIR="${RUN_DIR}/logs"
URDF_DIR="${RUN_DIR}/urdf_generated"
CACHE_DIR="${RUN_DIR}/.cache"

# Make sure output + cache dirs exist
mkdir -p "$LOG_DIR" "$URDF_DIR" "$CACHE_DIR"
mkdir -p "$URDF_DIR/meshes"
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
# - Bind the repo read-only (safer)
# - Bind output read-write
# ------------------------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:ro,${RUN_DIR}:/workspace/out:rw"

# ------------------------------------------------------------
# Run
# Approximate per-process OpenMP threads as:
#   floor(cpus_assigned / active_processes), with a minimum of 1.
# In logical super-scene mode, active_processes ~= workers + parent.
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
OMP_PROCESSES=1
if [ "${LOGICAL_SUPER_SCENE}" -ne 0 ]; then
  if [ "${NUM_WORKERS}" -gt 0 ]; then
    OMP_PROCESSES="${NUM_WORKERS}"
  else
    OMP_PROCESSES=$(( (N_URDF + URDF_SHARD_SIZE - 1) / URDF_SHARD_SIZE ))
  fi
  OMP_PROCESSES=$(( OMP_PROCESSES + 1 ))
fi
OMP_THREADS=$(( CPUS_ASSIGNED / OMP_PROCESSES ))
if [ "${OMP_THREADS}" -lt 1 ]; then
  OMP_THREADS=1
fi
export OMP_NUM_THREADS="${OMP_THREADS}"
echo "[RUNTIME] cpus_assigned=${CPUS_ASSIGNED} omp_processes=${OMP_PROCESSES} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ------------------------------------------------------------
# Optional: start CUDA MPS daemon on host
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

srun apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  # Make the code importable
  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}

  # Keep caches on scratch
  export XDG_CACHE_HOME=/workspace/out/.cache
  export TI_CACHE_DIR=/workspace/out/.cache/taichi
  export TAICHI_CACHE_DIR=/workspace/out/.cache/taichi
  export GSTAICHI_CACHE_DIR=/workspace/out/.cache/gstaichi
  export MPLCONFIGDIR=/workspace/out/.cache/mpl
  export LOGICAL_SUPER_SCENE_SHM=${LOGICAL_SUPER_SCENE_SHM}
  export LOGICAL_SUPER_SCENE_MPS_THREAD_PERCENT=${MPS_THREAD_PCT}
  if [ \"${ENABLE_CUDA_MPS}\" -ne 0 ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/workspace/out/.mps/pipe
    export CUDA_MPS_LOG_DIRECTORY=/workspace/out/.mps/log
  fi

  mkdir -p /workspace/out/logs /workspace/out/urdf_generated /workspace/out/.cache /workspace/out/.mps/pipe /workspace/out/.mps/log
  cd /workspace/out

  # Runtime placement debug (node + GPU)
  echo \"[RUNTIME] Node: \$(hostname)\"
  echo \"[RUNTIME] CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-<unset>}\"
  nvidia-smi -L || true

  # Ensure meshes exist for generated URDFs (required by package://meshes)
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/

  # Aero config files required next to the URDFs
  for fname in aero_parameters.yaml actuators.csv; do
    if [ ! -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      echo \"[ERROR] Missing \${SRC_AERO_DIR}/\${fname}\"
      exit 1
    fi
    cp -a \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
  done

  # If not generating new URDFs, require a non-empty catalog already in place
  if [ \"${N_URDF}\" -le 0 ]; then
    shopt -s nullglob
    urdfs=(/workspace/out/urdf_generated/*.urdf)
    shopt -u nullglob
    if [ \"\${#urdfs[@]}\" -eq 0 ]; then
      echo \"[ERROR] No URDFs in /workspace/out/urdf_generated. Set N_URDF>0 to generate a catalog.\"
      exit 1
    fi
  fi

  # URDF args
  URDF_ARGS=\"\"
  if [ \"${N_URDF}\" -gt 0 ]; then
    URDF_ARGS=\"--n-urdf ${N_URDF} --urdf-seed ${URDF_SEED} --catalog-dir /workspace/out/urdf_generated\"
  else
    URDF_ARGS=\"--catalog-dir /workspace/out/urdf_generated\"
  fi

  # Logical super-scene args (optional, train_gen.py specific)
  LSS_ARGS=\"\"
  if [ \"${LOGICAL_SUPER_SCENE}\" -ne 0 ]; then
    LSS_ARGS=\"--logical-super-scene --urdf-shard-size ${URDF_SHARD_SIZE} --collection-gpus ${COLLECTION_GPUS}\"
    if [ \"${NUM_WORKERS}\" -gt 0 ]; then
      LSS_ARGS=\"\${LSS_ARGS} --num-workers ${NUM_WORKERS}\"
    fi
  fi

  VIS_ARGS=\"\"
  if [ \"${VIS}\" -ne 0 ]; then
    VIS_ARGS=\"--vis\"
  fi

  # IMPORTANT: adjust this path if your script lives under Genesis-dev/src/...
  python /workspace/bind/src/general_policy/train_gen.py \
    --exp-name \"${EXP_NAME}\" \
    --num-envs \"${NUM_ENVS}\" \
    --max-iterations \"${MAX_ITERS}\" \
    --device cuda \
    \$URDF_ARGS \
    \$LSS_ARGS \
    \$VIS_ARGS
"
