#!/bin/bash -l
#SBATCH --job-name=nsga
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# Paths on SCITAS
# ------------------------------------------------------------
IMG="/scratch/$USER/sif/mygenesis.sif"
SRC="$HOME/binding/Genesis-dev"

# ------------------------------------------------------------
# Run params (override via sbatch --export=...)
# ------------------------------------------------------------
POP_SIZE=${POP_SIZE:-16}
NUM_GEN=${NUM_GEN:-6}
TRAIN_ITERS=${TRAIN_ITERS:-50}
RUN_TAG=${RUN_TAG:-nsga_${SLURM_JOB_ID}}
RUN_NAME=${RUN_NAME:-${RUN_TAG}}
DEVICE=${DEVICE:-cuda:0}
GA_PARALLEL=${GA_PARALLEL:-1}
GEN_POLICY=${GEN_POLICY:-0}
POLICY_PATH=${POLICY_PATH:-}

DEFAULT_POLICY_CLUSTER="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502412/logs/ea/foundation-mixture/model_2000.pt"
if [ -z "${POLICY_PATH}" ]; then
  if [ -f "${DEFAULT_POLICY_CLUSTER}" ]; then
    POLICY_PATH="${DEFAULT_POLICY_CLUSTER}"
  fi
fi
if [ "${GEN_POLICY}" = "0" ] && [ -n "${POLICY_PATH}" ]; then
  GEN_POLICY=1
fi

POLICY_BIND=""
POLICY_PATH_INNER=""
if [ -n "${POLICY_PATH}" ]; then
  POLICY_DIR="$(dirname "$POLICY_PATH")"
  POLICY_BASE="$(basename "$POLICY_PATH")"
  POLICY_BIND=",${POLICY_DIR}:/workspace/policy:ro"
  POLICY_PATH_INNER="/workspace/policy/${POLICY_BASE}"
fi

# ------------------------------------------------------------
# Organized output layout (base + per-run)
# ------------------------------------------------------------
OUT_BASE="$HOME/binding/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/evolution_nsga}"
RUN_ROOT="${RUN_DIR}/${RUN_NAME}"

LOG_DIR="${RUN_ROOT}/logs"
URDF_DIR="${RUN_ROOT}/urdf_generated"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
CACHE_DIR="${RUN_ROOT}/.cache"

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
if [ "${GEN_POLICY}" != "0" ] && [ -z "${POLICY_PATH}" ]; then
  echo "[ERROR] POLICY_PATH is required when GEN_POLICY is enabled"
  exit 1
fi
if [ -n "${POLICY_PATH}" ] && [ ! -f "${POLICY_PATH}" ]; then
  echo "[ERROR] POLICY_PATH not found: ${POLICY_PATH}"
  exit 1
fi
if [ "${GEN_POLICY}" != "0" ] && [ -n "${POLICY_PATH}" ]; then
  POLICY_CFG="${POLICY_DIR}/cfgs.pkl"
  if [ ! -f "${POLICY_CFG}" ]; then
    echo "[ERROR] cfgs.pkl not found next to policy: ${POLICY_CFG}"
    exit 1
  fi
fi

# ------------------------------------------------------------
# Create output dirs on host
# ------------------------------------------------------------
mkdir -p "$LOG_DIR" "$URDF_DIR" "$ANALYSIS_DIR" "$CACHE_DIR"
mkdir -p "$URDF_DIR/meshes"

# ------------------------------------------------------------
# Bind mounts
# - Repo in RO
# - Output base in RW
# ------------------------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:ro,${RUN_DIR}:/workspace/out:rw${POLICY_BIND}"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-1}}"

srun apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  # Make the code importable
  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}

  # Keep caches on output folder
  export XDG_CACHE_HOME=/workspace/out/${RUN_NAME}/.cache
  export TI_CACHE_DIR=/workspace/out/${RUN_NAME}/.cache/taichi
  export TAICHI_CACHE_DIR=/workspace/out/${RUN_NAME}/.cache/taichi
  export GSTAICHI_CACHE_DIR=/workspace/out/${RUN_NAME}/.cache/gstaichi
  export MPLCONFIGDIR=/workspace/out/${RUN_NAME}/.cache/mpl
  export RAY_TMPDIR=/workspace/out/${RUN_NAME}/.cache/ray
  export GA_PARALLEL=\"${GA_PARALLEL}\"

  GEN_POLICY_ARGS=\"\"
  if [ \"${GEN_POLICY}\" != \"0\" ]; then
    GEN_POLICY_ARGS=\"--gen_policy\"
    if [ -n \"${POLICY_PATH_INNER}\" ]; then
      GEN_POLICY_ARGS=\"\$GEN_POLICY_ARGS --policy_path ${POLICY_PATH_INNER}\"
    fi
  fi

  mkdir -p /workspace/out/${RUN_NAME}/logs /workspace/out/${RUN_NAME}/analysis \
    /workspace/out/${RUN_NAME}/urdf_generated /workspace/out/${RUN_NAME}/.cache
  cd /workspace/out/${RUN_NAME}

  # Ensure meshes exist for generated URDFs (required by package://meshes)
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/${RUN_NAME}/urdf_generated/meshes/

  # Aero config + model files required next to the URDFs
  for fname in aero_parameters.yaml actuators.csv drone.py; do
    if [ ! -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      echo \"[ERROR] Missing \${SRC_AERO_DIR}/\${fname}\"
      exit 1
    fi
    cp -a \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/${RUN_NAME}/urdf_generated/
  done

  python /workspace/bind/src/morph_evolution/evolution_nsga.py \
    --pop \"${POP_SIZE}\" \
    --gen \"${NUM_GEN}\" \
    --train_it \"${TRAIN_ITERS}\" \
    --run_name \"${RUN_NAME}\" \
    --base_dir /workspace/out \
    --device \"${DEVICE}\" \
    \$GEN_POLICY_ARGS
"
