#!/bin/bash -l
#SBATCH --job-name=gen_adapt_eval
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
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
GENERAL_EXP=${GENERAL_EXP:-}
ADAPT_EXP=${ADAPT_EXP:-adapt_eval_${SLURM_JOB_ID}}
RUN_TAG=${RUN_TAG:-${ADAPT_EXP}}

EVAL_ENVS=${EVAL_ENVS:-8192}
VMIN=${VMIN:-6.0}
VMAX=${VMAX:-30.0}

ADAPT_ENVS=${ADAPT_ENVS:-16384}
ADAPT_ITERS=${ADAPT_ITERS:-100}
ADAPT_REPEATS=${ADAPT_REPEATS:-1}
DEVICE=${DEVICE:-cuda}

N_URDF=${N_URDF:-50}
URDF_SEED=${URDF_SEED:-1}
NSGA_CSV=${NSGA_CSV:-}

GENERAL_CKPT=${GENERAL_CKPT:-2100}
GENERAL_RUN_DIR=${GENERAL_RUN_DIR:-}
GENERAL_LOG_DIR=${GENERAL_LOG_DIR:-}
GENERAL_POLICY_PATH=${GENERAL_POLICY_PATH:-}
GENERAL_CFG_PATH=${GENERAL_CFG_PATH:-}

DEFAULT_GENERAL_POLICY="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2590413/logs/ea/foundation-mixture/model_2100.pt"

if [ -z "${GENERAL_EXP}" ] && [ -n "${GENERAL_POLICY_PATH}" ]; then
  GENERAL_EXP="$(basename "$(dirname "${GENERAL_POLICY_PATH}")")"
fi

if [ -z "${GENERAL_LOG_DIR}" ] && [ -n "${GENERAL_RUN_DIR}" ] && [ -n "${GENERAL_EXP}" ]; then
  GENERAL_LOG_DIR="${GENERAL_RUN_DIR}/logs/ea/${GENERAL_EXP}"
fi

if [ -z "${GENERAL_LOG_DIR}" ] && [ -n "${GENERAL_POLICY_PATH}" ]; then
  GENERAL_LOG_DIR="$(dirname "${GENERAL_POLICY_PATH}")"
fi

if [ -z "${GENERAL_POLICY_PATH}" ] && [ -n "${GENERAL_LOG_DIR}" ] && [ -n "${GENERAL_CKPT}" ]; then
  GENERAL_POLICY_PATH="${GENERAL_LOG_DIR}/model_${GENERAL_CKPT}.pt"
fi

if [ -z "${GENERAL_POLICY_PATH}" ]; then
  GENERAL_POLICY_PATH="${DEFAULT_GENERAL_POLICY}"
fi

if [ -z "${GENERAL_CFG_PATH}" ] && [ -n "${GENERAL_LOG_DIR}" ] && [ -f "${GENERAL_LOG_DIR}/cfgs.pkl" ]; then
  GENERAL_CFG_PATH="${GENERAL_LOG_DIR}/cfgs.pkl"
fi

GENERAL_MODEL_BASE=""
if [ -n "${GENERAL_POLICY_PATH}" ]; then
  GENERAL_MODEL_BASE="$(basename "${GENERAL_POLICY_PATH}")"
fi

# ------------------------------------------------------------
# Organized output layout (single bind target)
# ------------------------------------------------------------
OUT_BASE="$HOME/binding/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/adapt_and_eval_general/${RUN_TAG}}"

LOG_DIR="${RUN_DIR}/logs"
URDF_DIR="${RUN_DIR}/urdf_generated"
ANALYSIS_DIR="${RUN_DIR}/analysis"
CACHE_DIR="${RUN_DIR}/.cache"

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
if [ -z "${GENERAL_POLICY_PATH}" ]; then
  echo "[ERROR] GENERAL_POLICY_PATH is required"
  exit 1
fi
if [ ! -f "${GENERAL_POLICY_PATH}" ]; then
  echo "[ERROR] GENERAL_POLICY_PATH not found: ${GENERAL_POLICY_PATH}"
  exit 1
fi
if [ -z "${GENERAL_CFG_PATH}" ]; then
  echo "[ERROR] Could not resolve GENERAL_CFG_PATH"
  exit 1
fi
if [ ! -f "${GENERAL_CFG_PATH}" ]; then
  echo "[ERROR] GENERAL_CFG_PATH not found: ${GENERAL_CFG_PATH}"
  exit 1
fi

# ------------------------------------------------------------
# Create output dirs on host
# ------------------------------------------------------------
mkdir -p "$LOG_DIR" "$URDF_DIR" "$ANALYSIS_DIR" "$CACHE_DIR"
mkdir -p "$URDF_DIR/meshes"

MODEL_STAGE_DIR="${RUN_DIR}/models/general"
mkdir -p "${MODEL_STAGE_DIR}"
cp -a "${GENERAL_POLICY_PATH}" "${MODEL_STAGE_DIR}/${GENERAL_MODEL_BASE}"
cp -a "${GENERAL_CFG_PATH}" "${MODEL_STAGE_DIR}/cfgs.pkl"

# ------------------------------------------------------------
# Bind mounts
# ------------------------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:ro,${SRC}:/workspace/Genesis:ro,${RUN_DIR}:/workspace/out:rw"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

if ! ulimit -u 65535; then
  echo "[WARN] Could not raise max user processes (ulimit -u)."
fi
if ! ulimit -n 65535; then
  echo "[WARN] Could not raise max open files (ulimit -n)."
fi

srun apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export MALLOC_ARENA_MAX=2

  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}
  export XDG_CACHE_HOME=/workspace/out/.cache
  export TI_CACHE_DIR=/workspace/out/.cache/taichi
  export TAICHI_CACHE_DIR=/workspace/out/.cache/taichi
  export GSTAICHI_CACHE_DIR=/workspace/out/.cache/gstaichi
  export MPLCONFIGDIR=/workspace/out/.cache/mpl

  mkdir -p /workspace/out/logs /workspace/out/analysis /workspace/out/urdf_generated /workspace/out/.cache
  cd /workspace/out

  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone

  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/

  for fname in aero_parameters.yaml actuators.csv; do
    if [ ! -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      echo \"[ERROR] Missing \${SRC_AERO_DIR}/\${fname}\"
      exit 1
    fi
    cp -a \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
  done

  if [ \"${N_URDF}\" -le 0 ]; then
    if [ -f \"\${SRC_URDF_DIR}/catalog.txt\" ]; then
      cp -a \"\${SRC_URDF_DIR}/catalog.txt\" /workspace/out/urdf_generated/
    fi
    shopt -s nullglob
    urdfs=(\"\${SRC_URDF_DIR}\"/*.urdf)
    if [ \"\${#urdfs[@]}\" -gt 0 ]; then
      cp -a \"\${urdfs[@]}\" /workspace/out/urdf_generated/
    fi
    shopt -u nullglob
  fi

  NSGA_ARG=\"\"
  if [ -n \"${NSGA_CSV}\" ]; then
    NSGA_ARG=\"--nsga-csv ${NSGA_CSV}\"
  fi

  CFG_ARG=\"\"
  if [ -f /workspace/out/models/general/cfgs.pkl ]; then
    CFG_ARG=\"--cfg-path /workspace/out/models/general/cfgs.pkl\"
  fi

  python /workspace/bind/src/general_policy/adapt_and_eval_gen.py \
    --general-policy /workspace/out/models/general/${GENERAL_MODEL_BASE} \
    \${CFG_ARG} \
    --catalog-dir /workspace/out/urdf_generated \
    --n-urdf ${N_URDF} \
    --urdf-seed ${URDF_SEED} \
    --exp ${ADAPT_EXP} \
    --csv evaluation_results.csv \
    --eval-envs ${EVAL_ENVS} \
    --vmin ${VMIN} \
    --vmax ${VMAX} \
    --adapt-envs ${ADAPT_ENVS} \
    --adapt-iters ${ADAPT_ITERS} \
    --adapt-repeats ${ADAPT_REPEATS} \
    --device ${DEVICE} \
    \${NSGA_ARG}
"
