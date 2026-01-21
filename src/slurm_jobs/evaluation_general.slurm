#!/bin/bash -l
#SBATCH --job-name=gen_eval
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=10:00:00
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
FOUNDATION_EXP=${FOUNDATION_EXP:-}
EVAL_EXP=${EVAL_EXP:-eval_general_${SLURM_JOB_ID}}
RUN_TAG=${RUN_TAG:-${EVAL_EXP}}

EVAL_ENVS=${EVAL_ENVS:-4096}
VMIN=${VMIN:-6.0}
VMAX=${VMAX:-24.0}

TRAIN_ENVS=${TRAIN_ENVS:-32768}
TRAIN_ITERS=${TRAIN_ITERS:-700}
TRAIN_REPEATS=${TRAIN_REPEATS:-5}
DEVICE=${DEVICE:-cuda}

N_URDF=${N_URDF:-10}
URDF_SEED=${URDF_SEED:-0}
MODEL_CKPT=${MODEL_CKPT:-2000}

# Optional: provide baseline logs dir explicitly or via FOUNDATION_RUN_DIR
FOUNDATION_RUN_DIR=${FOUNDATION_RUN_DIR:-}
BASELINE_LOG_DIR=${BASELINE_LOG_DIR:-}

# Optional: model paths (used to infer baseline log dir and foundation exp)
MODEL_PATH1=${MODEL_PATH1:-}
MODEL_PATH2=${MODEL_PATH2:-}
MODEL_PATH3=${MODEL_PATH3:-}
MODEL_PATH4=${MODEL_PATH4:-}
MODEL_PATH5=${MODEL_PATH5:-}

DEFAULT_MODEL_CLUSTER_1="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502403/logs/ea/foundation-mixture/model_2000.pt"
DEFAULT_MODEL_CLUSTER_2="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502405/logs/ea/foundation-mixture/model_2000.pt"
DEFAULT_MODEL_CLUSTER_3="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502407/logs/ea/foundation-mixture/model_2000.pt"
DEFAULT_MODEL_CLUSTER_4="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502410/logs/ea/foundation-mixture/model_2000.pt"
DEFAULT_MODEL_CLUSTER_5="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502412/logs/ea/foundation-mixture/model_2000.pt"

if [ -z "${MODEL_PATH1}" ]; then MODEL_PATH1="${DEFAULT_MODEL_CLUSTER_1}"; fi
if [ -z "${MODEL_PATH2}" ]; then MODEL_PATH2="${DEFAULT_MODEL_CLUSTER_2}"; fi
if [ -z "${MODEL_PATH3}" ]; then MODEL_PATH3="${DEFAULT_MODEL_CLUSTER_3}"; fi
if [ -z "${MODEL_PATH4}" ]; then MODEL_PATH4="${DEFAULT_MODEL_CLUSTER_4}"; fi
if [ -z "${MODEL_PATH5}" ]; then MODEL_PATH5="${DEFAULT_MODEL_CLUSTER_5}"; fi

FIRST_MODEL_PATH=""
for p in "$MODEL_PATH1" "$MODEL_PATH2" "$MODEL_PATH3" "$MODEL_PATH4" "$MODEL_PATH5"; do
  if [ -n "$p" ]; then
    FIRST_MODEL_PATH="$p"
    break
  fi
done

if [ -z "${FOUNDATION_EXP}" ] && [ -n "${FIRST_MODEL_PATH}" ]; then
  FOUNDATION_EXP="$(basename "$(dirname "$FIRST_MODEL_PATH")")"
fi

if [ -z "${BASELINE_LOG_DIR}" ] && [ -n "${FOUNDATION_RUN_DIR}" ] && [ -n "${FOUNDATION_EXP}" ]; then
  BASELINE_LOG_DIR="${FOUNDATION_RUN_DIR}/logs/ea/${FOUNDATION_EXP}"
fi

if [ -z "${BASELINE_LOG_DIR}" ] && [ -n "${FIRST_MODEL_PATH}" ]; then
  BASELINE_LOG_DIR="$(dirname "$FIRST_MODEL_PATH")"
fi

if [ -z "${FOUNDATION_EXP}" ] && [ -n "${BASELINE_LOG_DIR}" ]; then
  FOUNDATION_EXP="$(basename "$BASELINE_LOG_DIR")"
fi

MODEL_BASE1=""
MODEL_BASE2=""
MODEL_BASE3=""
MODEL_BASE4=""
MODEL_BASE5=""
if [ -n "$MODEL_PATH1" ]; then MODEL_BASE1="$(basename "$MODEL_PATH1")"; fi
if [ -n "$MODEL_PATH2" ]; then MODEL_BASE2="$(basename "$MODEL_PATH2")"; fi
if [ -n "$MODEL_PATH3" ]; then MODEL_BASE3="$(basename "$MODEL_PATH3")"; fi
if [ -n "$MODEL_PATH4" ]; then MODEL_BASE4="$(basename "$MODEL_PATH4")"; fi
if [ -n "$MODEL_PATH5" ]; then MODEL_BASE5="$(basename "$MODEL_PATH5")"; fi

# ------------------------------------------------------------
# Organized output layout (single bind target)
# ------------------------------------------------------------
OUT_BASE="$HOME/binding/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/evaluation_general/${RUN_TAG}}"

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
if [ -z "${FOUNDATION_EXP}" ]; then
  echo "[ERROR] FOUNDATION_EXP is required (logs/ea/<FOUNDATION_EXP>)"
  exit 1
fi
if [ -z "${BASELINE_LOG_DIR}" ]; then
  echo "[ERROR] BASELINE_LOG_DIR is required (path to logs/ea/<FOUNDATION_EXP>)"
  exit 1
fi
if [ ! -d "$BASELINE_LOG_DIR" ]; then
  echo "[ERROR] BASELINE_LOG_DIR not found: $BASELINE_LOG_DIR"
  exit 1
fi
if [ -n "${MODEL_PATH1}" ] && [ ! -f "${MODEL_PATH1}" ]; then
  echo "[ERROR] MODEL_PATH1 not found: ${MODEL_PATH1}"
  exit 1
fi
if [ -n "${MODEL_PATH2}" ] && [ ! -f "${MODEL_PATH2}" ]; then
  echo "[ERROR] MODEL_PATH2 not found: ${MODEL_PATH2}"
  exit 1
fi
if [ -n "${MODEL_PATH3}" ] && [ ! -f "${MODEL_PATH3}" ]; then
  echo "[ERROR] MODEL_PATH3 not found: ${MODEL_PATH3}"
  exit 1
fi
if [ -n "${MODEL_PATH4}" ] && [ ! -f "${MODEL_PATH4}" ]; then
  echo "[ERROR] MODEL_PATH4 not found: ${MODEL_PATH4}"
  exit 1
fi
if [ -n "${MODEL_PATH5}" ] && [ ! -f "${MODEL_PATH5}" ]; then
  echo "[ERROR] MODEL_PATH5 not found: ${MODEL_PATH5}"
  exit 1
fi
if [ -n "${FIRST_MODEL_PATH}" ]; then
  MODEL_BASE_CHECK="$(basename "$FIRST_MODEL_PATH")"
  if [ ! -f "${BASELINE_LOG_DIR}/${MODEL_BASE_CHECK}" ]; then
    echo "[ERROR] Model file not found in BASELINE_LOG_DIR: ${BASELINE_LOG_DIR}/${MODEL_BASE_CHECK}"
    exit 1
  fi
fi

# ------------------------------------------------------------
# Create output dirs on host
# ------------------------------------------------------------
mkdir -p "$LOG_DIR" "$URDF_DIR" "$ANALYSIS_DIR" "$CACHE_DIR"
mkdir -p "$URDF_DIR/meshes"

MODEL_STAGE_DIR="${RUN_DIR}/models"
MODEL_STAGE1=""
MODEL_STAGE2=""
MODEL_STAGE3=""
MODEL_STAGE4=""
MODEL_STAGE5=""
if [ -n "${MODEL_PATH1}" ]; then
  MODEL_STAGE1="${MODEL_STAGE_DIR}/m1/${MODEL_BASE1}"
  mkdir -p "$(dirname "$MODEL_STAGE1")"
  cp -a "${MODEL_PATH1}" "${MODEL_STAGE1}"
fi
if [ -n "${MODEL_PATH2}" ]; then
  MODEL_STAGE2="${MODEL_STAGE_DIR}/m2/${MODEL_BASE2}"
  mkdir -p "$(dirname "$MODEL_STAGE2")"
  cp -a "${MODEL_PATH2}" "${MODEL_STAGE2}"
fi
if [ -n "${MODEL_PATH3}" ]; then
  MODEL_STAGE3="${MODEL_STAGE_DIR}/m3/${MODEL_BASE3}"
  mkdir -p "$(dirname "$MODEL_STAGE3")"
  cp -a "${MODEL_PATH3}" "${MODEL_STAGE3}"
fi
if [ -n "${MODEL_PATH4}" ]; then
  MODEL_STAGE4="${MODEL_STAGE_DIR}/m4/${MODEL_BASE4}"
  mkdir -p "$(dirname "$MODEL_STAGE4")"
  cp -a "${MODEL_PATH4}" "${MODEL_STAGE4}"
fi
if [ -n "${MODEL_PATH5}" ]; then
  MODEL_STAGE5="${MODEL_STAGE_DIR}/m5/${MODEL_BASE5}"
  mkdir -p "$(dirname "$MODEL_STAGE5")"
  cp -a "${MODEL_PATH5}" "${MODEL_STAGE5}"
fi

# ------------------------------------------------------------
# Bind mounts
# - Repo in RO
# - Output in RW
# - Baseline logs in RO (we copy them into the run dir)
# ------------------------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:ro,${RUN_DIR}:/workspace/out:rw,${BASELINE_LOG_DIR}:/workspace/baseline:ro"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-1}}"

srun apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  # Make the code importable
  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\$PYTHONPATH

  # Keep caches on output folder
  export XDG_CACHE_HOME=/workspace/out/.cache
  export TI_CACHE_DIR=/workspace/out/.cache/taichi
  export TAICHI_CACHE_DIR=/workspace/out/.cache/taichi
  export GSTAICHI_CACHE_DIR=/workspace/out/.cache/gstaichi
  export MPLCONFIGDIR=/workspace/out/.cache/mpl

  mkdir -p /workspace/out/logs /workspace/out/analysis /workspace/out/urdf_generated /workspace/out/.cache
  cd /workspace/out

  # Ensure /workspace/Genesis exists for eval_gen mesh copy
  if [ ! -e /workspace/Genesis ]; then
    ln -s /workspace/bind /workspace/Genesis
  fi

  # Ensure meshes exist for generated URDFs (required by package://meshes)
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/

  # Aero config + model files required next to the URDFs
  for fname in aero_parameters.yaml actuators.csv drone.py; do
    if [ ! -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      echo \"[ERROR] Missing \${SRC_AERO_DIR}/\${fname}\"
      exit 1
    fi
    cp -a \"\${SRC_AERO_DIR}/\${fname}\" /workspace/out/urdf_generated/
  done

  # If not generating new URDFs, seed the catalog from repo defaults
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

  # Copy baseline logs into the run dir (so eval outputs stay organized)
  mkdir -p /workspace/out/logs/ea/${FOUNDATION_EXP}
  if [ ! -f /workspace/baseline/cfgs.pkl ]; then
    echo \"[ERROR] cfgs.pkl not found in baseline logs: /workspace/baseline\"
    exit 1
  fi
  cp -a /workspace/baseline/. /workspace/out/logs/ea/${FOUNDATION_EXP}/

  # Model args: either explicit MODEL_PATH* or a single MODEL_CKPT
  MODEL_ARGS=\"\"
  if [ -n \"${MODEL_PATH1:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path1 /workspace/out/models/m1/${MODEL_BASE1}\"; fi
  if [ -n \"${MODEL_PATH2:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path2 /workspace/out/models/m2/${MODEL_BASE2}\"; fi
  if [ -n \"${MODEL_PATH3:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path3 /workspace/out/models/m3/${MODEL_BASE3}\"; fi
  if [ -n \"${MODEL_PATH4:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path4 /workspace/out/models/m4/${MODEL_BASE4}\"; fi
  if [ -n \"${MODEL_PATH5:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path5 /workspace/out/models/m5/${MODEL_BASE5}\"; fi

  if [ -z \"\$MODEL_ARGS\" ]; then
    if [ -z \"${MODEL_CKPT:-}\" ]; then
      echo \"[ERROR] Provide MODEL_PATH1..MODEL_PATH5 or MODEL_CKPT\"
      exit 1
    fi
    MODEL_ARGS=\"--model-path1 /workspace/out/logs/ea/${FOUNDATION_EXP}/model_${MODEL_CKPT}.pt\"
  fi

  # Run evaluation pipeline
  python /workspace/bind/src/general_policy/eval_gen.py \
    --catalog-dir /workspace/out/urdf_generated \
    --n-urdf ${N_URDF} \
    --seed ${URDF_SEED} \
    --exp ${EVAL_EXP} \
    --foundation-exp ${FOUNDATION_EXP} \
    --csv /workspace/out/analysis/foundation_eval_lean \
    --eval-envs ${EVAL_ENVS} \
    --vmin ${VMIN} \
    --vmax ${VMAX} \
    --train-envs ${TRAIN_ENVS} \
    --train-iters ${TRAIN_ITERS} \
    --train-repeats ${TRAIN_REPEATS} \
    --device ${DEVICE} \
    \$MODEL_ARGS
"
