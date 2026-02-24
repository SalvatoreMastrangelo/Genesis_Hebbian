#!/bin/bash -l
#SBATCH --job-name=gen_train
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:50:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err

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
NUM_ENVS=${NUM_ENVS:-32768}
MAX_ITERS=${MAX_ITERS:-4000}
N_URDF=${N_URDF:-64}
URDF_SEED=${URDF_SEED:-0}

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
# Note: cpus-per-gpu=4 => total CPUs per task typically 4 on 1 GPU.
# Use SLURM_CPUS_PER_TASK if set, otherwise fall back to cpus-per-gpu.
# ------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-1}}"

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

  mkdir -p /workspace/out/logs /workspace/out/urdf_generated /workspace/out/.cache
  cd /workspace/out

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

  # IMPORTANT: adjust this path if your script lives under Genesis-dev/src/...
  python /workspace/bind/src/general_policy/train_gen.py \
    --exp-name \"${EXP_NAME}\" \
    --num-envs \"${NUM_ENVS}\" \
    --max-iterations \"${MAX_ITERS}\" \
    --device cuda \
    \$URDF_ARGS
"
