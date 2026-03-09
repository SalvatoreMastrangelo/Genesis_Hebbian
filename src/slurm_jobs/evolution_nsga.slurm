#!/bin/bash -l
#SBATCH --job-name=nsga
#SBATCH --partition=l40s
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=5:00:00
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
POP_SIZE=${POP_SIZE:-4}
NUM_GEN=${NUM_GEN:-4}
TRAIN_ITERS=${TRAIN_ITERS:-850}
TRAIN_REPETITION=${TRAIN_REPETITION:-2}
RUN_TAG=${RUN_TAG:-nsga_${SLURM_JOB_ID}}
RUN_NAME=${RUN_NAME:-${RUN_TAG}}
DEVICE=${DEVICE:-cuda:0}
GA_PARALLEL=${GA_PARALLEL:-1}
GEN_POLICY=${GEN_POLICY:-0}
POLICY_PATH=${POLICY_PATH:-}

DEFAULT_POLICY_CLUSTER="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2502405/logs/ea/foundation-mixture/model_2000.pt"
if [ "${GEN_POLICY}" != "0" ] && [ -z "${POLICY_PATH}" ]; then
  if [ -f "${DEFAULT_POLICY_CLUSTER}" ]; then
    POLICY_PATH="${DEFAULT_POLICY_CLUSTER}"
  fi
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
# Limit thread over-subscription per worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export RAY_DISABLE_METRICS=1

# Try to raise thread/file limits; don't fail if the cluster blocks it.
if ! ulimit -u 65535; then
  echo "[WARN] Could not raise max user processes (ulimit -u)."
fi
if ! ulimit -n 65535; then
  echo "[WARN] Could not raise max open files (ulimit -n)."
fi
ulimit -a

JOB_NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
TOTAL_GPUS=$((JOB_NODES * GPUS_PER_NODE))
MAX_GPUS=${MAX_GPUS:-40}
if [ "$TOTAL_GPUS" -gt "$MAX_GPUS" ]; then
  TOTAL_GPUS="$MAX_GPUS"
fi
RAY_TEMP_DIR=${SLURM_TMPDIR:-/tmp/${SLURM_JOB_ID}}
RAY_TMPDIR="${RAY_TEMP_DIR}/ray"

head_node=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
head_ip=$(getent hosts "$head_node" | awk '{print $1}' | head -n1)
if [ -z "$head_ip" ]; then
  echo "[ERROR] Could not resolve head node IP for $head_node"
  exit 1
fi
RAY_PORT=${RAY_PORT:-$((6379 + (SLURM_JOB_ID % 20000)))}
RAY_HEAD_ADDR="${head_ip}:${RAY_PORT}"

echo "[ray] head_node=${head_node} head_ip=${head_ip} gpus_per_node=${GPUS_PER_NODE} total_gpus=${TOTAL_GPUS}"

# Start Ray head/worker and driver within a single Slurm step so processes stay alive
srun --nodes=${JOB_NODES} --ntasks=${JOB_NODES} --ntasks-per-node=1 apptainer exec --nv $BINDS "$IMG" bash -lc "
  set -euo pipefail

  # Ensure thread limits inside the container too.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export MALLOC_ARENA_MAX=2
  export RAY_DISABLE_METRICS=1
  export MUJOCO_GL=${MUJOCO_GL:-egl}
  export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
  export GLFW_USE_EGL=${GLFW_USE_EGL:-1}

  if ! ulimit -u 65535; then
    echo \"[WARN] Could not raise max user processes (ulimit -u) inside container.\"
  fi
  if ! ulimit -n 65535; then
    echo \"[WARN] Could not raise max open files (ulimit -n) inside container.\"
  fi
  ulimit -a

  # Make the code importable
  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}

  # Keep caches on output folder
  NODE_TMP_DIR=${RAY_TEMP_DIR}/\${SLURMD_NODENAME}
  export XDG_CACHE_HOME=\${NODE_TMP_DIR}/cache
  export TI_CACHE_DIR=\${NODE_TMP_DIR}/taichi
  export TAICHI_CACHE_DIR=\${NODE_TMP_DIR}/taichi
  export GSTAICHI_CACHE_DIR=\${NODE_TMP_DIR}/gstaichi
  export MPLCONFIGDIR=\${NODE_TMP_DIR}/mpl
  export RAY_TMPDIR=\${NODE_TMP_DIR}/ray
  export GA_PARALLEL=\"${GA_PARALLEL}\"
  export RAY_ADDRESS=\"${RAY_HEAD_ADDR}\"
  export RAY_LOG_TO_DRIVER=0
  export RAY_MAX_CALLS=${RAY_MAX_CALLS:-0}
  export WORKER_START_JITTER=${WORKER_START_JITTER:-2}
  export PYTHONFAULTHANDLER=0
  export TORCH_SHOW_CPP_STACKTRACES=0
  export URDF_DIR=\${NODE_TMP_DIR}/urdf_generated
  export URDF_GS_INIT=0
  export DEBUG_URDF=1
  export URDF_MIRROR_DIR=/workspace/out/${RUN_NAME}/urdf_generated

  mkdir -p \"\${URDF_DIR}/meshes\"

  # Ensure meshes and aero configs exist for generated URDFs on each node
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  cp -a \"\${SRC_URDF_DIR}/meshes/.\" \"\${URDF_DIR}/meshes/\"
  for fname in aero_parameters.yaml actuators.csv; do
    if [ ! -f \"\${SRC_AERO_DIR}/\${fname}\" ]; then
      echo \"[ERROR] Missing \${SRC_AERO_DIR}/\${fname}\"
      exit 1
    fi
    cp -a \"\${SRC_AERO_DIR}/\${fname}\" \"\${URDF_DIR}/\"
  done

  SELF_HOST=\$(hostname -s)
  if [ \"\${SELF_HOST}\" = \"${head_node}\" ]; then
    ray stop --force >/dev/null 2>&1 || true
    rm -rf \"\${RAY_TMPDIR}\" || true
    ray start --head --node-ip-address ${head_ip} --port ${RAY_PORT} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --temp-dir \${NODE_TMP_DIR}
    sleep 5

    GEN_POLICY_ARGS=\"--gen_policy ${GEN_POLICY}\"
    if [ \"${GEN_POLICY}\" != \"0\" ] && [ -n \"${POLICY_PATH_INNER}\" ]; then
      GEN_POLICY_ARGS=\"\$GEN_POLICY_ARGS --policy_path ${POLICY_PATH_INNER}\"
    fi

    mkdir -p /workspace/out/${RUN_NAME}/logs /workspace/out/${RUN_NAME}/analysis \
      /workspace/out/${RUN_NAME}/urdf_generated /workspace/out/${RUN_NAME}/.cache
    cd /workspace/out/${RUN_NAME}

    python /workspace/bind/src/morph_evolution/evolution_nsga.py \
      --pop \"${POP_SIZE}\" \
      --gen \"${NUM_GEN}\" \
      --train_it \"${TRAIN_ITERS}\" \
      --train_repetition \"${TRAIN_REPETITION}\" \
      --run_name \"${RUN_NAME}\" \
      --base_dir /workspace/out \
      --device \"${DEVICE}\" \
      \$GEN_POLICY_ARGS
  else
    ray stop --force >/dev/null 2>&1 || true
    rm -rf \"\${RAY_TMPDIR}\" || true
    sleep 8
    ray start --address ${RAY_HEAD_ADDR} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --block
  fi
"

# Cleanup Ray on all nodes
srun --nodes=${JOB_NODES} --ntasks=${JOB_NODES} apptainer exec --nv $BINDS "$IMG" bash -lc "
  ray stop --force >/dev/null 2>&1 || true
" || true
