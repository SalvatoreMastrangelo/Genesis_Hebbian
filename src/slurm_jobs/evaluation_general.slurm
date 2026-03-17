#!/bin/bash -l
#SBATCH --job-name=gen_eval
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=5
#SBATCH --cpus-per-gpu=8
#SBATCH --time=1:00:00
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err

set -euo pipefail

# Increase process/thread limits for Ray workers (cluster policies may cap these)
if ! ulimit -u 65535; then
  echo "[WARN] Could not raise max user processes (ulimit -u)."
fi
ulimit -n 65535 || true
export RAY_DISABLE_METRICS=1
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
EVAL_BASELINES=${EVAL_BASELINES:-0}

EVAL_ENVS=${EVAL_ENVS:-8192}
VMIN=${VMIN:-5.0}
VMAX=${VMAX:-25.0}

TRAIN_ENVS=${TRAIN_ENVS:-16384}
TRAIN_ITERS=${TRAIN_ITERS:-850}
TRAIN_REPEATS=${TRAIN_REPEATS:-5}
DEVICE=${DEVICE:-cuda}
GA_PARALLEL=${GA_PARALLEL:-auto}
RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER:-1}
RAY_ADDRESS=${RAY_ADDRESS:-}
RAY_MAX_CALLS=${RAY_MAX_CALLS:-0}
RAY_TMPDIR=${RAY_TMPDIR:-}
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}
WORKER_START_JITTER=${WORKER_START_JITTER:-2}

N_URDF=${N_URDF:-50}
URDF_SEED=${URDF_SEED:-1}
MODEL_CKPT=${MODEL_CKPT:-900}
NSGA_CSV=${NSGA_CSV:-}

# Optional: provide baseline logs dir explicitly or via FOUNDATION_RUN_DIR
FOUNDATION_RUN_DIR=${FOUNDATION_RUN_DIR:-}
BASELINE_LOG_DIR=${BASELINE_LOG_DIR:-}

# Optional: model paths (used to infer baseline log dir and foundation exp)
MODEL_PATH1=${MODEL_PATH1:-}
MODEL_PATH2=${MODEL_PATH2:-}
MODEL_PATH3=${MODEL_PATH3:-}
MODEL_PATH4=${MODEL_PATH4:-}
MODEL_PATH5=${MODEL_PATH5:-}
MODEL_PATH6=${MODEL_PATH6:-}

DEFAULT_MODEL_CLUSTER_1="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2638135/logs/ea/foundation-mixture/model_900.pt"
DEFAULT_MODEL_CLUSTER_2="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2590414/logs/ea/foundation-mixture/model_2100.pt"
DEFAULT_MODEL_CLUSTER_3="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2590415/logs/ea/foundation-mixture/model_2100.pt"
DEFAULT_MODEL_CLUSTER_4="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2579212/logs/ea/foundation-mixture/model_1600.pt"
DEFAULT_MODEL_CLUSTER_5="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2579233/logs/ea/foundation-mixture/model_1600.pt"
DEFAULT_MODEL_CLUSTER_6="/home/avicari/binding/genesis_runs/training_general/foundation-mixture_2579207/logs/ea/foundation-mixture/model_1600.pt"

if [ -z "${MODEL_PATH1}" ]; then MODEL_PATH1="${DEFAULT_MODEL_CLUSTER_1}"; fi
if [ -z "${MODEL_PATH2}" ]; then MODEL_PATH2="${DEFAULT_MODEL_CLUSTER_2}"; fi
if [ -z "${MODEL_PATH3}" ]; then MODEL_PATH3="${DEFAULT_MODEL_CLUSTER_3}"; fi
if [ -z "${MODEL_PATH4}" ]; then MODEL_PATH4="${DEFAULT_MODEL_CLUSTER_4}"; fi
if [ -z "${MODEL_PATH5}" ]; then MODEL_PATH5="${DEFAULT_MODEL_CLUSTER_5}"; fi
if [ -z "${MODEL_PATH6}" ]; then MODEL_PATH6="${DEFAULT_MODEL_CLUSTER_6}"; fi

FIRST_MODEL_PATH=""
for p in "$MODEL_PATH1" "$MODEL_PATH2" "$MODEL_PATH3" "$MODEL_PATH4" "$MODEL_PATH5" "$MODEL_PATH6"; do
  if [ -n "$p" ]; then
    FIRST_MODEL_PATH="$p"
    break
  fi
done

if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${FOUNDATION_EXP}" ] && [ -n "${FIRST_MODEL_PATH}" ]; then
  FOUNDATION_EXP="$(basename "$(dirname "$FIRST_MODEL_PATH")")"
fi

if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${BASELINE_LOG_DIR}" ] && [ -n "${FOUNDATION_RUN_DIR}" ] && [ -n "${FOUNDATION_EXP}" ]; then
  BASELINE_LOG_DIR="${FOUNDATION_RUN_DIR}/logs/ea/${FOUNDATION_EXP}"
fi

if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${BASELINE_LOG_DIR}" ] && [ -n "${FIRST_MODEL_PATH}" ]; then
  BASELINE_LOG_DIR="$(dirname "$FIRST_MODEL_PATH")"
fi

if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${FOUNDATION_EXP}" ] && [ -n "${BASELINE_LOG_DIR}" ]; then
  FOUNDATION_EXP="$(basename "$BASELINE_LOG_DIR")"
fi

MODEL_BASE1=""
MODEL_BASE2=""
MODEL_BASE3=""
MODEL_BASE4=""
MODEL_BASE5=""
MODEL_BASE6=""
if [ -n "$MODEL_PATH1" ]; then MODEL_BASE1="$(basename "$MODEL_PATH1")"; fi
if [ -n "$MODEL_PATH2" ]; then MODEL_BASE2="$(basename "$MODEL_PATH2")"; fi
if [ -n "$MODEL_PATH3" ]; then MODEL_BASE3="$(basename "$MODEL_PATH3")"; fi
if [ -n "$MODEL_PATH4" ]; then MODEL_BASE4="$(basename "$MODEL_PATH4")"; fi
if [ -n "$MODEL_PATH5" ]; then MODEL_BASE5="$(basename "$MODEL_PATH5")"; fi
if [ -n "$MODEL_PATH6" ]; then MODEL_BASE6="$(basename "$MODEL_PATH6")"; fi

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
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${FOUNDATION_EXP}" ]; then
  echo "[ERROR] FOUNDATION_EXP is required (logs/ea/<FOUNDATION_EXP>)"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -z "${BASELINE_LOG_DIR}" ]; then
  echo "[ERROR] BASELINE_LOG_DIR is required (path to logs/ea/<FOUNDATION_EXP>)"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ ! -d "$BASELINE_LOG_DIR" ]; then
  echo "[ERROR] BASELINE_LOG_DIR not found: $BASELINE_LOG_DIR"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH1}" ] && [ ! -f "${MODEL_PATH1}" ]; then
  echo "[ERROR] MODEL_PATH1 not found: ${MODEL_PATH1}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH2}" ] && [ ! -f "${MODEL_PATH2}" ]; then
  echo "[ERROR] MODEL_PATH2 not found: ${MODEL_PATH2}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH3}" ] && [ ! -f "${MODEL_PATH3}" ]; then
  echo "[ERROR] MODEL_PATH3 not found: ${MODEL_PATH3}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH4}" ] && [ ! -f "${MODEL_PATH4}" ]; then
  echo "[ERROR] MODEL_PATH4 not found: ${MODEL_PATH4}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH5}" ] && [ ! -f "${MODEL_PATH5}" ]; then
  echo "[ERROR] MODEL_PATH5 not found: ${MODEL_PATH5}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH6}" ] && [ ! -f "${MODEL_PATH6}" ]; then
  echo "[ERROR] MODEL_PATH6 not found: ${MODEL_PATH6}"
  exit 1
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${FIRST_MODEL_PATH}" ]; then
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
MODEL_STAGE6=""
stage_model_with_cfg() {
  local src_model="$1"
  local dst_model="$2"
  if [ -z "$src_model" ]; then
    return 0
  fi

  local dst_dir
  dst_dir="$(dirname "$dst_model")"
  mkdir -p "$dst_dir"
  cp -a "$src_model" "$dst_model"

  local src_dir cfg_src
  src_dir="$(dirname "$src_model")"
  cfg_src=""
  if [ -f "${src_dir}/cfgs.pkl" ]; then
    cfg_src="${src_dir}/cfgs.pkl"
  elif [ -f "${src_dir}/../cfgs.pkl" ]; then
    cfg_src="${src_dir}/../cfgs.pkl"
  fi

  if [ -n "$cfg_src" ]; then
    cp -a "$cfg_src" "${dst_dir}/cfgs.pkl"
  else
    echo "[WARN] cfgs.pkl not found near ${src_model} (tried ${src_dir}/cfgs.pkl and ${src_dir}/../cfgs.pkl)"
  fi
}
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH1}" ]; then
  MODEL_STAGE1="${MODEL_STAGE_DIR}/m1/${MODEL_BASE1}"
  stage_model_with_cfg "${MODEL_PATH1}" "${MODEL_STAGE1}"
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH2}" ]; then
  MODEL_STAGE2="${MODEL_STAGE_DIR}/m2/${MODEL_BASE2}"
  stage_model_with_cfg "${MODEL_PATH2}" "${MODEL_STAGE2}"
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH3}" ]; then
  MODEL_STAGE3="${MODEL_STAGE_DIR}/m3/${MODEL_BASE3}"
  stage_model_with_cfg "${MODEL_PATH3}" "${MODEL_STAGE3}"
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH4}" ]; then
  MODEL_STAGE4="${MODEL_STAGE_DIR}/m4/${MODEL_BASE4}"
  stage_model_with_cfg "${MODEL_PATH4}" "${MODEL_STAGE4}"
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH5}" ]; then
  MODEL_STAGE5="${MODEL_STAGE_DIR}/m5/${MODEL_BASE5}"
  stage_model_with_cfg "${MODEL_PATH5}" "${MODEL_STAGE5}"
fi
if [ "${EVAL_BASELINES}" -ne 0 ] && [ -n "${MODEL_PATH6}" ]; then
  MODEL_STAGE6="${MODEL_STAGE_DIR}/m6/${MODEL_BASE6}"
  stage_model_with_cfg "${MODEL_PATH6}" "${MODEL_STAGE6}"
fi

# ------------------------------------------------------------
# Bind mounts
# - Repo in RO
# - Output in RW
# - Baseline logs in RO (we copy them into the run dir)
# ------------------------------------------------------------
if [ "${EVAL_BASELINES}" -ne 0 ]; then
  BINDS="--bind ${SRC}:/workspace/bind:ro,${SRC}:/workspace/Genesis:ro,${RUN_DIR}:/workspace/out:rw,${BASELINE_LOG_DIR}:/workspace/baseline:ro"
else
  BINDS="--bind ${SRC}:/workspace/bind:ro,${SRC}:/workspace/Genesis:ro,${RUN_DIR}:/workspace/out:rw"
fi

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

JOB_NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
RAY_TEMP_DIR=${SLURM_TMPDIR:-/tmp/${SLURM_JOB_ID}}

head_node=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
head_ip=$(getent hosts "$head_node" | awk '{print $1}' | head -n1)
if [ -z "$head_ip" ]; then
  echo "[ERROR] Could not resolve head node IP for $head_node"
  exit 1
fi
RAY_PORT=${RAY_PORT:-$((6379 + (SLURM_JOB_ID % 20000)))}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-30000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-39999}
RAY_HEAD_ADDR="${head_ip}:${RAY_PORT}"

echo "[ray] head_node=${head_node} head_ip=${head_ip} gpus_per_node=${GPUS_PER_NODE}"

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

  # Make the code importable
  export PYTHONPATH=/workspace/bind/src:/workspace/bind:\${PYTHONPATH:-}

  if ! ulimit -u 65535; then
    echo \"[WARN] Could not raise max user processes (ulimit -u) inside container.\"
  fi
  if ! ulimit -n 65535; then
    echo \"[WARN] Could not raise max open files (ulimit -n) inside container.\"
  fi

  # Keep caches on per-node tmp to avoid contention
  NODE_TMP_DIR=${RAY_TEMP_DIR}/\${SLURMD_NODENAME}
  mkdir -p \"\${NODE_TMP_DIR}\"
  export XDG_CACHE_HOME=\${NODE_TMP_DIR}/cache
  export TI_CACHE_DIR=\${NODE_TMP_DIR}/taichi
  export TAICHI_CACHE_DIR=\${NODE_TMP_DIR}/taichi
  export GSTAICHI_CACHE_DIR=\${NODE_TMP_DIR}/gstaichi
  export MPLCONFIGDIR=\${NODE_TMP_DIR}/mpl
  export GA_PARALLEL=${GA_PARALLEL}
  export RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER}
  export RAY_ADDRESS=\"${RAY_HEAD_ADDR}\"
  export RAY_MAX_CALLS=${RAY_MAX_CALLS}
  export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO}
  export RAY_TMPDIR=\${NODE_TMP_DIR}/ray
  export WORKER_START_JITTER=${WORKER_START_JITTER}
  export PYTHONFAULTHANDLER=0
  export TORCH_SHOW_CPP_STACKTRACES=0

  mkdir -p /workspace/out/logs /workspace/out/analysis /workspace/out/urdf_generated /workspace/out/.cache
  cd /workspace/out

  # Ensure meshes exist for generated URDFs (required by package://meshes)
  SRC_URDF_DIR=/workspace/bind/src/urdf_generated
  SRC_AERO_DIR=/workspace/bind/genesis/assets/urdf/mydrone
  if [ ! -d \"\${SRC_URDF_DIR}/meshes\" ]; then
    echo \"[ERROR] Missing meshes in \${SRC_URDF_DIR}/meshes\"
    exit 1
  fi
  # Copy meshes once with a lock to avoid races across tasks
  MESH_SENTINEL=/workspace/out/urdf_generated/.meshes_copied
  MESH_LOCK=/workspace/out/urdf_generated/.meshes_copy_lock
  if [ ! -f \"\${MESH_SENTINEL}\" ]; then
    if mkdir \"\${MESH_LOCK}\" 2>/dev/null; then
      cp -a \"\${SRC_URDF_DIR}/meshes/.\" /workspace/out/urdf_generated/meshes/
      touch \"\${MESH_SENTINEL}\"
      rmdir \"\${MESH_LOCK}\"
    else
      # Another task is copying; wait for sentinel
      while [ ! -f \"\${MESH_SENTINEL}\" ]; do
        sleep 1
      done
    fi
  fi

  # Aero config files required next to the URDFs
  for fname in aero_parameters.yaml actuators.csv; do
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
  if [ \"${EVAL_BASELINES}\" -ne 0 ]; then
    mkdir -p /workspace/out/logs/ea/${FOUNDATION_EXP}
    if [ ! -f /workspace/baseline/cfgs.pkl ]; then
      echo \"[ERROR] cfgs.pkl not found in baseline logs: /workspace/baseline\"
      exit 1
    fi
    cp -a /workspace/baseline/. /workspace/out/logs/ea/${FOUNDATION_EXP}/
  fi

  # Model args: either explicit MODEL_PATH* or a single MODEL_CKPT
  MODEL_ARGS=\"\"
  if [ \"${EVAL_BASELINES}\" -ne 0 ]; then
    if [ -n \"${MODEL_PATH1:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path1 /workspace/out/models/m1/${MODEL_BASE1}\"; fi
    #if [ -n \"${MODEL_PATH2:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path2 /workspace/out/models/m2/${MODEL_BASE2}\"; fi
    #if [ -n \"${MODEL_PATH3:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path3 /workspace/out/models/m3/${MODEL_BASE3}\"; fi
    #if [ -n \"${MODEL_PATH4:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path4 /workspace/out/models/m4/${MODEL_BASE4}\"; fi
    #if [ -n \"${MODEL_PATH5:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path5 /workspace/out/models/m5/${MODEL_BASE5}\"; fi
    #if [ -n \"${MODEL_PATH6:-}\" ]; then MODEL_ARGS=\"\$MODEL_ARGS --model-path6 /workspace/out/models/m6/${MODEL_BASE6}\"; fi

    if [ -z \"\$MODEL_ARGS\" ]; then
      if [ -z \"${MODEL_CKPT:-}\" ]; then
        echo \"[ERROR] Provide MODEL_PATH1..MODEL_PATH6 or MODEL_CKPT\"
        exit 1
      fi
      MODEL_ARGS=\"--model-path1 /workspace/out/logs/ea/${FOUNDATION_EXP}/model_${MODEL_CKPT}.pt\"
    fi
  fi

  # Run evaluation pipeline
  NSGA_ARG=\"\"
  if [ -n \"${NSGA_CSV}\" ]; then
    NSGA_ARG=\"--nsga-csv ${NSGA_CSV}\"
  fi

  FOUNDATION_ARG=\"\"
  if [ \"${EVAL_BASELINES}\" -ne 0 ]; then
    FOUNDATION_ARG=\"--foundation-exp ${FOUNDATION_EXP}\"
  fi

  SELF_HOST=\$(hostname -s)
  if [ \"\${SELF_HOST}\" = \"${head_node}\" ]; then
    ray stop --force >/dev/null 2>&1 || true
    rm -rf \"\${RAY_TMPDIR}\" || true
    ray start --head --node-ip-address ${head_ip} --port ${RAY_PORT} --min-worker-port ${RAY_MIN_WORKER_PORT} --max-worker-port ${RAY_MAX_WORKER_PORT} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --temp-dir \${NODE_TMP_DIR}
    sleep 5

    python /workspace/bind/src/general_policy/eval_gen.py \
      --catalog-dir /workspace/out/urdf_generated \
      --n-urdf ${N_URDF} \
      --urdf-seed ${URDF_SEED} \
      --exp ${EVAL_EXP} \
      --csv evaluation_results.csv \
      --eval-envs ${EVAL_ENVS} \
      --vmin ${VMIN} \
      --vmax ${VMAX} \
      --train-envs ${TRAIN_ENVS} \
      --train-iters ${TRAIN_ITERS} \
      --train-repeats ${TRAIN_REPEATS} \
      --device ${DEVICE} \
      --eval-baselines ${EVAL_BASELINES} \
      \${FOUNDATION_ARG} \
      \$MODEL_ARGS \
      \${NSGA_ARG}
  else
    ray stop --force >/dev/null 2>&1 || true
    rm -rf \"\${RAY_TMPDIR}\" || true
    sleep 8
    ray start --address ${RAY_HEAD_ADDR} --min-worker-port ${RAY_MIN_WORKER_PORT} --max-worker-port ${RAY_MAX_WORKER_PORT} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --block
  fi
"

# Cleanup Ray on all nodes
srun --nodes=${JOB_NODES} --ntasks=${JOB_NODES} apptainer exec --nv $BINDS "$IMG" bash -lc "
  ray stop --force >/dev/null 2>&1 || true
" || true
