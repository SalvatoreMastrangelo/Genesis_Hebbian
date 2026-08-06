#!/bin/bash -l
#SBATCH --job-name=outer_nsga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=144:00:00
# To use multiple GPUs for ParallelMultiSceneEvalEnv, override --gpus on
# submission (e.g. `bash run_parallel.sh ... --gpus=2`). The Python code
# auto-detects via torch.cuda.device_count() when evaluation.num_gpus is 0,
# so no other change is needed. CPU count and OMP threads scale with --gpus
# automatically via the SLURM_CPUS_PER_GPU block below.
#SBATCH --output=/home/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/%u/slurm_logs/%x-%j.err
#SBATCH --mail-user=salvatore.mastrangelo@epfl.ch
#SBATCH --mail-type=END

set -euo pipefail

# ------------------------------------------------------------
# Paths (adjust to your cluster layout)
# ------------------------------------------------------------
# The container image is searched in order; override with GENESIS_SIF=/path.
# NOTE: /scratch is purged by SCITAS on an untouched-file policy — an image
# left there goes away during any quiet spell between batches (it did on
# 2026-08-04, taking down the whole 2026-08-05 submission). Keep a copy
# somewhere durable and the fallback below will find it.
SIF_CANDIDATES=(
  "/scratch/izar/$USER/sif/mygenesis.sif"
  "$HOME/sif/mygenesis.sif"
)
SRC="$HOME/Genesis-dev"

# ------------------------------------------------------------
# Run params (override via sbatch --export=...)
# ------------------------------------------------------------
# EXP_REL: path to the experiment folder relative to src/WP2_Outer_Loop
#   e.g.  experiments/batch_0/my_first_experiment
# RUN_CFG_NAME / CKPT_NAME / WP1_CFG_NAME: filenames inside EXP_REL
#   run.yaml is a SINGLE-FILE OuterNSGA2Config (inner YAML + outer: section)
EXP_REL=${EXP_REL:-experiments/default}
RUN_CFG_NAME=${RUN_CFG_NAME:-run.yaml}
CKPT_NAME=${CKPT_NAME:-checkpoint.pt}
WP1_CFG_NAME=${WP1_CFG_NAME:-wp1_config.yaml}
RUN_TAG=${RUN_TAG:-outer_nsga_${SLURM_JOB_ID}}
CLI_OVERRIDES=${CLI_OVERRIDES:-}

# ------------------------------------------------------------
# Output layout
# ------------------------------------------------------------
OUT_BASE="/scratch/izar/$USER/genesis_runs"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/outer_nsga/${RUN_TAG}}"
CACHE_DIR="${RUN_DIR}/.cache"

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# Safety checks
IMG="${GENESIS_SIF:-}"
if [ -n "$IMG" ]; then
  if [ ! -f "$IMG" ]; then
    echo "[ERROR] GENESIS_SIF is set but no image is there: $IMG"
    exit 1
  fi
else
  for candidate in "${SIF_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
      IMG="$candidate"
      break
    fi
  done
fi
if [ -z "$IMG" ]; then
  echo "[ERROR] Genesis container image not found. Looked in:"
  for candidate in "${SIF_CANDIDATES[@]}"; do
    echo "[ERROR]   - $candidate"
  done
  echo "[ERROR]"
  echo "[ERROR] /scratch is purged by SCITAS, so a previously working image"
  echo "[ERROR] disappears after a quiet spell. Rebuild it from the archive:"
  echo "[ERROR]   mkdir -p /scratch/izar/\$USER/sif"
  echo "[ERROR]   apptainer build /scratch/izar/\$USER/sif/mygenesis.sif \\"
  echo "[ERROR]                   docker-archive://\$HOME/tar/mygenesis.tar"
  echo "[ERROR]"
  echo "[ERROR] Or point at an existing image: sbatch --export=ALL,GENESIS_SIF=/path/to.sif ..."
  exit 1
fi
echo "[CONFIG] container image  = ${IMG}"
if [ ! -d "$SRC" ]; then
  echo "[ERROR] Bind source directory not found: $SRC"
  exit 1
fi

HOST_EXP_DIR="${SRC}/src/WP2_Outer_Loop/${EXP_REL}"
if [ ! -d "${HOST_EXP_DIR}" ]; then
  echo "[ERROR] Experiment folder not found on host: ${HOST_EXP_DIR}"
  exit 1
fi
for fname in "${RUN_CFG_NAME}" "${CKPT_NAME}" "${WP1_CFG_NAME}"; do
  if [ ! -f "${HOST_EXP_DIR}/${fname}" ]; then
    echo "[ERROR] Missing ${fname} in ${HOST_EXP_DIR}"
    exit 1
  fi
done

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
# Container-internal paths
# ------------------------------------------------------------
EXP_DIR_IN_CT="/workspace/bind/src/WP2_Outer_Loop/${EXP_REL}"
RUN_CFG_IN_CT="${EXP_DIR_IN_CT}/${RUN_CFG_NAME}"
CKPT_IN_CT="${EXP_DIR_IN_CT}/${CKPT_NAME}"
WP1_CFG_IN_CT="${EXP_DIR_IN_CT}/${WP1_CFG_NAME}"

echo "[CONFIG] EXP_REL          = ${EXP_REL}"
echo "[CONFIG] run.yaml         = ${RUN_CFG_IN_CT}"
echo "[CONFIG] checkpoint       = ${CKPT_IN_CT}"
echo "[CONFIG] wp1_config       = ${WP1_CFG_IN_CT}"
echo "[CONFIG] RUN_TAG          = ${RUN_TAG}"
echo "[CONFIG] CLI_OVERRIDES    = ${CLI_OVERRIDES}"

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

  # Stage URDF meshes + aero assets so WingedDroneEnv can load them
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

  python -m WP2_Outer_Loop.run \
    --cfg ${RUN_CFG_IN_CT} \
    --cfg.checkpoint_path ${CKPT_IN_CT} \
    --cfg.checkpoint_config_path ${WP1_CFG_IN_CT} \
    --cfg.base_dir /workspace/out \
    ${CLI_OVERRIDES}
"

# ------------------------------------------------------------
# Copy run output to home for long-term storage
# ------------------------------------------------------------
# The science is finished by this point, so archiving is best-effort: a full
# quota must not turn a successful run into a Slurm FAILED (it did for jobs
# 3092665 / 3092828). The `||` branch swallows the failure so the job's exit
# status stays tied to the run itself; archive_run.sh prints the manual
# re-run command on failure.
echo "[POST] Archiving run to $HOME/genesis_runs/outer_nsga/ ..."
mkdir -p "$HOME/genesis_runs/outer_nsga"
bash "${SRC}/src/WP2_Outer_Loop/slurm_jobs/archive_run.sh" \
  "${RUN_DIR}" "$HOME/genesis_runs/outer_nsga" \
  || echo "[POST][WARN] Archiving failed — run preserved in ${RUN_DIR} (scratch is purged, copy it out before it ages away)."
