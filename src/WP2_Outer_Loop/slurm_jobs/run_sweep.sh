#!/usr/bin/env bash
# run_sweep.sh — submit every experiment folder in a batch as SEQUENTIAL SLURM jobs.
#
# Jobs are chained with --dependency=afterok so they run one at a time; useful when
# you want predictable resource usage on the cluster.
#
# Folder convention matches run_parallel.sh — each experiment folder contains:
#   run.yaml (single-file OuterNSGA2Config), wp1_config.yaml, <something>.pt
#
# Usage:
#   bash run_sweep.sh [BATCH_DIR] [EXTRA_SBATCH_ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/train.slurm"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BATCH_DIR="${1:-${REPO_ROOT}/src/WP2_Outer_Loop/experiments}"
shift || true

if [ ! -d "${BATCH_DIR}" ]; then
  echo "[ERROR] Batch directory not found: ${BATCH_DIR}"
  exit 1
fi

mapfile -t EXP_DIRS < <(find "${BATCH_DIR}" -mindepth 1 -maxdepth 2 -name "run.yaml" -printf '%h\n' | sort -u)

if [ ${#EXP_DIRS[@]} -eq 0 ]; then
  echo "[WARN] No experiment folders (containing run.yaml) found in ${BATCH_DIR}"
  exit 0
fi

echo "[SWEEP] Found ${#EXP_DIRS[@]} experiment folder(s) in ${BATCH_DIR}"
echo "[SWEEP] Using slurm script: ${SLURM_SCRIPT}"
echo ""

PREV_JOB_ID=""
SUBMITTED=0

for EXP_DIR in "${EXP_DIRS[@]}"; do
  RUN_YAML="${EXP_DIR}/run.yaml"
  EXP_FOLDER_NAME="$(basename "${EXP_DIR}")"

  CKPT_PATH=""
  if [ -f "${EXP_DIR}/checkpoint.pt" ]; then
    CKPT_PATH="${EXP_DIR}/checkpoint.pt"
  else
    mapfile -t PT_FILES < <(find "${EXP_DIR}" -maxdepth 1 -name "*.pt" | sort)
    if [ ${#PT_FILES[@]} -eq 1 ]; then
      CKPT_PATH="${PT_FILES[0]}"
    elif [ ${#PT_FILES[@]} -eq 0 ]; then
      echo "[SKIP] ${EXP_FOLDER_NAME}: no *.pt checkpoint found"
      continue
    else
      echo "[SKIP] ${EXP_FOLDER_NAME}: multiple *.pt files; rename one to checkpoint.pt"
      continue
    fi
  fi
  CKPT_NAME="$(basename "${CKPT_PATH}")"

  WP1_CFG_PATH=""
  if [ -f "${EXP_DIR}/wp1_config.yaml" ]; then
    WP1_CFG_PATH="${EXP_DIR}/wp1_config.yaml"
  else
    mapfile -t YAML_FILES < <(find "${EXP_DIR}" -maxdepth 1 -name "*.yaml" ! -name "run.yaml" | sort)
    if [ ${#YAML_FILES[@]} -eq 1 ]; then
      WP1_CFG_PATH="${YAML_FILES[0]}"
    elif [ ${#YAML_FILES[@]} -eq 0 ]; then
      echo "[SKIP] ${EXP_FOLDER_NAME}: no WP1 config YAML found"
      continue
    else
      echo "[SKIP] ${EXP_FOLDER_NAME}: multiple sibling YAMLs; rename the WP1 config to wp1_config.yaml"
      continue
    fi
  fi
  WP1_CFG_NAME="$(basename "${WP1_CFG_PATH}")"

  EXP_NAME="$(grep -E '^exp_name:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  EXP_NAME="${EXP_NAME:-${EXP_FOLDER_NAME}}"
  REPEAT="$(grep -E '^repeat:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  REPEAT="${REPEAT:-1}"
  BASE_SEED="$(grep -E '^seed:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  BASE_SEED="${BASE_SEED:-1}"

  EXP_REL="$(realpath --relative-to="${REPO_ROOT}/src/WP2_Outer_Loop" "${EXP_DIR}")"

  for ((i = 0; i < REPEAT; i++)); do
    RUN_TAG="outer_${EXP_NAME}_r${i}"
    SEED=$((BASE_SEED + i))

    DEPENDENCY_ARG=""
    if [ -n "${PREV_JOB_ID}" ]; then
      DEPENDENCY_ARG="--dependency=afterok:${PREV_JOB_ID}"
    fi

    JOB_ID=$(sbatch \
      --job-name="outer_${EXP_NAME}_r${i}" \
      --output="/home/%u/slurm_logs/${RUN_TAG}-%j.out" \
      --error="/home/%u/slurm_logs/${RUN_TAG}-%j.err" \
      ${DEPENDENCY_ARG} \
      --export=ALL,EXP_REL="${EXP_REL}",RUN_CFG_NAME="run.yaml",CKPT_NAME="${CKPT_NAME}",WP1_CFG_NAME="${WP1_CFG_NAME}",RUN_TAG="${RUN_TAG}",CLI_OVERRIDES="--cfg.seed ${SEED}" \
      "$@" \
      "${SLURM_SCRIPT}" \
      | awk '{print $NF}')

    echo "[SWEEP] Submitted ${JOB_ID} <- ${EXP_REL} (run ${i}/${REPEAT}, seed=${SEED})"
    PREV_JOB_ID="${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
  done
done

echo ""
echo "[SWEEP] All ${SUBMITTED} job(s) queued. Last job ID: ${PREV_JOB_ID:-(none)}"
echo "[SWEEP] Monitor with: squeue -u \$USER"
