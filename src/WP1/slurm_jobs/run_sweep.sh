#!/usr/bin/env bash
# run_sweep.sh — submit all YAMLs in a folder as sequential SLURM jobs.
#
# Jobs are chained with --dependency=afterok so they run one at a time;
# useful when you want predictable resource usage on the cluster.
#
# Usage:
#   bash run_sweep.sh [EXPERIMENTS_DIR] [EXTRA_SBATCH_ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/train.slurm"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXPERIMENTS_DIR="${1:-${REPO_ROOT}/src/WP1/experiments}"
shift || true

if [ ! -d "${EXPERIMENTS_DIR}" ]; then
  echo "[ERROR] Experiments directory not found: ${EXPERIMENTS_DIR}"
  exit 1
fi

mapfile -t YAMLS < <(find "${EXPERIMENTS_DIR}" -maxdepth 1 -name "*.yaml" | sort)

if [ ${#YAMLS[@]} -eq 0 ]; then
  echo "[WARN] No .yaml files found in ${EXPERIMENTS_DIR}"
  exit 0
fi

echo "[SWEEP] Found ${#YAMLS[@]} config(s) in ${EXPERIMENTS_DIR}"
echo "[SWEEP] Using slurm script: ${SLURM_SCRIPT}"
echo ""

PREV_JOB_ID=""

for YAML_PATH in "${YAMLS[@]}"; do
  CFG_FILE="$(realpath --relative-to="${REPO_ROOT}/src/WP1" "${YAML_PATH}")"
  EXP_NAME="$(grep -E '^exp_name:' "${YAML_PATH}" | awk '{print $2}' || echo "$(basename "${YAML_PATH}" .yaml)")"
  REPEAT=$(grep -E '^repeat:' "${YAML_PATH}" | awk '{print $2}' || echo "1")
  REPEAT="${REPEAT:-1}"
  BASE_TRAIN_SEED=$(grep -A 10 "^training:" "${YAML_PATH}" | grep "seed:" | awk '{print $2}' || echo "1")
  BASE_TRAIN_SEED="${BASE_TRAIN_SEED:-1}"

  for ((i = 0; i < REPEAT; i++)); do
    RUN_TAG="wp1_${EXP_NAME}_r${i}"
    TRAIN_SEED=$((BASE_TRAIN_SEED + i))

    DEPENDENCY_ARG=""
    if [ -n "${PREV_JOB_ID}" ]; then
      DEPENDENCY_ARG="--dependency=afterok:${PREV_JOB_ID}"
    fi

    JOB_ID=$(sbatch \
      --job-name="wp1_${EXP_NAME}_r${i}" \
      --output="/home/%u/slurm_logs/${RUN_TAG}-%j.out" \
      --error="/home/%u/slurm_logs/${RUN_TAG}-%j.err" \
      ${DEPENDENCY_ARG} \
      --export=ALL,CFG_FILE="${CFG_FILE}",RUN_TAG="${RUN_TAG}",CLI_OVERRIDES="--cfg.training.seed ${TRAIN_SEED}" \
      "$@" \
      "${SLURM_SCRIPT}" \
      | awk '{print $NF}')

    echo "[SWEEP] Submitted job ${JOB_ID} <- ${CFG_FILE} (run ${i}/${REPEAT}, train_seed=${TRAIN_SEED})"
    PREV_JOB_ID="${JOB_ID}"
  done
done

echo ""
echo "[SWEEP] All ${#YAMLS[@]} job(s) queued. Last job ID: ${PREV_JOB_ID}"
echo "[SWEEP] Monitor with: squeue -u \$USER"
