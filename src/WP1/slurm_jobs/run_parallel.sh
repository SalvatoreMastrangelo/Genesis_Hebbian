#!/usr/bin/env bash
# run_parallel.sh — submit all YAMLs in a folder as independent SLURM jobs.
#
# Usage:
#   bash run_parallel.sh [EXPERIMENTS_DIR] [EXTRA_SBATCH_ARGS...]
#
# Examples:
#   bash run_parallel.sh                                  # uses src/WP1/experiments/
#   bash run_parallel.sh src/WP1/experiments/batch_3/     # explicit dir
#   bash run_parallel.sh src/WP1/experiments/ --gpus=2    # extra sbatch flags
#
# Each yaml may declare:
#   - exp_name   : str   (used in the job/run tag)
#   - repeat     : int   (number of independent seeds to submit, default 1)
#   - training.seed
#
# Resume: pass RESUME=1 to restart all jobs from their latest checkpoint:
#   RESUME=1 bash run_parallel.sh src/WP1/experiments/batch_5/

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

echo "[PARALLEL] Found ${#YAMLS[@]} config(s) in ${EXPERIMENTS_DIR}"
echo "[PARALLEL] Using slurm script: ${SLURM_SCRIPT}"
echo ""

JOB_IDS=()

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

    RESUME_EXPORT=""
    if [ -n "${RESUME:-}" ] && [ "${RESUME}" -ne 0 ]; then
      RESUME_EXPORT=",RESUME=1"
    fi

    JOB_ID=$(sbatch -q long \
      --job-name="wp1_${EXP_NAME}_r${i}" \
      --output="/home/%u/slurm_logs/${RUN_TAG}-%j.out" \
      --error="/home/%u/slurm_logs/${RUN_TAG}-%j.err" \
      --export=ALL,CFG_FILE="${CFG_FILE}",RUN_TAG="${RUN_TAG}",CLI_OVERRIDES="--cfg.training.seed ${TRAIN_SEED}"${RESUME_EXPORT} \
      "$@" \
      "${SLURM_SCRIPT}" \
      | awk '{print $NF}')

    echo "[PARALLEL] Submitted job ${JOB_ID} <- ${CFG_FILE} (run ${i}/${REPEAT}, train_seed=${TRAIN_SEED})"
    JOB_IDS+=("${JOB_ID}")
  done
done

echo ""
echo "[PARALLEL] All ${#YAMLS[@]} job(s) queued: ${JOB_IDS[*]}"
echo "[PARALLEL] Monitor with: squeue -u \$USER"
