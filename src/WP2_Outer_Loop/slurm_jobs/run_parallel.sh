#!/usr/bin/env bash
# run_parallel.sh — submit every experiment folder in a batch as independent SLURM jobs.
#
# An outer-loop experiment lives in its OWN FOLDER and contains three files:
#   - run.yaml         (single-file OuterNSGA2Config: inner-loop YAML + outer:
#                       section — checkpoint paths can stay blank)
#   - wp1_config.yaml  (WP1 RunConfig the checkpoint was trained with)
#   - <something>.pt   (frozen WP1 actor checkpoint; auto-discovered)
#
# Usage:
#   bash run_parallel.sh [BATCH_DIR] [EXTRA_SBATCH_ARGS...]
#
# Examples:
#   bash run_parallel.sh                                                # uses src/WP2_Outer_Loop/experiments/
#   bash run_parallel.sh src/WP2_Outer_Loop/experiments/batch_0/       # explicit batch
#   bash run_parallel.sh src/WP2_Outer_Loop/experiments/batch_0/ --gpus-per-node=2
#   # multi-node (URDF-sharded eval, see WP2/dist_eval.py — per-run wall time
#   # drops roughly by the node count):
#   bash run_parallel.sh src/WP2_Outer_Loop/experiments/batch_0/ --nodes=4 --gpus-per-node=2
# NOTE: train.slurm uses per-node directives (--ntasks-per-node/--gpus-per-node);
# pass --gpus-per-node=2, NOT the old job-total --gpus=2, to use both V100s.
#
# Each run.yaml may declare:
#   - exp_name : str   (used in the job/run tag; default = experiment folder name)
#   - seed     : int   (top-level; default 1)
#   - repeat   : int   (number of independent seeds to submit, default 1)

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

# Discover experiment folders: any subdirectory that contains a run.yaml
mapfile -t EXP_DIRS < <(find "${BATCH_DIR}" -mindepth 1 -maxdepth 2 -name "run.yaml" -printf '%h\n' | sort -u)

if [ ${#EXP_DIRS[@]} -eq 0 ]; then
  echo "[WARN] No experiment folders (containing run.yaml) found in ${BATCH_DIR}"
  exit 0
fi

echo "[PARALLEL] Found ${#EXP_DIRS[@]} experiment folder(s) in ${BATCH_DIR}"
echo "[PARALLEL] Using slurm script: ${SLURM_SCRIPT}"
echo ""

JOB_IDS=()

for EXP_DIR in "${EXP_DIRS[@]}"; do
  RUN_YAML="${EXP_DIR}/run.yaml"
  EXP_FOLDER_NAME="$(basename "${EXP_DIR}")"

  # --- Locate the checkpoint (.pt) ---
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

  # --- Locate the WP1 config YAML (any yaml that isn't run.yaml) ---
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

  # --- Read exp_name / seed / repeat from run.yaml ---
  EXP_NAME="$(grep -E '^exp_name:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  EXP_NAME="${EXP_NAME:-${EXP_FOLDER_NAME}}"
  REPEAT="$(grep -E '^repeat:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  REPEAT="${REPEAT:-1}"
  BASE_SEED="$(grep -E '^seed:' "${RUN_YAML}" | head -1 | awk '{print $2}' || true)"
  BASE_SEED="${BASE_SEED:-1}"

  # --- Path relative to src/WP2_Outer_Loop (what train.slurm expects in EXP_REL) ---
  EXP_REL="$(realpath --relative-to="${REPO_ROOT}/src/WP2_Outer_Loop" "${EXP_DIR}")"

  for ((i = 0; i < REPEAT; i++)); do
    RUN_TAG="outer_${EXP_NAME}_r${i}"
    SEED=$((BASE_SEED + i))

    JOB_ID=$(sbatch \
      --job-name="outer_${EXP_NAME}_r${i}" \
      --output="/home/%u/slurm_logs/${RUN_TAG}-%j.out" \
      --error="/home/%u/slurm_logs/${RUN_TAG}-%j.err" \
      --export=ALL,EXP_REL="${EXP_REL}",RUN_CFG_NAME="run.yaml",CKPT_NAME="${CKPT_NAME}",WP1_CFG_NAME="${WP1_CFG_NAME}",RUN_TAG="${RUN_TAG}",CLI_OVERRIDES="--cfg.seed ${SEED}" \
      "$@" \
      "${SLURM_SCRIPT}" \
      | awk '{print $NF}')

    echo "[PARALLEL] Submitted ${JOB_ID} <- ${EXP_REL} (run ${i}/${REPEAT}, seed=${SEED})"
    JOB_IDS+=("${JOB_ID}")
  done
done

echo ""
echo "[PARALLEL] All ${#JOB_IDS[@]} job(s) queued: ${JOB_IDS[*]:-(none)}"
echo "[PARALLEL] Monitor with: squeue -u \$USER"
