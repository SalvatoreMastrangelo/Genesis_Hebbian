#!/bin/bash -l
#SBATCH --job-name=nsga
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00

set -euo pipefail

# ----- percorsi -------------------------------------------------
IMG=$HOME/Docker_RL/genesis-rl.sif
SRC=$HOME/Docker_RL/binding
OUT=/scratch/$USER/genesis_runs/$SLURM_JOB_ID
mkdir -p "$OUT" logs

# ----- bind string ----------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:rw,${OUT}:/workspace/out:rw"

# ----- run params -----------------------------------------------
POP_SIZE=${POP_SIZE:-16}
NUM_GEN=${NUM_GEN:-6}
TRAIN_ITERS=${TRAIN_ITERS:-50}
RUN_NAME=${RUN_NAME:-nsga_${SLURM_JOB_ID}}

# ----- run -------------------------------------------------------
srun apptainer exec --nv $BINDS "$IMG" bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/bind/Genesis:$PYTHONPATH
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    export XDG_CACHE_HOME=/workspace/out/.cache
    export TI_CACHE_DIR=/workspace/out/.cache/taichi
    export TAICHI_CACHE_DIR=/workspace/out/.cache/taichi
    export GSTAICHI_CACHE_DIR=/workspace/out/.cache/gstaichi
    export MPLCONFIGDIR=/workspace/out/.cache/mpl
    export RAY_TMPDIR=/workspace/out/.cache/ray
    export GA_PARALLEL=1

    mkdir -p /workspace/out
    cd /workspace/out

    python /workspace/bind/Genesis/src/morph_evolution/evolution_nsga.py \
        --pop "${POP_SIZE}" \
        --gen "${NUM_GEN}" \
        --train_it "${TRAIN_ITERS}" \
        --run_name "${RUN_NAME}" \
        --device cuda
'
