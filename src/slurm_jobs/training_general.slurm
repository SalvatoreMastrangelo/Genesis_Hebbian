#!/bin/bash -l
#SBATCH --job-name=gen_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=06:00:00

set -euo pipefail

# ----- percorsi -------------------------------------------------
IMG=$HOME/Docker_RL/genesis-rl.sif
SRC=$HOME/Docker_RL/binding
OUT=/scratch/$USER/genesis_runs/$SLURM_JOB_ID
mkdir -p "$OUT" logs

# ----- bind string ----------------------------------------------
BINDS="--bind ${SRC}:/workspace/bind:rw,${OUT}:/workspace/out:rw"

# ----- run params -----------------------------------------------
EXP_NAME=${EXP_NAME:-foundation-mixture}
NUM_ENVS=${NUM_ENVS:-16384}
MAX_ITERS=${MAX_ITERS:-4000}
N_URDF=${N_URDF:-0}
URDF_SEED=${URDF_SEED:-0}

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

    mkdir -p /workspace/out
    cd /workspace/out

    URDF_ARGS=""
    if [ "${N_URDF}" -gt 0 ]; then
        URDF_ARGS="--n-urdf ${N_URDF} --urdf-seed ${URDF_SEED} --catalog-dir /workspace/out/urdf_generated"
    else
        URDF_ARGS="--catalog-dir /workspace/out/urdf_generated"
    fi

    python /workspace/bind/Genesis/src/general_policy/train_gen.py \
        --exp-name "${EXP_NAME}" \
        --num-envs "${NUM_ENVS}" \
        --max-iterations "${MAX_ITERS}" \
        --device cuda \
        $URDF_ARGS
'
