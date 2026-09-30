#!/usr/bin/env bash
# Remaining fronts of the D ablation (POSSIBLE_EXPERIMENTS item 1 b), ONE DOCKER PROCESS PER RUN because
# tesis/scripts/d_ablation_fronts.py leaks about 0.45 GB of host RAM per chunk (Genesis scenes are not fully
# released by gs.destroy()); a single process over 8 runs reached 12 GB and had to be stopped on 2026-09-25.
# Run from anywhere after the local queue has finished (the GPU must be free of the queue).
set -u
cd "$(dirname "$0")/../../../.."
for r in outer_exam_4_64_64_300_r0 outer_exam_4_64_64_300_r1 outer_exam_6_64_64_300_r0 outer_exam_6_64_64_300_r1; do
  echo "=== $(date '+%F %T') START fronts $r" >> logs/thesis_inner_loop/d_ablation_fronts.log
  docker run --rm --name "thesis_d_ablation_$r" --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/workspace/bind -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
    mygenesis:latest python tesis/scripts/d_ablation_fronts.py "logs/remote/outer_nsga/$r" >> logs/thesis_inner_loop/d_ablation_fronts.log 2>&1
  echo "=== $(date '+%F %T') END fronts $r exit=$?" >> logs/thesis_inner_loop/d_ablation_fronts.log
done
