#!/usr/bin/env bash
# Sequential local queue (one GPU) of the inner-loop runs of thesis section 6.1.
# Usage, from anywhere:  src/WP2/experiments/thesis_inner_loop/run_queue.sh cfg1.yaml cfg2.yaml ...
# Each run goes to logs/runs_hebbian/<stamp>_<exp_name>; stdout/stderr to logs/thesis_inner_loop/<name>.log
set -u
cd "$(dirname "$0")/../../../.."
mkdir -p logs/thesis_inner_loop
QLOG=logs/thesis_inner_loop/queue.log
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "=== $(date '+%F %T') START $name" >> "$QLOG"
  docker run --rm --name "thesis_$name" --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/workspace/bind -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
    mygenesis:latest python -m WP2.run --cfg "$cfg" > "logs/thesis_inner_loop/$name.log" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') END   $name exit=$rc" >> "$QLOG"
done
echo "=== $(date '+%F %T') QUEUE DONE" >> "$QLOG"
