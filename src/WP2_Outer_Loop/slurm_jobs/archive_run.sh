#!/usr/bin/env bash
# archive_run.sh — copy a finished run from scratch to durable storage.
#
# Usage:
#   bash archive_run.sh <RUN_DIR> <DEST_PARENT>
#
# Example:
#   bash archive_run.sh /scratch/izar/$USER/genesis_runs/outer_nsga/outer_exam_6_64_64_r2 \
#                       $HOME/genesis_runs/outer_nsga
#
# Why this exists
# ---------------
# The run directory doubles as the Taichi/gstaichi cache root (train.slurm points
# XDG_CACHE_HOME / TI_CACHE_DIR / GSTAICHI_CACHE_DIR into it), so it accumulates
# hundreds of MB of kernel-compilation and shader cache per run — measured at
# 373-694 MB, roughly half of every run directory. Archiving that to $HOME is
# pure waste: the cache is a regenerable build artifact, not a result.
#
# Copying it verbatim exhausted the home quota on 2026-08-05 and, because
# train.slurm runs under `set -euo pipefail`, the failing copy aborted the job —
# Slurm reported FAILED for runs whose science had finished cleanly
# (jobs 3092665, 3092828).
#
# Caches are excluded at EVERY depth: the cluster writes both <run>/.cache and
# <run>/logs/.cache.
#
# Re-running this script over an existing archive is safe (rsync updates in
# place), which makes it the recovery path for runs whose archive step failed.

set -uo pipefail

# Directory names excluded from the archive: regenerable, never results.
EXCLUDES=(".cache/")

if [ "$#" -ne 2 ]; then
  echo "[ARCHIVE][ERROR] usage: archive_run.sh <RUN_DIR> <DEST_PARENT>" >&2
  exit 2
fi

SRC="${1%/}"
DEST_PARENT="${2%/}"

if [ ! -d "$SRC" ]; then
  echo "[ARCHIVE][ERROR] run directory not found: $SRC" >&2
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ARCHIVE][ERROR] rsync not found; cannot archive without it." >&2
  echo "[ARCHIVE][ERROR] The run itself is intact at: $SRC" >&2
  exit 2
fi

RUN_NAME="$(basename "$SRC")"
DEST="${DEST_PARENT}/${RUN_NAME}"

if ! mkdir -p "$DEST"; then
  echo "[ARCHIVE][ERROR] cannot create destination: $DEST" >&2
  echo "[ARCHIVE][ERROR] The run itself is intact at: $SRC" >&2
  exit 1
fi

RSYNC_ARGS=(-a)
for pattern in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=(--exclude="$pattern")
done

echo "[ARCHIVE] ${SRC} -> ${DEST}"
echo "[ARCHIVE] excluding: ${EXCLUDES[*]}"

rsync "${RSYNC_ARGS[@]}" "${SRC}/" "${DEST}/"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo "[ARCHIVE][ERROR] rsync exited ${STATUS} (quota? permissions?)." >&2
  echo "[ARCHIVE][ERROR] The run itself is intact at: $SRC" >&2
  echo "[ARCHIVE][ERROR] Re-run after freeing space:" >&2
  echo "[ARCHIVE][ERROR]   bash $0 '$SRC' '$DEST_PARENT'" >&2
  exit 1
fi

# Report what the exclusion actually saved, so quota pressure stays visible.
SRC_SIZE="$(du -sh "$SRC" 2>/dev/null | cut -f1)"
DEST_SIZE="$(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "[ARCHIVE] Done: ${DEST}  (source ${SRC_SIZE:-?}, archived ${DEST_SIZE:-?})"
