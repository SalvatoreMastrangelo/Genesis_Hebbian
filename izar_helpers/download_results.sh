#!/bin/bash
rsync -av \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='*.egg-info' \
  --exclude='**/.cache' \
  --exclude='**/.mps' \
  smastran@izar.hpc.epfl.ch:/home/smastran/genesis_runs/wp1_training \
  logs/remote/
