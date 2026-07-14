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
  --exclude='**/*.urdf' \
  --exclude='**/cmaes_state.pkl' \
  --exclude='**/*.stl' \
  smastran@izar.hpc.epfl.ch:/home/smastran/genesis_runs/ \
  logs/remote/
