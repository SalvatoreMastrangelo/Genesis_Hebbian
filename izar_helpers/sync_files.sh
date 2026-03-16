rsync -av \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='*.egg-info' \
  --exclude='dist/' \
  --exclude='build/' \
  --exclude='logs/' \
  --exclude='.claude/' \
  --exclude='slurm_logs/' \
  --exclude='my_container/' \
  --exclude='izar_helpers/' \
  /home/salvatore/Desktop/code/hebbian/Genesis_Hebbian/ \
  smastran@izar.hpc.epfl.ch:/home/smastran/Genesis-dev/
