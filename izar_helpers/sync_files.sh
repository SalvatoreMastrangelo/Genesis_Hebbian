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
  --exclude='tests/' \
  --exclude='.claude/' \
  --exclude='slurm_logs/' \
  --exclude='my_container/' \
  --exclude='izar_helpers/' \
  --exclude='src/WP2/multi_urdf_utils/test_catalog/' \
  /home/salvatore/Desktop/code/hebbian/Genesis_Hebbian/ \
  smastran@izar.hpc.epfl.ch:/home/smastran/Genesis-dev/
