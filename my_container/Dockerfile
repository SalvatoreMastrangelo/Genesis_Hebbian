FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# -------------------------------
# System Python + basic deps
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl pkg-config \
    libgl1-mesa-glx libglu1-mesa libegl1 libxrender1 \
    libgtk2.0-dev libvulkan-dev libosmesa6 libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# "python" -> python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Aggiorna pip (non tocca numpy, solo il tool)
RUN python -m pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------
# UN SOLO pip install con TUTTE le dipendenze Python
#   - numpy PINNATO a 1.26.4 (1.x ma recente)
#   - torch + cuda 11.8
#   - genesis-world 0.3.7
#   - rsl_rl + le tue extra
#   In questo modo il resolver vede tutto assieme e
#   NON può "silenziosamente" portare numpy a 2.x
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    "numpy==1.26.4" \
    "torch==2.1.0+cu118" \
    "torchvision==0.16.0+cu118" \
    "torchaudio==2.1.0" \
    "genesis-world==0.3.7" \
    "rsl-rl-lib==2.2.4" \
    deap \
    filelock \
    pyrr \
    plyfile \
    "open3d==0.17.0" \
    matplotlib \
    tensorboard \
    pyquaternion \
    rich \
    tqdm \
    jsonschema \
    click \
    pandas

# ------------------------------------------------------------
# Clona la tua fork di Genesis e installala SENZA deps
# (le dipendenze sono già coperte da genesis-world + extra)
# ------------------------------------------------------------
WORKDIR /workspace
RUN git clone --branch base_0_3_7 https://github.com/andrevic1/Genesis.git genesis-local

WORKDIR /workspace/genesis-local
RUN python -m pip install --no-cache-dir --no-deps .

WORKDIR /workspace/genesis-local
CMD ["/bin/bash"]
