FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev python3-venv \
    git wget curl pkg-config \
    libgl1-mesa-glx libglu1-mesa libegl1 libxrender1 libglfw3 \
    libgtk2.0-dev libvulkan-dev libosmesa6 libosmesa6-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir --upgrade pip

# Install all Python deps (the repo itself is bind-mounted at runtime)
COPY requirements_cluster.txt /tmp/requirements_cluster.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements_cluster.txt

WORKDIR /workspace
CMD ["/bin/bash"]
