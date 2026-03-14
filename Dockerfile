FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy full project
COPY pyproject.toml .
COPY src/ src/
COPY gcp/ gcp/
COPY config.yaml .

# Install package + cloud deps
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir google-cloud-storage google-cloud-batch

# Download MediaPipe hand landmarker model
RUN mkdir -p /app/models && \
    wget -q -O /app/models/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Verify install
RUN python -c "from ego_hand_pipeline.cli import main; print('Package installed OK')"

# Create data directories
RUN mkdir -p /data/raw /data/clips /data/depth /data/hand_poses \
    /data/objects /data/segmentation /data/trajectories /data/lerobot

# Default entrypoint for Cloud Batch
ENTRYPOINT ["python", "-m", "ego_hand_pipeline.cloud_worker"]
