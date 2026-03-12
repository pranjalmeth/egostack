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

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[cloud]"

# Copy source code
COPY src/ src/
COPY gcp/ gcp/
COPY config.yaml .

# Create data directories
RUN mkdir -p /data/raw /data/clips /data/depth /data/hand_poses \
    /data/objects /data/segmentation /data/trajectories /data/lerobot

# Default entrypoint for Cloud Batch
ENTRYPOINT ["python", "-m", "ego_hand_pipeline.cloud_worker"]
