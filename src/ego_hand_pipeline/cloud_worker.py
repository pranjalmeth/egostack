"""Cloud Batch worker entrypoint.

This module runs as the container entrypoint for GCP Cloud Batch tasks.
It reads configuration from environment variables, processes assigned
video URLs, and uploads results to GCS.

Environment variables:
    VIDEO_URLS: JSON-encoded list of video URLs to process
    GCS_BUCKET: GCS bucket for output
    GCS_OUTPUT_PREFIX: Prefix path in GCS bucket
    PIPELINE_CONFIG_GCS: Optional GCS path to config.yaml
    TASK_INDEX: Cloud Batch task index
    ENABLE_HAND_POSE: Enable 3D hand pose estimation ("true"/"false")
    ENABLE_OBJECT_DETECTION: Enable object detection ("true"/"false")
    ENABLE_SEGMENTATION: Enable segmentation ("true"/"false")
    EXPORT_FORMAT: Export format ("lerobot", "json", "csv")
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline_enhanced


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def main():
    """Cloud Batch worker entrypoint."""
    task_index = os.environ.get("TASK_INDEX", "0")
    print(f"=== Cloud Batch Worker - Task {task_index} ===")

    # Parse video URLs
    urls_json = os.environ.get("VIDEO_URLS", "[]")
    try:
        urls = json.loads(urls_json)
    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse VIDEO_URLS: {urls_json}")
        sys.exit(1)

    if not urls:
        print("No URLs to process. Exiting.")
        return

    print(f"Processing {len(urls)} video(s)")

    # Load config
    config_path = os.environ.get("PIPELINE_CONFIG_GCS")
    local_config = "/app/config.yaml"

    if config_path and config_path.startswith("gs://"):
        try:
            from gcp.gcs_storage import download_file, parse_gcs_uri
            bucket, path = parse_gcs_uri(config_path)
            download_file(bucket, path, local_config)
            print(f"Downloaded config from {config_path}")
        except Exception as e:
            print(f"Failed to download config from GCS: {e}. Using default.")

    config = load_config(local_config if Path(local_config).exists() else None)

    # Override data dirs to use /data mount
    config.download.output_dir = "/data/raw"
    config.clipping.output_dir = "/data/clips"
    config.depth.output_dir = "/data/depth"
    config.trajectories.output_dir = "/data/trajectories"
    config.base_dir = "/"

    # Build enhanced pipeline options from env
    options = {
        "enable_hand_pose": _env_bool("ENABLE_HAND_POSE", True),
        "enable_object_detection": _env_bool("ENABLE_OBJECT_DETECTION", True),
        "enable_segmentation": _env_bool("ENABLE_SEGMENTATION", True),
        "export_format": os.environ.get("EXPORT_FORMAT", "lerobot"),
        "hand_pose_output_dir": "/data/hand_poses",
        "object_output_dir": "/data/objects",
        "segmentation_output_dir": "/data/segmentation",
        "lerobot_output_dir": "/data/lerobot",
    }

    try:
        run_pipeline_enhanced(config, urls, **options)
        print("Pipeline completed successfully.")
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Upload results to GCS
    bucket_name = os.environ.get("GCS_BUCKET")
    output_prefix = os.environ.get("GCS_OUTPUT_PREFIX", "output")

    if bucket_name:
        try:
            from gcp.gcs_storage import upload_directory
            gcs_prefix = f"{output_prefix}/task_{task_index}"

            print(f"Uploading results to gs://{bucket_name}/{gcs_prefix}/")
            uploaded = upload_directory("/data", bucket_name, gcs_prefix)
            print(f"Uploaded {len(uploaded)} files to GCS")
        except Exception as e:
            print(f"ERROR: Failed to upload to GCS: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("No GCS_BUCKET set. Results saved locally only.")

    print(f"=== Task {task_index} complete ===")


if __name__ == "__main__":
    main()
