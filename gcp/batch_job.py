"""GCP Cloud Batch job configuration and submission.

Deploys the ego-hand pipeline as Cloud Batch tasks with GPU support,
using Spot/Preemptible VMs for cost optimization (60-91% cheaper).

Architecture:
    1. URL list is split into N chunks
    2. Each chunk becomes a Cloud Batch task
    3. Each task runs on a Spot VM with T4 GPU
    4. Results are uploaded to GCS
    5. Final merge aggregates all episode data

Estimated costs (T4 GPU Spot):
    - ~$0.11/hr per VM
    - ~15 min per video
    - 1000 videos @ 20 concurrent = ~12.5 hrs = ~$27.50 compute
    - GCS storage: ~$0.02/GB/month
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchJobConfig:
    """Configuration for a Cloud Batch pipeline job."""
    project_id: str
    region: str = "us-central1"
    bucket_name: str = "ego-hand-pipeline"
    job_name_prefix: str = "ego-hand"

    # VM configuration
    machine_type: str = "n1-standard-4"
    gpu_type: str = "nvidia-tesla-t4"
    gpu_count: int = 1
    boot_disk_gb: int = 100
    use_spot: bool = True

    # Task configuration
    max_concurrent_tasks: int = 20
    max_retry_count: int = 2
    task_timeout_seconds: int = 3600  # 1 hour per video

    # Container image
    container_image: str = "gcr.io/{project_id}/ego-hand-pipeline:latest"

    # Pipeline configuration
    pipeline_config_gcs: str = ""  # gs:// path to config.yaml
    enable_hand_pose: bool = True
    enable_object_detection: bool = True
    enable_segmentation: bool = True
    export_format: str = "lerobot"  # "lerobot", "json", "csv"


def generate_batch_job_json(
    config: BatchJobConfig,
    urls: list[str],
    chunk_size: int = 1,
) -> dict:
    """Generate a Cloud Batch job specification.

    Args:
        config: Batch job configuration.
        urls: List of video URLs to process.
        chunk_size: Number of URLs per task.

    Returns:
        Cloud Batch job JSON specification.
    """
    num_tasks = math.ceil(len(urls) / chunk_size)
    container_image = config.container_image.format(project_id=config.project_id)

    # Write URL chunks as task environment variables
    task_environments = []
    for i in range(num_tasks):
        chunk = urls[i * chunk_size:(i + 1) * chunk_size]
        task_environments.append({
            "variables": {
                "TASK_INDEX": str(i),
                "VIDEO_URLS": json.dumps(chunk),
                "GCS_BUCKET": config.bucket_name,
                "GCS_OUTPUT_PREFIX": f"output/batch_{config.job_name_prefix}",
                "ENABLE_HAND_POSE": str(config.enable_hand_pose).lower(),
                "ENABLE_OBJECT_DETECTION": str(config.enable_object_detection).lower(),
                "ENABLE_SEGMENTATION": str(config.enable_segmentation).lower(),
                "EXPORT_FORMAT": config.export_format,
            }
        })

    job_spec = {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": container_image,
                                "commands": [
                                    "python", "-m", "ego_hand_pipeline.cloud_worker",
                                ],
                                "volumes": ["/mnt/disks/data:/data"],
                            },
                        }
                    ],
                    "computeResource": {
                        "cpuMilli": 4000,
                        "memoryMib": 16384,
                    },
                    "maxRunDuration": f"{config.task_timeout_seconds}s",
                    "maxRetryCount": config.max_retry_count,
                    "volumes": [
                        {
                            "deviceName": "data-disk",
                            "mountPath": "/mnt/disks/data",
                        }
                    ],
                    "environment": {
                        "variables": {
                            "PIPELINE_CONFIG_GCS": config.pipeline_config_gcs,
                        }
                    },
                },
                "taskCount": num_tasks,
                "parallelism": min(num_tasks, config.max_concurrent_tasks),
                "taskEnvironments": task_environments,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "installGpuDrivers": True,
                    "policy": {
                        "machineType": config.machine_type,
                        "provisioningModel": "SPOT" if config.use_spot else "STANDARD",
                        "accelerators": [
                            {
                                "type": config.gpu_type,
                                "count": config.gpu_count,
                            }
                        ],
                        "disks": [
                            {
                                "newDisk": {
                                    "type": "pd-balanced",
                                    "sizeGb": config.boot_disk_gb,
                                },
                                "deviceName": "data-disk",
                            }
                        ],
                    },
                }
            ],
            "location": {
                "allowedLocations": [f"regions/{config.region}"],
            },
        },
        "logsPolicy": {
            "destination": "CLOUD_LOGGING",
        },
    }

    return job_spec


def submit_batch_job(
    config: BatchJobConfig,
    urls: list[str],
    chunk_size: int = 1,
    dry_run: bool = False,
) -> dict:
    """Submit a Cloud Batch job to process video URLs.

    Args:
        config: Batch job configuration.
        urls: List of video URLs.
        chunk_size: URLs per task.
        dry_run: If True, generate but don't submit.

    Returns:
        Job specification (dry_run) or submission response.
    """
    job_spec = generate_batch_job_json(config, urls, chunk_size)

    if dry_run:
        return job_spec

    try:
        from google.cloud import batch_v1
    except ImportError:
        raise ImportError(
            "google-cloud-batch not installed. "
            "Install with: pip install google-cloud-batch"
        )

    from google.protobuf import json_format

    client = batch_v1.BatchServiceClient()

    job = batch_v1.Job()
    json_format.ParseDict(job_spec, batch_v1.Job.pb(job))

    import uuid
    job_id = f"{config.job_name_prefix}-{uuid.uuid4().hex[:8]}"

    request = batch_v1.CreateJobRequest(
        parent=f"projects/{config.project_id}/locations/{config.region}",
        job_id=job_id,
        job=job,
    )

    response = client.create_job(request=request)
    print(f"Job submitted: {response.name}")
    print(f"  Tasks: {len(urls)} videos in {math.ceil(len(urls) / chunk_size)} tasks")
    print(f"  Concurrency: {min(len(urls), config.max_concurrent_tasks)}")
    print(f"  GPU: {config.gpu_type} x{config.gpu_count}")
    print(f"  Spot VMs: {config.use_spot}")

    return {"job_name": response.name, "job_id": job_id, "spec": job_spec}


def estimate_cost(
    num_videos: int,
    config: BatchJobConfig | None = None,
    avg_video_duration_min: float = 15.0,
) -> dict:
    """Estimate the cost of processing videos on Cloud Batch.

    Args:
        num_videos: Number of videos to process.
        config: Optional batch config for VM specs.
        avg_video_duration_min: Average processing time per video in minutes.

    Returns:
        Dict with cost breakdown.
    """
    if config is None:
        config = BatchJobConfig(project_id="estimate")

    # T4 GPU pricing (us-central1)
    gpu_rates = {
        "nvidia-tesla-t4": {"standard": 0.35, "spot": 0.11},
        "nvidia-l4": {"standard": 0.81, "spot": 0.24},
        "nvidia-tesla-v100": {"standard": 2.48, "spot": 0.74},
        "nvidia-a100-80gb": {"standard": 3.67, "spot": 1.10},
    }

    vm_rates = {
        "n1-standard-4": {"standard": 0.19, "spot": 0.04},
        "n1-standard-8": {"standard": 0.38, "spot": 0.08},
    }

    mode = "spot" if config.use_spot else "standard"
    gpu_rate = gpu_rates.get(config.gpu_type, {"standard": 0.35, "spot": 0.11})[mode]
    vm_rate = vm_rates.get(config.machine_type, {"standard": 0.19, "spot": 0.04})[mode]

    total_gpu_hours = (num_videos * avg_video_duration_min) / 60
    concurrent = min(num_videos, config.max_concurrent_tasks)
    wall_clock_hours = total_gpu_hours / concurrent

    compute_cost = total_gpu_hours * (gpu_rate + vm_rate) * config.gpu_count
    storage_gb = num_videos * 0.5  # ~500MB per video output
    storage_cost = storage_gb * 0.023  # GCS standard pricing

    return {
        "num_videos": num_videos,
        "avg_processing_min": avg_video_duration_min,
        "total_gpu_hours": round(total_gpu_hours, 1),
        "wall_clock_hours": round(wall_clock_hours, 1),
        "concurrent_tasks": concurrent,
        "vm_type": config.machine_type,
        "gpu_type": config.gpu_type,
        "pricing_mode": mode,
        "compute_cost_usd": round(compute_cost, 2),
        "storage_cost_usd_monthly": round(storage_cost, 2),
        "total_cost_usd": round(compute_cost + storage_cost, 2),
    }
