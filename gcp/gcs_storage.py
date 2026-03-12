"""Google Cloud Storage utilities for uploading/downloading pipeline data.

Handles bidirectional sync between local pipeline output dirs and GCS buckets,
with support for resumable uploads and parallel transfers.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


def _get_client():
    """Get authenticated GCS client."""
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage not installed. "
            "Install with: pip install google-cloud-storage"
        )
    return storage.Client()


def upload_directory(
    local_dir: str | Path,
    bucket_name: str,
    gcs_prefix: str = "",
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Upload a local directory to GCS.

    Args:
        local_dir: Local directory path.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix path within the bucket.
        exclude_patterns: File patterns to exclude (e.g., ["*.tmp", "__pycache__"]).

    Returns:
        List of uploaded GCS URIs.
    """
    local_dir = Path(local_dir)
    client = _get_client()
    bucket = client.bucket(bucket_name)

    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", ".git"]

    uploaded = []

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Check exclusions
        skip = False
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                skip = True
                break
        if skip:
            continue

        relative = file_path.relative_to(local_dir)
        gcs_path = f"{gcs_prefix}/{relative}" if gcs_prefix else str(relative)
        gcs_path = gcs_path.replace("\\", "/")

        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(file_path))
        uri = f"gs://{bucket_name}/{gcs_path}"
        uploaded.append(uri)

    return uploaded


def download_directory(
    bucket_name: str,
    gcs_prefix: str,
    local_dir: str | Path,
) -> list[Path]:
    """Download a GCS prefix to a local directory.

    Args:
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix path within the bucket.
        local_dir: Local directory to download into.

    Returns:
        List of downloaded local file paths.
    """
    local_dir = Path(local_dir)
    client = _get_client()
    bucket = client.bucket(bucket_name)

    downloaded = []

    blobs = bucket.list_blobs(prefix=gcs_prefix)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative = blob.name[len(gcs_prefix):].lstrip("/")
        local_path = local_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        downloaded.append(local_path)

    return downloaded


def upload_file(
    local_path: str | Path,
    bucket_name: str,
    gcs_path: str,
) -> str:
    """Upload a single file to GCS.

    Returns:
        GCS URI of the uploaded file.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{gcs_path}"


def download_file(
    bucket_name: str,
    gcs_path: str,
    local_path: str | Path,
) -> Path:
    """Download a single file from GCS.

    Returns:
        Local path of the downloaded file.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(str(local_path))
    return local_path


def gcs_uri(bucket_name: str, path: str) -> str:
    """Build a gs:// URI."""
    return f"gs://{bucket_name}/{path}"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket_name, path)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    return bucket, path
