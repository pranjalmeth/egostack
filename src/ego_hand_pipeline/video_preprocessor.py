"""Video preprocessing: resolution standardization and format normalization.

Converts all input videos to a consistent resolution before processing,
ensuring uniform data quality across the pipeline regardless of source.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Ideal processing resolution for robotics training data.
# 480p (640x480) balances accuracy and speed:
# - High enough for accurate hand landmark detection (MediaPipe, HaMeR)
# - Standard robotics dataset resolution (LeRobot, DROID, Open X-Embodiment)
# - Models resize internally anyway (DepthAnything->518x518, HaMeR->224x224)
# - 2-3x faster than 720p processing with minimal quality loss
IDEAL_WIDTH = 640
IDEAL_HEIGHT = 480
IDEAL_FPS = 30


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata (resolution, fps, duration, codec)."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


def needs_preprocessing(
    video_path: str | Path,
    target_width: int = IDEAL_WIDTH,
    target_height: int = IDEAL_HEIGHT,
    target_fps: int | None = None,
) -> bool:
    """Check if a video needs resolution/fps conversion."""
    info = get_video_info(video_path)
    if info["width"] != target_width or info["height"] != target_height:
        return True
    if target_fps and abs(info["fps"] - target_fps) > 1.0:
        return True
    return False


def preprocess_video(
    video_path: str | Path,
    output_dir: str | Path = "data/preprocessed",
    target_width: int = IDEAL_WIDTH,
    target_height: int = IDEAL_HEIGHT,
    target_fps: int | None = None,
    max_frames: int | None = None,
) -> Path:
    """Convert a video to the target resolution and fps.

    Uses letterboxing/pillarboxing to preserve aspect ratio rather than
    stretching, which would distort hand landmark positions.

    Args:
        video_path: Input video path.
        output_dir: Directory for preprocessed output.
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        target_fps: Target fps (None = keep original).
        max_frames: Maximum frames to process (for test mode).

    Returns:
        Path to the preprocessed video.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_preprocessed.mp4"

    # Skip if already preprocessed and exists
    if output_path.exists():
        existing_info = get_video_info(output_path)
        if (existing_info["width"] == target_width and
                existing_info["height"] == target_height):
            if max_frames is None or existing_info["total_frames"] <= max_frames:
                return output_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = target_fps if target_fps else src_fps
    frame_skip = max(1, round(src_fps / out_fps)) if target_fps else 1

    if max_frames:
        total_frames = min(total_frames, max_frames * frame_skip)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps,
                             (target_width, target_height))

    frames_written = 0
    try:
        for frame_idx in tqdm(range(total_frames), desc="Preprocessing video",
                              unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                continue

            resized = _resize_with_padding(frame, target_width, target_height)
            writer.write(resized)
            frames_written += 1

            if max_frames and frames_written >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    return output_path


def _resize_with_padding(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize frame preserving aspect ratio with black padding.

    This avoids distorting spatial proportions which would affect
    landmark coordinate accuracy.
    """
    h, w = frame.shape[:2]
    target_ratio = target_width / target_height
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 0.01:
        # Aspect ratios match, just resize
        return cv2.resize(frame, (target_width, target_height),
                          interpolation=cv2.INTER_AREA)

    if src_ratio > target_ratio:
        # Source is wider — fit to width, pad top/bottom
        new_w = target_width
        new_h = int(target_width / src_ratio)
    else:
        # Source is taller — fit to height, pad left/right
        new_h = target_height
        new_w = int(target_height * src_ratio)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and center the resized frame
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas
