"""Render sample visualization videos for test/benchmark mode.

Produces annotated video clips showing pipeline outputs overlaid on the
original footage: hand landmarks, object bounding boxes, depth colormaps,
segmentation masks, and scene captions — all in a single composite video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def render_sample_video(
    video_path: str | Path,
    output_path: str | Path,
    hand_detections: list | None = None,
    object_detections: list | None = None,
    depth_frames: list | None = None,
    segmentations: list | None = None,
    scene_annotations: list | None = None,
    max_frames: int | None = None,
) -> Path:
    """Render a composite sample video with all pipeline outputs overlaid.

    Creates a side-by-side layout:
    - Left: original video with hand landmarks and object boxes
    - Right: depth map with segmentation mask overlay

    Scene captions are shown as text at the bottom.

    Args:
        video_path: Source video.
        output_path: Where to write the sample video.
        hand_detections: Per-frame hand detections.
        object_detections: Per-frame object detections.
        depth_frames: Depth estimation results.
        segmentations: Segmentation results.
        scene_annotations: Scene captions.
        max_frames: Maximum frames to render.

    Returns:
        Path to the rendered sample video.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Output is side-by-side (2x width) + caption bar (60px bottom)
    caption_height = 60
    out_width = width * 2
    out_height = height + caption_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                             (out_width, out_height))

    # Build lookups
    hand_by_frame = _build_lookup(hand_detections)
    obj_by_frame = _build_lookup(object_detections)
    depth_by_frame = _build_lookup(depth_frames)
    seg_by_frame = _build_lookup(segmentations)
    ann_by_frame = _build_lookup(scene_annotations)

    try:
        for frame_idx in tqdm(range(total_frames), desc="Rendering sample",
                              unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            # --- Left panel: original + annotations ---
            left = frame.copy()
            _draw_hands(left, hand_by_frame.get(frame_idx))
            _draw_objects(left, obj_by_frame.get(frame_idx))

            # --- Right panel: depth + segmentation ---
            depth = depth_by_frame.get(frame_idx)
            if depth is not None:
                depth_map = depth.depth_map if hasattr(depth, "depth_map") else None
                if depth_map is not None:
                    right = _colorize_depth(depth_map, width, height)
                else:
                    right = np.zeros_like(frame)
            else:
                right = np.zeros_like(frame)

            _draw_segmentation_overlay(right, seg_by_frame.get(frame_idx))

            # --- Composite ---
            canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            canvas[:height, :width] = left
            canvas[:height, width:] = right

            # --- Caption bar ---
            ann = ann_by_frame.get(frame_idx)
            caption = ""
            if ann is not None:
                caption = ann.caption if hasattr(ann, "caption") else str(ann)

            _draw_caption_bar(canvas, caption, height, out_width, caption_height)

            # Frame counter
            cv2.putText(canvas, f"Frame {frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            writer.write(canvas)
    finally:
        cap.release()
        writer.release()

    return output_path


def _build_lookup(items: list | None) -> dict:
    """Build a frame_idx -> item lookup from a list of frame results."""
    if not items:
        return {}
    lookup = {}
    for item in items:
        fidx = item.frame_idx if hasattr(item, "frame_idx") else item.get("frame_idx", -1)
        lookup[fidx] = item
    return lookup


def _draw_hands(frame: np.ndarray, detection) -> None:
    """Draw hand landmarks and connections on a frame."""
    if detection is None:
        return
    hands = detection.hands if hasattr(detection, "hands") else []
    h, w = frame.shape[:2]

    # Colors: left=blue, right=green
    colors = {"left": (255, 150, 0), "right": (0, 255, 100), "unknown": (200, 200, 200)}

    # MediaPipe hand connections (pairs of joint indices)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),    # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),          # palm
    ]

    for hand in hands:
        label = hand.hand_label if hasattr(hand, "hand_label") else hand.get("hand_label", "unknown")
        landmarks = hand.landmarks if hasattr(hand, "landmarks") else hand.get("landmarks", [])
        color = colors.get(label, (200, 200, 200))

        points = []
        for lm in landmarks:
            px = int(lm["x"] * w)
            py = int(lm["y"] * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 3, color, -1)

        # Draw connections
        for i, j in connections:
            if i < len(points) and j < len(points):
                cv2.line(frame, points[i], points[j], color, 1)

        # Label
        if points:
            cv2.putText(frame, label, (points[0][0] + 5, points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_objects(frame: np.ndarray, objects_frame) -> None:
    """Draw object bounding boxes on a frame."""
    if objects_frame is None:
        return
    objects = objects_frame.objects if hasattr(objects_frame, "objects") else []
    h, w = frame.shape[:2]

    for obj in objects:
        bbox = obj.bbox if hasattr(obj, "bbox") else obj.get("bbox", [])
        label = obj.label if hasattr(obj, "label") else obj.get("label", "")
        conf = obj.confidence if hasattr(obj, "confidence") else obj.get("confidence", 0)

        if len(bbox) < 4:
            continue

        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)


def _colorize_depth(depth_map: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Colorize a depth map and resize to target dimensions."""
    normalized = depth_map - depth_map.min()
    max_val = normalized.max()
    if max_val > 0:
        normalized = (normalized / max_val * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    if colored.shape[:2] != (target_h, target_w):
        colored = cv2.resize(colored, (target_w, target_h))
    return colored


def _draw_segmentation_overlay(frame: np.ndarray, seg_frame) -> None:
    """Overlay semi-transparent segmentation masks on a frame."""
    if seg_frame is None:
        return
    masks = seg_frame.masks if hasattr(seg_frame, "masks") else []
    h, w = frame.shape[:2]

    # Distinct colors for different objects
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]

    for i, mask_obj in enumerate(masks):
        mask = mask_obj.mask if hasattr(mask_obj, "mask") else None
        if mask is None:
            continue

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)

        color = palette[i % len(palette)]
        overlay = np.zeros_like(frame)
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.3, frame, 1.0, 0, frame)


def _draw_caption_bar(
    canvas: np.ndarray,
    caption: str,
    y_start: int,
    width: int,
    bar_height: int,
) -> None:
    """Draw a caption text bar at the bottom of the canvas."""
    # Dark background
    canvas[y_start:y_start + bar_height, :] = (30, 30, 30)

    if caption:
        # Truncate if too long
        max_chars = width // 8
        if len(caption) > max_chars:
            caption = caption[:max_chars - 3] + "..."

        cv2.putText(canvas, caption, (10, y_start + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
