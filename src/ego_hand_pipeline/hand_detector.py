"""Hand detection and landmark extraction using MediaPipe HandLandmarker (Tasks API)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from tqdm import tqdm

# MediaPipe hand landmark names (21 joints)
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Resolve model path: check env var, then common locations
def _find_model_path() -> str:
    if os.environ.get("HAND_LANDMARKER_MODEL"):
        return os.environ["HAND_LANDMARKER_MODEL"]
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "hand_landmarker.task"),
        "/app/models/hand_landmarker.task",
        os.path.join(os.getcwd(), "models", "hand_landmarker.task"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]

_DEFAULT_MODEL_PATH = _find_model_path()


@dataclass
class HandDetection:
    """Detection result for a single hand in a single frame."""
    hand_label: str  # "left" or "right"
    confidence: float
    landmarks: list[dict]  # list of {joint_id, name, x, y, z}


@dataclass
class FrameDetection:
    """Detection results for a single frame."""
    frame_idx: int
    timestamp: float
    detected: bool
    num_hands: int
    hands: list[HandDetection] = field(default_factory=list)


def detect_hands(
    video_path: str | Path,
    confidence_threshold: float = 0.7,
    model_complexity: int = 1,
    model_path: str | None = None,
) -> list[FrameDetection]:
    """Run hand detection on every frame of a video.

    Args:
        video_path: Path to the video file.
        confidence_threshold: Minimum detection confidence.
        model_complexity: Unused (kept for config compat). Tasks API uses the model file directly.
        model_path: Path to hand_landmarker.task model file.

    Returns:
        List of FrameDetection, one per frame.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=confidence_threshold,
        min_hand_presence_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold,
    )

    detections: list[FrameDetection] = []

    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in tqdm(range(total_frames), desc="Detecting hands", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            timestamp_ms = int(frame_idx * 1000 / fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_det = FrameDetection(
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                detected=False,
                num_hands=0,
            )

            if result.hand_landmarks:
                frame_det.detected = True
                frame_det.num_hands = len(result.hand_landmarks)

                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    # Get handedness
                    label = "unknown"
                    conf = 0.0
                    if result.handedness and i < len(result.handedness):
                        category = result.handedness[i][0]
                        label = category.category_name.lower()
                        conf = category.score

                    landmarks = []
                    for joint_id, lm in enumerate(hand_landmarks):
                        landmarks.append({
                            "joint_id": joint_id,
                            "name": LANDMARK_NAMES[joint_id],
                            "x": round(float(lm.x), 6),
                            "y": round(float(lm.y), 6),
                            "z": round(float(lm.z), 6),
                        })

                    frame_det.hands.append(HandDetection(
                        hand_label=label,
                        confidence=round(float(conf), 4),
                        landmarks=landmarks,
                    ))

            detections.append(frame_det)

    cap.release()
    return detections
