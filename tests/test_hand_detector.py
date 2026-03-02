"""Tests for the hand_detector module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ego_hand_pipeline.hand_detector import (
    LANDMARK_NAMES,
    FrameDetection,
    HandDetection,
)


def test_landmark_names_count():
    assert len(LANDMARK_NAMES) == 21


def test_frame_detection_defaults():
    fd = FrameDetection(frame_idx=0, timestamp=0.0, detected=False, num_hands=0)
    assert fd.hands == []
    assert not fd.detected


def test_hand_detection_structure():
    landmarks = [
        {"joint_id": i, "name": LANDMARK_NAMES[i], "x": 0.5, "y": 0.5, "z": 0.0}
        for i in range(21)
    ]
    hd = HandDetection(hand_label="left", confidence=0.95, landmarks=landmarks)
    assert hd.hand_label == "left"
    assert len(hd.landmarks) == 21
    assert hd.landmarks[0]["name"] == "WRIST"


@patch("ego_hand_pipeline.hand_detector.HandLandmarker")
@patch("ego_hand_pipeline.hand_detector.cv2.VideoCapture")
def test_detect_hands_no_detections(mock_cap_class, mock_landmarker_class):
    """Test detection on a video where no hands are found."""
    mock_cap = MagicMock()
    mock_cap_class.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 30.0,  # CAP_PROP_FPS
        7: 3,     # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, fake_frame),
        (True, fake_frame),
        (True, fake_frame),
        (False, None),
    ]

    # Mock the landmarker context manager
    mock_landmarker = MagicMock()
    mock_landmarker_class.create_from_options.return_value.__enter__ = MagicMock(return_value=mock_landmarker)
    mock_landmarker_class.create_from_options.return_value.__exit__ = MagicMock(return_value=False)

    mock_result = MagicMock()
    mock_result.hand_landmarks = []
    mock_result.handedness = []
    mock_landmarker.detect_for_video.return_value = mock_result

    from ego_hand_pipeline.hand_detector import detect_hands
    detections = detect_hands("fake_video.mp4", confidence_threshold=0.5, model_path="fake.task")

    assert len(detections) == 3
    assert all(not d.detected for d in detections)
    assert all(d.num_hands == 0 for d in detections)
