"""Tests for the trajectory_mapper module."""

import numpy as np
import pytest

from ego_hand_pipeline.depth_estimator import DepthFrame
from ego_hand_pipeline.hand_detector import (
    LANDMARK_NAMES,
    FrameDetection,
    HandDetection,
)
from ego_hand_pipeline.trajectory_mapper import (
    _compute_derivatives,
    _smooth_positions,
    map_trajectories,
)


def _make_landmarks(x=0.5, y=0.5, z=0.0):
    return [
        {"joint_id": i, "name": LANDMARK_NAMES[i], "x": x, "y": y, "z": z}
        for i in range(21)
    ]


def _make_detection_sequence(n_frames: int, fps: float = 30.0):
    """Create a sequence of detections with one left hand moving across frames."""
    detections = []
    for i in range(n_frames):
        x = 0.3 + (i / n_frames) * 0.4  # move from 0.3 to 0.7
        detections.append(FrameDetection(
            frame_idx=i,
            timestamp=round(i / fps, 4),
            detected=True,
            num_hands=1,
            hands=[HandDetection(
                hand_label="left",
                confidence=0.95,
                landmarks=_make_landmarks(x=round(x, 4), y=0.5, z=0.0),
            )],
        ))
    return detections


def test_compute_derivatives():
    positions = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "frame_idx": 0, "timestamp": 0.0, "depth": None},
        {"x": 1.0, "y": 0.0, "z": 0.0, "frame_idx": 1, "timestamp": 0.033, "depth": None},
        {"x": 3.0, "y": 0.0, "z": 0.0, "frame_idx": 2, "timestamp": 0.067, "depth": None},
    ]
    velocities, accelerations = _compute_derivatives(positions, fps=30.0)
    assert len(velocities) == 3
    assert velocities[0]["vx"] == 0.0
    assert velocities[1]["vx"] == pytest.approx(30.0, rel=1e-3)
    assert len(accelerations) == 3


def test_smooth_positions():
    positions = [
        {"x": float(i), "y": 0.0, "z": 0.0, "frame_idx": i, "timestamp": i / 30.0, "depth": None}
        for i in range(10)
    ]
    smoothed = _smooth_positions(positions, window=3)
    assert len(smoothed) == 10
    # Smoothed values should be close to original for a linear sequence
    for i, p in enumerate(smoothed):
        assert abs(p["x"] - i) < 1.0


def test_smooth_positions_short():
    positions = [{"x": 1.0, "y": 0.0, "z": 0.0}]
    result = _smooth_positions(positions, window=5)
    assert result == positions


def test_map_trajectories_basic():
    detections = _make_detection_sequence(10, fps=30.0)
    traj = map_trajectories(
        video_path="test.mp4",
        detections=detections,
        clip_id="test_clip",
        fps=30.0,
    )
    assert traj.clip_id == "test_clip"
    assert len(traj.hands) == 1
    assert traj.hands[0].hand_label == "left"
    assert len(traj.hands[0].joints) == 21
    assert len(traj.frames) == 10


def test_map_trajectories_with_depth():
    detections = _make_detection_sequence(5, fps=30.0)
    depth_frames = [
        DepthFrame(
            frame_idx=i,
            timestamp=round(i / 30.0, 4),
            depth_map=np.ones((480, 640), dtype=np.float32) * (i + 1),
        )
        for i in range(5)
    ]

    traj = map_trajectories(
        video_path="test.mp4",
        detections=detections,
        depth_frames=depth_frames,
        clip_id="depth_test",
        fps=30.0,
    )

    # Check that depth values are present in frame data
    for frame in traj.frames:
        for hand in frame["hands"]:
            for lm in hand["landmarks"]:
                assert lm["depth"] is not None


def test_map_trajectories_velocity():
    detections = _make_detection_sequence(10, fps=30.0)
    traj = map_trajectories(
        video_path="test.mp4",
        detections=detections,
        fps=30.0,
        compute_velocity=True,
        compute_acceleration=True,
    )

    for ht in traj.hands:
        for jt in ht.joints:
            assert jt.velocities is not None
            assert jt.accelerations is not None
            assert len(jt.velocities) == len(jt.positions)
