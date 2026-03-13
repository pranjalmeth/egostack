"""Tests for the enhanced benchmark module."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ego_hand_pipeline.enhanced_benchmark import (
    ConfidenceStats,
    DepthQualityMetrics,
    DetectionQualityMetrics,
    EnhancedBenchmarkResult,
    HandPoseQualityMetrics,
    LeRobotExportQualityMetrics,
    ObjectDetectionQualityMetrics,
    PerfResult,
    SceneAnnotationQualityMetrics,
    SegmentationQualityMetrics,
    StageTiming,
    StageQuality,
    TemporalConsistency,
    TrajectoryQualityMetrics,
    _compute_confidence_stats,
    _compute_temporal_consistency,
    analyze_detection,
    analyze_clips,
    analyze_depth,
    analyze_hand_pose,
    analyze_lerobot_export,
    analyze_object_detection,
    analyze_scene_annotations,
    analyze_segmentation,
    analyze_trajectories,
    format_enhanced_report,
    save_enhanced_report,
)


# ── Mock dataclass helpers ──────────────────────────────────────────


@dataclass
class MockHandDetection:
    hand_label: str = "right"
    confidence: float = 0.9
    landmarks: list = field(default_factory=list)


@dataclass
class MockFrameDetection:
    frame_idx: int = 0
    timestamp: float = 0.0
    detected: bool = True
    num_hands: int = 1
    hands: list = field(default_factory=list)


@dataclass
class MockClipInfo:
    clip_id: str = "clip_001"
    path: Path = field(default_factory=lambda: Path("/tmp/fake_clip.mp4"))
    source_video: str = "test.mp4"
    start_time: float = 0.0
    end_time: float = 5.0
    duration: float = 5.0
    start_frame: int = 0
    end_frame: int = 150


@dataclass
class MockDepthFrame:
    frame_idx: int = 0
    timestamp: float = 0.0
    depth_map: np.ndarray = field(default_factory=lambda: np.random.rand(48, 64).astype(np.float32))
    npz_path: Path | None = None
    viz_path: Path | None = None


@dataclass
class MockHandPose3D:
    hand_label: str = "right"
    confidence: float = 0.85
    global_orient: list = field(default_factory=lambda: [0.1, 0.2, 0.3])
    hand_pose: list = field(default_factory=lambda: [0.0] * 45)
    betas: list = field(default_factory=lambda: [0.0] * 10)
    joints_3d: list = field(default_factory=list)
    joints_2d: list = field(default_factory=list)
    wrist_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wrist_orientation: list = field(default_factory=lambda: [0.0] * 6)
    fingertip_positions: list = field(default_factory=lambda: [
        {"x": 0.1, "y": 0.2, "z": 0.0},
        {"x": 0.3, "y": 0.1, "z": 0.0},
        {"x": 0.5, "y": 0.3, "z": 0.0},
        {"x": 0.2, "y": 0.4, "z": 0.0},
        {"x": 0.4, "y": 0.5, "z": 0.0},
    ])


@dataclass
class MockFrameHandPose:
    frame_idx: int = 0
    timestamp: float = 0.0
    hands: list = field(default_factory=list)


@dataclass
class MockDetectedObject:
    label: str = "cup"
    confidence: float = 0.8
    bbox: list = field(default_factory=lambda: [0.1, 0.1, 0.3, 0.3])
    center: list = field(default_factory=lambda: [0.2, 0.2])
    area: float = 0.04


@dataclass
class MockFrameObjects:
    frame_idx: int = 0
    timestamp: float = 0.0
    objects: list = field(default_factory=list)


@dataclass
class MockSegmentationMask:
    object_id: int = 0
    label: str = "hand"
    mask: np.ndarray = field(default_factory=lambda: np.ones((48, 64), dtype=bool))
    confidence: float = 0.9
    bbox: list = field(default_factory=lambda: [0.1, 0.1, 0.3, 0.3])
    area_fraction: float = 0.05


@dataclass
class MockFrameSegmentation:
    frame_idx: int = 0
    timestamp: float = 0.0
    masks: list = field(default_factory=list)


@dataclass
class MockSceneAnnotation:
    frame_idx: int = 0
    timestamp: float = 0.0
    caption: str = "A hand is holding a cup"
    objects_present: list = field(default_factory=lambda: ["cup"])
    hands_present: list = field(default_factory=lambda: ["right"])
    action_description: str = "holding"
    confidence: float = 0.8


@dataclass
class MockJointTrajectory:
    joint_id: int = 0
    name: str = "WRIST"
    positions: list = field(default_factory=list)
    velocities: list | None = None
    accelerations: list | None = None


@dataclass
class MockHandTrajectory:
    hand_label: str = "right"
    joints: list = field(default_factory=list)


@dataclass
class MockTrajectoryData:
    source_video: str = "test.mp4"
    clip_id: str = "clip_001"
    start_time: float = 0.0
    end_time: float = 5.0
    fps: float = 30.0
    hands: list = field(default_factory=list)
    frames: list = field(default_factory=list)


# ── Tests: _compute_confidence_stats ────────────────────────────────


class TestComputeConfidenceStats:
    def test_empty_input(self):
        result = _compute_confidence_stats([])
        assert result.count == 0
        assert result.mean == 0.0

    def test_single_value(self):
        result = _compute_confidence_stats([0.5])
        assert result.count == 1
        assert result.mean == 0.5
        assert result.median == 0.5
        assert result.std == 0.0
        assert result.min == 0.5
        assert result.max == 0.5

    def test_known_values(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = _compute_confidence_stats(values)
        assert result.count == 10
        assert abs(result.mean - 0.55) < 0.01
        assert abs(result.median - 0.55) < 0.01
        assert result.min == 0.1
        assert result.max == 1.0
        assert result.p25 == 0.3  # index 2
        assert result.p75 == 0.8  # index 7
        assert result.p90 == 1.0  # index 9
        assert result.std > 0

    def test_identical_values(self):
        result = _compute_confidence_stats([0.5, 0.5, 0.5])
        assert result.mean == 0.5
        assert result.std == 0.0


# ── Tests: _compute_temporal_consistency ────────────────────────────


class TestComputeTemporalConsistency:
    def test_empty_input(self):
        result = _compute_temporal_consistency([])
        assert result.flicker_count == 0
        assert result.longest_streak == 0

    def test_all_detected(self):
        flags = [True] * 10
        result = _compute_temporal_consistency(flags)
        assert result.longest_streak == 10
        assert result.flicker_count == 0
        assert result.gap_segment_count == 0
        assert result.frame_to_frame_change_rate == 0.0

    def test_all_undetected(self):
        flags = [False] * 10
        result = _compute_temporal_consistency(flags)
        assert result.longest_streak == 0
        assert result.gap_segment_count == 1
        assert result.frame_to_frame_change_rate == 0.0

    def test_alternating(self):
        flags = [True, False, True, False, True]
        result = _compute_temporal_consistency(flags)
        assert result.longest_streak == 1
        assert result.flicker_count == 2  # two gaps of length 1 (<=5)
        assert result.gap_segment_count == 2
        assert result.frame_to_frame_change_rate == 1.0

    def test_flicker_detection(self):
        # One short gap (flicker) and one long gap
        flags = [True, True, False, False, True, True, False] * 2 + [False] * 10
        result = _compute_temporal_consistency(flags)
        assert result.flicker_count >= 1  # short gaps exist
        assert result.longest_streak >= 2

    def test_single_element(self):
        result = _compute_temporal_consistency([True])
        assert result.longest_streak == 1
        assert result.frame_to_frame_change_rate == 0.0


# ── Tests: analyze_detection ────────────────────────────────────────


class TestAnalyzeDetection:
    def test_empty(self):
        result = analyze_detection([])
        assert result.total_frames == 0
        assert result.detection_rate == 0.0

    def test_all_detected(self):
        detections = []
        for i in range(10):
            d = MockFrameDetection(
                frame_idx=i, detected=True, num_hands=2,
                hands=[
                    MockHandDetection(hand_label="left", confidence=0.9),
                    MockHandDetection(hand_label="right", confidence=0.95),
                ],
            )
            detections.append(d)
        result = analyze_detection(detections)
        assert result.total_frames == 10
        assert result.detection_rate == 1.0
        assert result.both_hands_rate == 1.0
        assert result.left_count == 10
        assert result.right_count == 10
        assert result.lr_balance == 1.0
        assert result.confidence.count == 20

    def test_partial_detection(self):
        detections = [
            MockFrameDetection(frame_idx=0, detected=True, num_hands=1,
                               hands=[MockHandDetection(hand_label="right", confidence=0.8)]),
            MockFrameDetection(frame_idx=1, detected=False, num_hands=0, hands=[]),
            MockFrameDetection(frame_idx=2, detected=True, num_hands=1,
                               hands=[MockHandDetection(hand_label="right", confidence=0.9)]),
        ]
        result = analyze_detection(detections)
        assert result.detection_rate == pytest.approx(2 / 3)
        assert result.both_hands_rate == 0.0
        assert result.left_count == 0
        assert result.right_count == 2


# ── Tests: analyze_depth ────────────────────────────────────────────


class TestAnalyzeDepth:
    def test_empty(self):
        result = analyze_depth([], None, "test_model")
        assert result.model_name == "test_model"
        assert result.num_frames == 0

    def test_single_frame(self):
        depth_map = np.ones((48, 64), dtype=np.float32) * 5.0
        frames = [MockDepthFrame(frame_idx=0, depth_map=depth_map)]
        result = analyze_depth(frames, None, "test_model")
        assert result.num_frames == 1
        assert result.depth_mean == pytest.approx(5.0)
        assert result.depth_range == pytest.approx(0.0)

    def test_multiple_frames(self):
        frames = [
            MockDepthFrame(frame_idx=0, depth_map=np.ones((10, 10), dtype=np.float32) * 1.0),
            MockDepthFrame(frame_idx=1, depth_map=np.ones((10, 10), dtype=np.float32) * 3.0),
            MockDepthFrame(frame_idx=2, depth_map=np.ones((10, 10), dtype=np.float32) * 5.0),
        ]
        result = analyze_depth(frames, None, "da_v2")
        assert result.num_frames == 3
        assert result.depth_range == pytest.approx(4.0)
        assert result.depth_mean == pytest.approx(3.0)
        assert result.temporal_mean_delta == pytest.approx(2.0)
        assert result.temporal_max_delta == pytest.approx(2.0)


# ── Tests: analyze_hand_pose ───────────────────────────────────────


class TestAnalyzeHandPose:
    def test_empty(self):
        result = analyze_hand_pose([], 30.0)
        assert result.total_frames == 0

    def test_basic_poses(self):
        poses = []
        for i in range(5):
            hp = MockHandPose3D(
                wrist_position=[float(i) * 0.01, 0.0, 0.0],
                hand_pose=[float(i) * 0.1] * 45,
                global_orient=[float(i) * 0.05] * 3,
            )
            poses.append(MockFrameHandPose(frame_idx=i, hands=[hp]))

        result = analyze_hand_pose(poses, 30.0)
        assert result.total_frames == 5
        assert result.frames_with_poses == 5
        assert result.pose_detection_rate == 1.0
        assert result.mano_pose_range > 0
        assert result.wrist_velocity_mean > 0

    def test_no_hands_in_frames(self):
        poses = [MockFrameHandPose(frame_idx=i, hands=[]) for i in range(3)]
        result = analyze_hand_pose(poses, 30.0)
        assert result.total_frames == 3
        assert result.frames_with_poses == 0
        assert result.pose_detection_rate == 0.0


# ── Tests: analyze_object_detection ────────────────────────────────


class TestAnalyzeObjectDetection:
    def test_empty(self):
        result = analyze_object_detection([])
        assert result.total_frames == 0

    def test_basic_objects(self):
        frames = [
            MockFrameObjects(frame_idx=0, objects=[
                MockDetectedObject(label="cup", confidence=0.9),
                MockDetectedObject(label="bottle", confidence=0.8),
            ]),
            MockFrameObjects(frame_idx=1, objects=[
                MockDetectedObject(label="cup", confidence=0.85),
            ]),
            MockFrameObjects(frame_idx=2, objects=[]),
        ]
        result = analyze_object_detection(frames)
        assert result.total_frames == 3
        assert result.frames_with_objects == 2
        assert result.detection_rate == pytest.approx(2 / 3)
        assert result.unique_labels == 2
        assert result.avg_objects_per_frame == pytest.approx(1.0)
        assert result.confidence.count == 3

    def test_label_persistence(self):
        # "cup" appears in all 3 frames, "bottle" in 1
        frames = [
            MockFrameObjects(frame_idx=i, objects=[
                MockDetectedObject(label="cup", confidence=0.9),
            ])
            for i in range(3)
        ]
        frames[0].objects.append(MockDetectedObject(label="bottle", confidence=0.7))
        result = analyze_object_detection(frames)
        # cup: 3/3=1.0, bottle: 1/3=0.333, avg = 0.667
        assert result.label_persistence_rate == pytest.approx(2 / 3, abs=0.01)


# ── Tests: analyze_segmentation ────────────────────────────────────


class TestAnalyzeSegmentation:
    def test_empty(self):
        result = analyze_segmentation([], None, None)
        assert result.total_frames == 0

    def test_basic_segmentation(self):
        segs = [
            MockFrameSegmentation(frame_idx=0, masks=[MockSegmentationMask(area_fraction=0.1)]),
            MockFrameSegmentation(frame_idx=1, masks=[MockSegmentationMask(area_fraction=0.2)]),
            MockFrameSegmentation(frame_idx=2, masks=[]),
        ]
        result = analyze_segmentation(segs, None, None)
        assert result.total_frames == 3
        assert result.frames_with_masks == 2
        assert result.mask_rate == pytest.approx(2 / 3)
        assert result.avg_mask_area == pytest.approx(0.15)

    def test_with_hand_overlap(self):
        segs = [
            MockFrameSegmentation(frame_idx=0, masks=[MockSegmentationMask()]),
            MockFrameSegmentation(frame_idx=1, masks=[MockSegmentationMask()]),
        ]
        hand_dets = [
            MockFrameDetection(frame_idx=0, detected=True),
            MockFrameDetection(frame_idx=1, detected=True),
        ]
        result = analyze_segmentation(segs, hand_dets, None)
        assert result.hand_mask_overlap_rate == 1.0


# ── Tests: analyze_scene_annotations ───────────────────────────────


class TestAnalyzeSceneAnnotations:
    def test_empty(self):
        result = analyze_scene_annotations([], None)
        assert result.total_frames == 0

    def test_basic_annotations(self):
        annotations = [
            MockSceneAnnotation(frame_idx=0, caption="A hand is holding a cup"),
            MockSceneAnnotation(frame_idx=1, caption="A hand is picking up a bottle"),
            MockSceneAnnotation(frame_idx=2, caption="A hand is placing a cup"),
        ]
        result = analyze_scene_annotations(annotations, None)
        assert result.total_frames == 3
        assert result.avg_caption_length > 0
        assert result.hand_mention_rate == 1.0  # all have hands_present
        assert result.action_mention_rate == 1.0  # all have action_description
        assert result.diversity_ratio == 1.0  # all unique captions

    def test_object_caption_alignment(self):
        annotations = [
            MockSceneAnnotation(frame_idx=0, caption="A hand holds a cup on the table"),
        ]
        obj_dets = [
            MockFrameObjects(frame_idx=0, objects=[
                MockDetectedObject(label="cup", confidence=0.9),
                MockDetectedObject(label="table", confidence=0.8),
            ]),
        ]
        result = analyze_scene_annotations(annotations, obj_dets)
        # Both "cup" and "table" appear in caption
        assert result.object_caption_alignment == pytest.approx(1.0)

    def test_low_diversity(self):
        annotations = [
            MockSceneAnnotation(frame_idx=i, caption="A hand is visible")
            for i in range(5)
        ]
        result = analyze_scene_annotations(annotations, None)
        assert result.diversity_ratio == pytest.approx(1 / 5)


# ── Tests: analyze_trajectories ────────────────────────────────────


class TestAnalyzeTrajectories:
    def test_empty_detections(self):
        result = analyze_trajectories(None, [], 30.0)
        assert result.total_frames == 0

    def test_no_trajectory_data(self):
        dets = [MockFrameDetection(frame_idx=i, detected=True, hands=[MockHandDetection()])
                for i in range(5)]
        result = analyze_trajectories(None, dets, 30.0)
        assert result.total_frames == 5
        assert result.frames_with_data == 5
        assert result.velocity_mean == 0.0

    def test_with_trajectory_data(self):
        positions = [{"x": float(i) * 0.01, "y": 0.0, "z": 0.0} for i in range(10)]
        joint = MockJointTrajectory(positions=positions)
        hand = MockHandTrajectory(joints=[joint])
        traj = MockTrajectoryData(hands=[hand])

        dets = [MockFrameDetection(frame_idx=i, detected=True, hands=[MockHandDetection()])
                for i in range(10)]
        result = analyze_trajectories(traj, dets, 30.0)
        assert result.velocity_mean > 0
        assert result.frames_with_data == 10


# ── Tests: analyze_lerobot_export ──────────────────────────────────


class TestAnalyzeLeRobotExport:
    def test_nonexistent_path(self):
        result = analyze_lerobot_export(Path("/nonexistent/path"))
        assert result.export_success is False
        assert result.parquet_row_count == 0

    def test_empty_directory(self, tmp_path):
        result = analyze_lerobot_export(tmp_path)
        assert result.export_success is False

    def test_with_metadata(self, tmp_path):
        # Create a metadata file
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        meta = {"state_dim": 168, "action_dim": 48}
        (meta_dir / "info.json").write_text(json.dumps(meta))

        # Create a fake video file
        (tmp_path / "video.mp4").write_text("fake")

        result = analyze_lerobot_export(tmp_path)
        assert result.export_success is True
        assert result.video_count == 1
        assert result.state_dim == 168
        assert result.action_dim == 48
        assert result.state_dim_valid is True
        assert result.action_dim_valid is True


# ── Tests: save_enhanced_report round-trip ─────────────────────────


class TestSaveEnhancedReport:
    def test_round_trip(self, tmp_path):
        results = [
            EnhancedBenchmarkResult(
                video_name="test_video",
                perf=PerfResult(
                    video_name="test_video",
                    file_size_mb=10.0,
                    duration_s=30.0,
                    total_frames=900,
                    stages=[
                        StageTiming(stage="detect", elapsed_s=1.5),
                        StageTiming(stage="depth", elapsed_s=2.3),
                        StageTiming(stage="hand_pose", skipped=True),
                    ],
                ),
                quality=[
                    StageQuality(
                        stage="detect",
                        detection=DetectionQualityMetrics(
                            total_frames=900,
                            detected_frames=800,
                            detection_rate=800 / 900,
                        ),
                    ),
                ],
            ),
        ]

        json_path, txt_path = save_enhanced_report(results, tmp_path)

        assert json_path.exists()
        assert txt_path.exists()

        # Read back JSON
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["video_name"] == "test_video"
        assert data[0]["perf"]["file_size_mb"] == 10.0
        assert data[0]["perf"]["total_time"] == pytest.approx(1.5 + 2.3)
        assert len(data[0]["perf"]["stages"]) == 3
        assert data[0]["quality"][0]["stage"] == "detect"
        assert data[0]["quality"][0]["detection"]["detection_rate"] == pytest.approx(800 / 900)

        # Read back text
        text = txt_path.read_text()
        assert "ENHANCED BENCHMARK REPORT" in text
        assert "test_video" in text

    def test_with_run_dir(self, tmp_path):
        run_dir = tmp_path / "run_001"
        results = [EnhancedBenchmarkResult(video_name="test")]
        json_path, txt_path = save_enhanced_report(results, tmp_path, run_dir=run_dir)
        assert json_path.parent == run_dir
        assert txt_path.parent == run_dir


# ── Tests: format_enhanced_report ──────────────────────────────────


class TestFormatEnhancedReport:
    def test_empty_results(self):
        text = format_enhanced_report([])
        assert "ENHANCED BENCHMARK REPORT" in text

    def test_full_report(self):
        results = [
            EnhancedBenchmarkResult(
                video_name="test",
                perf=PerfResult(
                    video_name="test", file_size_mb=5.0, duration_s=10.0, total_frames=300,
                    stages=[StageTiming(stage="detect", elapsed_s=1.0)],
                ),
                quality=[
                    StageQuality(stage="detect", detection=DetectionQualityMetrics(
                        total_frames=300, detected_frames=250, detection_rate=250/300,
                        confidence=ConfidenceStats(count=250, mean=0.9),
                        temporal=TemporalConsistency(longest_streak=200),
                    )),
                    StageQuality(stage="depth", depth=DepthQualityMetrics(
                        model_name="da_v2", num_frames=300, depth_range=10.0,
                    )),
                    StageQuality(stage="hand_pose", hand_pose=HandPoseQualityMetrics(
                        total_frames=300, frames_with_poses=280, pose_detection_rate=280/300,
                    )),
                    StageQuality(stage="object_detection", object_detection=ObjectDetectionQualityMetrics(
                        total_frames=300, unique_labels=5,
                    )),
                    StageQuality(stage="segmentation", segmentation=SegmentationQualityMetrics(
                        total_frames=300, mask_rate=0.9,
                    )),
                    StageQuality(stage="scene_annotation", scene_annotation=SceneAnnotationQualityMetrics(
                        total_frames=300, diversity_ratio=0.8,
                    )),
                    StageQuality(stage="trajectories", trajectories=TrajectoryQualityMetrics(
                        total_frames=300, frames_with_data=250,
                    )),
                    StageQuality(stage="lerobot_export", lerobot_export=LeRobotExportQualityMetrics(
                        export_success=True, parquet_row_count=1000,
                    )),
                ],
            ),
        ]
        text = format_enhanced_report(results)
        assert "PERFORMANCE TIMING" in text
        assert "QUALITY METRICS" in text
        assert "[detect]" in text
        assert "[depth]" in text
        assert "[hand_pose]" in text
        assert "[object_detection]" in text
        assert "[segmentation]" in text
        assert "[scene_annotation]" in text
        assert "[trajectories]" in text
        assert "[lerobot_export]" in text


# ── Tests: PerfResult properties ───────────────────────────────────


class TestPerfResult:
    def test_total_time(self):
        pr = PerfResult(
            file_size_mb=10.0,
            duration_s=30.0,
            stages=[
                StageTiming(stage="a", elapsed_s=1.0),
                StageTiming(stage="b", elapsed_s=2.0),
                StageTiming(stage="c", elapsed_s=0.0, skipped=True),
            ],
        )
        assert pr.total_time == pytest.approx(3.0)
        assert pr.time_per_mb == pytest.approx(0.3)
        assert pr.time_per_second_of_video == pytest.approx(0.1)

    def test_zero_size(self):
        pr = PerfResult(file_size_mb=0.0, duration_s=0.0, stages=[])
        assert pr.total_time == 0.0
        assert pr.time_per_mb == 0.0
        assert pr.time_per_second_of_video == 0.0
