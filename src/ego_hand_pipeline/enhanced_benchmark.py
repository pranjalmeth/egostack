"""Unified enhanced benchmark — performance timing + quality metrics for all 11 pipeline stages."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ── Shared helpers ──────────────────────────────────────────────────


@dataclass
class ConfidenceStats:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0


@dataclass
class TemporalConsistency:
    flicker_count: int = 0
    longest_streak: int = 0
    gap_segment_count: int = 0
    frame_to_frame_change_rate: float = 0.0


def _compute_confidence_stats(values: list[float]) -> ConfidenceStats:
    if not values:
        return ConfidenceStats()
    arr = sorted(values)
    n = len(arr)
    return ConfidenceStats(
        count=n,
        mean=float(statistics.mean(arr)),
        median=float(statistics.median(arr)),
        std=float(statistics.stdev(arr)) if n > 1 else 0.0,
        min=float(arr[0]),
        max=float(arr[-1]),
        p25=float(arr[int(n * 0.25)]),
        p75=float(arr[int(n * 0.75)]),
        p90=float(arr[int(n * 0.90)]),
    )


def _compute_temporal_consistency(detected_flags: list[bool]) -> TemporalConsistency:
    if not detected_flags:
        return TemporalConsistency()

    longest_streak = 0
    current_streak = 0
    gap_segments: list[int] = []
    current_gap = 0
    changes = 0

    for i, flag in enumerate(detected_flags):
        if flag:
            if current_gap > 0:
                gap_segments.append(current_gap)
                current_gap = 0
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
            current_gap += 1
        if i > 0 and detected_flags[i] != detected_flags[i - 1]:
            changes += 1

    if current_gap > 0:
        gap_segments.append(current_gap)

    flicker_count = sum(1 for g in gap_segments if g <= 5)
    n = len(detected_flags)
    change_rate = changes / (n - 1) if n > 1 else 0.0

    return TemporalConsistency(
        flicker_count=flicker_count,
        longest_streak=longest_streak,
        gap_segment_count=len(gap_segments),
        frame_to_frame_change_rate=change_rate,
    )


# ── Performance dataclasses ─────────────────────────────────────────


@dataclass
class StageTiming:
    stage: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False
    error: str | None = None


@dataclass
class PerfResult:
    video_name: str = ""
    file_size_mb: float = 0.0
    duration_s: float = 0.0
    total_frames: int = 0
    stages: list[StageTiming] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return sum(s.elapsed_s for s in self.stages if not s.skipped)

    @property
    def time_per_mb(self) -> float:
        return self.total_time / self.file_size_mb if self.file_size_mb > 0 else 0.0

    @property
    def time_per_second_of_video(self) -> float:
        return self.total_time / self.duration_s if self.duration_s > 0 else 0.0


# ── Quality dataclasses per stage ───────────────────────────────────


@dataclass
class DetectionQualityMetrics:
    total_frames: int = 0
    detected_frames: int = 0
    detection_rate: float = 0.0
    both_hands_frames: int = 0
    both_hands_rate: float = 0.0
    confidence: ConfidenceStats = field(default_factory=ConfidenceStats)
    temporal: TemporalConsistency = field(default_factory=TemporalConsistency)
    left_count: int = 0
    right_count: int = 0
    lr_balance: float = 0.0


@dataclass
class ClipQualityMetrics:
    clip_id: str = ""
    expected_frames: int = 0
    actual_frames: int = 0
    frame_delta: int = 0
    expected_duration: float = 0.0
    actual_duration: float = 0.0
    duration_delta: float = 0.0
    detection_coverage: float = 0.0


@dataclass
class DepthQualityMetrics:
    model_name: str = ""
    num_frames: int = 0
    depth_range: float = 0.0
    depth_mean: float = 0.0
    depth_std: float = 0.0
    temporal_mean_delta: float = 0.0
    temporal_max_delta: float = 0.0


@dataclass
class HandPoseQualityMetrics:
    total_frames: int = 0
    frames_with_poses: int = 0
    pose_detection_rate: float = 0.0
    mano_pose_range: float = 0.0
    mano_orient_range: float = 0.0
    wrist_velocity_mean: float = 0.0
    wrist_velocity_max: float = 0.0
    wrist_jerk_mean: float = 0.0
    joint_stability: float = 0.0
    fingertip_spread_std: float = 0.0


@dataclass
class ObjectDetectionQualityMetrics:
    total_frames: int = 0
    frames_with_objects: int = 0
    detection_rate: float = 0.0
    unique_labels: int = 0
    avg_objects_per_frame: float = 0.0
    confidence: ConfidenceStats = field(default_factory=ConfidenceStats)
    label_persistence_rate: float = 0.0


@dataclass
class SegmentationQualityMetrics:
    total_frames: int = 0
    frames_with_masks: int = 0
    mask_rate: float = 0.0
    avg_mask_area: float = 0.0
    hand_mask_overlap_rate: float = 0.0
    object_mask_overlap_rate: float = 0.0
    mask_count_std: float = 0.0


@dataclass
class SceneAnnotationQualityMetrics:
    total_frames: int = 0
    avg_caption_length: float = 0.0
    hand_mention_rate: float = 0.0
    object_mention_rate: float = 0.0
    action_mention_rate: float = 0.0
    diversity_ratio: float = 0.0
    object_caption_alignment: float = 0.0


@dataclass
class TrajectoryQualityMetrics:
    total_frames: int = 0
    frames_with_data: int = 0
    velocity_mean: float = 0.0
    velocity_max: float = 0.0
    acceleration_mean: float = 0.0
    jerk_mean: float = 0.0
    jerk_max: float = 0.0
    stability: float = 0.0


@dataclass
class LeRobotExportQualityMetrics:
    export_success: bool = False
    parquet_row_count: int = 0
    video_count: int = 0
    state_dim: int = 0
    action_dim: int = 0
    state_dim_valid: bool = False
    action_dim_valid: bool = False


# ── Top-level result ────────────────────────────────────────────────


@dataclass
class StageQuality:
    stage: str = ""
    detection: DetectionQualityMetrics | None = None
    clips: list[ClipQualityMetrics] | None = None
    depth: DepthQualityMetrics | None = None
    hand_pose: HandPoseQualityMetrics | None = None
    object_detection: ObjectDetectionQualityMetrics | None = None
    segmentation: SegmentationQualityMetrics | None = None
    scene_annotation: SceneAnnotationQualityMetrics | None = None
    trajectories: TrajectoryQualityMetrics | None = None
    lerobot_export: LeRobotExportQualityMetrics | None = None


@dataclass
class EnhancedBenchmarkResult:
    video_name: str = ""
    perf: PerfResult = field(default_factory=PerfResult)
    quality: list[StageQuality] = field(default_factory=list)


# ── Analysis functions ──────────────────────────────────────────────


def _safe_mean(vals: list[float]) -> float:
    return float(statistics.mean(vals)) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def analyze_detection(detections: list) -> DetectionQualityMetrics:
    """Analyze hand detection quality."""
    total = len(detections)
    if total == 0:
        return DetectionQualityMetrics()

    detected = sum(1 for d in detections if d.detected)
    both_hands = sum(1 for d in detections if d.num_hands >= 2)

    confidences: list[float] = []
    left_count = 0
    right_count = 0
    for d in detections:
        for h in d.hands:
            confidences.append(h.confidence)
            if h.hand_label == "left":
                left_count += 1
            elif h.hand_label == "right":
                right_count += 1

    detected_flags = [d.detected for d in detections]
    temporal = _compute_temporal_consistency(detected_flags)

    lr_max = max(left_count, right_count)
    lr_min = min(left_count, right_count)

    return DetectionQualityMetrics(
        total_frames=total,
        detected_frames=detected,
        detection_rate=detected / total,
        both_hands_frames=both_hands,
        both_hands_rate=both_hands / total,
        confidence=_compute_confidence_stats(confidences),
        temporal=temporal,
        left_count=left_count,
        right_count=right_count,
        lr_balance=lr_min / lr_max if lr_max > 0 else 0.0,
    )


def analyze_clips(clips: list, detections: list, fps: float) -> list[ClipQualityMetrics]:
    """Analyze clip extraction quality."""
    results: list[ClipQualityMetrics] = []

    for clip in clips:
        expected_dur = clip.end_time - clip.start_time
        expected_frames = int(round(expected_dur * fps))

        cap = cv2.VideoCapture(str(clip.path))
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        actual_dur = actual_frames / actual_fps if actual_fps > 0 else 0.0
        cap.release()

        clip_dets = [
            d for d in detections
            if clip.start_frame <= d.frame_idx <= clip.end_frame
        ]
        covered = sum(1 for d in clip_dets if d.detected)
        coverage = covered / len(clip_dets) if clip_dets else 0.0

        results.append(ClipQualityMetrics(
            clip_id=clip.clip_id,
            expected_frames=expected_frames,
            actual_frames=actual_frames,
            frame_delta=actual_frames - expected_frames,
            expected_duration=round(expected_dur, 4),
            actual_duration=round(actual_dur, 4),
            duration_delta=round(actual_dur - expected_dur, 4),
            detection_coverage=round(coverage, 4),
        ))

    return results


def analyze_depth(
    depth_frames: list,
    detections: list | None,
    model_name: str,
) -> DepthQualityMetrics:
    """Analyze depth estimation quality for a single model."""
    if not depth_frames:
        return DepthQualityMetrics(model_name=model_name)

    all_mins: list[float] = []
    all_maxs: list[float] = []
    all_means: list[float] = []
    all_stds: list[float] = []

    for df in depth_frames:
        dm = df.depth_map
        all_mins.append(float(dm.min()))
        all_maxs.append(float(dm.max()))
        all_means.append(float(dm.mean()))
        all_stds.append(float(dm.std()))

    depth_range = float(max(all_maxs) - min(all_mins))
    depth_mean = float(statistics.mean(all_means))
    depth_std = float(statistics.mean(all_stds))

    # Temporal consistency
    deltas: list[float] = []
    if detections:
        det_by_frame = {d.frame_idx: d for d in detections}
        for i in range(1, len(depth_frames)):
            prev_df = depth_frames[i - 1]
            curr_df = depth_frames[i]
            # Compare mean depth at landmark positions
            for df_pair in [(prev_df, curr_df)]:
                pass  # just need the pair
            prev_mean = float(prev_df.depth_map.mean())
            curr_mean = float(curr_df.depth_map.mean())
            deltas.append(abs(curr_mean - prev_mean))
    else:
        for i in range(1, len(depth_frames)):
            prev_mean = float(depth_frames[i - 1].depth_map.mean())
            curr_mean = float(depth_frames[i].depth_map.mean())
            deltas.append(abs(curr_mean - prev_mean))

    temporal_mean = _safe_mean(deltas)
    temporal_max = float(max(deltas)) if deltas else 0.0

    return DepthQualityMetrics(
        model_name=model_name,
        num_frames=len(depth_frames),
        depth_range=depth_range,
        depth_mean=depth_mean,
        depth_std=depth_std,
        temporal_mean_delta=temporal_mean,
        temporal_max_delta=temporal_max,
    )


def analyze_hand_pose(hand_poses: list, fps: float) -> HandPoseQualityMetrics:
    """Analyze 3D hand pose estimation quality."""
    if not hand_poses:
        return HandPoseQualityMetrics()

    total = len(hand_poses)
    frames_with = sum(1 for fp in hand_poses if fp.hands)

    all_pose_vals: list[float] = []
    all_orient_vals: list[float] = []
    wrist_positions: list[list[float]] = []
    fingertip_spreads: list[float] = []

    for fp in hand_poses:
        for hp in fp.hands:
            all_pose_vals.extend(hp.hand_pose)
            all_orient_vals.extend(hp.global_orient)
            wrist_positions.append(hp.wrist_position)

            # Fingertip spread: std of distances between fingertip positions
            if hp.fingertip_positions:
                xs = [f["x"] for f in hp.fingertip_positions]
                ys = [f["y"] for f in hp.fingertip_positions]
                if len(xs) >= 2:
                    spread = statistics.stdev(xs) + statistics.stdev(ys)
                    fingertip_spreads.append(spread)

    pose_range = (max(all_pose_vals) - min(all_pose_vals)) if all_pose_vals else 0.0
    orient_range = (max(all_orient_vals) - min(all_orient_vals)) if all_orient_vals else 0.0

    # Wrist velocity and jerk
    dt = 1.0 / fps if fps > 0 else 1.0
    velocities: list[float] = []
    for i in range(1, len(wrist_positions)):
        dx = wrist_positions[i][0] - wrist_positions[i - 1][0]
        dy = wrist_positions[i][1] - wrist_positions[i - 1][1]
        dz = wrist_positions[i][2] - wrist_positions[i - 1][2]
        velocities.append(math.sqrt(dx**2 + dy**2 + dz**2) / dt)

    jerks: list[float] = []
    if len(velocities) >= 2:
        accels: list[float] = []
        for i in range(1, len(velocities)):
            accels.append(abs(velocities[i] - velocities[i - 1]) / dt)
        for i in range(1, len(accels)):
            jerks.append(abs(accels[i] - accels[i - 1]) / dt)

    # Joint stability: mean displacement when velocity is low
    velocity_threshold = 0.005
    still_disps: list[float] = []
    for i in range(1, len(wrist_positions)):
        dx = wrist_positions[i][0] - wrist_positions[i - 1][0]
        dy = wrist_positions[i][1] - wrist_positions[i - 1][1]
        dz = wrist_positions[i][2] - wrist_positions[i - 1][2]
        disp = math.sqrt(dx**2 + dy**2 + dz**2)
        vmag = disp / dt
        if vmag < velocity_threshold:
            still_disps.append(disp)

    return HandPoseQualityMetrics(
        total_frames=total,
        frames_with_poses=frames_with,
        pose_detection_rate=frames_with / total if total > 0 else 0.0,
        mano_pose_range=pose_range,
        mano_orient_range=orient_range,
        wrist_velocity_mean=_safe_mean(velocities),
        wrist_velocity_max=max(velocities) if velocities else 0.0,
        wrist_jerk_mean=_safe_mean(jerks),
        joint_stability=_safe_mean(still_disps),
        fingertip_spread_std=_safe_std(fingertip_spreads),
    )


def analyze_object_detection(frame_objects: list) -> ObjectDetectionQualityMetrics:
    """Analyze object detection quality."""
    if not frame_objects:
        return ObjectDetectionQualityMetrics()

    total = len(frame_objects)
    frames_with = sum(1 for fo in frame_objects if fo.objects)

    all_confidences: list[float] = []
    all_labels: set[str] = set()
    counts: list[int] = []
    label_sets: list[set[str]] = []

    for fo in frame_objects:
        counts.append(len(fo.objects))
        frame_labels: set[str] = set()
        for obj in fo.objects:
            all_confidences.append(obj.confidence)
            all_labels.add(obj.label)
            frame_labels.add(obj.label)
        label_sets.append(frame_labels)

    # Label persistence: fraction of frames where each label appears (averaged)
    persistence = 0.0
    if all_labels and total > 0:
        per_label = []
        for label in all_labels:
            present = sum(1 for ls in label_sets if label in ls)
            per_label.append(present / total)
        persistence = _safe_mean(per_label)

    return ObjectDetectionQualityMetrics(
        total_frames=total,
        frames_with_objects=frames_with,
        detection_rate=frames_with / total if total > 0 else 0.0,
        unique_labels=len(all_labels),
        avg_objects_per_frame=_safe_mean([float(c) for c in counts]),
        confidence=_compute_confidence_stats(all_confidences),
        label_persistence_rate=persistence,
    )


def analyze_segmentation(
    segmentations: list,
    hand_detections: list | None,
    object_detections: list | None,
) -> SegmentationQualityMetrics:
    """Analyze segmentation quality."""
    if not segmentations:
        return SegmentationQualityMetrics()

    total = len(segmentations)
    frames_with = sum(1 for s in segmentations if s.masks)

    areas: list[float] = []
    mask_counts: list[float] = []
    hand_overlaps = 0
    object_overlaps = 0
    hand_total = 0
    object_total = 0

    hand_by_frame = {}
    if hand_detections:
        hand_by_frame = {d.frame_idx: d for d in hand_detections}
    obj_by_frame = {}
    if object_detections:
        obj_by_frame = {fo.frame_idx: fo for fo in object_detections}

    for seg in segmentations:
        mask_counts.append(float(len(seg.masks)))
        for mask in seg.masks:
            areas.append(mask.area_fraction)

        # Check hand overlap
        hdet = hand_by_frame.get(seg.frame_idx)
        if hdet and hdet.detected:
            hand_total += 1
            if seg.masks:
                hand_overlaps += 1

        # Check object overlap
        odet = obj_by_frame.get(seg.frame_idx)
        if odet and odet.objects:
            object_total += 1
            if seg.masks:
                object_overlaps += 1

    return SegmentationQualityMetrics(
        total_frames=total,
        frames_with_masks=frames_with,
        mask_rate=frames_with / total if total > 0 else 0.0,
        avg_mask_area=_safe_mean(areas),
        hand_mask_overlap_rate=hand_overlaps / hand_total if hand_total > 0 else 0.0,
        object_mask_overlap_rate=object_overlaps / object_total if object_total > 0 else 0.0,
        mask_count_std=_safe_std(mask_counts),
    )


def analyze_scene_annotations(
    annotations: list,
    object_detections: list | None,
) -> SceneAnnotationQualityMetrics:
    """Analyze scene annotation quality."""
    if not annotations:
        return SceneAnnotationQualityMetrics()

    total = len(annotations)
    caption_lengths: list[float] = []
    hand_mentions = 0
    object_mentions = 0
    action_mentions = 0
    unique_captions: set[str] = set()

    obj_by_frame = {}
    if object_detections:
        obj_by_frame = {fo.frame_idx: fo for fo in object_detections}

    alignment_scores: list[float] = []

    for ann in annotations:
        caption = ann.caption
        caption_lengths.append(float(len(caption)))
        unique_captions.add(caption)

        lower = caption.lower()
        if ann.hands_present or "hand" in lower:
            hand_mentions += 1
        if ann.objects_present or "object" in lower:
            object_mentions += 1
        if ann.action_description or any(w in lower for w in ("pick", "grasp", "hold", "move", "touch", "place", "push", "pull")):
            action_mentions += 1

        # Object-caption alignment: do detected objects appear in caption?
        odet = obj_by_frame.get(ann.frame_idx)
        if odet and odet.objects:
            mentioned = sum(1 for obj in odet.objects if obj.label.lower() in lower)
            alignment_scores.append(mentioned / len(odet.objects))

    diversity = len(unique_captions) / total if total > 0 else 0.0

    return SceneAnnotationQualityMetrics(
        total_frames=total,
        avg_caption_length=_safe_mean(caption_lengths),
        hand_mention_rate=hand_mentions / total if total > 0 else 0.0,
        object_mention_rate=object_mentions / total if total > 0 else 0.0,
        action_mention_rate=action_mentions / total if total > 0 else 0.0,
        diversity_ratio=diversity,
        object_caption_alignment=_safe_mean(alignment_scores),
    )


def analyze_trajectories(
    traj_data,
    detections: list,
    fps: float,
) -> TrajectoryQualityMetrics:
    """Analyze trajectory quality (aggregated across joints)."""
    total_frames = len(detections)
    frames_with_data = sum(1 for d in detections if d.detected and d.hands)

    if not traj_data or not traj_data.hands:
        return TrajectoryQualityMetrics(
            total_frames=total_frames,
            frames_with_data=frames_with_data,
        )

    dt = 1.0 / fps if fps > 0 else 1.0
    velocity_threshold = 0.005
    all_velocities: list[float] = []
    all_accelerations: list[float] = []
    all_jerks: list[float] = []
    all_still_disps: list[float] = []

    for hand_traj in traj_data.hands:
        for jt in hand_traj.joints:
            positions = jt.positions
            if len(positions) < 2:
                continue

            vel_mags: list[float] = []
            for i in range(1, len(positions)):
                dx = positions[i]["x"] - positions[i - 1]["x"]
                dy = positions[i]["y"] - positions[i - 1]["y"]
                dz = positions[i]["z"] - positions[i - 1]["z"]
                vmag = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                vel_mags.append(vmag)
                disp = math.sqrt(dx**2 + dy**2 + dz**2)
                if vmag < velocity_threshold:
                    all_still_disps.append(disp)

            all_velocities.extend(vel_mags)

            if len(vel_mags) >= 2:
                accel_mags: list[float] = []
                for i in range(1, len(vel_mags)):
                    accel_mags.append(abs(vel_mags[i] - vel_mags[i - 1]) / dt)
                all_accelerations.extend(accel_mags)

                for i in range(1, len(accel_mags)):
                    all_jerks.append(abs(accel_mags[i] - accel_mags[i - 1]) / dt)

    return TrajectoryQualityMetrics(
        total_frames=total_frames,
        frames_with_data=frames_with_data,
        velocity_mean=_safe_mean(all_velocities),
        velocity_max=max(all_velocities) if all_velocities else 0.0,
        acceleration_mean=_safe_mean(all_accelerations),
        jerk_mean=_safe_mean(all_jerks),
        jerk_max=max(all_jerks) if all_jerks else 0.0,
        stability=_safe_mean(all_still_disps),
    )


def analyze_lerobot_export(
    dataset_path: Path | str,
    expected_episodes: int = 1,
) -> LeRobotExportQualityMetrics:
    """Analyze LeRobot export quality by inspecting output files."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return LeRobotExportQualityMetrics()

    # Count parquet rows
    parquet_files = list(dataset_path.glob("**/*.parquet"))
    row_count = 0
    if parquet_files:
        try:
            import pyarrow.parquet as pq
            for pf in parquet_files:
                table = pq.read_table(pf)
                row_count += table.num_rows
                # Check dims from first file
                if table.num_rows > 0:
                    cols = table.column_names
                    state_cols = [c for c in cols if c.startswith("observation.state")]
                    action_cols = [c for c in cols if c.startswith("action")]
        except ImportError:
            # Fallback: just count files
            row_count = len(parquet_files)

    # Count video files
    video_files = list(dataset_path.glob("**/*.mp4"))
    video_count = len(video_files)

    # Check metadata for dimensions
    state_dim = 0
    action_dim = 0
    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        meta_path = dataset_path / "metadata.json"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            state_dim = meta.get("state_dim", 0)
            action_dim = meta.get("action_dim", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    # If no metadata, check parquet columns
    if state_dim == 0 and parquet_files:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_files[0])
            cols = table.column_names
            state_dim = sum(1 for c in cols if "state" in c.lower())
            action_dim = sum(1 for c in cols if "action" in c.lower())
        except (ImportError, Exception):
            pass

    export_success = row_count > 0 or video_count > 0

    return LeRobotExportQualityMetrics(
        export_success=export_success,
        parquet_row_count=row_count,
        video_count=video_count,
        state_dim=state_dim,
        action_dim=action_dim,
        state_dim_valid=state_dim == 168,
        action_dim_valid=action_dim == 48,
    )


# ── Orchestrator ────────────────────────────────────────────────────


def _get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _get_video_info(video_path: Path) -> tuple[float, float, int]:
    """Return (file_size_mb, duration_s, total_frames)."""
    size_mb = video_path.stat().st_size / (1024 * 1024)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return size_mb, duration_s, total_frames


def _time_stage(name: str, func, *args, **kwargs) -> tuple[StageTiming, object]:
    """Time a pipeline stage, returning (timing, result)."""
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return StageTiming(stage=name, elapsed_s=elapsed), result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"    WARNING: {name} failed: {e}")
        return StageTiming(stage=name, elapsed_s=elapsed, error=str(e)), None


def run_enhanced_benchmark(
    config,
    video_paths: list[Path],
    *,
    skip_stages: set[str] | None = None,
    max_frames: int | None = None,
) -> list[EnhancedBenchmarkResult]:
    """Run the unified enhanced benchmark on each video.

    Measures both timing and quality for all 11 pipeline stages.
    """
    from .pipeline import (
        _run_parallel,
        run_clip,
        run_depth,
        run_detect,
        run_hand_poses,
        run_lerobot_export,
        run_object_detection,
        run_preprocess,
        run_sample_render,
        run_scene_annotation,
        run_segmentation,
        run_trajectories,
    )

    skip = skip_stages or set()
    results: list[EnhancedBenchmarkResult] = []

    for video_path in video_paths:
        video_name = video_path.stem
        print(f"\n{'='*60}")
        print(f"Enhanced benchmark: {video_name}")

        result = EnhancedBenchmarkResult(video_name=video_name)

        # --- Preprocess ---
        working_path = video_path
        if "preprocess" not in skip and config.preprocessing.enabled:
            timing, preprocessed = _time_stage(
                "preprocess", run_preprocess, video_path, config, max_frames,
            )
            result.perf.stages.append(timing)
            if preprocessed:
                working_path = preprocessed
                print(f"  Preprocessed: {working_path}")
        elif "preprocess" in skip:
            result.perf.stages.append(StageTiming(stage="preprocess", skipped=True))

        # Get video info
        size_mb, duration_s, total_frames = _get_video_info(working_path)
        result.perf.video_name = video_name
        result.perf.file_size_mb = round(size_mb, 2)
        result.perf.duration_s = round(duration_s, 2)
        result.perf.total_frames = total_frames
        fps = _get_fps(working_path)

        # --- Detect ---
        detections = []
        if "detect" not in skip:
            timing, detections_result = _time_stage("detect", run_detect, working_path, config)
            result.perf.stages.append(timing)
            if detections_result:
                detections = detections_result
                quality = analyze_detection(detections)
                result.quality.append(StageQuality(stage="detect", detection=quality))
                print(f"  Detection: {quality.detected_frames}/{quality.total_frames} frames ({quality.detection_rate:.2%})")
        else:
            result.perf.stages.append(StageTiming(stage="detect", skipped=True))

        # --- Clip ---
        clips = []
        if "clip" not in skip:
            timing, clips_result = _time_stage("clip", run_clip, working_path, detections, config)
            result.perf.stages.append(timing)
            if clips_result:
                clips = clips_result
                clip_quality = analyze_clips(clips, detections, fps)
                result.quality.append(StageQuality(stage="clip", clips=clip_quality))
                print(f"  Clips: {len(clips)} extracted")
        else:
            result.perf.stages.append(StageTiming(stage="clip", skipped=True))

        # Use first clip for remaining stages
        if clips:
            clip_path = clips[0].path
            clip_info = clips[0]
        else:
            clip_path = working_path
            clip_info = None

        # Re-detect on clip for downstream stages
        clip_detections = detections
        if clips:
            timing, clip_det_result = _time_stage("clip_detect", run_detect, clip_path, config)
            result.perf.stages.append(timing)
            if clip_det_result:
                clip_detections = clip_det_result
            clip_fps = _get_fps(clip_path)
        else:
            clip_fps = fps

        # --- Depth ---
        depth_frames = []
        if "depth" not in skip:
            timing, depth_result = _time_stage("depth", run_depth, clip_path, config)
            result.perf.stages.append(timing)
            if depth_result:
                depth_frames = depth_result
                depth_quality = analyze_depth(depth_frames, clip_detections, config.depth.model)
                result.quality.append(StageQuality(stage="depth", depth=depth_quality))
                print(f"  Depth: {len(depth_frames)} frames")
        else:
            result.perf.stages.append(StageTiming(stage="depth", skipped=True))

        # --- Hand pose + Object detection (parallel) ---
        hand_poses = None
        obj_detections = None
        parallel_tasks = {}

        if "hand_pose" not in skip:
            parallel_tasks["hand_pose"] = (run_hand_poses, (clip_path, clip_detections, config))
        else:
            result.perf.stages.append(StageTiming(stage="hand_pose", skipped=True))

        if "object_detection" not in skip:
            parallel_tasks["object_detection"] = (run_object_detection, (clip_path, config))
        else:
            result.perf.stages.append(StageTiming(stage="object_detection", skipped=True))

        if parallel_tasks:
            t0 = time.perf_counter()
            parallel_results = _run_parallel(parallel_tasks)
            parallel_elapsed = time.perf_counter() - t0

            if "hand_pose" in parallel_results:
                hand_poses = parallel_results["hand_pose"]
                result.perf.stages.append(StageTiming(stage="hand_pose", elapsed_s=parallel_elapsed))
                if hand_poses:
                    hp_quality = analyze_hand_pose(hand_poses, clip_fps)
                    result.quality.append(StageQuality(stage="hand_pose", hand_pose=hp_quality))
                    print(f"  Hand pose: {hp_quality.frames_with_poses}/{hp_quality.total_frames} ({hp_quality.pose_detection_rate:.2%})")

            if "object_detection" in parallel_results:
                obj_detections = parallel_results["object_detection"]
                result.perf.stages.append(StageTiming(stage="object_detection", elapsed_s=parallel_elapsed))
                if obj_detections:
                    od_quality = analyze_object_detection(obj_detections)
                    result.quality.append(StageQuality(stage="object_detection", object_detection=od_quality))
                    print(f"  Objects: {od_quality.unique_labels} labels, {od_quality.avg_objects_per_frame:.1f}/frame")

        # --- Segmentation ---
        segmentations = None
        if "segmentation" not in skip:
            timing, seg_result = _time_stage(
                "segmentation", run_segmentation,
                clip_path, clip_detections, obj_detections, config,
            )
            result.perf.stages.append(timing)
            if seg_result:
                segmentations = seg_result
                seg_quality = analyze_segmentation(segmentations, clip_detections, obj_detections)
                result.quality.append(StageQuality(stage="segmentation", segmentation=seg_quality))
                print(f"  Segmentation: {seg_quality.frames_with_masks}/{seg_quality.total_frames} masks ({seg_quality.mask_rate:.2%})")
        else:
            result.perf.stages.append(StageTiming(stage="segmentation", skipped=True))

        # --- Scene annotation ---
        scene_annotations = None
        if "scene_annotation" not in skip:
            timing, ann_result = _time_stage(
                "scene_annotation", run_scene_annotation,
                clip_path, clip_detections, obj_detections, config,
            )
            result.perf.stages.append(timing)
            if ann_result:
                scene_annotations = ann_result
                ann_quality = analyze_scene_annotations(scene_annotations, obj_detections)
                result.quality.append(StageQuality(stage="scene_annotation", scene_annotation=ann_quality))
                print(f"  Scene annotation: {ann_quality.total_frames} frames, diversity={ann_quality.diversity_ratio:.2f}")
        else:
            result.perf.stages.append(StageTiming(stage="scene_annotation", skipped=True))

        # --- Trajectories ---
        traj_data = None
        if "trajectories" not in skip:
            timing, traj_result = _time_stage(
                "trajectories", run_trajectories,
                clip_path, clip_detections, depth_frames, clip_info, config,
            )
            result.perf.stages.append(timing)
            if traj_result:
                traj_data = traj_result
                traj_quality = analyze_trajectories(traj_data, clip_detections, clip_fps)
                result.quality.append(StageQuality(stage="trajectories", trajectories=traj_quality))
                print(f"  Trajectories: {traj_quality.frames_with_data}/{traj_quality.total_frames} frames")
        else:
            result.perf.stages.append(StageTiming(stage="trajectories", skipped=True))

        # --- LeRobot export ---
        if "lerobot_export" not in skip and traj_data is not None:
            episode = {
                "traj_data": traj_data,
                "clip_info": clip_info,
                "hand_poses": hand_poses,
                "object_detections": obj_detections,
                "segmentations": segmentations,
                "scene_annotations": scene_annotations,
                "video_path": clip_path,
            }
            timing, dataset_path = _time_stage(
                "lerobot_export", run_lerobot_export, [episode], config,
            )
            result.perf.stages.append(timing)
            if dataset_path:
                lr_quality = analyze_lerobot_export(dataset_path)
                result.quality.append(StageQuality(stage="lerobot_export", lerobot_export=lr_quality))
                print(f"  LeRobot: success={lr_quality.export_success}, rows={lr_quality.parquet_row_count}")
        elif "lerobot_export" in skip:
            result.perf.stages.append(StageTiming(stage="lerobot_export", skipped=True))

        # --- Sample render (optional, timed only) ---
        if "sample_render" not in skip:
            timing, _ = _time_stage(
                "sample_render", run_sample_render,
                clip_path, config,
                hand_detections=clip_detections,
                object_detections=obj_detections,
                depth_frames=depth_frames,
                segmentations=segmentations,
                scene_annotations=scene_annotations,
                max_frames=max_frames,
            )
            result.perf.stages.append(timing)
            print(f"  Sample render: {timing.elapsed_s:.2f}s")
        else:
            result.perf.stages.append(StageTiming(stage="sample_render", skipped=True))

        print(f"\n  TOTAL: {result.perf.total_time:.2f}s")
        results.append(result)

    return results


# ── Report formatting ───────────────────────────────────────────────


def format_enhanced_report(results: list[EnhancedBenchmarkResult]) -> str:
    """Format a human-readable enhanced benchmark report."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append("ENHANCED BENCHMARK REPORT")
    lines.append(sep)

    for result in results:
        lines.append(f"\nVideo: {result.video_name}")
        lines.append(f"  Size: {result.perf.file_size_mb:.1f} MB  Duration: {result.perf.duration_s:.1f}s  Frames: {result.perf.total_frames}")
        lines.append("")

        # ── Performance section ──
        lines.append("PERFORMANCE TIMING")
        lines.append(dash)
        lines.append(f"  {'Stage':<25} {'Time (s)':>10} {'Status':>10}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*10}")

        for st in result.perf.stages:
            if st.skipped:
                status = "SKIPPED"
                time_str = "-"
            elif st.error:
                status = "ERROR"
                time_str = f"{st.elapsed_s:.2f}"
            else:
                status = "OK"
                time_str = f"{st.elapsed_s:.2f}"
            lines.append(f"  {st.stage:<25} {time_str:>10} {status:>10}")

        lines.append(f"  {'-'*25} {'-'*10} {'-'*10}")
        lines.append(f"  {'TOTAL':<25} {result.perf.total_time:>10.2f}")
        lines.append(f"  Time/MB: {result.perf.time_per_mb:.2f}s   Time/sec_video: {result.perf.time_per_second_of_video:.2f}s")
        lines.append("")

        # ── Quality section ──
        lines.append("QUALITY METRICS")
        lines.append(dash)

        for sq in result.quality:
            lines.append(f"\n  [{sq.stage}]")

            if sq.detection:
                d = sq.detection
                lines.append(f"    Detection rate:     {d.detection_rate:.4f} ({d.detected_frames}/{d.total_frames})")
                lines.append(f"    Both-hands rate:    {d.both_hands_rate:.4f}")
                lines.append(f"    Confidence mean:    {d.confidence.mean:.4f}")
                lines.append(f"    L/R balance:        {d.lr_balance:.4f}")
                lines.append(f"    Flicker count:      {d.temporal.flicker_count}")
                lines.append(f"    Longest streak:     {d.temporal.longest_streak}")

            if sq.clips:
                for cq in sq.clips:
                    lines.append(f"    {cq.clip_id}: frames={cq.actual_frames} delta={cq.frame_delta:+d} coverage={cq.detection_coverage:.4f}")

            if sq.depth:
                dp = sq.depth
                lines.append(f"    Model: {dp.model_name}  Frames: {dp.num_frames}")
                lines.append(f"    Range: {dp.depth_range:.4f}  Mean: {dp.depth_mean:.4f}  Std: {dp.depth_std:.4f}")
                lines.append(f"    Temporal delta: mean={dp.temporal_mean_delta:.4f} max={dp.temporal_max_delta:.4f}")

            if sq.hand_pose:
                hp = sq.hand_pose
                lines.append(f"    Pose detection rate: {hp.pose_detection_rate:.4f} ({hp.frames_with_poses}/{hp.total_frames})")
                lines.append(f"    MANO pose range:     {hp.mano_pose_range:.4f}")
                lines.append(f"    Wrist vel mean:      {hp.wrist_velocity_mean:.6f}")
                lines.append(f"    Wrist jerk mean:     {hp.wrist_jerk_mean:.6f}")
                lines.append(f"    Joint stability:     {hp.joint_stability:.6f}")
                lines.append(f"    Fingertip spread:    {hp.fingertip_spread_std:.6f}")

            if sq.object_detection:
                od = sq.object_detection
                lines.append(f"    Detection rate:      {od.detection_rate:.4f} ({od.frames_with_objects}/{od.total_frames})")
                lines.append(f"    Unique labels:       {od.unique_labels}")
                lines.append(f"    Avg objects/frame:   {od.avg_objects_per_frame:.2f}")
                lines.append(f"    Confidence mean:     {od.confidence.mean:.4f}")
                lines.append(f"    Label persistence:   {od.label_persistence_rate:.4f}")

            if sq.segmentation:
                sg = sq.segmentation
                lines.append(f"    Mask rate:           {sg.mask_rate:.4f} ({sg.frames_with_masks}/{sg.total_frames})")
                lines.append(f"    Avg mask area:       {sg.avg_mask_area:.6f}")
                lines.append(f"    Hand overlap rate:   {sg.hand_mask_overlap_rate:.4f}")
                lines.append(f"    Object overlap rate: {sg.object_mask_overlap_rate:.4f}")

            if sq.scene_annotation:
                sa = sq.scene_annotation
                lines.append(f"    Avg caption length:  {sa.avg_caption_length:.1f}")
                lines.append(f"    Hand mention rate:   {sa.hand_mention_rate:.4f}")
                lines.append(f"    Object mention rate: {sa.object_mention_rate:.4f}")
                lines.append(f"    Action mention rate: {sa.action_mention_rate:.4f}")
                lines.append(f"    Diversity ratio:     {sa.diversity_ratio:.4f}")
                lines.append(f"    Object alignment:    {sa.object_caption_alignment:.4f}")

            if sq.trajectories:
                tr = sq.trajectories
                lines.append(f"    Frames with data:    {tr.frames_with_data}/{tr.total_frames}")
                lines.append(f"    Velocity mean:       {tr.velocity_mean:.6f}")
                lines.append(f"    Accel mean:          {tr.acceleration_mean:.6f}")
                lines.append(f"    Jerk mean:           {tr.jerk_mean:.6f}")
                lines.append(f"    Stability:           {tr.stability:.6f}")

            if sq.lerobot_export:
                lr = sq.lerobot_export
                lines.append(f"    Export success:       {lr.export_success}")
                lines.append(f"    Parquet rows:        {lr.parquet_row_count}")
                lines.append(f"    Video count:         {lr.video_count}")
                lines.append(f"    State dim:           {lr.state_dim} (valid={lr.state_dim_valid})")
                lines.append(f"    Action dim:          {lr.action_dim} (valid={lr.action_dim_valid})")

        lines.append("")

    return "\n".join(lines)


# ── Persistence ─────────────────────────────────────────────────────


def _sanitize_for_json(obj):
    """Recursively convert numpy scalars and handle NaN for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if math.isnan(v) else v
    if isinstance(obj, float):
        return 0.0 if math.isnan(obj) else obj
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def save_enhanced_report(
    results: list[EnhancedBenchmarkResult],
    output_dir: Path,
    *,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the enhanced benchmark report as JSON and text.

    Returns (json_path, txt_path).
    """
    dest = run_dir if run_dir is not None else output_dir
    dest.mkdir(parents=True, exist_ok=True)

    # Build serializable data
    data = []
    for r in results:
        entry = {
            "video_name": r.video_name,
            "perf": {
                "video_name": r.perf.video_name,
                "file_size_mb": r.perf.file_size_mb,
                "duration_s": r.perf.duration_s,
                "total_frames": r.perf.total_frames,
                "total_time": r.perf.total_time,
                "time_per_mb": r.perf.time_per_mb,
                "time_per_second_of_video": r.perf.time_per_second_of_video,
                "stages": [asdict(s) for s in r.perf.stages],
            },
            "quality": [],
        }
        for sq in r.quality:
            sq_data = {"stage": sq.stage}
            # Only include the populated quality field
            for attr in (
                "detection", "clips", "depth", "hand_pose",
                "object_detection", "segmentation", "scene_annotation",
                "trajectories", "lerobot_export",
            ):
                val = getattr(sq, attr, None)
                if val is not None:
                    if isinstance(val, list):
                        sq_data[attr] = [asdict(item) for item in val]
                    else:
                        sq_data[attr] = asdict(val)
            entry["quality"].append(sq_data)
        data.append(entry)

    data = _sanitize_for_json(data)

    json_path = dest / "enhanced_benchmark.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    txt_path = dest / "enhanced_benchmark.txt"
    txt_path.write_text(format_enhanced_report(results) + "\n")

    return json_path, txt_path
