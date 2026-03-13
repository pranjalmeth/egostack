"""Pipeline quality benchmark — measures accuracy and consistency of outputs."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .benchmark import resize_video
from .clipper import ClipInfo
from .depth_estimator import DepthFrame, estimate_depth
from .hand_detector import FrameDetection, LANDMARK_NAMES
from .pipeline import (
    _slice_detections_for_clip,
    run_clip,
    run_detect,
    run_trajectories,
)
from .trajectory_mapper import TrajectoryData


# Joints to highlight in reports
KEY_JOINTS = ["WRIST", "INDEX_FINGER_TIP", "THUMB_TIP", "MIDDLE_FINGER_TIP", "PINKY_TIP"]
KEY_JOINT_IDS = [LANDMARK_NAMES.index(j) for j in KEY_JOINTS]


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class ConfidenceStats:
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0


@dataclass
class DetectionQuality:
    total_frames: int = 0
    detected_frames: int = 0
    detection_rate: float = 0.0
    both_hands_frames: int = 0
    both_hands_rate: float = 0.0
    confidence: ConfidenceStats = field(default_factory=ConfidenceStats)
    # Temporal consistency
    flicker_count: int = 0  # gaps <= 5 frames between detections
    longest_streak: int = 0
    gap_segment_count: int = 0
    # Left/right balance
    left_count: int = 0
    right_count: int = 0
    lr_balance: float = 0.0  # min(L,R) / max(L,R)


@dataclass
class DepthModelStats:
    model_name: str = ""
    num_frames: int = 0
    depth_range: float = 0.0
    depth_mean: float = 0.0
    depth_std: float = 0.0
    temporal_mean_delta: float = 0.0
    temporal_max_delta: float = 0.0


@dataclass
class DepthQuality:
    midas: DepthModelStats = field(default_factory=DepthModelStats)
    depth_anything_v2: DepthModelStats = field(default_factory=DepthModelStats)
    cross_model_correlation: float = 0.0


@dataclass
class JointMotionStats:
    joint_id: int = 0
    name: str = ""
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    velocity_max: float = 0.0
    acceleration_mean: float = 0.0
    acceleration_std: float = 0.0
    acceleration_max: float = 0.0
    jerk_mean: float = 0.0
    jerk_max: float = 0.0
    stability: float = 0.0  # mean displacement when hand is "still"


@dataclass
class TrajectoryQuality:
    total_frames: int = 0
    frames_with_data: int = 0
    frames_missing: int = 0
    joints: list[JointMotionStats] = field(default_factory=list)


@dataclass
class ClipQuality:
    clip_id: str = ""
    expected_frames: int = 0
    actual_frames: int = 0
    frame_delta: int = 0
    expected_duration: float = 0.0
    actual_duration: float = 0.0
    duration_delta: float = 0.0
    detection_coverage: float = 0.0  # fraction of clip frames with hands


@dataclass
class QualityResult:
    video_name: str = ""
    detection: DetectionQuality = field(default_factory=DetectionQuality)
    depth: DepthQuality = field(default_factory=DepthQuality)
    trajectory: TrajectoryQuality = field(default_factory=TrajectoryQuality)
    clips: list[ClipQuality] = field(default_factory=list)


# ── Analysis functions ──────────────────────────────────────────────


def _compute_confidence_stats(values: list[float]) -> ConfidenceStats:
    if not values:
        return ConfidenceStats()
    arr = sorted(values)
    n = len(arr)
    return ConfidenceStats(
        mean=float(statistics.mean(arr)),
        median=float(statistics.median(arr)),
        std=float(statistics.stdev(arr)) if n > 1 else 0.0,
        min=float(arr[0]),
        max=float(arr[-1]),
        p25=float(arr[int(n * 0.25)]),
        p75=float(arr[int(n * 0.75)]),
        p90=float(arr[int(n * 0.90)]),
    )


def analyze_detection_quality(detections: list[FrameDetection]) -> DetectionQuality:
    """Analyze detection rate, confidence, temporal consistency, and L/R balance."""
    total = len(detections)
    if total == 0:
        return DetectionQuality()

    detected = sum(1 for d in detections if d.detected)
    both_hands = sum(1 for d in detections if d.num_hands >= 2)

    # Confidence from all hands across all frames
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

    # Temporal consistency: streaks, gaps, flickers
    longest_streak = 0
    current_streak = 0
    gap_segments: list[int] = []  # lengths of gap segments
    current_gap = 0

    for d in detections:
        if d.detected:
            if current_gap > 0:
                gap_segments.append(current_gap)
                current_gap = 0
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
            current_gap += 1
    if current_gap > 0:
        gap_segments.append(current_gap)

    flicker_count = sum(1 for g in gap_segments if g <= 5)

    lr_max = max(left_count, right_count)
    lr_min = min(left_count, right_count)

    return DetectionQuality(
        total_frames=total,
        detected_frames=detected,
        detection_rate=detected / total,
        both_hands_frames=both_hands,
        both_hands_rate=both_hands / total,
        confidence=_compute_confidence_stats(confidences),
        flicker_count=flicker_count,
        longest_streak=longest_streak,
        gap_segment_count=len(gap_segments),
        left_count=left_count,
        right_count=right_count,
        lr_balance=lr_min / lr_max if lr_max > 0 else 0.0,
    )


def _depth_at_landmarks(
    depth_frames: list[DepthFrame],
    detections: list[FrameDetection],
) -> list[list[float]]:
    """Extract depth values at hand landmark positions for each depth frame.

    Returns a list (one per depth frame) of depth values at all landmark positions
    found in the corresponding detection frame.
    """
    det_by_frame = {d.frame_idx: d for d in detections}
    per_frame_depths: list[list[float]] = []

    for df in depth_frames:
        det = det_by_frame.get(df.frame_idx)
        vals: list[float] = []
        if det and det.detected:
            h, w = df.depth_map.shape
            for hand in det.hands:
                for lm in hand.landmarks:
                    px = min(int(lm["x"] * w), w - 1)
                    py = min(int(lm["y"] * h), h - 1)
                    vals.append(float(df.depth_map[py, px]))
        per_frame_depths.append(vals)

    return per_frame_depths


def _analyze_single_depth_model(
    depth_frames: list[DepthFrame],
    detections: list[FrameDetection],
    model_name: str,
) -> DepthModelStats:
    """Compute stats for a single depth model's output."""
    if not depth_frames:
        return DepthModelStats(model_name=model_name)

    # Global depth stats across all frames
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

    # Temporal consistency: frame-to-frame depth delta at landmark positions
    landmark_depths = _depth_at_landmarks(depth_frames, detections)
    deltas: list[float] = []
    for i in range(1, len(landmark_depths)):
        prev = landmark_depths[i - 1]
        curr = landmark_depths[i]
        # Compare matching positions (take min length)
        n = min(len(prev), len(curr))
        if n > 0:
            for j in range(n):
                deltas.append(abs(curr[j] - prev[j]))

    temporal_mean = float(statistics.mean(deltas)) if deltas else 0.0
    temporal_max = float(max(deltas)) if deltas else 0.0

    return DepthModelStats(
        model_name=model_name,
        num_frames=len(depth_frames),
        depth_range=depth_range,
        depth_mean=depth_mean,
        depth_std=depth_std,
        temporal_mean_delta=temporal_mean,
        temporal_max_delta=temporal_max,
    )


def analyze_depth_quality(
    midas_frames: list[DepthFrame],
    da_v2_frames: list[DepthFrame],
    detections: list[FrameDetection],
) -> DepthQuality:
    """Analyze depth quality for both models and cross-model correlation."""
    midas_stats = _analyze_single_depth_model(midas_frames, detections, "midas")
    da_v2_stats = _analyze_single_depth_model(da_v2_frames, detections, "depth_anything_v2")

    # Cross-model correlation at shared landmark positions
    correlation = _cross_model_correlation(midas_frames, da_v2_frames, detections)

    return DepthQuality(
        midas=midas_stats,
        depth_anything_v2=da_v2_stats,
        cross_model_correlation=correlation,
    )


def _cross_model_correlation(
    midas_frames: list[DepthFrame],
    da_v2_frames: list[DepthFrame],
    detections: list[FrameDetection],
) -> float:
    """Pearson correlation of depth values at landmark positions across both models."""
    midas_by_frame = {df.frame_idx: df for df in midas_frames}
    da_v2_by_frame = {df.frame_idx: df for df in da_v2_frames}
    det_by_frame = {d.frame_idx: d for d in detections}

    midas_vals: list[float] = []
    da_v2_vals: list[float] = []

    shared_frames = set(midas_by_frame) & set(da_v2_by_frame)
    for fidx in sorted(shared_frames):
        det = det_by_frame.get(fidx)
        if not det or not det.detected:
            continue

        mdf = midas_by_frame[fidx]
        ddf = da_v2_by_frame[fidx]

        for hand in det.hands:
            for lm in hand.landmarks:
                mh, mw = mdf.depth_map.shape
                mpx = min(int(lm["x"] * mw), mw - 1)
                mpy = min(int(lm["y"] * mh), mh - 1)

                dh, dw = ddf.depth_map.shape
                dpx = min(int(lm["x"] * dw), dw - 1)
                dpy = min(int(lm["y"] * dh), dh - 1)

                midas_vals.append(float(mdf.depth_map[mpy, mpx]))
                da_v2_vals.append(float(ddf.depth_map[dpy, dpx]))

    if len(midas_vals) < 2:
        return 0.0

    # Pearson correlation
    n = len(midas_vals)
    mean_m = sum(midas_vals) / n
    mean_d = sum(da_v2_vals) / n
    cov = sum((m - mean_m) * (d - mean_d) for m, d in zip(midas_vals, da_v2_vals)) / n
    std_m = math.sqrt(sum((m - mean_m) ** 2 for m in midas_vals) / n)
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in da_v2_vals) / n)

    if std_m == 0 or std_d == 0:
        return 0.0

    return cov / (std_m * std_d)


def analyze_trajectory_quality(
    traj_data: TrajectoryData,
    detections: list[FrameDetection],
    fps: float,
) -> TrajectoryQuality:
    """Analyze trajectory smoothness and stability."""
    total_frames = len(detections)
    frames_with_data = sum(1 for d in detections if d.detected and d.hands)

    dt = 1.0 / fps if fps > 0 else 1.0
    velocity_threshold = 0.005  # normalized coords/frame — "still" threshold

    joint_stats: list[JointMotionStats] = []

    for jid in KEY_JOINT_IDS:
        jname = LANDMARK_NAMES[jid]
        all_velocities: list[float] = []
        all_accelerations: list[float] = []
        all_jerks: list[float] = []
        still_displacements: list[float] = []

        for hand_traj in traj_data.hands:
            jt = hand_traj.joints[jid]
            positions = jt.positions

            if len(positions) < 2:
                continue

            # Velocity magnitudes
            for i in range(1, len(positions)):
                dx = positions[i]["x"] - positions[i - 1]["x"]
                dy = positions[i]["y"] - positions[i - 1]["y"]
                dz = positions[i]["z"] - positions[i - 1]["z"]
                vmag = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                all_velocities.append(vmag)

            # Acceleration magnitudes
            if len(all_velocities) >= 2:
                accel_mags: list[float] = []
                # Recompute velocity vectors for acceleration
                vel_vectors: list[tuple[float, float, float]] = []
                for i in range(1, len(positions)):
                    dx = (positions[i]["x"] - positions[i - 1]["x"]) / dt
                    dy = (positions[i]["y"] - positions[i - 1]["y"]) / dt
                    dz = (positions[i]["z"] - positions[i - 1]["z"]) / dt
                    vel_vectors.append((dx, dy, dz))

                for i in range(1, len(vel_vectors)):
                    ax = (vel_vectors[i][0] - vel_vectors[i - 1][0]) / dt
                    ay = (vel_vectors[i][1] - vel_vectors[i - 1][1]) / dt
                    az = (vel_vectors[i][2] - vel_vectors[i - 1][2]) / dt
                    amag = math.sqrt(ax**2 + ay**2 + az**2)
                    accel_mags.append(amag)

                all_accelerations.extend(accel_mags)

                # Jerk = derivative of acceleration magnitude
                for i in range(1, len(accel_mags)):
                    jerk = abs(accel_mags[i] - accel_mags[i - 1]) / dt
                    all_jerks.append(jerk)

            # Stability: displacement when velocity is below threshold
            for i in range(1, len(positions)):
                dx = positions[i]["x"] - positions[i - 1]["x"]
                dy = positions[i]["y"] - positions[i - 1]["y"]
                dz = positions[i]["z"] - positions[i - 1]["z"]
                vmag = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                if vmag < velocity_threshold:
                    disp = math.sqrt(dx**2 + dy**2 + dz**2)
                    still_displacements.append(disp)

        joint_stats.append(JointMotionStats(
            joint_id=jid,
            name=jname,
            velocity_mean=_safe_mean(all_velocities),
            velocity_std=_safe_std(all_velocities),
            velocity_max=max(all_velocities) if all_velocities else 0.0,
            acceleration_mean=_safe_mean(all_accelerations),
            acceleration_std=_safe_std(all_accelerations),
            acceleration_max=max(all_accelerations) if all_accelerations else 0.0,
            jerk_mean=_safe_mean(all_jerks),
            jerk_max=max(all_jerks) if all_jerks else 0.0,
            stability=_safe_mean(still_displacements),
        ))

    return TrajectoryQuality(
        total_frames=total_frames,
        frames_with_data=frames_with_data,
        frames_missing=total_frames - frames_with_data,
        joints=joint_stats,
    )


def _safe_mean(vals: list[float]) -> float:
    return float(statistics.mean(vals)) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def analyze_clip_quality(
    clips: list[ClipInfo],
    detections: list[FrameDetection],
    fps: float,
) -> list[ClipQuality]:
    """Analyze frame/duration alignment and detection coverage per clip."""
    results: list[ClipQuality] = []

    for clip in clips:
        # Expected frames from time range
        expected_dur = clip.end_time - clip.start_time
        expected_frames = int(round(expected_dur * fps))

        # Actual frames from the clip file
        cap = cv2.VideoCapture(str(clip.path))
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        actual_dur = actual_frames / actual_fps if actual_fps > 0 else 0.0
        cap.release()

        # Detection coverage within clip's frame range
        clip_dets = [
            d for d in detections
            if clip.start_frame <= d.frame_idx <= clip.end_frame
        ]
        covered = sum(1 for d in clip_dets if d.detected)
        coverage = covered / len(clip_dets) if clip_dets else 0.0

        results.append(ClipQuality(
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


# ── Orchestrator ────────────────────────────────────────────────────


def _get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def run_quality_benchmark(
    config,
    video_paths: list[Path],
) -> list[QualityResult]:
    """Run the quality benchmark on each video at 360p.

    For each video:
    1. Resize to 360p
    2. Detect hands → detection quality
    3. Extract clips → clip quality
    4. Depth estimation (both models) on first clip → depth quality
    5. Trajectory mapping on first clip → trajectory quality

    .. deprecated::
        Use :func:`enhanced_benchmark.run_enhanced_benchmark` instead,
        which covers all 11 pipeline stages with both timing and quality.
    """
    import warnings

    warnings.warn(
        "run_quality_benchmark() is deprecated. Use enhanced_benchmark.run_enhanced_benchmark() "
        "which covers all 11 pipeline stages with both timing and quality metrics.",
        DeprecationWarning,
        stacklevel=2,
    )
    results: list[QualityResult] = []

    for video_path in video_paths:
        video_name = video_path.stem
        print(f"\nQuality benchmark: {video_name}")

        # Step 1: Resize to 360p
        resized_dir = config.resolve_path("data/benchmark/resized")
        resized_dir.mkdir(parents=True, exist_ok=True)
        resized_path = resized_dir / f"{video_name}_360p.mp4"
        print("  Resizing to 360p...")
        resize_video(video_path, resized_path, 360)

        fps = _get_fps(resized_path)

        # Step 2: Detect hands on full video
        print("  Detecting hands...")
        detections = run_detect(resized_path, config)

        # Step 3: Detection quality
        print("  Analyzing detection quality...")
        det_quality = analyze_detection_quality(detections)

        # Step 4: Extract clips
        print("  Extracting clips...")
        clips = run_clip(resized_path, detections, config)

        # Step 5: Clip quality
        print("  Analyzing clip quality...")
        clip_quality = analyze_clip_quality(clips, detections, fps)

        # Steps 6-10: Depth and trajectory on first clip
        depth_quality = DepthQuality()
        traj_quality = TrajectoryQuality()

        if clips:
            clip_path = clips[0].path
            clip_info = clips[0]
            clip_detections = _slice_detections_for_clip(detections, clip_info)
            clip_fps = _get_fps(clip_path)

            # Step 7: Depth estimation — MiDaS
            print("  Running depth (MiDaS) on first clip...")
            depth_output_dir = config.resolve_path(config.depth.output_dir)
            midas_frames = estimate_depth(
                clip_path,
                output_dir=depth_output_dir / "quality_midas",
                sample_rate=config.depth.sample_rate,
                model_name="midas",
                save_visualization=False,
            )

            # Step 7b: Depth estimation — Depth Anything V2
            print("  Running depth (DA V2) on first clip...")
            da_v2_frames = estimate_depth(
                clip_path,
                output_dir=depth_output_dir / "quality_da_v2",
                sample_rate=config.depth.sample_rate,
                model_name="depth_anything_v2",
                save_visualization=False,
            )

            # Step 8: Depth quality
            print("  Analyzing depth quality...")
            depth_quality = analyze_depth_quality(
                midas_frames, da_v2_frames, clip_detections,
            )

            # Step 9: Trajectories using DA V2 depth
            print("  Computing trajectories...")
            traj_data = run_trajectories(
                clip_path, clip_detections, da_v2_frames, clip_info, config,
            )

            # Step 10: Trajectory quality
            print("  Analyzing trajectory quality...")
            traj_quality = analyze_trajectory_quality(
                traj_data, clip_detections, clip_fps,
            )
        else:
            print("  No clips found — skipping depth/trajectory analysis")

        results.append(QualityResult(
            video_name=video_name,
            detection=det_quality,
            depth=depth_quality,
            trajectory=traj_quality,
            clips=clip_quality,
        ))

    return results


# ── Report formatting ───────────────────────────────────────────────


def format_quality_report(results: list[QualityResult]) -> str:
    """Format a human-readable quality benchmark report."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append("QUALITY BENCHMARK REPORT")
    lines.append(sep)

    for result in results:
        lines.append(f"\nVideo: {result.video_name}")
        lines.append(sep)

        # ── Detection Quality ──
        d = result.detection
        lines.append("\nDETECTION QUALITY")
        lines.append(dash)
        lines.append(f"  Total frames:       {d.total_frames}")
        lines.append(f"  Detected frames:    {d.detected_frames}")
        lines.append(f"  Detection rate:     {d.detection_rate:.4f}")
        lines.append(f"  Both-hands frames:  {d.both_hands_frames}")
        lines.append(f"  Both-hands rate:    {d.both_hands_rate:.4f}")
        lines.append(f"  Left detections:    {d.left_count}")
        lines.append(f"  Right detections:   {d.right_count}")
        lines.append(f"  L/R balance:        {d.lr_balance:.4f}")
        lines.append("")
        lines.append("  Confidence stats:")
        c = d.confidence
        lines.append(f"    Mean:   {c.mean:.4f}    Median: {c.median:.4f}")
        lines.append(f"    Std:    {c.std:.4f}    Min:    {c.min:.4f}    Max: {c.max:.4f}")
        lines.append(f"    P25:    {c.p25:.4f}    P75:    {c.p75:.4f}    P90: {c.p90:.4f}")
        lines.append("")
        lines.append("  Temporal consistency:")
        lines.append(f"    Flicker count (gap<=5): {d.flicker_count}")
        lines.append(f"    Longest streak:         {d.longest_streak}")
        lines.append(f"    Gap segments:           {d.gap_segment_count}")

        # ── Depth Quality ──
        lines.append("")
        lines.append("DEPTH QUALITY")
        lines.append(dash)
        for model_stats in [result.depth.midas, result.depth.depth_anything_v2]:
            lines.append(f"  Model: {model_stats.model_name}")
            lines.append(f"    Frames processed:   {model_stats.num_frames}")
            lines.append(f"    Depth range:        {model_stats.depth_range:.4f}")
            lines.append(f"    Depth mean:         {model_stats.depth_mean:.4f}")
            lines.append(f"    Depth std:          {model_stats.depth_std:.4f}")
            lines.append(f"    Temporal mean delta: {model_stats.temporal_mean_delta:.4f}")
            lines.append(f"    Temporal max delta:  {model_stats.temporal_max_delta:.4f}")
            lines.append("")
        lines.append(f"  Cross-model correlation: {result.depth.cross_model_correlation:.4f}")

        # ── Trajectory Quality ──
        t = result.trajectory
        lines.append("")
        lines.append("TRAJECTORY QUALITY")
        lines.append(dash)
        lines.append(f"  Total frames:       {t.total_frames}")
        lines.append(f"  Frames with data:   {t.frames_with_data}")
        lines.append(f"  Frames missing:     {t.frames_missing}")
        lines.append("")

        if t.joints:
            header = f"  {'Joint':<22} {'Vel(mean)':>10} {'Vel(max)':>10} {'Acc(mean)':>10} {'Jerk(mean)':>11} {'Stability':>10}"
            lines.append(header)
            lines.append("  " + "-" * 66)
            for j in t.joints:
                row = (
                    f"  {j.name:<22}"
                    f" {j.velocity_mean:>10.6f}"
                    f" {j.velocity_max:>10.6f}"
                    f" {j.acceleration_mean:>10.4f}"
                    f" {j.jerk_mean:>11.4f}"
                    f" {j.stability:>10.6f}"
                )
                lines.append(row)

        # ── Clip Quality ──
        lines.append("")
        lines.append("CLIP EXTRACTION QUALITY")
        lines.append(dash)
        if result.clips:
            header = f"  {'Clip':<40} {'Exp.Fr':>7} {'Act.Fr':>7} {'Delta':>6} {'Coverage':>9}"
            lines.append(header)
            lines.append("  " + "-" * 66)
            for cq in result.clips:
                row = (
                    f"  {cq.clip_id:<40}"
                    f" {cq.expected_frames:>7}"
                    f" {cq.actual_frames:>7}"
                    f" {cq.frame_delta:>+6}"
                    f" {cq.detection_coverage:>9.4f}"
                )
                lines.append(row)
            lines.append("")
            for cq in result.clips:
                lines.append(
                    f"  {cq.clip_id}: duration expected={cq.expected_duration:.4f}s "
                    f"actual={cq.actual_duration:.4f}s delta={cq.duration_delta:+.4f}s"
                )
        else:
            lines.append("  No clips extracted.")

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


def save_quality_report(
    results: list[QualityResult],
    output_dir: Path,
    *,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write quality benchmark results as JSON and human-readable text.

    When *run_dir* is provided the files are written there instead of
    *output_dir* (used by the versioned run storage).

    Returns (json_path, txt_path).
    """
    dest = run_dir if run_dir is not None else output_dir
    dest.mkdir(parents=True, exist_ok=True)

    # JSON
    data = _sanitize_for_json([asdict(r) for r in results])
    json_path = dest / "quality_benchmark.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    # Text
    txt_path = dest / "quality_benchmark.txt"
    txt_path.write_text(format_quality_report(results) + "\n")

    return json_path, txt_path
