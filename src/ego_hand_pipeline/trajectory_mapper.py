"""Hand joint trajectory tracking and velocity/acceleration computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .depth_estimator import DepthFrame
from .hand_detector import FrameDetection, LANDMARK_NAMES

# MediaPipe hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

# Colors (BGR): green for left, blue for right
_HAND_COLORS = {
    "left": (0, 255, 0),
    "right": (255, 0, 0),
}
_DEFAULT_COLOR = (0, 255, 255)  # yellow fallback


@dataclass
class JointTrajectory:
    """Trajectory of a single joint across frames."""
    joint_id: int
    name: str
    positions: list[dict]  # [{frame_idx, timestamp, x, y, z, depth}]
    velocities: list[dict] | None = None  # [{vx, vy, vz}]
    accelerations: list[dict] | None = None  # [{ax, ay, az}]


@dataclass
class HandTrajectory:
    """All joint trajectories for a single hand."""
    hand_label: str
    joints: list[JointTrajectory]


@dataclass
class TrajectoryData:
    """Complete trajectory data for a clip."""
    source_video: str
    clip_id: str
    start_time: float
    end_time: float
    fps: float
    hands: list[HandTrajectory]
    frames: list[dict] = field(default_factory=list)  # per-frame combined data


def _lookup_depth_at(
    depth_frames: list[DepthFrame],
    frame_idx: int,
    x_norm: float,
    y_norm: float,
) -> float | None:
    """Look up depth value at a normalized (x, y) position in a depth map."""
    for df in depth_frames:
        if df.frame_idx == frame_idx:
            h, w = df.depth_map.shape
            px = min(int(x_norm * w), w - 1)
            py = min(int(y_norm * h), h - 1)
            return round(float(df.depth_map[py, px]), 4)
    return None


def _build_depth_index(depth_frames: list[DepthFrame]) -> dict[int, DepthFrame]:
    """Build a frame_idx -> DepthFrame lookup dict."""
    return {df.frame_idx: df for df in depth_frames}


def _compute_derivatives(
    positions: list[dict],
    fps: float,
) -> tuple[list[dict], list[dict]]:
    """Compute velocities and accelerations from position time series."""
    dt = 1.0 / fps if fps > 0 else 1.0
    velocities: list[dict] = []
    accelerations: list[dict] = []

    for i in range(len(positions)):
        if i == 0:
            velocities.append({"vx": 0.0, "vy": 0.0, "vz": 0.0})
        else:
            vx = (positions[i]["x"] - positions[i - 1]["x"]) / dt
            vy = (positions[i]["y"] - positions[i - 1]["y"]) / dt
            vz = (positions[i]["z"] - positions[i - 1]["z"]) / dt
            velocities.append({
                "vx": round(vx, 6),
                "vy": round(vy, 6),
                "vz": round(vz, 6),
            })

    for i in range(len(velocities)):
        if i == 0:
            accelerations.append({"ax": 0.0, "ay": 0.0, "az": 0.0})
        else:
            ax = (velocities[i]["vx"] - velocities[i - 1]["vx"]) / dt
            ay = (velocities[i]["vy"] - velocities[i - 1]["vy"]) / dt
            az = (velocities[i]["vz"] - velocities[i - 1]["vz"]) / dt
            accelerations.append({
                "ax": round(ax, 6),
                "ay": round(ay, 6),
                "az": round(az, 6),
            })

    return velocities, accelerations


def _smooth_positions(positions: list[dict], window: int = 5) -> list[dict]:
    """Apply a simple moving average smoothing to positions."""
    if len(positions) < window:
        return positions

    smoothed = []
    for key in ("x", "y", "z"):
        vals = np.array([p[key] for p in positions])
        kernel = np.ones(window) / window
        padded = np.pad(vals, (window // 2, window // 2), mode="edge")
        smooth_vals = np.convolve(padded, kernel, mode="valid")[:len(vals)]
        for i, v in enumerate(smooth_vals):
            if i >= len(smoothed):
                smoothed.append(dict(positions[i]))
            smoothed[i][key] = round(float(v), 6)

    return smoothed


def render_trajectory_video(
    video_path: str | Path,
    detections: list[FrameDetection],
    output_path: str | Path,
) -> Path:
    """Render hand landmarks and skeleton connections overlaid on the original video.

    Args:
        video_path: Path to the source video.
        detections: Per-frame hand detections.
        output_path: Where to write the trajectory overlay video.

    Returns:
        Path to the written video file.
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

    # Index detections by frame_idx for fast lookup
    det_by_frame: dict[int, FrameDetection] = {d.frame_idx: d for d in detections}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for frame_idx in tqdm(range(total_frames), desc="Rendering trajectories", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            det = det_by_frame.get(frame_idx)
            if det and det.detected:
                for hand_det in det.hands:
                    color = _HAND_COLORS.get(hand_det.hand_label, _DEFAULT_COLOR)
                    lm_pixels = {}

                    # Draw landmarks as circles
                    for lm in hand_det.landmarks:
                        px = int(lm["x"] * width)
                        py = int(lm["y"] * height)
                        lm_pixels[lm["joint_id"]] = (px, py)
                        cv2.circle(frame, (px, py), 4, color, -1)

                    # Draw skeleton connections
                    for start_id, end_id in HAND_CONNECTIONS:
                        if start_id in lm_pixels and end_id in lm_pixels:
                            cv2.line(frame, lm_pixels[start_id], lm_pixels[end_id], color, 2)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path


def map_trajectories(
    video_path: str | Path,
    detections: list[FrameDetection],
    depth_frames: list[DepthFrame] | None = None,
    clip_id: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
    fps: float = 30.0,
    smooth: bool = True,
    compute_velocity: bool = True,
    compute_acceleration: bool = True,
) -> TrajectoryData:
    """Map hand joint trajectories across frames.

    Args:
        video_path: Source video (for metadata).
        detections: Per-frame hand detections.
        depth_frames: Optional depth frames for depth-enhanced trajectories.
        clip_id: Identifier for this clip.
        start_time: Clip start time.
        end_time: Clip end time.
        fps: Video frame rate.
        smooth: Apply smoothing filter.
        compute_velocity: Compute per-joint velocities.
        compute_acceleration: Compute per-joint accelerations.

    Returns:
        TrajectoryData with per-hand, per-joint trajectories.
    """
    video_path = Path(video_path)
    depth_index = _build_depth_index(depth_frames) if depth_frames else {}

    # Collect positions per hand per joint
    # hand_label -> joint_id -> list of position dicts
    hand_joints: dict[str, dict[int, list[dict]]] = {}

    # Also build per-frame combined data for export
    per_frame: list[dict] = []

    for det in detections:
        frame_data = {
            "frame_idx": det.frame_idx,
            "timestamp": det.timestamp,
            "hands": [],
        }

        for hand_det in det.hands:
            label = hand_det.hand_label
            if label not in hand_joints:
                hand_joints[label] = {j: [] for j in range(21)}

            hand_frame = {
                "hand": label,
                "confidence": hand_det.confidence,
                "landmarks": [],
            }

            for lm in hand_det.landmarks:
                jid = lm["joint_id"]
                depth_val = None
                if det.frame_idx in depth_index:
                    df = depth_index[det.frame_idx]
                    h, w = df.depth_map.shape
                    px = min(int(lm["x"] * w), w - 1)
                    py = min(int(lm["y"] * h), h - 1)
                    depth_val = round(float(df.depth_map[py, px]), 4)

                pos = {
                    "frame_idx": det.frame_idx,
                    "timestamp": det.timestamp,
                    "x": lm["x"],
                    "y": lm["y"],
                    "z": lm["z"],
                    "depth": depth_val,
                }
                hand_joints[label][jid].append(pos)

                hand_frame["landmarks"].append({
                    **lm,
                    "depth": depth_val,
                })

            frame_data["hands"].append(hand_frame)

        per_frame.append(frame_data)

    # Build trajectories
    hand_trajectories: list[HandTrajectory] = []

    for label, joints_dict in hand_joints.items():
        joint_trajs: list[JointTrajectory] = []

        for joint_id in range(21):
            positions = joints_dict.get(joint_id, [])

            if smooth and len(positions) > 1:
                positions = _smooth_positions(positions)

            velocities = None
            accelerations = None
            if compute_velocity and len(positions) > 1:
                velocities, accelerations_data = _compute_derivatives(positions, fps)
                if compute_acceleration:
                    accelerations = accelerations_data

            joint_trajs.append(JointTrajectory(
                joint_id=joint_id,
                name=LANDMARK_NAMES[joint_id],
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
            ))

        hand_trajectories.append(HandTrajectory(
            hand_label=label,
            joints=joint_trajs,
        ))

    # Attach velocity/acceleration to per-frame data
    if compute_velocity:
        _attach_derivatives_to_frames(per_frame, hand_trajectories)

    return TrajectoryData(
        source_video=video_path.name,
        clip_id=clip_id,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
        hands=hand_trajectories,
        frames=per_frame,
    )


def _attach_derivatives_to_frames(
    frames: list[dict],
    hand_trajectories: list[HandTrajectory],
) -> None:
    """Attach velocity/acceleration data back to per-frame structures."""
    # Build lookup: hand_label -> joint_id -> frame_idx -> (vel, acc)
    vel_acc_lookup: dict[str, dict[int, dict[int, tuple[dict | None, dict | None]]]] = {}

    for ht in hand_trajectories:
        vel_acc_lookup[ht.hand_label] = {}
        for jt in ht.joints:
            vel_acc_lookup[ht.hand_label][jt.joint_id] = {}
            for i, pos in enumerate(jt.positions):
                fidx = pos["frame_idx"]
                vel = jt.velocities[i] if jt.velocities and i < len(jt.velocities) else None
                acc = jt.accelerations[i] if jt.accelerations and i < len(jt.accelerations) else None
                vel_acc_lookup[ht.hand_label][jt.joint_id][fidx] = (vel, acc)

    for frame in frames:
        for hand_data in frame["hands"]:
            label = hand_data["hand"]
            if label not in vel_acc_lookup:
                continue
            vel_list = []
            acc_list = []
            for lm in hand_data["landmarks"]:
                jid = lm["joint_id"]
                fidx = frame["frame_idx"]
                entry = vel_acc_lookup.get(label, {}).get(jid, {}).get(fidx)
                if entry:
                    vel, acc = entry
                    vel_list.append(vel)
                    acc_list.append(acc)
                else:
                    vel_list.append(None)
                    acc_list.append(None)
            hand_data["velocity"] = vel_list
            hand_data["acceleration"] = acc_list
