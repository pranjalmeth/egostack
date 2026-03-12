"""Export pipeline data in LeRobot-compatible dataset format.

LeRobot (HuggingFace) is the emerging standard for PyTorch-native robotics
training datasets. This exporter produces Parquet files for trajectory/state
data and preserves video clips as MP4, compatible with LeRobotDataset v2/v3.

Reference: https://github.com/huggingface/lerobot
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode (clip)."""
    episode_id: int
    source_video: str
    clip_id: str
    task_description: str
    start_time: float
    end_time: float
    fps: float
    num_frames: int


def _trajectory_to_rows(
    traj_data,
    episode_id: int,
    hand_poses: list | None = None,
    object_detections: list | None = None,
    segmentations: list | None = None,
) -> list[dict]:
    """Convert trajectory data into flat row dicts for Parquet export.

    Each row represents one timestep in the episode, containing:
    - observation.state: hand joint positions (flattened)
    - observation.hand_pose_3d: 3D hand pose (MANO params + joints)
    - action: EgoDex-style action vector (wrist pos + orient + fingertips)
    - object_detections: detected objects at this timestep
    - segmentation_labels: labels of segmented objects
    """
    rows = []
    frame_idx_global = 0

    # Build lookups for hand poses and objects by frame_idx
    pose_by_frame = {}
    if hand_poses:
        for fp in hand_poses:
            fidx = fp.frame_idx if hasattr(fp, "frame_idx") else fp.get("frame_idx", -1)
            pose_by_frame[fidx] = fp

    obj_by_frame = {}
    if object_detections:
        for fo in object_detections:
            fidx = fo.frame_idx if hasattr(fo, "frame_idx") else fo.get("frame_idx", -1)
            obj_by_frame[fidx] = fo

    seg_by_frame = {}
    if segmentations:
        for fs in segmentations:
            fidx = fs.frame_idx if hasattr(fs, "frame_idx") else fs.get("frame_idx", -1)
            seg_by_frame[fidx] = fs

    for frame in traj_data.frames:
        fidx = frame["frame_idx"]
        timestamp = frame["timestamp"]

        # Build state vector: flatten all hand landmarks into a single vector
        state = []
        for hand_data in frame.get("hands", []):
            for lm in hand_data.get("landmarks", []):
                state.extend([lm["x"], lm["y"], lm["z"]])
                depth = lm.get("depth")
                state.append(depth if depth is not None else 0.0)

        # Pad state to fixed size (2 hands x 21 joints x 4 values = 168)
        while len(state) < 168:
            state.append(0.0)
        state = state[:168]

        # Build action vector from 3D hand poses (EgoDex-style 48D)
        action = [0.0] * 48
        pose_frame = pose_by_frame.get(fidx)
        if pose_frame:
            hands = pose_frame.hands if hasattr(pose_frame, "hands") else []
            for i, hp in enumerate(hands[:2]):
                from .hand_pose_estimator import to_egodex_action
                act = to_egodex_action(hp)
                offset = i * 24
                action[offset:offset + 24] = act

        # Object detection summary
        obj_labels = []
        obj_frame = obj_by_frame.get(fidx)
        if obj_frame:
            objects = obj_frame.objects if hasattr(obj_frame, "objects") else []
            obj_labels = [
                obj.label if hasattr(obj, "label") else obj.get("label", "")
                for obj in objects
            ]

        # Segmentation summary
        seg_labels = []
        seg_frame = seg_by_frame.get(fidx)
        if seg_frame:
            masks = seg_frame.masks if hasattr(seg_frame, "masks") else []
            seg_labels = [
                m.label if hasattr(m, "label") else m.get("label", "")
                for m in masks
            ]

        row = {
            "episode_id": episode_id,
            "frame_idx": frame_idx_global,
            "timestamp": timestamp,
            "observation.state": state,
            "action": action,
            "object_labels": json.dumps(obj_labels),
            "segmentation_labels": json.dumps(seg_labels),
            "done": False,
        }

        rows.append(row)
        frame_idx_global += 1

    # Mark last frame as done
    if rows:
        rows[-1]["done"] = True

    return rows


def export_lerobot(
    episodes: list[dict],
    output_dir: str | Path,
    dataset_name: str = "ego_hand_manipulation",
    task_description: str = "manipulation",
) -> Path:
    """Export processed pipeline data as a LeRobot-compatible dataset.

    Args:
        episodes: List of episode dicts, each containing:
            - traj_data: TrajectoryData
            - clip_info: ClipInfo (optional)
            - hand_poses: list[FrameHandPose] (optional)
            - object_detections: list[FrameObjects] (optional)
            - segmentations: list[FrameSegmentation] (optional)
            - video_path: Path to the clip video
        output_dir: Root directory for the dataset.
        dataset_name: Name of the dataset.
        task_description: Default task description for episodes.

    Returns:
        Path to the dataset root directory.
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = data_dir / "data"
    video_dir = data_dir / "videos"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    episode_metadata = []

    for ep_idx, episode in enumerate(episodes):
        traj_data = episode["traj_data"]
        clip_info = episode.get("clip_info")
        hand_poses = episode.get("hand_poses")
        object_detections = episode.get("object_detections")
        segmentations = episode.get("segmentations")
        video_path = episode.get("video_path")

        # Convert trajectory data to rows
        rows = _trajectory_to_rows(
            traj_data,
            episode_id=ep_idx,
            hand_poses=hand_poses,
            object_detections=object_detections,
            segmentations=segmentations,
        )
        all_rows.extend(rows)

        # Copy video clip
        if video_path and Path(video_path).exists():
            dst = video_dir / f"episode_{ep_idx:06d}.mp4"
            shutil.copy2(str(video_path), str(dst))

        # Build episode metadata
        clip_id = clip_info.clip_id if clip_info else traj_data.clip_id
        episode_metadata.append(EpisodeMetadata(
            episode_id=ep_idx,
            source_video=traj_data.source_video,
            clip_id=clip_id,
            task_description=task_description,
            start_time=traj_data.start_time,
            end_time=traj_data.end_time,
            fps=traj_data.fps,
            num_frames=len(rows),
        ))

    # Write Parquet
    if HAS_ARROW and all_rows:
        # Separate scalar and array columns
        scalar_keys = ["episode_id", "frame_idx", "timestamp", "object_labels",
                       "segmentation_labels", "done"]
        array_keys = ["observation.state", "action"]

        table_dict = {}
        for key in scalar_keys:
            table_dict[key] = [row[key] for row in all_rows]

        for key in array_keys:
            # Store arrays as list-of-lists (nested Parquet)
            table_dict[key] = [row[key] for row in all_rows]

        table = pa.table(table_dict)
        pq.write_table(table, str(parquet_dir / "train.parquet"))
    elif all_rows:
        # Fallback: save as JSON lines
        jsonl_path = parquet_dir / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row, default=str) + "\n")

    # Write dataset metadata
    meta = {
        "dataset_name": dataset_name,
        "version": "2.1",
        "format": "lerobot",
        "num_episodes": len(episodes),
        "total_frames": len(all_rows),
        "task_description": task_description,
        "observation_space": {
            "state_dim": 168,
            "state_description": "2 hands x 21 joints x 4 (x, y, z, depth)",
        },
        "action_space": {
            "action_dim": 48,
            "action_description": "EgoDex-style: 2 hands x 24 (wrist_pos[3] + wrist_orient[6] + fingertips[15])",
        },
        "episodes": [
            {
                "episode_id": em.episode_id,
                "source_video": em.source_video,
                "clip_id": em.clip_id,
                "task_description": em.task_description,
                "start_time": em.start_time,
                "end_time": em.end_time,
                "fps": em.fps,
                "num_frames": em.num_frames,
            }
            for em in episode_metadata
        ],
    }

    meta_path = data_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Write info.json (LeRobot convention)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "human_hand",
        "total_episodes": len(episodes),
        "total_frames": len(all_rows),
        "fps": episodes[0]["traj_data"].fps if episodes else 30.0,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [168],
                "names": ["hand_landmarks"],
            },
            "action": {
                "dtype": "float32",
                "shape": [48],
                "names": ["egodex_action"],
            },
            "observation.image": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": episodes[0]["traj_data"].fps if episodes else 30.0,
                    "video.codec": "mp4v",
                },
            },
        },
    }

    info_path = data_dir / "info.json"
    info_path.write_text(json.dumps(info, indent=2))

    print(f"LeRobot dataset exported to {data_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {len(all_rows)}")

    return data_dir
