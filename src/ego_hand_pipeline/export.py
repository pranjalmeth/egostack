"""JSON and CSV export for trajectory data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .trajectory_mapper import TrajectoryData


def export_json(data: TrajectoryData, path: str | Path) -> Path:
    """Export trajectory data as structured JSON.

    Args:
        data: TrajectoryData from trajectory_mapper.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "source_video": data.source_video,
        "clip_id": data.clip_id,
        "start_time": data.start_time,
        "end_time": data.end_time,
        "fps": data.fps,
        "frames": data.frames,
    }

    path.write_text(json.dumps(output, indent=2, default=str))
    return path


def export_csv(data: TrajectoryData, path: str | Path) -> Path:
    """Export trajectory data as a flat CSV table.

    One row per frame per joint per hand.
    Columns: frame, timestamp, clip_id, hand, joint_id, joint_name, x, y, z, depth, vx, vy, vz

    Args:
        data: TrajectoryData from trajectory_mapper.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame", "timestamp", "clip_id", "hand", "joint_id", "joint_name",
        "x", "y", "z", "depth", "vx", "vy", "vz",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame in data.frames:
            for hand_data in frame.get("hands", []):
                hand_label = hand_data["hand"]
                velocities = hand_data.get("velocity", [])
                accelerations = hand_data.get("acceleration", [])

                for i, lm in enumerate(hand_data.get("landmarks", [])):
                    vel = velocities[i] if i < len(velocities) and velocities[i] else {}
                    row = {
                        "frame": frame["frame_idx"],
                        "timestamp": frame["timestamp"],
                        "clip_id": data.clip_id,
                        "hand": hand_label,
                        "joint_id": lm["joint_id"],
                        "joint_name": lm["name"],
                        "x": lm["x"],
                        "y": lm["y"],
                        "z": lm["z"],
                        "depth": lm.get("depth", ""),
                        "vx": vel.get("vx", ""),
                        "vy": vel.get("vy", ""),
                        "vz": vel.get("vz", ""),
                    }
                    writer.writerow(row)

    return path
