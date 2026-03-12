"""3D hand mesh recovery using HaMeR (Hand Mesh Recovery).

HaMeR uses a ViT-based architecture to reconstruct 3D hand meshes from
monocular images, producing MANO hand model parameters (pose, shape, camera)
and 3D joint positions.

Reference: https://github.com/geopavlakos/hamer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class HandPose3D:
    """3D hand pose result for a single hand in a single frame."""
    hand_label: str  # "left" or "right"
    confidence: float
    # MANO parameters
    global_orient: list[float]  # 3 rotation values (axis-angle)
    hand_pose: list[float]  # 45 values (15 joints x 3 axis-angle)
    betas: list[float]  # 10 shape parameters
    # 3D joint positions in camera space (21 joints x 3)
    joints_3d: list[dict]  # [{joint_id, name, x, y, z}]
    # 2D projected joint positions (normalized 0-1)
    joints_2d: list[dict]  # [{joint_id, name, x, y}]
    # Wrist position and orientation for robotics (6D: xyz + rpy)
    wrist_position: list[float]  # [x, y, z] in camera frame
    wrist_orientation: list[float]  # [r1..r6] 6D rotation representation
    # Fingertip positions for EgoDex-style action representation
    fingertip_positions: list[dict]  # [{finger, x, y, z}]


@dataclass
class FrameHandPose:
    """3D hand pose results for a single frame."""
    frame_idx: int
    timestamp: float
    hands: list[HandPose3D] = field(default_factory=list)


# MANO joint names matching MediaPipe ordering
MANO_JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

FINGERTIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


def _load_hamer_model(device: torch.device, model_path: str | None = None):
    """Load HaMeR model.

    Tries to load from the hamer package first, falls back to a lightweight
    ViT-based hand pose estimator if HaMeR is not installed.
    """
    try:
        from hamer.models import load_hamer
        model, model_cfg = load_hamer(checkpoint_path=model_path)
        model.to(device).eval()
        return model, model_cfg, "hamer"
    except ImportError:
        pass

    try:
        from hand_detector_hamer import HandPoseEstimator
        model = HandPoseEstimator(model_path=model_path, device=device)
        return model, None, "hand_detector_hamer"
    except ImportError:
        pass

    # Fallback: use a minimal ViT-based estimator via timm
    print("HaMeR not installed. Using fallback MediaPipe-based 3D hand estimation.")
    return None, None, "fallback"


def _estimate_with_fallback(
    frame_bgr: np.ndarray,
    hand_detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> list[HandPose3D]:
    """Fallback 3D estimation using MediaPipe landmark z-values with synthetic MANO params."""
    results = []
    for hand_det in hand_detections:
        landmarks = hand_det.get("landmarks", [])
        if not landmarks:
            continue

        joints_3d = []
        joints_2d = []
        for lm in landmarks:
            joints_3d.append({
                "joint_id": lm["joint_id"],
                "name": MANO_JOINT_NAMES[lm["joint_id"]] if lm["joint_id"] < 21 else f"joint_{lm['joint_id']}",
                "x": round(float(lm["x"]), 6),
                "y": round(float(lm["y"]), 6),
                "z": round(float(lm["z"]), 6),
            })
            joints_2d.append({
                "joint_id": lm["joint_id"],
                "name": MANO_JOINT_NAMES[lm["joint_id"]] if lm["joint_id"] < 21 else f"joint_{lm['joint_id']}",
                "x": round(float(lm["x"]), 6),
                "y": round(float(lm["y"]), 6),
            })

        wrist = landmarks[0] if landmarks else {"x": 0, "y": 0, "z": 0}
        wrist_pos = [float(wrist["x"]), float(wrist["y"]), float(wrist["z"])]

        fingertips = []
        for finger_name, tip_id in FINGERTIP_IDS.items():
            if tip_id < len(landmarks):
                lm = landmarks[tip_id]
                fingertips.append({
                    "finger": finger_name,
                    "x": round(float(lm["x"]), 6),
                    "y": round(float(lm["y"]), 6),
                    "z": round(float(lm["z"]), 6),
                })

        results.append(HandPose3D(
            hand_label=hand_det.get("hand_label", "unknown"),
            confidence=hand_det.get("confidence", 0.0),
            global_orient=[0.0, 0.0, 0.0],
            hand_pose=[0.0] * 45,
            betas=[0.0] * 10,
            joints_3d=joints_3d,
            joints_2d=joints_2d,
            wrist_position=wrist_pos,
            wrist_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # identity rotation
            fingertip_positions=fingertips,
        ))

    return results


def _estimate_with_hamer(
    frame_bgr: np.ndarray,
    model,
    model_cfg,
    device: torch.device,
) -> list[HandPose3D]:
    """Run HaMeR model on a frame."""
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Use ViTDet to detect hands, then run HaMeR on each crop
    dataset = ViTDetDataset(model_cfg, rgb)
    if len(dataset) == 0:
        return []

    results = []
    for idx in range(len(dataset)):
        batch = dataset[idx]
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out = model(batch)

        pred_vertices = out["pred_vertices"].cpu().numpy()
        pred_joints = out["pred_keypoints_3d"].cpu().numpy()
        pred_cam = out["pred_cam"].cpu().numpy()

        # Extract MANO parameters
        global_orient = out.get("pred_mano_params", {}).get("global_orient", torch.zeros(1, 3))
        hand_pose_param = out.get("pred_mano_params", {}).get("hand_pose", torch.zeros(1, 45))
        betas_param = out.get("pred_mano_params", {}).get("betas", torch.zeros(1, 10))

        joints_3d_array = pred_joints[0] if pred_joints.ndim > 2 else pred_joints
        h, w = frame_bgr.shape[:2]

        joints_3d = []
        joints_2d = []
        for jid in range(min(21, len(joints_3d_array))):
            j = joints_3d_array[jid]
            joints_3d.append({
                "joint_id": jid,
                "name": MANO_JOINT_NAMES[jid],
                "x": round(float(j[0]), 6),
                "y": round(float(j[1]), 6),
                "z": round(float(j[2]), 6),
            })
            # Project to 2D (simple weak perspective)
            joints_2d.append({
                "joint_id": jid,
                "name": MANO_JOINT_NAMES[jid],
                "x": round(float(j[0] / w), 6),
                "y": round(float(j[1] / h), 6),
            })

        wrist_j = joints_3d_array[0]
        wrist_pos = [round(float(wrist_j[0]), 6), round(float(wrist_j[1]), 6), round(float(wrist_j[2]), 6)]

        fingertips = []
        for finger_name, tip_id in FINGERTIP_IDS.items():
            if tip_id < len(joints_3d_array):
                j = joints_3d_array[tip_id]
                fingertips.append({
                    "finger": finger_name,
                    "x": round(float(j[0]), 6),
                    "y": round(float(j[1]), 6),
                    "z": round(float(j[2]), 6),
                })

        is_right = batch.get("right", torch.tensor([1])).item()
        hand_label = "right" if is_right else "left"

        results.append(HandPose3D(
            hand_label=hand_label,
            confidence=round(float(batch.get("personScore", 0.9)), 4),
            global_orient=global_orient[0].cpu().tolist() if torch.is_tensor(global_orient) else [0.0] * 3,
            hand_pose=hand_pose_param[0].cpu().tolist() if torch.is_tensor(hand_pose_param) else [0.0] * 45,
            betas=betas_param[0].cpu().tolist() if torch.is_tensor(betas_param) else [0.0] * 10,
            joints_3d=joints_3d,
            joints_2d=joints_2d,
            wrist_position=wrist_pos,
            wrist_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            fingertip_positions=fingertips,
        ))

    return results


def estimate_hand_poses(
    video_path: str | Path,
    hand_detections: list | None = None,
    sample_rate: int = 1,
    model_path: str | None = None,
    output_dir: str | Path = "data/hand_poses",
) -> list[FrameHandPose]:
    """Estimate 3D hand poses for frames of a video.

    Args:
        video_path: Path to the video file.
        hand_detections: Optional pre-computed hand detections (FrameDetection list)
            for fallback mode. If HaMeR is available, it runs its own detection.
        sample_rate: Process every Nth frame.
        model_path: Optional path to HaMeR checkpoint.
        output_dir: Directory for saving results.

    Returns:
        List of FrameHandPose results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, backend = _load_hamer_model(device, model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build detection lookup for fallback mode
    det_by_frame = {}
    if hand_detections:
        for det in hand_detections:
            frame_idx = det.frame_idx if hasattr(det, "frame_idx") else det.get("frame_idx", -1)
            det_by_frame[frame_idx] = det

    frame_poses: list[FrameHandPose] = []

    try:
        for frame_idx in tqdm(range(total_frames), desc="Estimating hand poses", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                continue

            timestamp = round(frame_idx / fps, 4)

            if backend == "hamer" and model is not None:
                hands = _estimate_with_hamer(frame, model, model_cfg, device)
            elif backend == "fallback":
                det = det_by_frame.get(frame_idx)
                if det is None:
                    frame_poses.append(FrameHandPose(frame_idx=frame_idx, timestamp=timestamp))
                    continue

                hand_list = det.hands if hasattr(det, "hands") else []
                hand_dicts = []
                for h in hand_list:
                    hand_dicts.append({
                        "hand_label": h.hand_label if hasattr(h, "hand_label") else h.get("hand_label", "unknown"),
                        "confidence": h.confidence if hasattr(h, "confidence") else h.get("confidence", 0.0),
                        "landmarks": h.landmarks if hasattr(h, "landmarks") else h.get("landmarks", []),
                    })
                hands = _estimate_with_fallback(frame, hand_dicts, width, height)
            else:
                hands = []

            frame_poses.append(FrameHandPose(
                frame_idx=frame_idx,
                timestamp=timestamp,
                hands=hands,
            ))
    finally:
        cap.release()

    return frame_poses


def to_egodex_action(hand_pose: HandPose3D) -> list[float]:
    """Convert a HandPose3D to EgoDex-style 48D action vector.

    The action representation is: wrist_position (3D) + wrist_orientation (6D rotation)
    + 5 fingertip positions (5 x 3D) = 3 + 6 + 15 = 24D per hand.
    For both hands: 48D total.

    This follows the EgoDex paper's action representation.
    """
    action = []
    action.extend(hand_pose.wrist_position[:3])
    action.extend(hand_pose.wrist_orientation[:6])
    for ft in hand_pose.fingertip_positions:
        action.extend([ft["x"], ft["y"], ft["z"]])

    # Pad to 24 if needed
    while len(action) < 24:
        action.append(0.0)

    return action[:24]
