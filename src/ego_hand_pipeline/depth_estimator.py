"""Monocular depth estimation using Depth Anything V2 or MiDaS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class DepthFrame:
    """Depth estimation result for a single frame."""
    frame_idx: int
    timestamp: float
    depth_map: np.ndarray  # H x W float32 (relative depth)
    npz_path: Path | None = None
    viz_path: Path | None = None


def _load_depth_anything_v2(device: torch.device) -> torch.nn.Module:
    """Load Depth Anything V2 model via torch hub."""
    model = torch.hub.load(
        "huggingface/pytorch-image-models",
        "model",
        "vit_small_patch14_dinov2.depth_anything_v2",
        pretrained=True,
    )
    model.to(device).eval()
    return model


def _load_midas(device: torch.device):
    """Load MiDaS model and transforms via torch hub."""
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    return model, transform


def _estimate_depth_midas(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Run MiDaS on a single BGR frame, returns H x W float32 depth map."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy().astype(np.float32)


def _estimate_depth_da_v2(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Run Depth Anything V2 on a single BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Resize to model input (518x518 for ViT-S)
    input_size = 518
    img = cv2.resize(rgb, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = model.forward_features(tensor)
        # Use CLS token or mean pooling — for depth we need a spatial map
        # Reshape patch tokens back to spatial grid
        patch_size = 14
        grid_size = input_size // patch_size
        patch_tokens = features[:, 1:, :]  # remove CLS token
        depth_features = patch_tokens.mean(dim=-1).reshape(1, 1, grid_size, grid_size)
        depth_map = torch.nn.functional.interpolate(
            depth_features, size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()

    return depth_map.cpu().numpy().astype(np.float32)


def _save_depth_viz(depth_map: np.ndarray, path: Path) -> None:
    """Save a colorized depth visualization."""
    colored = _colorize_depth(depth_map)
    cv2.imwrite(str(path), colored)


def _colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Return a BGR colorized depth image (uint8)."""
    normalized = depth_map - depth_map.min()
    max_val = normalized.max()
    if max_val > 0:
        normalized = (normalized / max_val * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def render_depth_video(
    depth_frames: list[DepthFrame],
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 1,
) -> Path:
    """Render colorized depth maps as an mp4 video.

    Args:
        depth_frames: List of DepthFrame results from estimate_depth().
        video_path: Original video (used to read fps).
        output_path: Where to write the depth video.
        sample_rate: The sample_rate used during estimation (adjusts output fps).

    Returns:
        Path to the written video file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not depth_frames:
        raise ValueError("No depth frames to render")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_fps = fps / sample_rate
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    try:
        for df in depth_frames:
            colored = _colorize_depth(df.depth_map)
            # Resize to original video dimensions in case depth map differs
            if colored.shape[:2] != (height, width):
                colored = cv2.resize(colored, (width, height))
            writer.write(colored)
    finally:
        writer.release()

    return output_path


def estimate_depth(
    video_path: str | Path,
    output_dir: str | Path = "data/depth",
    sample_rate: int = 1,
    model_name: str = "midas",
    save_visualization: bool = True,
) -> list[DepthFrame]:
    """Estimate depth for frames of a video.

    Args:
        video_path: Path to the video file.
        output_dir: Directory for depth outputs.
        sample_rate: Process every Nth frame (1 = all).
        model_name: "depth_anything_v2" or "midas".
        save_visualization: Whether to save colorized PNG visualizations.

    Returns:
        List of DepthFrame results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    npz_dir = output_dir / video_path.stem
    npz_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    midas_model = None
    midas_transform = None
    da_model = None

    if model_name == "depth_anything_v2":
        try:
            da_model = _load_depth_anything_v2(device)
        except Exception:
            print("Depth Anything V2 failed to load, falling back to MiDaS")
            model_name = "midas"

    if model_name == "midas":
        midas_model, midas_transform = _load_midas(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    depth_frames: list[DepthFrame] = []

    try:
        for frame_idx in tqdm(range(total_frames), desc="Estimating depth", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                continue

            timestamp = round(frame_idx / fps, 4)

            if model_name == "midas" and midas_model is not None:
                depth_map = _estimate_depth_midas(frame, midas_model, midas_transform, device)
            elif da_model is not None:
                depth_map = _estimate_depth_da_v2(frame, da_model, device)
            else:
                raise RuntimeError("No depth model loaded")

            # Save depth as npz
            npz_path = npz_dir / f"frame_{frame_idx:06d}.npz"
            np.savez_compressed(str(npz_path), depth=depth_map)

            viz_path = None
            if save_visualization:
                viz_path = npz_dir / f"frame_{frame_idx:06d}.png"
                _save_depth_viz(depth_map, viz_path)

            depth_frames.append(DepthFrame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                depth_map=depth_map,
                npz_path=npz_path,
                viz_path=viz_path,
            ))
    finally:
        cap.release()

    return depth_frames
