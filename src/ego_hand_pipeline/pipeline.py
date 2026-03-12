"""Orchestrator that chains all pipeline stages."""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

from .clipper import ClipInfo, extract_clips
from .config import PipelineConfig
from .depth_estimator import DepthFrame, estimate_depth, render_depth_video
from .downloader import download_video
from .export import export_csv, export_json
from .hand_detector import FrameDetection, detect_hands
from .trajectory_mapper import TrajectoryData, map_trajectories, render_trajectory_video


def _get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def run_download(url: str, config: PipelineConfig) -> Path:
    """Download a single video."""
    output_dir = config.resolve_path(config.download.output_dir)
    return download_video(
        url,
        output_dir=output_dir,
        resolution=config.download.resolution,
        format=config.download.format,
    )


def run_detect(video_path: Path, config: PipelineConfig) -> list[FrameDetection]:
    """Run hand detection on a video."""
    return detect_hands(
        video_path,
        confidence_threshold=config.detection.confidence_threshold,
        model_complexity=config.detection.model_complexity,
    )


def run_clip(
    video_path: Path,
    detections: list[FrameDetection],
    config: PipelineConfig,
) -> list[ClipInfo]:
    """Extract hand-visible clips."""
    output_dir = config.resolve_path(config.clipping.output_dir)
    return extract_clips(
        video_path,
        detections,
        output_dir=output_dir,
        min_duration=config.clipping.min_duration,
        merge_gap=config.clipping.merge_gap,
    )


def run_depth(
    video_path: Path,
    config: PipelineConfig,
) -> list[DepthFrame]:
    """Run depth estimation on a video/clip."""
    output_dir = config.resolve_path(config.depth.output_dir)
    return estimate_depth(
        video_path,
        output_dir=output_dir,
        sample_rate=config.depth.sample_rate,
        model_name=config.depth.model,
        save_visualization=config.depth.save_visualization,
    )


def run_depth_video(
    video_path: Path,
    depth_frames: list[DepthFrame],
    config: PipelineConfig,
) -> Path:
    """Render colorized depth video."""
    output_dir = config.resolve_path(config.depth.output_dir)
    output_path = output_dir / f"{video_path.stem}_depth.mp4"
    return render_depth_video(
        depth_frames,
        video_path,
        output_path,
        sample_rate=config.depth.sample_rate,
    )


def run_trajectories(
    video_path: Path,
    detections: list[FrameDetection],
    depth_frames: list[DepthFrame] | None,
    clip_info: ClipInfo | None,
    config: PipelineConfig,
) -> TrajectoryData:
    """Compute trajectories for a clip."""
    fps = _get_fps(video_path)
    return map_trajectories(
        video_path,
        detections,
        depth_frames=depth_frames,
        clip_id=clip_info.clip_id if clip_info else video_path.stem,
        start_time=clip_info.start_time if clip_info else 0.0,
        end_time=clip_info.end_time if clip_info else 0.0,
        fps=fps,
        smooth=config.trajectories.smooth,
        compute_velocity=config.trajectories.compute_velocity,
        compute_acceleration=config.trajectories.compute_acceleration,
    )


def run_trajectory_video(
    video_path: Path,
    detections: list[FrameDetection],
    config: PipelineConfig,
) -> Path:
    """Render hand trajectory overlay video."""
    output_dir = config.resolve_path(config.trajectories.output_dir)
    output_path = output_dir / f"{video_path.stem}_trajectories.mp4"
    return render_trajectory_video(video_path, detections, output_path)


def run_export(
    traj_data: TrajectoryData,
    config: PipelineConfig,
) -> list[Path]:
    """Export trajectory data in configured formats."""
    output_dir = config.resolve_path(config.trajectories.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if "json" in config.export.formats:
        json_path = output_dir / f"{traj_data.clip_id}.json"
        export_json(traj_data, json_path)
        outputs.append(json_path)

    if "csv" in config.export.formats:
        csv_path = output_dir / f"{traj_data.clip_id}.csv"
        export_csv(traj_data, csv_path)
        outputs.append(csv_path)

    return outputs


def run_pipeline(config: PipelineConfig, urls: list[str]) -> None:
    """Run the full pipeline: download -> detect -> clip -> depth -> trajectories -> export.

    Args:
        config: Pipeline configuration.
        urls: List of YouTube URLs to process.
    """
    for url in tqdm(urls, desc="Processing videos"):
        print(f"\n{'='*60}")
        print(f"Downloading: {url}")
        video_path = run_download(url, config)
        print(f"  Saved: {video_path}")

        print("Detecting hands...")
        detections = run_detect(video_path, config)
        hand_frames = sum(1 for d in detections if d.detected)
        print(f"  Hands detected in {hand_frames}/{len(detections)} frames")

        print("Extracting clips...")
        clips = run_clip(video_path, detections, config)
        print(f"  Extracted {len(clips)} clips")

        for clip in tqdm(clips, desc="Processing clips", leave=False):
            print(f"\n  Processing clip: {clip.clip_id}")

            print("    Detecting hands in clip...")
            clip_detections = run_detect(clip.path, config)

            print("    Estimating depth...")
            depth_frames = run_depth(clip.path, config)
            depth_video = run_depth_video(clip.path, depth_frames, config)
            print(f"    Depth video: {depth_video}")

            print("    Computing trajectories...")
            traj_data = run_trajectories(
                clip.path, clip_detections, depth_frames, clip, config
            )
            traj_video = run_trajectory_video(clip.path, clip_detections, config)
            print(f"    Trajectory video: {traj_video}")

            print("    Exporting data...")
            exported = run_export(traj_data, config)
            for p in exported:
                print(f"      {p}")

    print(f"\n{'='*60}")
    print("Pipeline complete.")


# ---------------------------------------------------------------------------
# Enhanced pipeline for SOTA robotics training data generation
# ---------------------------------------------------------------------------

def run_hand_poses(
    video_path: Path,
    hand_detections: list[FrameDetection] | None,
    config: PipelineConfig,
) -> list:
    """Run 3D hand pose estimation (HaMeR)."""
    from .hand_pose_estimator import estimate_hand_poses

    output_dir = config.resolve_path(
        getattr(config, "hand_pose", None) and config.hand_pose.output_dir
        or "data/hand_poses"
    )
    sample_rate = getattr(config, "hand_pose", None) and config.hand_pose.sample_rate or 1

    return estimate_hand_poses(
        video_path,
        hand_detections=hand_detections,
        sample_rate=sample_rate,
        output_dir=output_dir,
    )


def run_object_detection(
    video_path: Path,
    config: PipelineConfig,
) -> list:
    """Run open-vocabulary object detection (GroundingDINO)."""
    from .object_detector import detect_objects

    output_dir = config.resolve_path(
        getattr(config, "object_detection", None) and config.object_detection.output_dir
        or "data/objects"
    )
    sample_rate = (
        getattr(config, "object_detection", None) and config.object_detection.sample_rate or 30
    )
    text_prompts = (
        getattr(config, "object_detection", None) and config.object_detection.text_prompts or None
    )
    box_threshold = (
        getattr(config, "object_detection", None) and config.object_detection.box_threshold or 0.3
    )

    return detect_objects(
        video_path,
        text_prompts=text_prompts,
        sample_rate=sample_rate,
        box_threshold=box_threshold,
        output_dir=output_dir,
    )


def run_segmentation(
    video_path: Path,
    hand_detections: list[FrameDetection] | None,
    object_detections: list | None,
    config: PipelineConfig,
) -> list:
    """Run segmentation (SAM2)."""
    from .segmenter import segment_video

    output_dir = config.resolve_path(
        getattr(config, "segmentation", None) and config.segmentation.output_dir
        or "data/segmentation"
    )
    sample_rate = (
        getattr(config, "segmentation", None) and config.segmentation.sample_rate or 1
    )
    model_size = (
        getattr(config, "segmentation", None) and config.segmentation.model_size or "small"
    )

    return segment_video(
        video_path,
        hand_detections=hand_detections,
        object_detections=object_detections,
        sample_rate=sample_rate,
        model_size=model_size,
        output_dir=output_dir,
    )


def run_lerobot_export(
    episodes: list[dict],
    config: PipelineConfig,
    output_dir: str | None = None,
) -> Path:
    """Export to LeRobot dataset format."""
    from .lerobot_exporter import export_lerobot

    if output_dir is None:
        output_dir = config.resolve_path(
            getattr(config, "lerobot", None) and config.lerobot.output_dir
            or "data/lerobot"
        )

    return export_lerobot(episodes, output_dir)


def run_pipeline_enhanced(
    config: PipelineConfig,
    urls: list[str],
    enable_hand_pose: bool = True,
    enable_object_detection: bool = True,
    enable_segmentation: bool = True,
    export_format: str = "lerobot",
    hand_pose_output_dir: str | None = None,
    object_output_dir: str | None = None,
    segmentation_output_dir: str | None = None,
    lerobot_output_dir: str | None = None,
) -> None:
    """Run the enhanced SOTA robotics pipeline.

    Extended pipeline:
        download → detect → clip → depth → hand_pose → objects → segmentation
        → trajectories → export (LeRobot / JSON / CSV)

    Args:
        config: Pipeline configuration.
        urls: List of YouTube URLs to process.
        enable_hand_pose: Run HaMeR 3D hand pose estimation.
        enable_object_detection: Run GroundingDINO object detection.
        enable_segmentation: Run SAM2 segmentation.
        export_format: "lerobot", "json", "csv", or "all".
        hand_pose_output_dir: Override for hand pose output directory.
        object_output_dir: Override for object detection output directory.
        segmentation_output_dir: Override for segmentation output directory.
        lerobot_output_dir: Override for LeRobot export directory.
    """
    all_episodes = []

    for url in tqdm(urls, desc="Processing videos"):
        print(f"\n{'='*60}")
        print(f"Downloading: {url}")
        video_path = run_download(url, config)
        print(f"  Saved: {video_path}")

        print("Detecting hands...")
        detections = run_detect(video_path, config)
        hand_frames = sum(1 for d in detections if d.detected)
        print(f"  Hands detected in {hand_frames}/{len(detections)} frames")

        print("Extracting clips...")
        clips = run_clip(video_path, detections, config)
        print(f"  Extracted {len(clips)} clips")

        for clip in tqdm(clips, desc="Processing clips", leave=False):
            print(f"\n  Processing clip: {clip.clip_id}")

            # --- Core stages (same as original) ---
            print("    Detecting hands in clip...")
            clip_detections = run_detect(clip.path, config)

            print("    Estimating depth...")
            depth_frames = run_depth(clip.path, config)
            depth_video = run_depth_video(clip.path, depth_frames, config)
            print(f"    Depth video: {depth_video}")

            # --- Enhanced stages ---
            hand_poses = None
            if enable_hand_pose:
                print("    Estimating 3D hand poses (HaMeR)...")
                hand_poses = run_hand_poses(clip.path, clip_detections, config)
                print(f"    Hand poses: {len(hand_poses)} frames")

            obj_detections = None
            if enable_object_detection:
                print("    Detecting objects (GroundingDINO)...")
                obj_detections = run_object_detection(clip.path, config)
                total_objects = sum(len(f.objects) for f in obj_detections)
                print(f"    Objects: {total_objects} detections across {len(obj_detections)} frames")

            segmentations = None
            if enable_segmentation:
                print("    Segmenting (SAM2)...")
                segmentations = run_segmentation(
                    clip.path, clip_detections, obj_detections, config
                )
                total_masks = sum(len(f.masks) for f in segmentations)
                print(f"    Segmentation: {total_masks} masks across {len(segmentations)} frames")

            # --- Trajectories ---
            print("    Computing trajectories...")
            traj_data = run_trajectories(
                clip.path, clip_detections, depth_frames, clip, config
            )
            traj_video = run_trajectory_video(clip.path, clip_detections, config)
            print(f"    Trajectory video: {traj_video}")

            # --- Traditional export ---
            if export_format in ("json", "csv", "all"):
                print("    Exporting data...")
                exported = run_export(traj_data, config)
                for p in exported:
                    print(f"      {p}")

            # --- Collect episode for LeRobot export ---
            all_episodes.append({
                "traj_data": traj_data,
                "clip_info": clip,
                "hand_poses": hand_poses,
                "object_detections": obj_detections,
                "segmentations": segmentations,
                "video_path": clip.path,
            })

    # --- LeRobot export ---
    if export_format in ("lerobot", "all") and all_episodes:
        print(f"\n{'='*60}")
        print("Exporting LeRobot dataset...")
        dataset_path = run_lerobot_export(
            all_episodes, config, output_dir=lerobot_output_dir
        )
        print(f"  Dataset: {dataset_path}")

    print(f"\n{'='*60}")
    print(f"Enhanced pipeline complete. Processed {len(all_episodes)} episodes.")
