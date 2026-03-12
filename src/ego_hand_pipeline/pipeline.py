"""Orchestrator that chains all pipeline stages."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _is_local_file(input_str: str) -> bool:
    """Determine if the input is a local file path or a URL."""
    p = Path(input_str)
    if p.exists() and p.is_file():
        return True
    # URL patterns
    if input_str.startswith(("http://", "https://", "www.")):
        return False
    # Could be a relative path that doesn't exist yet
    if p.suffix in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"):
        return True
    return False


# ---------------------------------------------------------------------------
# Individual pipeline stages
# ---------------------------------------------------------------------------

def run_download(url: str, config: PipelineConfig) -> Path:
    """Download a single video."""
    output_dir = config.resolve_path(config.download.output_dir)
    return download_video(
        url,
        output_dir=output_dir,
        resolution=config.download.resolution,
        format=config.download.format,
    )


def run_preprocess(
    video_path: Path,
    config: PipelineConfig,
    max_frames: int | None = None,
) -> Path:
    """Preprocess video to ideal resolution."""
    from .video_preprocessor import preprocess_video

    prep = config.preprocessing
    output_dir = config.resolve_path(prep.output_dir)

    return preprocess_video(
        video_path,
        output_dir=output_dir,
        target_width=prep.target_width,
        target_height=prep.target_height,
        target_fps=prep.target_fps,
        max_frames=max_frames,
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
    """Run the full pipeline: download -> detect -> clip -> depth -> trajectories -> export."""
    for url in tqdm(urls, desc="Processing videos"):
        print(f"\n{'='*60}")

        # Support local files
        if _is_local_file(url):
            video_path = Path(url)
            print(f"Using local file: {video_path}")
        else:
            print(f"Downloading: {url}")
            video_path = run_download(url, config)
            print(f"  Saved: {video_path}")

        # Preprocess to standard resolution
        if config.preprocessing.enabled:
            print("Preprocessing video to standard resolution...")
            video_path = run_preprocess(video_path, config)
            print(f"  Preprocessed: {video_path}")

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


def run_scene_annotation(
    video_path: Path,
    hand_detections: list[FrameDetection] | None,
    object_detections: list | None,
    config: PipelineConfig,
) -> list:
    """Run scene annotation / captioning."""
    from .scene_annotator import annotate_scenes

    sa = config.scene_annotation
    output_dir = config.resolve_path(sa.output_dir)

    return annotate_scenes(
        video_path,
        hand_detections=hand_detections,
        object_detections=object_detections,
        sample_rate=sa.sample_rate,
        model_name=sa.model_name,
        output_dir=output_dir,
    )


def run_sample_render(
    video_path: Path,
    config: PipelineConfig,
    hand_detections: list | None = None,
    object_detections: list | None = None,
    depth_frames: list | None = None,
    segmentations: list | None = None,
    scene_annotations: list | None = None,
    max_frames: int | None = None,
) -> Path:
    """Render a sample visualization video (test/benchmark mode)."""
    from .sample_renderer import render_sample_video

    output_dir = config.resolve_path(config.test_mode.sample_output_dir)
    output_path = output_dir / f"{video_path.stem}_sample.mp4"

    return render_sample_video(
        video_path,
        output_path,
        hand_detections=hand_detections,
        object_detections=object_detections,
        depth_frames=depth_frames,
        segmentations=segmentations,
        scene_annotations=scene_annotations,
        max_frames=max_frames,
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


def _resolve_input(input_str: str, config: PipelineConfig) -> Path:
    """Resolve an input string to a video file path.

    Supports:
    - Local file paths (absolute or relative)
    - YouTube URLs (downloads first)
    """
    if _is_local_file(input_str):
        path = Path(input_str)
        if not path.exists():
            raise FileNotFoundError(f"Local video file not found: {path}")
        return path
    else:
        return run_download(input_str, config)


def run_pipeline_enhanced(
    config: PipelineConfig,
    urls: list[str],
    enable_hand_pose: bool = True,
    enable_object_detection: bool = True,
    enable_segmentation: bool = True,
    enable_scene_annotation: bool = True,
    export_format: str = "lerobot",
    test_mode: bool = False,
    hand_pose_output_dir: str | None = None,
    object_output_dir: str | None = None,
    segmentation_output_dir: str | None = None,
    lerobot_output_dir: str | None = None,
) -> None:
    """Run the enhanced SOTA robotics pipeline.

    Extended pipeline:
        [download/load] → preprocess → detect → clip → depth → hand_pose
        → objects → segmentation → scene_annotation → trajectories
        → export (LeRobot / JSON / CSV)

    Supports both YouTube URLs and local video files as input.
    Independent stages (hand_pose, object_detection) run in parallel.

    Args:
        config: Pipeline configuration.
        urls: List of YouTube URLs or local file paths to process.
        enable_hand_pose: Run HaMeR 3D hand pose estimation.
        enable_object_detection: Run GroundingDINO object detection.
        enable_segmentation: Run SAM2 segmentation.
        enable_scene_annotation: Run scene captioning.
        export_format: "lerobot", "json", "csv", or "all".
        test_mode: Enable test mode (limited frames, sample video output).
        hand_pose_output_dir: Override for hand pose output directory.
        object_output_dir: Override for object detection output directory.
        segmentation_output_dir: Override for segmentation output directory.
        lerobot_output_dir: Override for LeRobot export directory.
    """
    all_episodes = []

    # Test mode settings
    is_test = test_mode or config.test_mode.enabled
    max_frames = config.test_mode.max_frames if is_test else None

    if is_test:
        print(f"TEST MODE: processing max {max_frames} frames per video")
        print(f"  Sample videos will be rendered to {config.test_mode.sample_output_dir}")

    timings: dict[str, list[float]] = {}

    for input_str in tqdm(urls, desc="Processing videos"):
        print(f"\n{'='*60}")

        # --- Resolve input (local file or URL) ---
        t0 = time.time()
        if _is_local_file(input_str):
            video_path = Path(input_str)
            if not video_path.exists():
                print(f"  ERROR: File not found: {video_path}")
                continue
            print(f"Using local file: {video_path}")
        else:
            print(f"Downloading: {input_str}")
            video_path = run_download(input_str, config)
            print(f"  Saved: {video_path}")
        _record_timing(timings, "download/load", time.time() - t0)

        # --- Preprocess to standard resolution ---
        if config.preprocessing.enabled:
            t0 = time.time()
            print(f"Preprocessing to {config.preprocessing.target_width}x{config.preprocessing.target_height}...")
            video_path = run_preprocess(video_path, config, max_frames=max_frames)
            print(f"  Preprocessed: {video_path}")
            _record_timing(timings, "preprocess", time.time() - t0)

        # --- Hand detection ---
        t0 = time.time()
        print("Detecting hands...")
        detections = run_detect(video_path, config)
        hand_frames = sum(1 for d in detections if d.detected)
        print(f"  Hands detected in {hand_frames}/{len(detections)} frames")
        _record_timing(timings, "hand_detection", time.time() - t0)

        # --- Clip extraction ---
        t0 = time.time()
        print("Extracting clips...")
        clips = run_clip(video_path, detections, config)
        print(f"  Extracted {len(clips)} clips")
        _record_timing(timings, "clip_extraction", time.time() - t0)

        for clip in tqdm(clips, desc="Processing clips", leave=False):
            print(f"\n  Processing clip: {clip.clip_id}")

            # --- Core stages ---
            t0 = time.time()
            print("    Detecting hands in clip...")
            clip_detections = run_detect(clip.path, config)
            _record_timing(timings, "clip_hand_detection", time.time() - t0)

            t0 = time.time()
            print("    Estimating depth...")
            depth_frames = run_depth(clip.path, config)
            depth_video = run_depth_video(clip.path, depth_frames, config)
            print(f"    Depth video: {depth_video}")
            _record_timing(timings, "depth_estimation", time.time() - t0)

            # --- Enhanced stages (parallel where possible) ---
            hand_poses = None
            obj_detections = None
            scene_annotations = None

            # Hand pose and object detection are independent — run in parallel
            parallel_tasks = {}

            if enable_hand_pose:
                parallel_tasks["hand_pose"] = (
                    run_hand_poses, (clip.path, clip_detections, config)
                )
            if enable_object_detection:
                parallel_tasks["object_detection"] = (
                    run_object_detection, (clip.path, config)
                )

            if parallel_tasks:
                t0 = time.time()
                results = _run_parallel(parallel_tasks)
                hand_poses = results.get("hand_pose")
                obj_detections = results.get("object_detection")

                if hand_poses is not None:
                    print(f"    Hand poses: {len(hand_poses)} frames")
                    _record_timing(timings, "hand_pose", time.time() - t0)
                if obj_detections is not None:
                    total_objects = sum(len(f.objects) for f in obj_detections)
                    print(f"    Objects: {total_objects} detections across {len(obj_detections)} frames")
                    _record_timing(timings, "object_detection", time.time() - t0)

            # Segmentation depends on object detections
            segmentations = None
            if enable_segmentation:
                t0 = time.time()
                print("    Segmenting (SAM2)...")
                segmentations = run_segmentation(
                    clip.path, clip_detections, obj_detections, config
                )
                total_masks = sum(len(f.masks) for f in segmentations)
                print(f"    Segmentation: {total_masks} masks across {len(segmentations)} frames")
                _record_timing(timings, "segmentation", time.time() - t0)

            # Scene annotation
            if enable_scene_annotation and config.scene_annotation.enabled:
                t0 = time.time()
                print("    Annotating scenes...")
                scene_annotations = run_scene_annotation(
                    clip.path, clip_detections, obj_detections, config
                )
                print(f"    Scene annotations: {len(scene_annotations)} frames")
                _record_timing(timings, "scene_annotation", time.time() - t0)

            # --- Trajectories ---
            t0 = time.time()
            print("    Computing trajectories...")
            traj_data = run_trajectories(
                clip.path, clip_detections, depth_frames, clip, config
            )
            traj_video = run_trajectory_video(clip.path, clip_detections, config)
            print(f"    Trajectory video: {traj_video}")
            _record_timing(timings, "trajectories", time.time() - t0)

            # --- Traditional export ---
            if export_format in ("json", "csv", "all"):
                print("    Exporting data...")
                exported = run_export(traj_data, config)
                for p in exported:
                    print(f"      {p}")

            # --- Test mode: render sample video ---
            if is_test and config.test_mode.render_samples:
                t0 = time.time()
                print("    Rendering sample visualization video...")
                sample_path = run_sample_render(
                    clip.path, config,
                    hand_detections=clip_detections,
                    object_detections=obj_detections,
                    depth_frames=depth_frames,
                    segmentations=segmentations,
                    scene_annotations=scene_annotations,
                    max_frames=max_frames,
                )
                print(f"    Sample video: {sample_path}")
                _record_timing(timings, "sample_render", time.time() - t0)

            # --- Collect episode for LeRobot export ---
            all_episodes.append({
                "traj_data": traj_data,
                "clip_info": clip,
                "hand_poses": hand_poses,
                "object_detections": obj_detections,
                "segmentations": segmentations,
                "scene_annotations": scene_annotations,
                "video_path": clip.path,
            })

    # --- LeRobot export ---
    if export_format in ("lerobot", "all") and all_episodes:
        t0 = time.time()
        print(f"\n{'='*60}")
        print("Exporting LeRobot dataset...")
        dataset_path = run_lerobot_export(
            all_episodes, config, output_dir=lerobot_output_dir
        )
        print(f"  Dataset: {dataset_path}")
        _record_timing(timings, "lerobot_export", time.time() - t0)

    print(f"\n{'='*60}")
    print(f"Enhanced pipeline complete. Processed {len(all_episodes)} episodes.")

    # --- Print timing summary (always in test mode, optional otherwise) ---
    if is_test and timings:
        _print_timing_summary(timings)


def _run_parallel(tasks: dict[str, tuple]) -> dict:
    """Run multiple pipeline stages in parallel using threads.

    Args:
        tasks: dict mapping name -> (function, args_tuple)

    Returns:
        dict mapping name -> result
    """
    results = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {}
        for name, (func, args) in tasks.items():
            future = executor.submit(func, *args)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"    WARNING: {name} failed: {e}")
                results[name] = None

    return results


def _record_timing(timings: dict[str, list[float]], stage: str, elapsed: float) -> None:
    """Record timing for a pipeline stage."""
    timings.setdefault(stage, []).append(elapsed)


def _print_timing_summary(timings: dict[str, list[float]]) -> None:
    """Print a timing summary table."""
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'Stage':<25} {'Count':>6} {'Total':>10} {'Avg':>10}")
    print(f"{'-'*25} {'-'*6} {'-'*10} {'-'*10}")

    total_time = 0.0
    for stage, times in timings.items():
        total = sum(times)
        avg = total / len(times)
        total_time += total
        print(f"{stage:<25} {len(times):>6} {total:>9.2f}s {avg:>9.2f}s")

    print(f"{'-'*25} {'-'*6} {'-'*10} {'-'*10}")
    print(f"{'TOTAL':<25} {'':>6} {total_time:>9.2f}s")
