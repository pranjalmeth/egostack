"""Command-line interface for the ego-hand-pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .downloader import load_urls_from_file


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML (default: config.yaml in current directory)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for relative paths (default: current directory)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ego-hand-pipeline",
        description="Egocentric hand detection, depth estimation, and trajectory mapping pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run (full pipeline) ---
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--urls", nargs="+", help="YouTube video URLs")
    run_parser.add_argument("--url-file", type=str, help="File with one URL per line")
    _add_common_args(run_parser)

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Download videos only")
    dl_parser.add_argument("--urls", nargs="+", help="YouTube video URLs")
    dl_parser.add_argument("--url-file", type=str, help="File with one URL per line")
    _add_common_args(dl_parser)

    # --- detect ---
    det_parser = subparsers.add_parser("detect", help="Run hand detection on a video")
    det_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    _add_common_args(det_parser)

    # --- clip ---
    clip_parser = subparsers.add_parser("clip", help="Extract hand-visible clips")
    clip_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    _add_common_args(clip_parser)

    # --- depth ---
    depth_parser = subparsers.add_parser("depth", help="Run depth estimation on a video")
    depth_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    _add_common_args(depth_parser)

    # --- trajectories ---
    traj_parser = subparsers.add_parser("trajectories", help="Compute hand joint trajectories")
    traj_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    _add_common_args(traj_parser)

    return parser


def _collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if getattr(args, "urls", None):
        urls.extend(args.urls)
    if getattr(args, "url_file", None):
        urls.extend(load_urls_from_file(args.url_file))
    return urls


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        default = Path("config.yaml")
        config_path = str(default) if default.exists() else None

    config = load_config(config_path, overrides={"base_dir": args.base_dir})

    if args.command == "run":
        from .pipeline import run_pipeline

        urls = _collect_urls(args)
        if not urls:
            parser.error("Provide URLs via --urls or --url-file")
        run_pipeline(config, urls)

    elif args.command == "download":
        from .downloader import download_videos

        urls = _collect_urls(args)
        if not urls:
            parser.error("Provide URLs via --urls or --url-file")
        output_dir = config.resolve_path(config.download.output_dir)
        paths = download_videos(urls, output_dir, config.download.resolution, config.download.format)
        for p in paths:
            print(f"Downloaded: {p}")

    elif args.command == "detect":
        from .pipeline import run_detect

        video_path = Path(args.video)
        detections = run_detect(video_path, config)
        hand_frames = sum(1 for d in detections if d.detected)
        print(f"Hands detected in {hand_frames}/{len(detections)} frames")

    elif args.command == "clip":
        from .pipeline import run_clip, run_detect

        video_path = Path(args.video)
        print("Detecting hands...")
        detections = run_detect(video_path, config)
        print("Extracting clips...")
        clips = run_clip(video_path, detections, config)
        for c in clips:
            print(f"  {c.clip_id}: {c.start_time:.2f}s - {c.end_time:.2f}s ({c.duration:.2f}s)")

    elif args.command == "depth":
        from .pipeline import run_depth, run_depth_video

        video_path = Path(args.video)
        depth_frames = run_depth(video_path, config)
        print(f"Processed {len(depth_frames)} depth frames")
        depth_video = run_depth_video(video_path, depth_frames, config)
        print(f"Depth video: {depth_video}")

    elif args.command == "trajectories":
        from .pipeline import (
            run_depth, run_depth_video, run_detect, run_export,
            run_trajectories, run_trajectory_video,
        )

        video_path = Path(args.video)
        print("Detecting hands...")
        detections = run_detect(video_path, config)
        print("Estimating depth...")
        depth_frames = run_depth(video_path, config)
        depth_video = run_depth_video(video_path, depth_frames, config)
        print(f"Depth video: {depth_video}")
        print("Computing trajectories...")
        traj_data = run_trajectories(video_path, detections, depth_frames, None, config)
        traj_video = run_trajectory_video(video_path, detections, config)
        print(f"Trajectory video: {traj_video}")
        print("Exporting...")
        exported = run_export(traj_data, config)
        for p in exported:
            print(f"  {p}")


if __name__ == "__main__":
    main()
