"""Pipeline performance benchmark across resolutions."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2


# ── Resolution helpers ───────────────────────────────────────────────

RESOLUTION_MAP = {
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}


def _video_height(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h


def _video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def resize_video(input_path: Path, output_path: Path, height: int) -> Path:
    """Downscale *input_path* to the given *height* using ffmpeg.

    If the video is already at (or below) the target height, a symlink is
    created instead so we skip re-encoding.
    """
    current_h = _video_height(input_path)
    if current_h <= height:
        if not output_path.exists():
            output_path.symlink_to(input_path.resolve())
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output_path


# ── Data structures ──────────────────────────────────────────────────

STAGE_NAMES = [
    "detect_hands_raw",
    "extract_clips",
    "filter_detections_clip",
    "estimate_depth",
    "map_trajectories",
    "render_trajectory_video",
]


@dataclass
class ResolutionResult:
    resolution: str
    height: int
    file_size_mb: float
    duration_s: float
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds


@dataclass
class VideoReport:
    video_name: str
    results: list[ResolutionResult] = field(default_factory=list)


# ── Core benchmark ───────────────────────────────────────────────────

def run_benchmark(
    config,
    video_paths: list[Path],
    resolutions: list[str],
    skip_trajectory_render: bool = False,
) -> list[VideoReport]:
    """Benchmark every *video* at each *resolution*.

    Returns one :class:`VideoReport` per input video.

    .. deprecated::
        Use :func:`enhanced_benchmark.run_enhanced_benchmark` instead,
        which covers all 11 pipeline stages with both timing and quality.
    """
    import warnings

    warnings.warn(
        "run_benchmark() is deprecated. Use enhanced_benchmark.run_enhanced_benchmark() "
        "which covers all 11 pipeline stages with both timing and quality metrics.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .config import PipelineConfig
    from .pipeline import (
        _slice_detections_for_clip,
        run_clip,
        run_depth,
        run_detect,
        run_trajectories,
        run_trajectory_video,
    )

    reports: list[VideoReport] = []

    for video_path in video_paths:
        video_name = video_path.stem
        report = VideoReport(video_name=video_name)
        print(f"\nBenchmarking: {video_name}")

        for res_label in resolutions:
            height = RESOLUTION_MAP[res_label]
            print(f"  {res_label} ({height}p) ...")

            # Resize
            resized_dir = config.resolve_path("data/benchmark/resized")
            resized_dir.mkdir(parents=True, exist_ok=True)
            resized_path = resized_dir / f"{video_name}_{res_label}.mp4"
            resize_video(video_path, resized_path, height)

            size_mb = _file_size_mb(resized_path)
            dur_s = _video_duration(resized_path)

            rr = ResolutionResult(
                resolution=res_label,
                height=height,
                file_size_mb=round(size_mb, 2),
                duration_s=round(dur_s, 2),
            )

            # Stage 1: detect hands on full (resized) video
            t0 = time.perf_counter()
            detections = run_detect(resized_path, config)
            rr.timings["detect_hands_raw"] = time.perf_counter() - t0

            # Stage 2: extract clips
            t0 = time.perf_counter()
            clips = run_clip(resized_path, detections, config)
            rr.timings["extract_clips"] = time.perf_counter() - t0

            # Use first clip for remaining stages (representative sample)
            if clips:
                clip_path = clips[0].path
                clip_info = clips[0]
            else:
                # No clips found – use the full video as fallback
                clip_path = resized_path
                clip_info = None

            # Stage 3: filter detections for clip (reuses raw detections)
            t0 = time.perf_counter()
            if clip_info is not None:
                clip_detections = _slice_detections_for_clip(detections, clip_info)
            else:
                clip_detections = detections
            rr.timings["filter_detections_clip"] = time.perf_counter() - t0

            # Stage 4: estimate depth
            t0 = time.perf_counter()
            depth_frames = run_depth(clip_path, config)
            rr.timings["estimate_depth"] = time.perf_counter() - t0

            # Stage 5: map trajectories
            t0 = time.perf_counter()
            run_trajectories(
                clip_path, clip_detections, depth_frames, clip_info, config,
            )
            rr.timings["map_trajectories"] = time.perf_counter() - t0

            # Stage 6: render trajectory video
            if skip_trajectory_render:
                rr.timings["render_trajectory_video"] = 0.0
            else:
                t0 = time.perf_counter()
                run_trajectory_video(clip_path, clip_detections, config)
                rr.timings["render_trajectory_video"] = time.perf_counter() - t0

            report.results.append(rr)

        reports.append(report)

    return reports


# ── Report formatting ────────────────────────────────────────────────

def format_report_table(reports: list[VideoReport]) -> str:
    """Return a human-readable multi-table report string."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append("RESOLUTION BENCHMARK REPORT")
    lines.append(sep)

    for report in reports:
        res_labels = [r.resolution for r in report.results]
        col_w = 12  # width per resolution column

        lines.append(f"Video: {report.video_name}")

        sizes = "  ".join(
            f"{r.resolution}={r.file_size_mb:.1f}MB" for r in report.results
        )
        lines.append(f"File sizes: {sizes}")

        durs = "  ".join(
            f"{r.resolution}={r.duration_s:.1f}s" for r in report.results
        )
        lines.append(f"Duration:   {durs}")

        # ── Table 1: Elapsed time (seconds) ──
        lines.append("")
        lines.append("TABLE 1: Elapsed Time (seconds)")
        lines.append(dash)
        header = f"{'Component':<32}" + "".join(
            f"{r:>{col_w}}" for r in res_labels
        )
        lines.append(header)
        lines.append(dash)

        totals = {r.resolution: 0.0 for r in report.results}
        for stage in STAGE_NAMES:
            row = f"{stage:<32}"
            for r in report.results:
                t = r.timings.get(stage, 0.0)
                totals[r.resolution] += t
                row += f"{t:>{col_w - 1}.2f}s"
            lines.append(row)

        lines.append(dash)
        row = f"{'TOTAL':<32}"
        for r in report.results:
            row += f"{totals[r.resolution]:>{col_w - 1}.2f}s"
        lines.append(row)

        # ── Table 2: Processing time per MB ──
        lines.append("")
        lines.append("TABLE 2: Processing Time per MB (sec/MB)")
        lines.append(dash)
        lines.append(header)
        lines.append(dash)

        totals_pm = {r.resolution: 0.0 for r in report.results}
        for stage in STAGE_NAMES:
            row = f"{stage:<32}"
            for r in report.results:
                t = r.timings.get(stage, 0.0)
                v = t / r.file_size_mb if r.file_size_mb > 0 else 0.0
                totals_pm[r.resolution] += v
                row += f"{v:>{col_w}.2f}"
            lines.append(row)

        lines.append(dash)
        row = f"{'TOTAL':<32}"
        for r in report.results:
            row += f"{totals_pm[r.resolution]:>{col_w}.2f}"
        lines.append(row)

        # ── Table 3: Processing time per second of video ──
        lines.append("")
        lines.append("TABLE 3: Processing Time per Second of Video (sec/sec)")
        lines.append(dash)
        lines.append(header)
        lines.append(dash)

        totals_ps = {r.resolution: 0.0 for r in report.results}
        for stage in STAGE_NAMES:
            row = f"{stage:<32}"
            for r in report.results:
                t = r.timings.get(stage, 0.0)
                v = t / r.duration_s if r.duration_s > 0 else 0.0
                totals_ps[r.resolution] += v
                row += f"{v:>{col_w}.2f}"
            lines.append(row)

        lines.append(dash)
        row = f"{'TOTAL':<32}"
        for r in report.results:
            row += f"{totals_ps[r.resolution]:>{col_w}.2f}"
        lines.append(row)

        # ── Speedup vs highest resolution ──
        if len(report.results) >= 2:
            baseline = report.results[-1]  # highest resolution = last
            lines.append("")
            lines.append(f"SPEEDUP vs {baseline.resolution}:")
            for stage in STAGE_NAMES:
                base_t = baseline.timings.get(stage, 0.0)
                if base_t <= 0:
                    continue
                parts: list[str] = []
                for r in report.results[:-1]:
                    t = r.timings.get(stage, 0.0)
                    speedup = base_t / t if t > 0 else float("inf")
                    parts.append(f"{r.resolution}: {speedup:.2f}x")
                if parts:
                    lines.append(f"  {stage:<28} {'  '.join(parts)}")

        lines.append("")

    return "\n".join(lines)


# ── Persistence ──────────────────────────────────────────────────────

def save_report(
    reports: list[VideoReport],
    output_dir: Path,
    *,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the benchmark report as both JSON and human-readable text.

    When *run_dir* is provided the files are written there instead of
    *output_dir* (used by the versioned run storage).

    Returns (json_path, txt_path).
    """
    dest = run_dir if run_dir is not None else output_dir
    dest.mkdir(parents=True, exist_ok=True)

    # JSON
    data = []
    for report in reports:
        entry = {
            "video_name": report.video_name,
            "results": [
                {
                    "resolution": r.resolution,
                    "height": r.height,
                    "file_size_mb": r.file_size_mb,
                    "duration_s": r.duration_s,
                    "timings": r.timings,
                }
                for r in report.results
            ],
        }
        data.append(entry)

    json_path = dest / "benchmark.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    # Text
    txt_path = dest / "benchmark.txt"
    txt_path.write_text(format_report_table(reports) + "\n")

    return json_path, txt_path
