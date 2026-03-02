"""Segment extraction based on hand visibility detections."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .hand_detector import FrameDetection


@dataclass
class Segment:
    """A contiguous range of frames where hands are visible."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass
class ClipInfo:
    """Metadata for an extracted clip."""
    clip_id: str
    path: Path
    source_video: str
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int


def _find_segments(detections: list[FrameDetection]) -> list[Segment]:
    """Find contiguous runs of frames where at least one hand is detected."""
    segments: list[Segment] = []
    current_start: int | None = None

    for det in detections:
        if det.detected:
            if current_start is None:
                current_start = det.frame_idx
        else:
            if current_start is not None:
                prev = detections[det.frame_idx - 1] if det.frame_idx > 0 else det
                segments.append(Segment(
                    start_frame=current_start,
                    end_frame=det.frame_idx - 1,
                    start_time=detections[current_start].timestamp,
                    end_time=prev.timestamp,
                ))
                current_start = None

    # Close trailing segment
    if current_start is not None:
        last = detections[-1]
        segments.append(Segment(
            start_frame=current_start,
            end_frame=last.frame_idx,
            start_time=detections[current_start].timestamp,
            end_time=last.timestamp,
        ))

    return segments


def _merge_segments(segments: list[Segment], merge_gap: float) -> list[Segment]:
    """Merge segments that are separated by less than merge_gap seconds."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start_time - prev.end_time <= merge_gap:
            merged[-1] = Segment(
                start_frame=prev.start_frame,
                end_frame=seg.end_frame,
                start_time=prev.start_time,
                end_time=seg.end_time,
            )
        else:
            merged.append(seg)

    return merged


def _filter_segments(segments: list[Segment], min_duration: float) -> list[Segment]:
    """Remove segments shorter than min_duration seconds."""
    return [s for s in segments if (s.end_time - s.start_time) >= min_duration]


def _extract_clip_ffmpeg(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
) -> None:
    """Extract a clip using ffmpeg stream copy (no re-encoding)."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_clips(
    video_path: str | Path,
    detections: list[FrameDetection],
    output_dir: str | Path = "data/clips",
    min_duration: float = 1.0,
    merge_gap: float = 0.5,
) -> list[ClipInfo]:
    """Extract hand-visible clips from a video based on detection results.

    Args:
        video_path: Source video path.
        detections: Per-frame detection results from hand_detector.
        output_dir: Directory to write clips.
        min_duration: Minimum clip duration in seconds.
        merge_gap: Maximum gap between segments to merge (seconds).

    Returns:
        List of ClipInfo describing each extracted clip.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = _find_segments(detections)
    segments = _merge_segments(segments, merge_gap)
    segments = _filter_segments(segments, min_duration)

    clips: list[ClipInfo] = []
    stem = video_path.stem

    for i, seg in enumerate(segments):
        clip_id = f"{stem}_clip_{i:03d}"
        clip_path = output_dir / f"{clip_id}.mp4"
        duration = seg.end_time - seg.start_time

        _extract_clip_ffmpeg(video_path, clip_path, seg.start_time, seg.end_time)

        clips.append(ClipInfo(
            clip_id=clip_id,
            path=clip_path,
            source_video=video_path.name,
            start_time=round(seg.start_time, 4),
            end_time=round(seg.end_time, 4),
            duration=round(duration, 4),
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
        ))

    return clips
