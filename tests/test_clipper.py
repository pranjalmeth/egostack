"""Tests for the clipper module."""

from unittest.mock import patch

import pytest

from ego_hand_pipeline.clipper import (
    ClipInfo,
    Segment,
    _filter_segments,
    _find_segments,
    _merge_segments,
    extract_clips,
)
from ego_hand_pipeline.hand_detector import FrameDetection


def _make_detections(pattern: list[bool], fps: float = 30.0) -> list[FrameDetection]:
    """Helper: create FrameDetection list from a bool pattern."""
    return [
        FrameDetection(
            frame_idx=i,
            timestamp=round(i / fps, 4),
            detected=d,
            num_hands=1 if d else 0,
        )
        for i, d in enumerate(pattern)
    ]


def test_find_segments_basic():
    # 10 frames at 30fps: hands visible in frames 2-5
    pattern = [False, False, True, True, True, True, False, False, False, False]
    detections = _make_detections(pattern)
    segments = _find_segments(detections)
    assert len(segments) == 1
    assert segments[0].start_frame == 2
    assert segments[0].end_frame == 5


def test_find_segments_multiple():
    pattern = [True, True, False, False, True, True, True, False]
    detections = _make_detections(pattern)
    segments = _find_segments(detections)
    assert len(segments) == 2


def test_find_segments_trailing():
    pattern = [False, True, True, True]
    detections = _make_detections(pattern)
    segments = _find_segments(detections)
    assert len(segments) == 1
    assert segments[0].end_frame == 3


def test_merge_segments():
    segs = [
        Segment(start_frame=0, end_frame=10, start_time=0.0, end_time=0.33),
        Segment(start_frame=15, end_frame=30, start_time=0.5, end_time=1.0),
    ]
    # Gap of 0.17s < merge_gap of 0.5s => should merge
    merged = _merge_segments(segs, merge_gap=0.5)
    assert len(merged) == 1
    assert merged[0].start_frame == 0
    assert merged[0].end_frame == 30


def test_merge_segments_no_merge():
    segs = [
        Segment(start_frame=0, end_frame=10, start_time=0.0, end_time=0.33),
        Segment(start_frame=60, end_frame=90, start_time=2.0, end_time=3.0),
    ]
    merged = _merge_segments(segs, merge_gap=0.5)
    assert len(merged) == 2


def test_filter_segments():
    segs = [
        Segment(start_frame=0, end_frame=10, start_time=0.0, end_time=0.3),
        Segment(start_frame=30, end_frame=90, start_time=1.0, end_time=3.0),
    ]
    filtered = _filter_segments(segs, min_duration=1.0)
    assert len(filtered) == 1
    assert filtered[0].start_frame == 30


def test_filter_segments_all_pass():
    segs = [
        Segment(start_frame=0, end_frame=60, start_time=0.0, end_time=2.0),
    ]
    filtered = _filter_segments(segs, min_duration=1.0)
    assert len(filtered) == 1


@patch("ego_hand_pipeline.clipper._extract_clip_ffmpeg")
def test_extract_clips(mock_ffmpeg, tmp_path):
    pattern = [True] * 60 + [False] * 30 + [True] * 90
    detections = _make_detections(pattern)

    clips = extract_clips(
        video_path="test_video.mp4",
        detections=detections,
        output_dir=tmp_path,
        min_duration=1.0,
        merge_gap=0.5,
    )

    assert len(clips) >= 1
    assert all(c.duration >= 1.0 for c in clips)
    assert mock_ffmpeg.called
