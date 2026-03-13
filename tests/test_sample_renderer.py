"""Tests for sample_renderer module."""

import numpy as np
import pytest


def test_build_lookup_empty():
    """Empty input should return empty dict."""
    from ego_hand_pipeline.sample_renderer import _build_lookup

    assert _build_lookup(None) == {}
    assert _build_lookup([]) == {}


def test_build_lookup_with_objects():
    """Should build a frame_idx -> item lookup."""
    from ego_hand_pipeline.sample_renderer import _build_lookup
    from dataclasses import dataclass

    @dataclass
    class FakeFrame:
        frame_idx: int
        value: str

    items = [FakeFrame(0, "a"), FakeFrame(5, "b"), FakeFrame(10, "c")]
    lookup = _build_lookup(items)

    assert lookup[0].value == "a"
    assert lookup[5].value == "b"
    assert lookup[10].value == "c"
    assert 3 not in lookup


def test_draw_caption_bar():
    """Caption bar should draw text on the canvas."""
    from ego_hand_pipeline.sample_renderer import _draw_caption_bar

    canvas = np.zeros((540, 1280, 3), dtype=np.uint8)
    _draw_caption_bar(canvas, "Test caption text", 480, 1280, 60)

    # Bottom 60 rows should not be all black (has dark background + text)
    assert canvas[480:, :].sum() > 0


def test_colorize_depth():
    """Depth colorization should produce correct dimensions."""
    from ego_hand_pipeline.sample_renderer import _colorize_depth

    depth = np.random.rand(100, 150).astype(np.float32)
    colored = _colorize_depth(depth, 640, 480)

    assert colored.shape == (480, 640, 3)
    assert colored.dtype == np.uint8


def test_colorize_depth_flat():
    """Flat depth map (all zeros) should not crash."""
    from ego_hand_pipeline.sample_renderer import _colorize_depth

    depth = np.zeros((100, 100), dtype=np.float32)
    colored = _colorize_depth(depth, 320, 240)
    assert colored.shape == (240, 320, 3)
