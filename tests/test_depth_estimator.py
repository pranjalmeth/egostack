"""Tests for the depth_estimator module."""

import numpy as np
import pytest

from ego_hand_pipeline.depth_estimator import DepthFrame, _save_depth_viz


def test_depth_frame_structure():
    depth_map = np.random.rand(480, 640).astype(np.float32)
    df = DepthFrame(frame_idx=0, timestamp=0.0, depth_map=depth_map)
    assert df.depth_map.shape == (480, 640)
    assert df.npz_path is None


def test_save_depth_viz(tmp_path):
    depth_map = np.random.rand(100, 100).astype(np.float32)
    viz_path = tmp_path / "depth_viz.png"
    _save_depth_viz(depth_map, viz_path)
    assert viz_path.exists()
    assert viz_path.stat().st_size > 0


def test_save_depth_viz_constant(tmp_path):
    """Test visualization with a constant depth map (max_val == 0)."""
    depth_map = np.ones((100, 100), dtype=np.float32) * 5.0
    viz_path = tmp_path / "constant.png"
    _save_depth_viz(depth_map, viz_path)
    assert viz_path.exists()
