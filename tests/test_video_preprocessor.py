"""Tests for video_preprocessor module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_resize_with_padding_matching_aspect():
    """Resize with matching aspect ratio should just resize."""
    from ego_hand_pipeline.video_preprocessor import _resize_with_padding

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 255
    result = _resize_with_padding(frame, 320, 240)

    assert result.shape == (240, 320, 3)


def test_resize_with_padding_wider_source():
    """Wider source should get pillarboxed (black top/bottom)."""
    from ego_hand_pipeline.video_preprocessor import _resize_with_padding

    # 16:9 source -> 4:3 target
    frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
    result = _resize_with_padding(frame, 640, 480)

    assert result.shape == (480, 640, 3)
    # Top should be black (padding)
    assert result[0, 320, 0] == 0


def test_resize_with_padding_taller_source():
    """Taller source should get letterboxed (black left/right)."""
    from ego_hand_pipeline.video_preprocessor import _resize_with_padding

    # 3:4 source -> 4:3 target
    frame = np.ones((640, 480, 3), dtype=np.uint8) * 128
    result = _resize_with_padding(frame, 640, 480)

    assert result.shape == (480, 640, 3)


def test_needs_preprocessing_matching():
    """Video at target resolution should not need preprocessing."""
    from ego_hand_pipeline.video_preprocessor import needs_preprocessing

    with patch("ego_hand_pipeline.video_preprocessor.get_video_info") as mock_info:
        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0}
        assert not needs_preprocessing("test.mp4")


def test_needs_preprocessing_different_resolution():
    """Video at different resolution should need preprocessing."""
    from ego_hand_pipeline.video_preprocessor import needs_preprocessing

    with patch("ego_hand_pipeline.video_preprocessor.get_video_info") as mock_info:
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0}
        assert needs_preprocessing("test.mp4")


def test_get_video_info_file_not_found():
    """Missing video file should raise FileNotFoundError."""
    from ego_hand_pipeline.video_preprocessor import get_video_info

    with pytest.raises(FileNotFoundError):
        get_video_info("/nonexistent/video.mp4")
