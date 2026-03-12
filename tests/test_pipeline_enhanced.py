"""Tests for enhanced pipeline features: local files, parallel execution, test mode."""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_is_local_file_with_path():
    """Local file paths should be detected."""
    from ego_hand_pipeline.pipeline import _is_local_file

    with patch("ego_hand_pipeline.pipeline.Path") as MockPath:
        instance = MockPath.return_value
        instance.exists.return_value = True
        instance.is_file.return_value = True
        instance.suffix = ".mp4"
        assert _is_local_file("/path/to/video.mp4")


def test_is_local_file_with_url():
    """URLs should not be detected as local files."""
    from ego_hand_pipeline.pipeline import _is_local_file

    assert not _is_local_file("https://youtube.com/watch?v=abc123")
    assert not _is_local_file("http://example.com/video.mp4")


def test_is_local_file_with_extension():
    """Files with video extensions should be treated as local."""
    from ego_hand_pipeline.pipeline import _is_local_file

    # This tests the extension-based fallback for non-existent local files
    with patch("ego_hand_pipeline.pipeline.Path") as MockPath:
        instance = MockPath.return_value
        instance.exists.return_value = False
        instance.is_file.return_value = False
        instance.suffix = ".mp4"
        assert _is_local_file("video.mp4")


def test_run_parallel_success():
    """Parallel execution should return results from all tasks."""
    from ego_hand_pipeline.pipeline import _run_parallel

    def task_a():
        return "result_a"

    def task_b():
        return "result_b"

    tasks = {
        "a": (task_a, ()),
        "b": (task_b, ()),
    }

    results = _run_parallel(tasks)
    assert results["a"] == "result_a"
    assert results["b"] == "result_b"


def test_run_parallel_with_failure():
    """Parallel execution should handle failures gracefully."""
    from ego_hand_pipeline.pipeline import _run_parallel

    def task_ok():
        return "ok"

    def task_fail():
        raise RuntimeError("boom")

    tasks = {
        "ok": (task_ok, ()),
        "fail": (task_fail, ()),
    }

    results = _run_parallel(tasks)
    assert results["ok"] == "ok"
    assert results["fail"] is None


def test_record_timing():
    """Timing recording should accumulate values."""
    from ego_hand_pipeline.pipeline import _record_timing

    timings = {}
    _record_timing(timings, "stage_a", 1.5)
    _record_timing(timings, "stage_a", 2.0)
    _record_timing(timings, "stage_b", 3.0)

    assert len(timings["stage_a"]) == 2
    assert sum(timings["stage_a"]) == pytest.approx(3.5)
    assert len(timings["stage_b"]) == 1


def test_config_new_sections():
    """Config should include new preprocessing, scene_annotation, test_mode sections."""
    from ego_hand_pipeline.config import load_config

    config = load_config(None)

    assert config.preprocessing.target_width == 640
    assert config.preprocessing.target_height == 480
    assert config.preprocessing.enabled is True

    assert config.scene_annotation.enabled is True
    assert config.scene_annotation.model_name == "rule_based"

    assert config.test_mode.enabled is False
    assert config.test_mode.max_frames == 900
    assert config.test_mode.render_samples is True


def test_config_load_yaml_with_new_sections(tmp_path):
    """Config should load new sections from YAML."""
    from ego_hand_pipeline.config import load_config

    yaml_content = """
preprocessing:
  enabled: false
  target_width: 1280
  target_height: 720

scene_annotation:
  model_name: blip2
  sample_rate: 15

test_mode:
  enabled: true
  max_frames: 300
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(str(config_file))

    assert config.preprocessing.enabled is False
    assert config.preprocessing.target_width == 1280
    assert config.preprocessing.target_height == 720

    assert config.scene_annotation.model_name == "blip2"
    assert config.scene_annotation.sample_rate == 15

    assert config.test_mode.enabled is True
    assert config.test_mode.max_frames == 300
