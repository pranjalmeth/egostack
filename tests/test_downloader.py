"""Tests for the downloader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ego_hand_pipeline.downloader import (
    _sanitize_filename,
    download_video,
    load_urls_from_file,
)


def test_sanitize_filename():
    assert _sanitize_filename('Hello: World?') == "Hello__World"
    assert _sanitize_filename("  spaces  ") == "spaces"
    assert _sanitize_filename("a/b\\c") == "a_b_c"


def test_load_urls_from_file(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://youtube.com/watch?v=abc\n\n# comment\nhttps://youtube.com/watch?v=def\n")
    urls = load_urls_from_file(url_file)
    assert urls == [
        "https://youtube.com/watch?v=abc",
        "https://youtube.com/watch?v=def",
    ]


def test_load_urls_from_file_empty(tmp_path):
    url_file = tmp_path / "empty.txt"
    url_file.write_text("\n\n# only comments\n")
    assert load_urls_from_file(url_file) == []


@patch("ego_hand_pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_video(mock_ydl_class, tmp_path):
    # Create a fake downloaded file
    fake_video = tmp_path / "raw" / "test_video.mp4"
    fake_video.parent.mkdir(parents=True, exist_ok=True)
    fake_video.write_bytes(b"fake video content")

    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

    # Simulate the postprocessor hook being called
    def fake_download(urls):
        opts = mock_ydl_class.call_args[0][0]
        for hook in opts.get("postprocessor_hooks", []):
            hook({"status": "finished", "info_dict": {"filepath": str(fake_video)}})

    mock_ydl.download.side_effect = fake_download

    result = download_video(
        "https://youtube.com/watch?v=test",
        output_dir=tmp_path / "raw",
    )
    assert result == fake_video
