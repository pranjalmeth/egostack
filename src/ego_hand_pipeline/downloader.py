"""YouTube video downloading via yt-dlp."""

from __future__ import annotations

import re
from pathlib import Path

import yt_dlp


def _sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe for filenames."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")[:200]


def download_video(
    url: str,
    output_dir: str | Path = "data/raw",
    resolution: int = 720,
    format: str = "mp4",
) -> Path:
    """Download a single YouTube video and return the path to the saved file.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the video.
        resolution: Maximum video height (e.g. 720).
        format: Container format.

    Returns:
        Path to the downloaded video file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture the final filename after merging via postprocessor hook
    final_path: list[Path] = []

    def _postprocessor_hook(d: dict) -> None:
        if d.get("status") == "finished":
            filepath = d.get("info_dict", {}).get("filepath")
            if filepath:
                final_path.append(Path(filepath))

    ydl_opts = {
        "format": f"bestvideo[height<={resolution}][ext={format}][vcodec!~='av0?1']+bestaudio[ext=m4a]/best[height<={resolution}][ext={format}][vcodec!~='av0?1']/best[height<={resolution}]",
        "merge_output_format": format,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "restrictfilenames": True,
        "postprocessor_hooks": [_postprocessor_hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if final_path:
        return final_path[-1]

    # Fallback: find the most recently modified file in output_dir
    files = sorted(output_dir.glob(f"*.{format}"), key=lambda p: p.stat().st_mtime)
    if files:
        return files[-1]

    raise RuntimeError(f"Download succeeded but could not locate file for {url}")


def download_videos(
    urls: list[str],
    output_dir: str | Path = "data/raw",
    resolution: int = 720,
    format: str = "mp4",
) -> list[Path]:
    """Download multiple videos, returning paths to all downloaded files."""
    paths = []
    for url in urls:
        path = download_video(url, output_dir, resolution, format)
        paths.append(path)
    return paths


def load_urls_from_file(path: str | Path) -> list[str]:
    """Read URLs from a text file (one per line, ignoring blanks and comments)."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
