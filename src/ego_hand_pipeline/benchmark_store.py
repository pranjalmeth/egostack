"""Versioned benchmark run storage — directory management, metadata, and comparison."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: str | Path, label: str | None = None) -> Path:
    """Create a timestamped run directory under ``data/benchmark/runs/``.

    Updates the ``latest`` symlink to point to the new directory.
    Returns the created run directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dir_name = f"{timestamp}_{label}" if label else timestamp

    runs_dir = Path(base_dir) / "data" / "benchmark" / "runs"
    run_dir = runs_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update "latest" symlink
    latest = runs_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    return run_dir


def _git_info() -> tuple[str, bool]:
    """Return (short commit hash, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(porcelain)
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False

    return commit, dirty


def save_run_meta(
    run_dir: Path,
    videos: list[Path],
    run_type: str,
    label: str | None = None,
) -> Path:
    """Write ``run_meta.json`` into *run_dir*.

    Returns the path to the written file.
    """
    commit, dirty = _git_info()

    meta = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "git_commit": commit,
        "git_dirty": dirty,
        "videos": [v.name for v in videos],
        "run_type": run_type,
    }

    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def find_previous_run(runs_dir: Path, current_run_name: str) -> Path | None:
    """Find the run directory immediately before *current_run_name*.

    Directories are sorted alphabetically (timestamps sort naturally).
    Returns ``None`` if there is no earlier run.
    """
    if not runs_dir.is_dir():
        return None

    dirs = sorted(
        d.name for d in runs_dir.iterdir()
        if d.is_dir() and d.name != "latest"
    )

    try:
        idx = dirs.index(current_run_name)
    except ValueError:
        return None

    if idx == 0:
        return None

    return runs_dir / dirs[idx - 1]


def compare_benchmark_runs(current_json: Path, previous_json: Path) -> str:
    """Compare two benchmark JSON files and return a formatted diff string.

    Works for both perf benchmark (``benchmark.json``) and quality benchmark
    (``quality_benchmark.json``).
    """
    current = json.loads(current_json.read_text())
    previous = json.loads(previous_json.read_text())

    # Detect which kind of benchmark by filename
    is_enhanced = "enhanced" in current_json.name
    is_quality = "quality" in current_json.name

    if is_enhanced:
        return _compare_enhanced(current, previous, previous_json.parent.name)
    if is_quality:
        return _compare_quality(current, previous, previous_json.parent.name)
    return _compare_perf(current, previous, previous_json.parent.name)


def _compare_perf(current: list, previous: list, prev_name: str) -> str:
    """Compare perf benchmark results (total time per video/resolution)."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append(f"COMPARISON vs previous run ({prev_name})")
    lines.append(dash)
    lines.append(f"  {'Metric':<30} {'Previous':>10} {'Current':>10} {'Δ':>10}")
    lines.append(dash)

    # Build lookup: video_name -> resolution -> total time
    def _totals(data: list) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for entry in data:
            name = entry["video_name"]
            out[name] = {}
            for r in entry["results"]:
                total = sum(r["timings"].values())
                out[name][r["resolution"]] = total
        return out

    prev_totals = _totals(previous)
    curr_totals = _totals(current)

    for video_name, resolutions in curr_totals.items():
        for res, curr_time in resolutions.items():
            prev_time = prev_totals.get(video_name, {}).get(res)
            if prev_time is None:
                continue
            delta = curr_time - prev_time
            # For time, lower is better
            arrow = "↑" if delta < 0 else "↓" if delta > 0 else " "
            metric = f"{video_name}/{res} total_time (s)"
            lines.append(
                f"  {metric:<30} {prev_time:>10.2f} {curr_time:>10.2f} "
                f"{delta:>+9.2f}  {arrow}"
            )

    lines.append(dash)
    return "\n".join(lines)


def _compare_quality(current: list, previous: list, prev_name: str) -> str:
    """Compare quality benchmark results (key metrics)."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append(f"COMPARISON vs previous run ({prev_name})")
    lines.append(dash)
    lines.append(f"  {'Metric':<30} {'Previous':>10} {'Current':>10} {'Δ':>10}")
    lines.append(dash)

    # Key quality metrics to compare (higher is better, except jerk)
    HIGHER_BETTER = {"detection_rate", "confidence_mean", "cross_model_correlation"}
    LOWER_BETTER = {"jerk_mean"}

    for curr_entry, prev_entry in zip(current, previous):
        video = curr_entry.get("video_name", "")

        metrics: list[tuple[str, float, float]] = []

        # Detection
        cd = curr_entry.get("detection", {})
        pd = prev_entry.get("detection", {})
        metrics.append(("detection_rate", pd.get("detection_rate", 0), cd.get("detection_rate", 0)))
        metrics.append(("confidence_mean", pd.get("confidence", {}).get("mean", 0), cd.get("confidence", {}).get("mean", 0)))

        # Depth
        metrics.append((
            "cross_model_correlation",
            prev_entry.get("depth", {}).get("cross_model_correlation", 0),
            curr_entry.get("depth", {}).get("cross_model_correlation", 0),
        ))

        # Trajectory — average jerk across joints
        curr_joints = curr_entry.get("trajectory", {}).get("joints", [])
        prev_joints = prev_entry.get("trajectory", {}).get("joints", [])
        if curr_joints and prev_joints:
            curr_jerk = sum(j.get("jerk_mean", 0) for j in curr_joints) / len(curr_joints)
            prev_jerk = sum(j.get("jerk_mean", 0) for j in prev_joints) / len(prev_joints)
            metrics.append(("jerk_mean", prev_jerk, curr_jerk))

        if video:
            lines.append(f"  [{video}]")

        for name, prev_val, curr_val in metrics:
            delta = curr_val - prev_val
            if name in HIGHER_BETTER:
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else " "
            elif name in LOWER_BETTER:
                arrow = "↑" if delta < 0 else "↓" if delta > 0 else " "
            else:
                arrow = " "
            lines.append(
                f"  {name:<30} {prev_val:>10.4f} {curr_val:>10.4f} "
                f"{delta:>+9.4f}  {arrow}"
            )

    lines.append(dash)
    return "\n".join(lines)


def _compare_enhanced(current: list, previous: list, prev_name: str) -> str:
    """Compare enhanced benchmark results (key metrics across all stages)."""
    lines: list[str] = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append(f"ENHANCED COMPARISON vs previous run ({prev_name})")
    lines.append(dash)
    lines.append(f"  {'Metric':<35} {'Previous':>10} {'Current':>10} {'Δ':>10}")
    lines.append(dash)

    # Higher is better for these, lower for time/jerk
    HIGHER_BETTER = {
        "detection_rate", "pose_detection_rate", "mask_rate",
        "diversity_ratio", "export_success",
    }
    LOWER_BETTER = {"total_time", "jerk_mean"}

    for curr_entry, prev_entry in zip(current, previous):
        video = curr_entry.get("video_name", "")
        if video:
            lines.append(f"  [{video}]")

        metrics: list[tuple[str, float, float]] = []

        # Perf: total time
        curr_time = curr_entry.get("perf", {}).get("total_time", 0)
        prev_time = prev_entry.get("perf", {}).get("total_time", 0)
        metrics.append(("total_time", prev_time, curr_time))

        # Quality stages
        curr_quality = {q["stage"]: q for q in curr_entry.get("quality", [])}
        prev_quality = {q["stage"]: q for q in prev_entry.get("quality", [])}

        # Detection
        cd = curr_quality.get("detect", {}).get("detection", {})
        pd = prev_quality.get("detect", {}).get("detection", {})
        if cd or pd:
            metrics.append(("detection_rate", pd.get("detection_rate", 0), cd.get("detection_rate", 0)))

        # Hand pose
        chp = curr_quality.get("hand_pose", {}).get("hand_pose", {})
        php = prev_quality.get("hand_pose", {}).get("hand_pose", {})
        if chp or php:
            metrics.append(("pose_detection_rate", php.get("pose_detection_rate", 0), chp.get("pose_detection_rate", 0)))

        # Segmentation
        cs = curr_quality.get("segmentation", {}).get("segmentation", {})
        ps = prev_quality.get("segmentation", {}).get("segmentation", {})
        if cs or ps:
            metrics.append(("mask_rate", ps.get("mask_rate", 0), cs.get("mask_rate", 0)))

        # Scene annotation
        csa = curr_quality.get("scene_annotation", {}).get("scene_annotation", {})
        psa = prev_quality.get("scene_annotation", {}).get("scene_annotation", {})
        if csa or psa:
            metrics.append(("diversity_ratio", psa.get("diversity_ratio", 0), csa.get("diversity_ratio", 0)))

        # Trajectories
        ct = curr_quality.get("trajectories", {}).get("trajectories", {})
        pt = prev_quality.get("trajectories", {}).get("trajectories", {})
        if ct or pt:
            metrics.append(("jerk_mean", pt.get("jerk_mean", 0), ct.get("jerk_mean", 0)))

        # LeRobot
        clr = curr_quality.get("lerobot_export", {}).get("lerobot_export", {})
        plr = prev_quality.get("lerobot_export", {}).get("lerobot_export", {})
        if clr or plr:
            metrics.append(("export_success", float(plr.get("export_success", False)), float(clr.get("export_success", False))))

        for name, prev_val, curr_val in metrics:
            delta = curr_val - prev_val
            if name in HIGHER_BETTER:
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else " "
            elif name in LOWER_BETTER:
                arrow = "↑" if delta < 0 else "↓" if delta > 0 else " "
            else:
                arrow = " "

            fmt = ".2f" if name == "total_time" else ".4f"
            lines.append(
                f"  {name:<35} {prev_val:>10{fmt}} {curr_val:>10{fmt}} "
                f"{delta:>+9{fmt}}  {arrow}"
            )

    lines.append(dash)
    return "\n".join(lines)


def list_runs(runs_dir: Path) -> list[dict]:
    """Read ``run_meta.json`` from each run directory.

    Returns a list of metadata dicts sorted by timestamp (oldest first).
    """
    if not runs_dir.is_dir():
        return []

    results: list[dict] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "latest":
            continue
        meta_path = d / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["_dir_name"] = d.name
            results.append(meta)
        else:
            # Partial run — include with minimal info
            results.append({
                "_dir_name": d.name,
                "timestamp": d.name.split("_")[0] if "_" in d.name else d.name,
                "label": None,
                "git_commit": "unknown",
                "run_type": "unknown",
                "videos": [],
            })

    return results
