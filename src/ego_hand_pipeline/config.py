"""Configuration loading from YAML with dataclass validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DownloadConfig:
    resolution: int = 720
    format: str = "mp4"
    output_dir: str = "data/raw"


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.7
    model_complexity: int = 1


@dataclass
class ClippingConfig:
    min_duration: float = 1.0
    merge_gap: float = 0.5
    output_dir: str = "data/clips"


@dataclass
class DepthConfig:
    model: str = "depth_anything_v2"
    sample_rate: int = 30
    output_dir: str = "data/depth"
    save_visualization: bool = True


@dataclass
class TrajectoryConfig:
    smooth: bool = True
    compute_velocity: bool = True
    compute_acceleration: bool = True
    output_dir: str = "data/trajectories"


@dataclass
class ExportConfig:
    formats: list[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class PipelineConfig:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    trajectories: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    base_dir: str = "."

    def resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against the base directory."""
        return Path(self.base_dir) / relative


def _build_section(cls, data: dict | None):
    if data is None:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file with optional overrides."""
    raw: dict = {}
    if path is not None:
        path = Path(path)
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}

    if overrides:
        for key, value in overrides.items():
            if "." in key:
                section, param = key.split(".", 1)
                raw.setdefault(section, {})[param] = value
            else:
                raw[key] = value

    cfg = PipelineConfig(
        download=_build_section(DownloadConfig, raw.get("download")),
        detection=_build_section(DetectionConfig, raw.get("detection")),
        clipping=_build_section(ClippingConfig, raw.get("clipping")),
        depth=_build_section(DepthConfig, raw.get("depth")),
        trajectories=_build_section(TrajectoryConfig, raw.get("trajectories")),
        export=_build_section(ExportConfig, raw.get("export")),
        base_dir=raw.get("base_dir", "."),
    )
    return cfg
