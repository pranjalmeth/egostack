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
class HandPoseConfig:
    """HaMeR 3D hand pose estimation settings."""
    enabled: bool = True
    sample_rate: int = 1
    model_path: str | None = None
    output_dir: str = "data/hand_poses"


@dataclass
class ObjectDetectionConfig:
    """GroundingDINO open-vocabulary object detection settings."""
    enabled: bool = True
    sample_rate: int = 30
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    text_prompts: list[str] | None = None
    model_path: str | None = None
    output_dir: str = "data/objects"


@dataclass
class SegmentationConfig:
    """SAM2 segmentation settings."""
    enabled: bool = True
    sample_rate: int = 1
    model_size: str = "small"  # tiny, small, base_plus, large
    model_path: str | None = None
    output_dir: str = "data/segmentation"
    save_masks: bool = True


@dataclass
class LeRobotConfig:
    """LeRobot dataset export settings."""
    enabled: bool = True
    dataset_name: str = "ego_hand_manipulation"
    task_description: str = "manipulation"
    output_dir: str = "data/lerobot"


@dataclass
class PreprocessingConfig:
    """Video preprocessing / resolution standardization settings."""
    enabled: bool = True
    target_width: int = 640    # 480p — ideal for robotics training data
    target_height: int = 480
    target_fps: int | None = None  # None = keep original fps
    output_dir: str = "data/preprocessed"


@dataclass
class SceneAnnotationConfig:
    """Scene captioning / annotation settings."""
    enabled: bool = True
    sample_rate: int = 30        # annotate every Nth frame (~1/sec at 30fps)
    model_name: str = "rule_based"  # "blip2", "instructblip", or "rule_based"
    output_dir: str = "data/scene_annotations"


@dataclass
class TestModeConfig:
    """Test / benchmark mode settings."""
    enabled: bool = False
    max_frames: int = 900         # ~30 seconds at 30fps
    render_samples: bool = True   # generate annotated sample videos
    sample_output_dir: str = "data/samples"


@dataclass
class PipelineConfig:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    trajectories: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    hand_pose: HandPoseConfig = field(default_factory=HandPoseConfig)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    lerobot: LeRobotConfig = field(default_factory=LeRobotConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    scene_annotation: SceneAnnotationConfig = field(default_factory=SceneAnnotationConfig)
    test_mode: TestModeConfig = field(default_factory=TestModeConfig)
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
        hand_pose=_build_section(HandPoseConfig, raw.get("hand_pose")),
        object_detection=_build_section(ObjectDetectionConfig, raw.get("object_detection")),
        segmentation=_build_section(SegmentationConfig, raw.get("segmentation")),
        lerobot=_build_section(LeRobotConfig, raw.get("lerobot")),
        preprocessing=_build_section(PreprocessingConfig, raw.get("preprocessing")),
        scene_annotation=_build_section(SceneAnnotationConfig, raw.get("scene_annotation")),
        test_mode=_build_section(TestModeConfig, raw.get("test_mode")),
        base_dir=raw.get("base_dir", "."),
    )
    return cfg
