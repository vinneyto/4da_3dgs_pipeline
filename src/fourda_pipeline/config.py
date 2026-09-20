"""Configuration for a reproducible 4DAnyone run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fourda_rerun.config import RerunConfig


DEFAULT_LAYER_PITCHES = (-15, 0, 15)
DEFAULT_FRAME_INDICES = (60,)
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
FOURDANYONE_DATASET_TYPE = "4danyone"


def extract_4danyone_dataset_config(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Return the concrete dataset configuration from a pipeline document.

    Schema v4 models the pipeline as typed, optional stages. Earlier schemas
    stored the 4DAnyone settings directly in the pipeline object.
    """
    if "dataset" not in pipeline:
        return dict(pipeline)

    stage = pipeline["dataset"]
    if not isinstance(stage, dict):
        raise TypeError("pipeline.dataset must be an object")
    dataset_enabled = stage.get("enabled", True)
    if not isinstance(dataset_enabled, bool):
        raise TypeError("pipeline.dataset.enabled must be a boolean")
    if not dataset_enabled:
        raise ValueError(
            "pipeline.dataset is disabled and no executable reconstruction stage exists yet"
        )
    stage_type = stage.get("type")
    if stage_type != FOURDANYONE_DATASET_TYPE:
        raise ValueError(
            f"unsupported pipeline.dataset.type {stage_type!r}; "
            f"expected {FOURDANYONE_DATASET_TYPE!r}"
        )
    config = stage.get("config", {})
    artifacts = stage.get("artifacts", {})
    if not isinstance(config, dict):
        raise TypeError("pipeline.dataset.config must be an object")
    if not isinstance(artifacts, dict):
        raise TypeError("pipeline.dataset.artifacts must be an object")

    values = dict(config)
    experiment_name = pipeline.get("experiment_name")
    if experiment_name is not None:
        configured_name = values.get("experiment_name")
        if configured_name is not None and configured_name != experiment_name:
            raise ValueError(
                "pipeline.experiment_name and "
                "pipeline.dataset.config.experiment_name must match"
            )
        values["experiment_name"] = experiment_name
    if "rerun" in values and "rerun" in artifacts:
        raise ValueError(
            "configure Rerun in pipeline.dataset.artifacts, not dataset.config"
        )
    values["rerun"] = artifacts.get("rerun")

    reconstruction = pipeline.get("reconstruction")
    if reconstruction is not None:
        if not isinstance(reconstruction, dict):
            raise TypeError("pipeline.reconstruction must be an object")
        reconstruction_enabled = reconstruction.get("enabled", True)
        if not isinstance(reconstruction_enabled, bool):
            raise TypeError("pipeline.reconstruction.enabled must be a boolean")
        if reconstruction_enabled:
            raise ValueError(
                "pipeline.reconstruction is enabled, but 3DGS reconstruction "
                "is not implemented yet"
            )
    return values


def load_pipeline_config(path: Path) -> "FourDAnyoneConfig":
    document = json.loads(path.read_text())
    if document.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"config schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    try:
        payload = document["pipeline"]
    except KeyError as error:
        raise ValueError("config must contain a pipeline object") from error
    return FourDAnyoneConfig.from_dict(extract_4danyone_dataset_config(payload))

@dataclass(frozen=True, slots=True)
class FourDAnyoneConfig:
    """One 4DAnyone inference followed by one or more static 3DGS exports."""

    video_path: Path
    experiment_name: str
    fourdanyone_root: Path
    model_dir: Path
    runs_dir: Path
    views_per_layer: int = 24
    layer_pitches: tuple[int, ...] = DEFAULT_LAYER_PITCHES
    start_yaw: int = 0
    yaw_span: int = 360
    target_fps: float = 30.0
    seed: int = 42
    enable_turbo: bool = True
    attention_backend: str = "auto"
    frame_indices: tuple[int, ...] = DEFAULT_FRAME_INDICES
    export_device: str = "cuda:0"
    resume: bool = False
    rerun: RerunConfig = RerunConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(self, "fourdanyone_root", Path(self.fourdanyone_root))
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "runs_dir", Path(self.runs_dir))
        object.__setattr__(self, "layer_pitches", tuple(int(value) for value in self.layer_pitches))
        object.__setattr__(self, "frame_indices", tuple(int(value) for value in self.frame_indices))
        if isinstance(self.rerun, dict):
            object.__setattr__(self, "rerun", RerunConfig.from_dict(self.rerun))
        self.validate_values()

    @property
    def num_views(self) -> int:
        return self.views_per_layer * len(self.layer_pitches)

    @property
    def experiment_dir(self) -> Path:
        return self.runs_dir / self.experiment_name

    @property
    def inference_dir(self) -> Path:
        return self.experiment_dir / "4danyone"

    @property
    def datasets_dir(self) -> Path:
        return self.experiment_dir / "nerfstudio"

    @property
    def rerun_dir(self) -> Path:
        return self.experiment_dir / "rerun"

    @property
    def rerun_path(self) -> Path:
        return self.rerun_dir / f"{self.experiment_name}.rrd"

    def dataset_dir(self, frame_index: int) -> Path:
        return self.datasets_dir / f"frame_{frame_index:03d}"

    def validate_values(self) -> None:
        paths = {
            "video_path": self.video_path,
            "fourdanyone_root": self.fourdanyone_root,
            "model_dir": self.model_dir,
            "runs_dir": self.runs_dir,
        }
        relative = [f"{name}: {path}" for name, path in paths.items() if not path.is_absolute()]
        if relative:
            raise ValueError("all paths must be absolute:\n" + "\n".join(f" - {item}" for item in relative))
        if not self.experiment_name or any(part in self.experiment_name for part in ("/", "\\", "..")):
            raise ValueError("experiment_name must be a simple directory name")
        if self.views_per_layer < 1:
            raise ValueError("views_per_layer must be positive")
        if not self.layer_pitches:
            raise ValueError("layer_pitches must not be empty")
        if any(pitch < -15 or pitch > 45 for pitch in self.layer_pitches):
            raise ValueError("every layer pitch must be between -15 and 45 degrees")
        if self.num_views % 6:
            raise ValueError("total target views must be divisible by 6")
        if not 1 <= self.yaw_span <= 360:
            raise ValueError("yaw_span must be between 1 and 360")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not self.frame_indices:
            raise ValueError("at least one frame index is required")
        if any(index < 0 or index > 120 for index in self.frame_indices):
            raise ValueError("frame indices must be in the inclusive range 0..120")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("frame indices must be unique")

    def validate_paths(self) -> None:
        required = (
            (self.video_path, "input video"),
            (self.fourdanyone_root / "inference.py", "4DAnyone inference.py"),
            (self.fourdanyone_root / "scripts/export_nerfstudio.py", "Nerfstudio exporter"),
            (self.fourdanyone_root / "third_party/GVHMR/hmr4d/__init__.py", "GVHMR submodule"),
        )
        missing = [f"{label}: {path}" for path, label in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required paths:\n" + "\n".join(f" - {item}" for item in missing))
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("video_path", "fourdanyone_root", "model_dir", "runs_dir"):
            payload[key] = str(payload[key])
        payload["layer_pitches"] = list(self.layer_pitches)
        payload["frame_indices"] = list(self.frame_indices)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FourDAnyoneConfig":
        values = dict(payload)
        values["rerun"] = RerunConfig.from_dict(values.get("rerun"))
        return cls(**values)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
