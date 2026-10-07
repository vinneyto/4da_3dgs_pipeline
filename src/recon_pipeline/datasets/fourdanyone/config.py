"""Configuration for a reproducible 4DAnyone run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from recon_pipeline.environment import PipelineEnvironment
from recon_pipeline.run_document import (
    SUPPORTED_SCHEMA_VERSIONS,
    validate_environment_free,
)
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.artifacts.rerun.config import RerunConfig
from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import (
    SplatfactoConfig,
    SplatfactoRerunConfig,
)

DEFAULT_LAYER_PITCHES = (-15, 0, 15)
FOURDANYONE_DATASET_TYPE = "4danyone"


def _migrate_legacy_nerfstudio_artifact(
    values: dict[str, Any],
    payload: Any = None,
    *,
    default_enabled: bool = False,
) -> Any:
    """Move pre-v6 export settings out of the 4DAnyone stage."""
    legacy: dict[str, Any] = {}
    if "frame" in values:
        if "frame_indices" in values:
            raise ValueError(
                "pipeline must use either frame or frame_indices, not both"
            )
        legacy["frames"] = [values.pop("frame")]
    elif "frame_indices" in values:
        legacy["frames"] = values.pop("frame_indices")
    if "export_device" in values:
        legacy["device"] = values.pop("export_device")
    if payload is not None and legacy:
        raise ValueError(
            "configure Nerfstudio export in artifacts.dataset.nerfstudio, "
            "not pipeline.dataset.config"
        )
    if payload is not None:
        return payload
    return {"enabled": bool(legacy) or default_enabled, **legacy}


def extract_4danyone_dataset_config(
    pipeline: dict[str, Any],
    *,
    experiment_name: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the concrete dataset configuration from a pipeline document.

    Schema v6 models the pipeline as typed stages and artifacts as independent
    root tasks. Earlier schemas stored both directly in the pipeline object.
    """
    if "dataset" not in pipeline:
        values = dict(pipeline)
        # Flat pre-v4 configurations always ran the static export.
        values["nerfstudio"] = _migrate_legacy_nerfstudio_artifact(
            values,
            default_enabled=True,
        )
        return values

    stage = pipeline["dataset"]
    if not isinstance(stage, dict):
        raise TypeError("pipeline.dataset must be an object")
    dataset_enabled = stage.get("enabled", True)
    if not isinstance(dataset_enabled, bool):
        raise TypeError("pipeline.dataset.enabled must be a boolean")
    stage_type = stage.get("type")
    if dataset_enabled and stage_type != FOURDANYONE_DATASET_TYPE:
        raise ValueError(
            f"unsupported pipeline.dataset.type {stage_type!r}; "
            f"expected {FOURDANYONE_DATASET_TYPE!r}"
        )
    config = stage.get("config", {})
    stage_artifacts = stage.get("artifacts", {})
    if not isinstance(config, dict):
        raise TypeError("pipeline.dataset.config must be an object")
    if not isinstance(stage_artifacts, dict):
        raise TypeError("pipeline.dataset.artifacts must be an object")

    values = dict(config)
    experiment_name = experiment_name or pipeline.get("experiment_name")
    if experiment_name is not None:
        configured_name = values.get("experiment_name")
        if configured_name is not None and configured_name != experiment_name:
            raise ValueError(
                "pipeline.experiment_name and "
                "pipeline.dataset.config.experiment_name must match"
            )
        values["experiment_name"] = experiment_name
    root_artifacts = artifacts or {}
    if not isinstance(root_artifacts, dict):
        raise TypeError("artifacts must be an object")
    dataset_artifacts = root_artifacts.get("dataset", {})
    if not isinstance(dataset_artifacts, dict):
        raise TypeError("artifacts.dataset must be an object")
    rerun_payload = dataset_artifacts.get("rerun", stage_artifacts.get("rerun"))
    nerfstudio_payload = _migrate_legacy_nerfstudio_artifact(
        values,
        dataset_artifacts.get("nerfstudio"),
    )
    if "rerun" in values and rerun_payload is not None:
        raise ValueError("configure Rerun in root artifacts, not dataset.config")
    values["dataset_enabled"] = dataset_enabled
    values["nerfstudio"] = nerfstudio_payload
    values["rerun"] = rerun_payload

    reconstruction = pipeline.get("reconstruction")
    if reconstruction is None:
        reconstruction = {}
    if not isinstance(reconstruction, dict):
        raise TypeError("pipeline.reconstruction must be an object")
    enabled = reconstruction.get("enabled", bool(reconstruction))
    if not isinstance(enabled, bool):
        raise TypeError("pipeline.reconstruction.enabled must be a boolean")
    if enabled and reconstruction.get("type") != "nerfstudio_splatfacto":
        raise ValueError("unsupported pipeline.reconstruction.type")
    settings = reconstruction.get("config", {})
    if not isinstance(settings, dict):
        raise TypeError("pipeline.reconstruction.config must be an object")
    values["reconstruction"] = {**(settings if enabled else {}), "enabled": enabled}
    reconstruction_artifacts = root_artifacts.get("reconstruction", {})
    if not isinstance(reconstruction_artifacts, dict):
        raise TypeError("artifacts.reconstruction must be an object")
    values["reconstruction_rerun"] = reconstruction_artifacts.get("rerun")
    if (
        SplatfactoRerunConfig.from_dict(values["reconstruction_rerun"]).enabled
        and not enabled
    ):
        raise ValueError(
            "artifacts.reconstruction.rerun requires an enabled reconstruction stage"
        )
    rerun_enabled = RerunConfig.from_dict(rerun_payload).enabled
    nerfstudio_enabled = NerfstudioArtifactConfig.from_dict(nerfstudio_payload).enabled
    if (
        not dataset_enabled
        and not nerfstudio_enabled
        and not rerun_enabled
        and not enabled
    ):
        raise ValueError("no pipeline stage or artifact is enabled")
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
    if document["schema_version"] >= 7:
        validate_environment_free(document)
    values = extract_4danyone_dataset_config(
        payload,
        experiment_name=document.get("experiment_name"),
        artifacts=document.get("artifacts"),
    )
    environment = PipelineEnvironment.from_environ()
    video = values.get("video")
    if video is None and values.get("video_path"):
        video = Path(values["video_path"]).name
    if not video or Path(video).is_absolute() or ".." in Path(video).parts:
        raise ValueError(
            "local dataset.config.video must be relative to RECON_DATA_ROOT/input"
        )
    if "turbo" in values:
        values["enable_turbo"] = values.pop("turbo")
    return FourDAnyoneConfig.from_dict(
        environment.materialize(
            values,
            environment.data_root / "input" / video,
        )
    )


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
    resume: bool = False
    dataset_enabled: bool = True
    python: str | None = None
    nerfstudio: NerfstudioArtifactConfig = NerfstudioArtifactConfig()
    rerun: RerunConfig = RerunConfig()
    reconstruction: SplatfactoConfig = SplatfactoConfig()
    reconstruction_rerun: SplatfactoRerunConfig = SplatfactoRerunConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(self, "fourdanyone_root", Path(self.fourdanyone_root))
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "runs_dir", Path(self.runs_dir))
        object.__setattr__(
            self, "layer_pitches", tuple(int(value) for value in self.layer_pitches)
        )
        if isinstance(self.nerfstudio, (dict, bool)):
            object.__setattr__(
                self,
                "nerfstudio",
                NerfstudioArtifactConfig.from_dict(self.nerfstudio),
            )
        if isinstance(self.rerun, dict):
            object.__setattr__(self, "rerun", RerunConfig.from_dict(self.rerun))
        if isinstance(self.reconstruction, dict):
            object.__setattr__(
                self, "reconstruction", SplatfactoConfig.from_dict(self.reconstruction)
            )
        if isinstance(self.reconstruction_rerun, (dict, bool)):
            object.__setattr__(
                self,
                "reconstruction_rerun",
                SplatfactoRerunConfig.from_dict(self.reconstruction_rerun),
            )
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
    def reconstruction_dir(self) -> Path:
        return self.experiment_dir / "splatfacto"

    @property
    def reconstruction_frames(self) -> tuple[int, ...]:
        return self.reconstruction.frames or self.nerfstudio.frames

    @property
    def nerfstudio_source_experiment_name(self) -> str:
        return self.nerfstudio.source_experiment_name or self.experiment_name

    @property
    def nerfstudio_source_experiment_dir(self) -> Path:
        return self.runs_dir / self.nerfstudio_source_experiment_name

    @property
    def nerfstudio_generation_dir(self) -> Path:
        return self.nerfstudio_source_experiment_dir / "4danyone"

    @property
    def rerun_dir(self) -> Path:
        return self.experiment_dir / "rerun"

    @property
    def rerun_path(self) -> Path:
        return self.rerun_dir / f"{self.experiment_name}.rrd"

    @property
    def rerun_source_experiment_name(self) -> str:
        return self.rerun.source_experiment_name or self.experiment_name

    @property
    def rerun_source_experiment_dir(self) -> Path:
        return self.runs_dir / self.rerun_source_experiment_name

    @property
    def rerun_generation_dir(self) -> Path:
        return self.rerun_source_experiment_dir / "4danyone"

    def dataset_dir(self, frame_index: int) -> Path:
        return self.datasets_dir / f"frame_{frame_index:03d}"

    def validate_values(self) -> None:
        paths = {
            "video_path": self.video_path,
            "fourdanyone_root": self.fourdanyone_root,
            "model_dir": self.model_dir,
            "runs_dir": self.runs_dir,
        }
        relative = [
            f"{name}: {path}" for name, path in paths.items() if not path.is_absolute()
        ]
        if relative:
            raise ValueError(
                "all paths must be absolute:\n"
                + "\n".join(f" - {item}" for item in relative)
            )
        if not self.experiment_name or any(
            part in self.experiment_name for part in ("/", "\\", "..")
        ):
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
        if self.reconstruction.enabled:
            if not self.nerfstudio.enabled:
                raise ValueError(
                    "Splatfacto reconstruction requires artifacts.dataset.nerfstudio"
                )
            if not set(self.reconstruction_frames) <= set(self.nerfstudio.frames):
                raise ValueError(
                    "reconstruction.frames must be exported Nerfstudio frames"
                )
        if self.reconstruction_rerun.enabled:
            if not self.reconstruction.enabled:
                raise ValueError(
                    "artifacts.reconstruction.rerun requires reconstruction"
                )
            source = self.reconstruction_rerun.source_experiment_name
            if source is not None and source != self.experiment_name:
                raise ValueError(
                    "reconstruction.rerun source must be the current experiment"
                )

    def validate_paths(self) -> None:
        required = [
            (
                self.fourdanyone_root / "third_party/GVHMR/hmr4d/__init__.py",
                "GVHMR submodule",
            ),
        ]
        if self.dataset_enabled:
            required.extend(
                [
                    (self.video_path, "input video"),
                    (self.fourdanyone_root / "inference.py", "4DAnyone inference.py"),
                ]
            )
        if self.nerfstudio.enabled:
            required.append(
                (
                    self.fourdanyone_root / "scripts/export_nerfstudio.py",
                    "Nerfstudio exporter",
                )
            )
            if not (
                self.dataset_enabled
                and self.nerfstudio_source_experiment_name == self.experiment_name
            ):
                required.extend(
                    [
                        (
                            self.nerfstudio_generation_dir / "metadata.json",
                            "Nerfstudio source metadata",
                        ),
                        (
                            self.nerfstudio_generation_dir / "cameras.json",
                            "Nerfstudio source cameras",
                        ),
                    ]
                )
        if self.rerun.enabled:
            if not (
                self.dataset_enabled
                and self.rerun_source_experiment_name == self.experiment_name
            ):
                required.extend(
                    [
                        (
                            self.rerun_generation_dir / "metadata.json",
                            "Rerun source metadata",
                        ),
                        (
                            self.rerun_generation_dir / "cameras.json",
                            "Rerun source cameras",
                        ),
                    ]
                )
        missing = [f"{label}: {path}" for path, label in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required paths:\n"
                + "\n".join(f" - {item}" for item in missing)
            )
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("video_path", "fourdanyone_root", "model_dir", "runs_dir"):
            payload[key] = str(payload[key])
        payload["layer_pitches"] = list(self.layer_pitches)
        return payload

    def settings_dict(self) -> dict[str, Any]:
        """Portable settings saved alongside the run, without execution paths."""
        payload = self.to_dict()
        for name in (
            "video_path",
            "fourdanyone_root",
            "model_dir",
            "runs_dir",
            "python",
        ):
            payload.pop(name)
        payload["video"] = self.video_path.name
        payload["reconstruction"].pop("nerfstudio_bin")
        payload["reconstruction_rerun"].pop("python")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FourDAnyoneConfig":
        values = dict(payload)
        values["nerfstudio"] = NerfstudioArtifactConfig.from_dict(
            values.get("nerfstudio")
        )
        values["rerun"] = RerunConfig.from_dict(values.get("rerun"))
        values["reconstruction"] = SplatfactoConfig.from_dict(
            values.get("reconstruction")
        )
        values["reconstruction_rerun"] = SplatfactoRerunConfig.from_dict(
            values.get("reconstruction_rerun")
        )
        return cls(**values)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(path)
