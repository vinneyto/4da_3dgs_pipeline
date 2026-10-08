"""Portable pipeline structure; no machine paths and no free-form settings."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

from recon_pipeline.config_types import object_value, parse_settings, checked_value
from recon_pipeline.artifacts.rerun.config import RerunConfig
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import (
    SplatfactoConfig,
    SplatfactoRerunConfig,
)
from recon_pipeline.postprocessing.config import (
    PostprocessingConfig,
    SplatConversionConfig,
)


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    video: str | None = None
    experiment_name: str | None = None
    views_per_layer: int = 24
    layer_pitches: tuple[int, ...] = (-15, 0, 15)
    start_yaw: int = 0
    yaw_span: int = 360
    target_fps: float = 30.0
    seed: int = 42
    enable_turbo: bool = True
    attention_backend: str = "auto"
    resume: bool = False

    @classmethod
    def from_dict(cls, payload: Any) -> DatasetSettings:
        values = dict(object_value(payload, "pipeline.dataset.config"))
        legacy = values.keys() & {"frame", "frame_indices", "export_device"}
        if legacy:
            raise ValueError(
                f"Unsupported legacy export fields {sorted(legacy)}; configure artifacts.dataset.nerfstudio (frames, device, enabled)"
            )
        if "turbo" in values:
            if "enable_turbo" in values:
                raise ValueError(
                    "pipeline must use either turbo or enable_turbo, not both"
                )
            values["enable_turbo"] = values.pop("turbo")
        return parse_settings(cls, values, "pipeline.dataset.config")


@dataclass(frozen=True, slots=True)
class DatasetStage:
    config: DatasetSettings = DatasetSettings()
    enabled: bool = True
    type: str | None = None
    # Retained pre-v6 Rerun compatibility; Nerfstudio is root-only.
    rerun: RerunConfig | None = None


@dataclass(frozen=True, slots=True)
class ReconstructionStage:
    config: SplatfactoConfig = SplatfactoConfig()
    enabled: bool = False
    type: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    dataset: DatasetStage
    reconstruction: ReconstructionStage = ReconstructionStage()
    experiment_name: str | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> PipelineSettings:
        values = object_value(payload, "pipeline")
        if "dataset" not in values:
            # Retain flat settings, but never implicitly enable static export.
            return cls(
                DatasetStage(config=DatasetSettings.from_dict(values), type="4danyone")
            )
        stage = object_value(values["dataset"], "pipeline.dataset")
        enabled = checked_value(
            stage.get("enabled", True), bool, "pipeline.dataset.enabled"
        )
        stage_type = checked_value(
            stage.get("type"), str | None, "pipeline.dataset.type"
        )
        if enabled and stage_type != "4danyone":
            raise ValueError(
                f"unsupported pipeline.dataset.type {stage_type!r}; expected 4danyone"
            )
        artifacts = object_value(
            stage.get("artifacts", {}), "pipeline.dataset.artifacts"
        )
        dataset = DatasetStage(
            DatasetSettings.from_dict(stage.get("config", {})),
            enabled,
            stage_type,
            (
                artifact_settings(
                    RerunConfig, artifacts["rerun"], "pipeline.dataset.artifacts.rerun"
                )
                if "rerun" in artifacts
                else None
            ),
        )
        recon = object_value(
            {} if values.get("reconstruction") is None else values["reconstruction"],
            "pipeline.reconstruction",
        )
        active = checked_value(
            recon.get("enabled", bool(recon)), bool, "pipeline.reconstruction.enabled"
        )
        recon_type = checked_value(
            recon.get("type"), str | None, "pipeline.reconstruction.type"
        )
        if active and recon_type != "nerfstudio_splatfacto":
            raise ValueError("unsupported pipeline.reconstruction.type")
        settings = object_value(
            recon.get("config", {}), "pipeline.reconstruction.config"
        )
        # Disabled stages intentionally tolerate settings from future versions.
        config = parse_settings(
            SplatfactoConfig,
            {
                **(
                    settings
                    if active
                    else {
                        k: v
                        for k, v in settings.items()
                        if k in {f.name for f in fields(SplatfactoConfig)}
                    }
                ),
                "enabled": active,
            },
            "pipeline.reconstruction.config",
        )
        name = checked_value(
            values.get("experiment_name"), str | None, "pipeline.experiment_name"
        )
        return cls(dataset, ReconstructionStage(config, active, recon_type), name)


ArtifactConfig = TypeVar(
    "ArtifactConfig", RerunConfig, NerfstudioArtifactConfig, SplatfactoRerunConfig
)


def artifact_settings(
    cls: type[ArtifactConfig], payload: Any, location: str
) -> ArtifactConfig:
    if payload is None:
        return cls(enabled=False)
    if isinstance(payload, bool):
        return cls(enabled=payload)
    values = dict(object_value(payload, location))
    # Existing artifact-level alias is independent of the removed migration.
    if cls is NerfstudioArtifactConfig and "frame_indices" in values:
        if "frames" in values:
            raise ValueError(
                "nerfstudio must use either frames or legacy frame_indices, not both"
            )
        values["frames"] = values.pop("frame_indices")
    return parse_settings(cls, values, location)


@dataclass(frozen=True, slots=True)
class DatasetArtifacts:
    nerfstudio: NerfstudioArtifactConfig = NerfstudioArtifactConfig(enabled=False)
    rerun: RerunConfig | None = None


@dataclass(frozen=True, slots=True)
class ReconstructionArtifacts:
    rerun: SplatfactoRerunConfig = SplatfactoRerunConfig()


@dataclass(frozen=True, slots=True)
class ArtifactSettings:
    dataset: DatasetArtifacts = DatasetArtifacts()
    reconstruction: ReconstructionArtifacts = ReconstructionArtifacts()

    @classmethod
    def from_dict(cls, payload: Any) -> ArtifactSettings:
        root = object_value({} if payload is None else payload, "artifacts")
        dataset = object_value(root.get("dataset", {}), "artifacts.dataset")
        recon = object_value(root.get("reconstruction", {}), "artifacts.reconstruction")
        return cls(
            DatasetArtifacts(
                artifact_settings(
                    NerfstudioArtifactConfig,
                    dataset.get("nerfstudio"),
                    "artifacts.dataset.nerfstudio",
                ),
                (
                    artifact_settings(
                        RerunConfig, dataset["rerun"], "artifacts.dataset.rerun"
                    )
                    if "rerun" in dataset
                    else None
                ),
            ),
            ReconstructionArtifacts(
                artifact_settings(
                    SplatfactoRerunConfig,
                    recon.get("rerun"),
                    "artifacts.reconstruction.rerun",
                )
            ),
        )


def parse_postprocessing(payload: Any) -> PostprocessingConfig:
    values = object_value({} if payload is None else payload, "postprocessing")
    if values.keys() - {"splat_conversion"}:
        raise ValueError("postprocessing contains unsupported fields")
    return PostprocessingConfig(
        parse_settings(
            SplatConversionConfig,
            (
                {}
                if values.get("splat_conversion") is None
                else values["splat_conversion"]
            ),
            "postprocessing.splat_conversion",
        )
    )
