"""Configuration and materialization for the AWS-specific background worker."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from fourda_pipeline.config import FourDAnyoneConfig


VALID_SHUTDOWN_POLICIES = frozenset({"never", "success", "always"})
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    version = document.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"config schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
    return document


@dataclass(frozen=True, slots=True)
class SnsConfig:
    topic_name: str
    email: str


@dataclass(frozen=True, slots=True)
class SageMakerAppConfig:
    domain_id: str
    space_name: str
    app_name: str = "default"


@dataclass(frozen=True, slots=True)
class LocalWorkspaceConfig:
    data_root: Path
    fourdanyone_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "fourdanyone_root", Path(self.fourdanyone_root))
        for name, path in (
            ("data_root", self.data_root),
            ("fourdanyone_root", self.fourdanyone_root),
        ):
            if not path.is_absolute():
                raise ValueError(f"aws_worker.local.{name} must be absolute: {path}")

    @property
    def input_dir(self) -> Path:
        return self.data_root / "input"

    @property
    def model_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"


@dataclass(frozen=True, slots=True)
class AwsWorkerConfig:
    job_id: str
    shutdown_on: str
    region: str
    bucket: str
    s3_video_path: str
    input_prefix: str
    runs_prefix: str
    models_prefix: str
    sync_models: bool
    upload_results: bool
    local: LocalWorkspaceConfig
    sns: SnsConfig
    sagemaker: SageMakerAppConfig

    def __post_init__(self) -> None:
        if self.shutdown_on not in VALID_SHUTDOWN_POLICIES:
            raise ValueError(f"invalid shutdown policy: {self.shutdown_on}")
        if not self.job_id or any(part in self.job_id for part in ("/", "\\", "..")):
            raise ValueError("job_id must be a simple directory name")
        source_bucket, source_key = self.video_s3_location
        if source_bucket != self.bucket:
            raise ValueError(
                f"s3_video_path bucket {source_bucket!r} must match aws_worker.bucket {self.bucket!r}"
            )
        if PurePosixPath(source_key).name in {"", ".", ".."}:
            raise ValueError(f"s3_video_path must identify an object: {self.s3_video_path}")

    @property
    def jobs_dir(self) -> Path:
        return self.local.jobs_dir

    @property
    def video_s3_location(self) -> tuple[str, str]:
        if self.s3_video_path.startswith("s3://"):
            parsed = urlparse(self.s3_video_path)
            if not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError(f"invalid s3_video_path: {self.s3_video_path}")
            return parsed.netloc, parsed.path.lstrip("/")
        key = self.s3_video_path.strip("/")
        if "/" not in key and self.input_prefix:
            key = f"{self.input_prefix}/{key}"
        return self.bucket, key

    @property
    def video_s3_uri(self) -> str:
        bucket, key = self.video_s3_location
        return f"s3://{bucket}/{key}"

    @property
    def local_video_path(self) -> Path:
        _, key = self.video_s3_location
        return self.local.input_dir / PurePosixPath(key).name

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AwsWorkerConfig":
        sns_payload = payload.get("sns") or {
            "topic_name": payload["sns_topic_name"],
            "email": payload["notification_email"],
        }
        sagemaker_payload = payload.get("sagemaker") or {
            "domain_id": payload["sagemaker_domain_id"],
            "space_name": payload["sagemaker_space_name"],
            "app_name": payload.get("sagemaker_app_name", "default"),
        }
        return cls(
            job_id=str(payload["job_id"]),
            shutdown_on=str(payload.get("shutdown_on", "never")),
            region=str(payload["region"]),
            bucket=str(payload["bucket"]),
            s3_video_path=str(payload["s3_video_path"]),
            input_prefix=str(payload.get("input_prefix", "input")).strip("/"),
            runs_prefix=str(payload.get("runs_prefix", "runs")).strip("/"),
            models_prefix=str(payload.get("models_prefix", "models")).strip("/"),
            sync_models=bool(payload.get("sync_models", True)),
            upload_results=bool(payload.get("upload_results", True)),
            local=LocalWorkspaceConfig(**payload["local"]),
            sns=SnsConfig(**sns_payload),
            sagemaker=SageMakerAppConfig(**sagemaker_payload),
        )

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "AwsWorkerConfig":
        try:
            payload = dict(document["aws_worker"])
        except KeyError as error:
            raise ValueError("config must contain an aws_worker object") from error
        # Migrate the first run-document shape: its local paths lived inside
        # pipeline and its input video was uploaded rather than downloaded.
        if "local" not in payload:
            pipeline = document.get("pipeline", {})
            jobs_dir = Path(payload["jobs_dir"])
            payload["local"] = {
                "data_root": str(jobs_dir.parent),
                "fourdanyone_root": pipeline["fourdanyone_root"],
            }
        if "s3_video_path" not in payload:
            pipeline = document.get("pipeline", {})
            filename = Path(pipeline["video_path"]).name
            prefix = str(payload.get("input_prefix", "input")).strip("/")
            key = "/".join(part for part in (prefix, filename) if part)
            payload["s3_video_path"] = f"s3://{payload['bucket']}/{key}"
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local"]["data_root"] = str(self.local.data_root)
        payload["local"]["fourdanyone_root"] = str(self.local.fourdanyone_root)
        return payload


def materialize_pipeline_config(
    pipeline_payload: dict[str, Any], worker: AwsWorkerConfig
) -> FourDAnyoneConfig:
    """Complete an AWS run's path-free pipeline section with staged local paths."""
    payload = dict(pipeline_payload)
    if "frame" in payload:
        if "frame_indices" in payload:
            raise ValueError("pipeline must use either frame or frame_indices, not both")
        payload["frame_indices"] = [payload.pop("frame")]
    if "turbo" in payload:
        if "enable_turbo" in payload:
            raise ValueError("pipeline must use either turbo or enable_turbo, not both")
        payload["enable_turbo"] = payload.pop("turbo")
    payload.update(
        video_path=worker.local_video_path,
        fourdanyone_root=worker.local.fourdanyone_root,
        model_dir=worker.local.model_dir,
        runs_dir=worker.local.runs_dir,
    )
    return FourDAnyoneConfig.from_dict(payload)


def load_aws_worker_config(path: Path) -> tuple[FourDAnyoneConfig, AwsWorkerConfig]:
    document = load_document(path)
    worker = AwsWorkerConfig.from_document(document)
    try:
        pipeline_payload = document["pipeline"]
    except KeyError as error:
        raise ValueError("config must contain a pipeline object") from error
    return materialize_pipeline_config(pipeline_payload, worker), worker
