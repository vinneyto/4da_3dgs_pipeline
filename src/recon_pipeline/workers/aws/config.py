"""Configuration and materialization for the AWS-specific background worker."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .requests import ExperimentRequest
from urllib.parse import urlparse

from recon_pipeline.environment import PipelineEnvironment
from recon_pipeline.run_document import (
    SUPPORTED_SCHEMA_VERSIONS,
    validate_environment_free,
)
from recon_pipeline.datasets.fourdanyone.config import (
    FourDAnyoneConfig,
    extract_4danyone_dataset_config,
)

VALID_SHUTDOWN_POLICIES = frozenset({"never", "success", "failure", "always"})


@dataclass(frozen=True, slots=True)
class CloudWatchConfig:
    log_group: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.log_group, str) or not re.fullmatch(
            r"[.\-_/A-Za-z0-9#]{1,512}", self.log_group
        ):
            raise ValueError(
                "cloudwatch.log_group must be a valid CloudWatch log group name"
            )


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    version = document.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"config schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if version >= 7:
        validate_environment_free(document)
    return document


@dataclass(frozen=True, slots=True)
class SnsConfig:
    topic_name: str
    email: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    chat_id: str
    bot_token: str | None = None
    bot_token_env: str | None = None
    enabled: bool = True
    shutdown_command: bool = True
    allowed_user_id: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not self.chat_id:
            raise ValueError("telegram chat_id must not be empty")
        if self.enabled and bool(self.bot_token) == bool(self.bot_token_env):
            raise ValueError(
                "telegram must define exactly one of bot_token or bot_token_env"
            )
        if self.allowed_user_id is not None and not self.allowed_user_id.strip():
            raise ValueError("telegram allowed_user_id must not be empty")

    @property
    def shutdown_user_id(self) -> str:
        """User allowed to stop the configured App; private chats need no extra ID."""
        return self.allowed_user_id or self.chat_id

    def resolve_bot_token(self) -> str:
        if self.bot_token:
            return self.bot_token
        if not self.bot_token_env:
            raise RuntimeError("Telegram bot token is not configured")
        token = os.environ.get(self.bot_token_env)
        if not token:
            raise RuntimeError(
                f"Telegram bot token environment variable {self.bot_token_env!r} is not set"
            )
        return token


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
class BucketConfig:
    """One S3 namespace used by the AWS worker."""

    name: str
    video: str
    input_prefix: str = "input"
    models_prefix: str = "models"
    runs_prefix: str = "runs"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_prefix", self.input_prefix.strip("/"))
        object.__setattr__(self, "models_prefix", self.models_prefix.strip("/"))
        object.__setattr__(self, "runs_prefix", self.runs_prefix.strip("/"))
        video = self.video.strip("/")
        object.__setattr__(self, "video", video)
        if not self.name:
            raise ValueError("aws_worker.bucket.name must not be empty")
        if self.video.startswith("s3://"):
            raise ValueError("aws_worker.bucket.video must be relative to input_prefix")
        path = PurePosixPath(video)
        if (
            not video
            or path.is_absolute()
            or ".." in path.parts
            or path.name in {"", ".", ".."}
        ):
            raise ValueError(
                "aws_worker.bucket.video must be a safe relative object path"
            )
        for field_name in ("input_prefix", "models_prefix", "runs_prefix"):
            prefix = getattr(self, field_name)
            prefix_path = PurePosixPath(prefix)
            if prefix_path.is_absolute() or ".." in prefix_path.parts:
                raise ValueError(
                    f"aws_worker.bucket.{field_name} must be a safe relative prefix"
                )

    @property
    def video_key(self) -> str:
        return "/".join(part for part in (self.input_prefix, self.video) if part)


@dataclass(frozen=True, slots=True)
class AwsWorkerConfig:
    job_id: str
    shutdown_on: str
    region: str
    bucket: BucketConfig
    sync_models: bool
    upload_results: bool
    local: LocalWorkspaceConfig
    sns: SnsConfig | None
    sagemaker: SageMakerAppConfig
    telegram: TelegramConfig | None = None
    cloudwatch: CloudWatchConfig | None = None

    def __post_init__(self) -> None:
        if self.shutdown_on not in VALID_SHUTDOWN_POLICIES:
            raise ValueError(f"invalid shutdown policy: {self.shutdown_on}")
        if not self.job_id or any(part in self.job_id for part in ("/", "\\", "..")):
            raise ValueError("job_id must be a simple directory name")
        if isinstance(self.bucket, dict):
            object.__setattr__(self, "bucket", BucketConfig(**self.bucket))
        if isinstance(self.cloudwatch, dict):
            object.__setattr__(self, "cloudwatch", CloudWatchConfig(**self.cloudwatch))

    @property
    def jobs_dir(self) -> Path:
        return self.local.jobs_dir

    @property
    def video_s3_location(self) -> tuple[str, str]:
        return self.bucket.name, self.bucket.video_key

    @property
    def video_s3_uri(self) -> str:
        bucket, key = self.video_s3_location
        return f"s3://{bucket}/{key}"

    @property
    def local_video_path(self) -> Path:
        _, key = self.video_s3_location
        return self.local.input_dir / PurePosixPath(key).name

    @property
    def bucket_name(self) -> str:
        return self.bucket.name

    @property
    def input_prefix(self) -> str:
        return self.bucket.input_prefix

    @property
    def models_prefix(self) -> str:
        return self.bucket.models_prefix

    @property
    def runs_prefix(self) -> str:
        return self.bucket.runs_prefix

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AwsWorkerConfig":
        notifications = payload.get("notifications") or {}
        sns_payload = notifications.get("email", payload.get("sns"))
        if (
            sns_payload is None
            and payload.get("sns_topic_name")
            and payload.get("notification_email")
        ):
            sns_payload = {
                "topic_name": payload["sns_topic_name"],
                "email": payload["notification_email"],
            }
        telegram_payload = notifications.get("telegram", payload.get("telegram"))
        if telegram_payload:
            telegram_payload = dict(telegram_payload)
            # Versions up to schema 6 exposed noisy runtime monitoring controls.
            # Accept old run documents, but intentionally ignore those settings.
            telegram_payload.pop("stream_logs", None)
            telegram_payload.pop("resource_status_interval_seconds", None)
        sagemaker_payload = payload.get("sagemaker") or {
            "domain_id": payload["sagemaker_domain_id"],
            "space_name": payload["sagemaker_space_name"],
            "app_name": payload.get("sagemaker_app_name", "default"),
        }
        bucket_payload = payload["bucket"]
        if isinstance(bucket_payload, dict):
            bucket = BucketConfig(**bucket_payload)
        else:
            bucket_name = str(bucket_payload)
            input_prefix = str(payload.get("input_prefix", "input")).strip("/")
            legacy_video = str(payload["s3_video_path"])
            if legacy_video.startswith("s3://"):
                parsed = urlparse(legacy_video)
                if parsed.netloc != bucket_name or not parsed.path.strip("/"):
                    raise ValueError(
                        "legacy s3_video_path must identify an object in aws_worker.bucket"
                    )
                video_key = parsed.path.lstrip("/")
            else:
                video_key = legacy_video.strip("/")
            prefix_with_slash = f"{input_prefix}/" if input_prefix else ""
            if prefix_with_slash and video_key.startswith(prefix_with_slash):
                video_key = video_key[len(prefix_with_slash) :]
            elif "/" in video_key and input_prefix:
                raise ValueError(
                    "legacy s3_video_path must be inside aws_worker.input_prefix"
                )
            bucket = BucketConfig(
                name=bucket_name,
                video=video_key,
                input_prefix=input_prefix,
                models_prefix=str(payload.get("models_prefix", "models")),
                runs_prefix=str(payload.get("runs_prefix", "runs")),
            )
        return cls(
            job_id=str(payload["job_id"]),
            shutdown_on=str(payload.get("shutdown_on", "never")),
            region=str(payload["region"]),
            bucket=bucket,
            sync_models=bool(payload.get("sync_models", True)),
            upload_results=bool(payload.get("upload_results", True)),
            local=(
                LocalWorkspaceConfig(**payload["local"])
                if "local" in payload
                else cls.workspace_from_environment()
            ),
            sns=SnsConfig(**sns_payload) if sns_payload else None,
            telegram=TelegramConfig(**telegram_payload) if telegram_payload else None,
            sagemaker=SageMakerAppConfig(**sagemaker_payload),
            cloudwatch=(
                CloudWatchConfig(**payload["cloudwatch"])
                if payload.get("cloudwatch") is not None
                else None
            ),
        )

    @classmethod
    def from_document(
        cls, document: ExperimentRequest, environment: PipelineEnvironment | None = None
    ) -> AwsWorkerConfig:
        settings = document.aws_worker
        return cls(
            job_id=settings.job_id,
            shutdown_on=settings.shutdown_on,
            region=settings.region,
            bucket=settings.bucket,
            sync_models=settings.sync_models,
            upload_results=settings.upload_results,
            local=cls.workspace_from_environment(environment),
            sns=settings.notifications.email,
            telegram=settings.notifications.telegram,
            sagemaker=settings.sagemaker,
            cloudwatch=settings.cloudwatch,
        )

    @staticmethod
    def workspace_from_environment(
        environment: PipelineEnvironment | None = None,
    ) -> LocalWorkspaceConfig:
        environment = environment or PipelineEnvironment.from_environ()
        return LocalWorkspaceConfig(environment.data_root, environment.fourdanyone_root)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("local")
        return payload


def materialize_pipeline_config(
    document: ExperimentRequest,
    worker: AwsWorkerConfig,
    environment: PipelineEnvironment | None = None,
) -> FourDAnyoneConfig:
    """Complete typed portable settings with staged local paths."""
    settings = extract_4danyone_dataset_config(
        document.pipeline,
        experiment_name=document.experiment_name,
        artifacts=document.artifacts,
        postprocessing=document.postprocessing,
    )
    environment = environment or PipelineEnvironment.from_environ()
    return environment.materialize(settings, worker.local_video_path)


def validate_experiment_request(document: ExperimentRequest) -> None:
    """Domain validation without resolving any machine environment."""
    settings = extract_4danyone_dataset_config(
        document.pipeline,
        experiment_name=document.experiment_name,
        artifacts=document.artifacts,
        postprocessing=document.postprocessing,
    )
    dataset = settings.dataset
    FourDAnyoneConfig(
        video_path=Path("/validation/input.mov"),
        experiment_name=settings.experiment_name,
        fourdanyone_root=Path("/validation"),
        model_dir=Path("/validation/models"),
        runs_dir=Path("/validation/runs"),
        views_per_layer=dataset.views_per_layer,
        layer_pitches=dataset.layer_pitches,
        start_yaw=dataset.start_yaw,
        yaw_span=dataset.yaw_span,
        target_fps=dataset.target_fps,
        seed=dataset.seed,
        enable_turbo=dataset.enable_turbo,
        attention_backend=dataset.attention_backend,
        resume=dataset.resume,
        dataset_enabled=settings.dataset_enabled,
        nerfstudio=settings.nerfstudio,
        rerun=settings.rerun,
        reconstruction=settings.reconstruction,
        reconstruction_rerun=settings.reconstruction_rerun,
        postprocessing=settings.postprocessing,
    )


def validate_run_settings(document: dict[str, Any]) -> None:
    """JSON-authoring boundary used by the portable config CLI."""
    from .requests import ExperimentRequest

    validate_environment_free(document)
    ExperimentRequest.from_dict(document)


def load_aws_worker_config(path: Path) -> tuple[FourDAnyoneConfig, AwsWorkerConfig]:
    from .requests import ExperimentRequest

    document = ExperimentRequest.from_dict(load_document(path))
    environment = PipelineEnvironment.from_environ()
    worker = AwsWorkerConfig.from_document(document, environment)
    return materialize_pipeline_config(document, worker, environment), worker
