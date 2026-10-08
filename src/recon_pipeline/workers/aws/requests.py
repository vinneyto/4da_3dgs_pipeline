"""Validated worker requests. JSON/dicts exist only at load/save boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

from recon_pipeline.config_types import checked_value, object_value, parse_settings
from recon_pipeline.run_document import (
    SUPPORTED_SCHEMA_VERSIONS,
    strip_environment,
    validate_environment_free,
)
from recon_pipeline.settings import (
    ArtifactSettings,
    PipelineSettings,
    parse_postprocessing,
)
from recon_pipeline.postprocessing.config import PostprocessingConfig
from .config import (
    AwsWorkerConfig,
    BucketConfig,
    CloudWatchConfig,
    SageMakerAppConfig,
    SnsConfig,
    TelegramConfig,
    VALID_SHUTDOWN_POLICIES,
)


@dataclass(frozen=True, slots=True)
class NotificationSettings:
    email: SnsConfig | None = None
    telegram: TelegramConfig | None = None


@dataclass(frozen=True, slots=True)
class AwsWorkerSettings:
    job_id: str
    region: str
    bucket: BucketConfig
    sagemaker: SageMakerAppConfig
    shutdown_on: str = "never"
    sync_models: bool = True
    upload_results: bool = True
    notifications: NotificationSettings = NotificationSettings()
    cloudwatch: CloudWatchConfig | None = None

    def __post_init__(self) -> None:
        if self.shutdown_on not in VALID_SHUTDOWN_POLICIES:
            raise ValueError(f"invalid shutdown policy: {self.shutdown_on}")
        if not self.job_id or any(part in self.job_id for part in ("/", "\\", "..")):
            raise ValueError("job_id must be a simple directory name")
        if not self.region:
            raise ValueError("aws_worker.region must not be empty")

    @classmethod
    def from_dict(cls, payload: Any, pipeline: PipelineSettings) -> AwsWorkerSettings:
        values = dict(object_value(payload, "aws_worker"))
        # Keep existing AWS aliases at the read boundary, independent of export migration.
        for name in ("sync_models", "upload_results"):
            if name in values:
                checked_value(values[name], bool, f"aws_worker.{name}")
        for name in ("job_id", "region", "shutdown_on"):
            if name in values:
                checked_value(values[name], str, f"aws_worker.{name}")
        if isinstance(values.get("bucket"), dict):
            parse_settings(BucketConfig, values["bucket"], "aws_worker.bucket")
        elif "s3_video_path" not in values:
            from pathlib import Path

            filename = Path(pipeline.dataset.config.video or "").name
            prefix = str(values.get("input_prefix", "input")).strip("/")
            key = "/".join(part for part in (prefix, filename) if part)
            values["s3_video_path"] = f"s3://{values['bucket']}/{key}"
        notifications = object_value(
            {} if values.get("notifications") is None else values["notifications"],
            "aws_worker.notifications",
        )
        for name, model in (("email", SnsConfig), ("telegram", TelegramConfig)):
            item = notifications.get(
                name, values.get("sns" if name == "email" else name)
            )
            if item is not None:
                item = dict(object_value(item, f"aws_worker.notifications.{name}"))
                if name == "telegram":
                    item.pop("stream_logs", None)
                    item.pop("resource_status_interval_seconds", None)
                parse_settings(model, item, f"aws_worker.notifications.{name}")
        if values.get("sagemaker") is not None:
            parse_settings(
                SageMakerAppConfig, values["sagemaker"], "aws_worker.sagemaker"
            )
        if values.get("cloudwatch") is not None:
            parse_settings(
                CloudWatchConfig, values["cloudwatch"], "aws_worker.cloudwatch"
            )
        values["local"] = {
            "data_root": "/validation",
            "fourdanyone_root": "/validation",
        }
        worker = AwsWorkerConfig.from_dict(values)
        return cls(
            worker.job_id,
            worker.region,
            worker.bucket,
            worker.sagemaker,
            worker.shutdown_on,
            worker.sync_models,
            worker.upload_results,
            NotificationSettings(worker.sns, worker.telegram),
            worker.cloudwatch,
        )


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    schema_version: int
    experiment_name: str
    pipeline: PipelineSettings
    aws_worker: AwsWorkerSettings
    artifacts: ArtifactSettings = ArtifactSettings()
    postprocessing: PostprocessingConfig = PostprocessingConfig()
    force: bool = False

    @classmethod
    def from_dict(cls, payload: Any) -> ExperimentRequest:
        values = object_value(payload, "experiment request")
        if "kind" in values:
            raise ValueError("experiment request must omit kind")
        version = checked_value(values.get("schema_version"), int, "schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"config schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if version >= 7:
            validate_environment_free(values)
        else:
            values = strip_environment(values)
        if "pipeline" not in values:
            raise ValueError("config must contain a pipeline object")
        if "aws_worker" not in values:
            raise ValueError("config must contain an aws_worker object")
        pipeline = PipelineSettings.from_dict(values["pipeline"])
        name = checked_value(
            values.get(
                "experiment_name",
                pipeline.experiment_name or pipeline.dataset.config.experiment_name,
            ),
            str,
            "experiment_name",
        )
        request = cls(
            version,
            name,
            pipeline,
            AwsWorkerSettings.from_dict(values["aws_worker"], pipeline),
            ArtifactSettings.from_dict(values.get("artifacts")),
            parse_postprocessing(values.get("postprocessing")),
            checked_value(values.get("force", False), bool, "force"),
        )
        # Domain checks need no machine environment or GPU and run before dispatch.
        from .config import validate_experiment_request

        validate_experiment_request(request)
        return request

    def to_dict(self) -> dict[str, Any]:
        dataset = asdict(self.pipeline.dataset.config)
        dataset = {k: v for k, v in dataset.items() if v is not None}
        reconstruction = asdict(self.pipeline.reconstruction.config)
        reconstruction.pop("enabled")
        reconstruction.pop("nerfstudio_bin")
        artifacts = asdict(self.artifacts)
        if (
            artifacts["dataset"]["rerun"] is None
            and self.pipeline.dataset.rerun is not None
        ):
            artifacts["dataset"]["rerun"] = asdict(self.pipeline.dataset.rerun)
        artifacts["reconstruction"]["rerun"].pop("python")
        postprocessing = asdict(self.postprocessing)
        postprocessing["splat_conversion"].pop("splat_transform")
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "pipeline": {
                "dataset": {
                    "type": self.pipeline.dataset.type,
                    "enabled": self.pipeline.dataset.enabled,
                    "config": dataset,
                },
                "reconstruction": {
                    "type": self.pipeline.reconstruction.type,
                    "enabled": self.pipeline.reconstruction.enabled,
                    "config": reconstruction,
                },
            },
            "artifacts": artifacts,
            "postprocessing": postprocessing,
            "aws_worker": asdict(self.aws_worker),
            "force": self.force,
        }


@dataclass(frozen=True, slots=True)
class QueueRequest:
    configs: tuple[ExperimentRequest, ...]
    shutdown_on: str
    region: str
    sagemaker: SageMakerAppConfig
    kind: str = "queue"

    def __post_init__(self) -> None:
        if self.kind != "queue":
            raise ValueError("queue kind must be queue")
        if not self.configs:
            raise ValueError("queue.configs must contain at least one experiment")
        if self.shutdown_on not in VALID_SHUTDOWN_POLICIES:
            raise ValueError(f"invalid shutdown policy: {self.shutdown_on}")
        if any(
            (r.aws_worker.region, r.aws_worker.sagemaker)
            != (self.region, self.sagemaker)
            for r in self.configs
        ):
            raise ValueError(
                "All queued configs must target the same SageMaker App and region"
            )
        ids = [r.aws_worker.job_id for r in self.configs]
        names = [r.experiment_name for r in self.configs]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError(
                "Queued configs must have distinct job IDs and experiment names"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> QueueRequest:
        values = object_value(payload, "queue request")
        configs = values.get("configs")
        if not isinstance(configs, list):
            raise TypeError("queue.configs must be an array")
        return cls(
            tuple(ExperimentRequest.from_dict(r) for r in configs),
            checked_value(values.get("shutdown_on"), str, "queue.shutdown_on"),
            checked_value(values.get("region"), str, "queue.region"),
            parse_settings(
                SageMakerAppConfig, values.get("sagemaker"), "queue.sagemaker"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "queue",
            "configs": [r.to_dict() for r in self.configs],
            "shutdown_on": self.shutdown_on,
            "region": self.region,
            "sagemaker": asdict(self.sagemaker),
        }


WorkerRequest: TypeAlias = ExperimentRequest | QueueRequest


def parse_worker_request(payload: Any) -> WorkerRequest:
    values = object_value(payload, "worker request")
    if "kind" not in values:
        return ExperimentRequest.from_dict(values)
    if values["kind"] == "queue":
        return QueueRequest.from_dict(values)
    raise ValueError(
        f"unknown worker request kind {values['kind']!r}; omit kind for an experiment or use 'queue'"
    )
