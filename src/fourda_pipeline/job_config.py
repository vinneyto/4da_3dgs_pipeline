"""Configuration for the AWS-aware background-job adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FourDAnyoneConfig


VALID_SHUTDOWN_POLICIES = frozenset({"never", "success", "always"})


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
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
class AwsConfig:
    region: str
    bucket: str
    input_prefix: str
    runs_prefix: str
    models_prefix: str
    upload_input: bool
    upload_results: bool
    sns: SnsConfig
    sagemaker: SageMakerAppConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AwsConfig":
        return cls(
            region=str(payload["region"]),
            bucket=str(payload["bucket"]),
            input_prefix=str(payload.get("input_prefix", "input")).strip("/"),
            runs_prefix=str(payload.get("runs_prefix", "runs")).strip("/"),
            models_prefix=str(payload.get("models_prefix", "models")).strip("/"),
            upload_input=bool(payload.get("upload_input", True)),
            upload_results=bool(payload.get("upload_results", True)),
            sns=SnsConfig(**payload["sns"]),
            sagemaker=SageMakerAppConfig(**payload["sagemaker"]),
        )


@dataclass(frozen=True, slots=True)
class JobConfig:
    job_id: str
    jobs_dir: Path
    shutdown_on: str
    aws: AwsConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "jobs_dir", Path(self.jobs_dir))
        if not self.jobs_dir.is_absolute():
            raise ValueError(f"job.jobs_dir must be absolute: {self.jobs_dir}")
        if self.shutdown_on not in VALID_SHUTDOWN_POLICIES:
            raise ValueError(f"invalid shutdown policy: {self.shutdown_on}")

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "JobConfig":
        payload = document["job"]
        return cls(
            job_id=str(payload["job_id"]),
            jobs_dir=Path(payload["jobs_dir"]),
            shutdown_on=str(payload.get("shutdown_on", "never")),
            aws=AwsConfig.from_dict(document["aws"]),
        )


def load_job_config(path: Path) -> tuple[FourDAnyoneConfig, JobConfig]:
    document = load_document(path)
    return FourDAnyoneConfig.from_dict(document["pipeline"]), JobConfig.from_document(document)
