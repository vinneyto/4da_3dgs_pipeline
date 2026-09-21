"""AWS composition root: assemble passes and observers from one run document."""

from __future__ import annotations

from pathlib import Path

from fourda_4danyone.config import FourDAnyoneConfig
from fourda_4danyone.passes import FourDAnyoneInferencePass, PrepareExperimentPass
from fourda_nerfstudio.passes import NerfstudioExportPass
from fourda_pipeline.core import Pipeline
from fourda_rerun.passes import RerunExportPass

from .config import AwsWorkerConfig
from .finalizers import SageMakerShutdownFinalizer
from .observers import (
    AwsNotificationObserver,
    ConsoleObserver,
    JobStatusObserver,
    RuntimeMonitoringObserver,
)
from .passes import (
    AwsPreflightPass,
    S3DownloadInputPass,
    S3RestoreExperimentPass,
    S3SyncModelsPass,
    S3UploadResultsPass,
    WriteRunManifestPass,
)
from .status import JobStatus


def required_source_experiments(
    config: FourDAnyoneConfig,
) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    if config.nerfstudio.enabled and not (
        config.dataset_enabled
        and config.nerfstudio_source_experiment_name == config.experiment_name
    ):
        sources[config.nerfstudio_source_experiment_name] = (
            config.nerfstudio_source_experiment_dir
        )
    if config.rerun.enabled and not (
        config.dataset_enabled
        and config.rerun_source_experiment_name == config.experiment_name
    ):
        sources[config.rerun_source_experiment_name] = config.rerun_source_experiment_dir
    return sources


def build_aws_pipeline(
    worker: AwsWorkerConfig,
    config: FourDAnyoneConfig,
    job_dir: Path,
    status: JobStatus,
) -> Pipeline:
    passes = [AwsPreflightPass(worker, config)]
    if config.dataset_enabled:
        passes.append(S3DownloadInputPass(worker))
    passes.append(S3SyncModelsPass(worker))
    passes.extend(
        S3RestoreExperimentPass(worker, name, destination)
        for name, destination in required_source_experiments(config).items()
    )
    passes.append(PrepareExperimentPass(config))
    if config.dataset_enabled:
        passes.append(FourDAnyoneInferencePass(config))
    if config.nerfstudio.enabled:
        passes.append(NerfstudioExportPass(config))
    if config.rerun.enabled:
        passes.append(RerunExportPass(config))
    passes.append(WriteRunManifestPass(config))
    if worker.upload_results:
        passes.append(S3UploadResultsPass(worker, config))

    observers = [
        ConsoleObserver(),
        JobStatusObserver(status, job_dir / "status.json"),
        AwsNotificationObserver(worker, config.experiment_name),
        RuntimeMonitoringObserver(worker, job_dir),
    ]
    return Pipeline(
        passes,
        finalizers=[SageMakerShutdownFinalizer(worker)],
        observers=observers,
    )
