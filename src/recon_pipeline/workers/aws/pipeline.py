"""AWS composition root: assemble passes and observers from one run document."""

from __future__ import annotations

from pathlib import Path

from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.datasets.fourdanyone.passes import FourDAnyoneInferencePass, PrepareExperimentPass
from recon_pipeline.reconstructions.nerfstudio.passes import (
    GaussianSplatExportPass, NerfstudioExportPass, SplatfactoTrainPass,
)
from recon_pipeline.core import JsonPassCheckpointStore, Pipeline
from recon_pipeline.artifacts.rerun.passes import RerunExportPass

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
    *,
    force: bool = False,
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
    if config.reconstruction.enabled:
        for frame in config.reconstruction.frames:
            passes.append(SplatfactoTrainPass(config, frame))
            passes.append(GaussianSplatExportPass(config, frame))
    passes.append(WriteRunManifestPass(config))
    if worker.upload_results:
        passes.append(S3UploadResultsPass(worker, config))

    observers = [
        ConsoleObserver(),
        JobStatusObserver(status, job_dir / "status.json"),
        AwsNotificationObserver(worker, config.experiment_name),
        RuntimeMonitoringObserver(worker),
    ]
    return Pipeline(
        passes,
        finalizers=[SageMakerShutdownFinalizer(worker)],
        observers=observers,
        checkpoint_store=JsonPassCheckpointStore(
            config.experiment_dir / ".recon-pipeline/pass-state.json",
            config.experiment_name,
        ),
        force=force,
    )
