"""AWS composition root: assemble passes and observers from one run document."""

from __future__ import annotations

from pathlib import Path

from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.datasets.fourdanyone.passes import (
    FourDAnyoneInferencePass,
    PrepareExperimentPass,
)
from recon_pipeline.reconstructions.nerfstudio.passes import NerfstudioExportPass
from recon_pipeline.core import JsonPassCheckpointStore, Pipeline
from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import SplatfactoPass
from recon_pipeline.artifacts.rerun.passes.splatfacto import SplatfactoRerunPass
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
    WriteRunManifestPass,
)
from .status import JobStatus
from .recovery import (
    RecoverablePass,
    RecoveryCheckpointStore,
    artifact_paths,
    fingerprint,
)
from .passes.upload_artifacts import S3UploadArtifactsPass
from .persistence import S3PersistenceObserver, S3DiagnosticsFinalizer


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
        sources[config.rerun_source_experiment_name] = (
            config.rerun_source_experiment_dir
        )
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
    if config.reconstruction.enabled:
        for frame in config.reconstruction_frames:
            passes.append(SplatfactoPass(config, frame))
            if config.reconstruction_rerun.enabled:
                passes.append(SplatfactoRerunPass(config, frame))
    if config.rerun.enabled:
        passes.append(RerunExportPass(config))
    passes.append(WriteRunManifestPass(config))
    settings = config.settings_dict()
    input_identity = (worker.bucket_name, worker.input_prefix, worker.bucket.video)
    legacy = None
    saved_settings = config.experiment_dir / "pipeline-config.json"
    if saved_settings.is_file():
        import json

        legacy = json.loads(saved_settings.read_text())
    composed, uploads = [], []
    for item in passes:
        if artifact_paths(item, config):
            signature = fingerprint(item.id, settings, input_identity)
            legacy_signature = (
                fingerprint(item.id, legacy, input_identity) if legacy else None
            )
            item = RecoverablePass(item, config, signature, legacy_signature)
        composed.append(item)
        if worker.upload_results and isinstance(item, RecoverablePass):
            upload = S3UploadArtifactsPass(worker, config, item)
            uploads.append(upload)
            composed.append(upload)
    passes = composed

    observers = [
        ConsoleObserver(),
        JobStatusObserver(status, job_dir / "status.json"),
        AwsNotificationObserver(worker, config.experiment_name),
        RuntimeMonitoringObserver(worker),
    ]
    finalizers = [SageMakerShutdownFinalizer(worker)]
    store = JsonPassCheckpointStore(
        config.experiment_dir / ".recon-pipeline/pass-state.json",
        config.experiment_name,
    )
    if worker.upload_results:
        persistence = S3PersistenceObserver(worker, config, job_dir)
        observers.append(persistence)
        finalizers.insert(0, S3DiagnosticsFinalizer(persistence))
        store = RecoveryCheckpointStore(worker, config, uploads)
    return Pipeline(
        passes,
        finalizers=finalizers,
        observers=observers,
        checkpoint_store=store,
        resume_by_id=True,
        force=force,
    )
