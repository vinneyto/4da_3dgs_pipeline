"""AWS staging, persistence, and cleanup passes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fourda_4danyone.config import FourDAnyoneConfig
from fourda_4danyone.artifacts import (
    EXPERIMENT_WORKSPACE,
    INPUT_VIDEO,
    MODEL_CACHE,
    experiment_artifact,
)
from fourda_pipeline.core import PassResult, PipelineContext, PipelineOutcome
from fourda_nerfstudio.passes import NERFSTUDIO_DATASETS
from fourda_rerun.passes import RERUN_RECORDING

from .artifacts import (
    AWS_HEALTH,
    RUN_RESULT,
    S3_RESULT,
)

from .aws import (
    download_input_video,
    run_health_check,
    stop_sagemaker_app,
    sync_experiment_results,
    sync_model_objects,
    upload_directory,
)
from .config import AwsWorkerConfig


class AwsPreflightPass:
    id = "aws-preflight"
    name = "AWS preflight"
    requires = frozenset()
    provides = frozenset({AWS_HEALTH})

    def __init__(
        self, worker: AwsWorkerConfig, pipeline_config: FourDAnyoneConfig
    ) -> None:
        self.worker = worker
        self.pipeline_config = pipeline_config

    def run(self, context: PipelineContext) -> PassResult:
        context.report_progress(0.1, "Validating AWS identity and S3 access")
        health = run_health_check(
            self.worker,
            self.worker.job_id,
            require_input_video=self.pipeline_config.dataset_enabled,
        )
        context.report_progress(1.0, "AWS dependencies are ready")
        return PassResult(
            artifacts={AWS_HEALTH: health},
            details={
                "caller": health.caller_arn,
                "bucket": health.bucket,
                "email": health.email_status,
                "telegram": health.telegram_status,
            },
        )


class S3DownloadInputPass:
    id = "s3-download-input"
    name = "Download input video"
    requires = frozenset({AWS_HEALTH})
    provides = frozenset({INPUT_VIDEO})

    def __init__(self, worker: AwsWorkerConfig) -> None:
        self.worker = worker

    def run(self, context: PipelineContext) -> PassResult:
        context.report_progress(0.0, f"Downloading {self.worker.video_s3_uri}")
        path = download_input_video(self.worker)
        context.report_progress(1.0, f"Input video ready at {path}")
        return PassResult(
            artifacts={INPUT_VIDEO: path},
            details={"s3_uri": self.worker.video_s3_uri, "path": str(path)},
        )


class S3SyncModelsPass:
    id = "s3-sync-models"
    name = "Synchronize model cache"
    requires = frozenset({AWS_HEALTH})
    provides = frozenset({MODEL_CACHE})

    def __init__(self, worker: AwsWorkerConfig) -> None:
        self.worker = worker

    def run(self, context: PipelineContext) -> PassResult:
        context.report_progress(0.0, "Synchronizing model objects")
        found, downloaded = sync_model_objects(self.worker)
        context.report_progress(1.0, "Model cache is ready")
        return PassResult(
            artifacts={MODEL_CACHE: self.worker.local.model_dir},
            details={"objects": found, "downloaded": downloaded},
        )


class S3RestoreExperimentPass:
    name = "Restore source experiment"
    requires = frozenset({AWS_HEALTH})

    def __init__(
        self,
        worker: AwsWorkerConfig,
        experiment_name: str,
        destination: Path,
    ) -> None:
        self.worker = worker
        self.experiment_name = experiment_name
        self.destination = destination
        self.id = f"s3-restore-experiment:{experiment_name}"
        self.name = f"Restore source experiment {experiment_name}"
        self.provides = frozenset({experiment_artifact(experiment_name)})

    def run(self, context: PipelineContext) -> PassResult:
        generation = self.destination / "4danyone"
        required = (generation / "metadata.json", generation / "cameras.json")
        if all(path.is_file() for path in required):
            found = downloaded = 0
            message = "Reusing source experiment from persistent storage"
        else:
            context.report_progress(0.0, "Restoring source experiment from S3")
            found, downloaded = sync_experiment_results(
                self.worker, self.experiment_name, self.destination
            )
            message = "Source experiment restored"
        context.report_progress(1.0, message)
        return PassResult(
            artifacts={experiment_artifact(self.experiment_name): generation},
            details={"objects": found, "downloaded": downloaded, "path": str(generation)},
        )


class WriteRunManifestPass:
    id = "write-run-manifest"
    name = "Write run manifest"
    requires = frozenset({EXPERIMENT_WORKSPACE})
    provides = frozenset({RUN_RESULT})

    def __init__(self, config: FourDAnyoneConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PassResult:
        durations = dict(context.values.get("pass_durations", {}))
        result = {
            "experiment_name": self.config.experiment_name,
            "experiment_dir": str(self.config.experiment_dir),
            "inference_dir": str(self.config.inference_dir),
            "datasets": context.artifacts.get(NERFSTUDIO_DATASETS, []),
            "rerun_file": (
                str(context.artifacts[RERUN_RECORDING])
                if RERUN_RECORDING in context.artifacts
                else None
            ),
            "num_views": self.config.num_views,
            "pass_durations_seconds": durations,
            "elapsed_seconds": sum(durations.values()),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        path = self.config.experiment_dir / "pipeline-result.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        context.report_progress(1.0, "Run manifest written")
        return PassResult(
            artifacts={RUN_RESULT: result},
            details={"path": str(path)},
        )


class S3UploadResultsPass:
    id = "s3-upload-results"
    name = "Upload results to S3"
    requires = frozenset({RUN_RESULT})
    provides = frozenset({S3_RESULT})

    def __init__(self, worker: AwsWorkerConfig, config: FourDAnyoneConfig) -> None:
        self.worker = worker
        self.config = config

    def run(self, context: PipelineContext) -> PassResult:
        context.report_progress(0.0, "Uploading experiment results")
        uri = upload_directory(
            self.worker,
            self.config.experiment_dir,
            self.config.experiment_name,
        )
        context.report_progress(1.0, f"Results uploaded to {uri}")
        return PassResult(
            artifacts={S3_RESULT: uri},
            details={"s3_uri": uri},
        )


class SageMakerShutdownFinalizer:
    id = "sagemaker-shutdown"
    name = "Apply SageMaker shutdown policy"

    def __init__(self, worker: AwsWorkerConfig) -> None:
        self.worker = worker

    def run(self, context: PipelineContext, outcome: PipelineOutcome) -> None:
        policy = self.worker.shutdown_on
        should_stop = policy == "always" or (
            policy == "success" and outcome.error is None
        ) or (policy == "failure" and outcome.error is not None)
        if should_stop:
            stop_sagemaker_app(self.worker.region, self.worker.sagemaker)
