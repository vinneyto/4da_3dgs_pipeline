"""Launch the persistent model synchronization utility."""

from pathlib import Path

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import MODEL_CACHE
from ..artifacts import AWS_HEALTH
from ..config import AwsWorkerConfig


class S3SyncModelsPass:
    id = "s3-sync-models"
    name = "Synchronize model cache"
    requires = frozenset({AWS_HEALTH})
    provides = frozenset({MODEL_CACHE})

    def __init__(
        self, worker: AwsWorkerConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.worker = worker
        self.runner = runner or CommandRunner()

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.storage.s3.utilities.sync_models",
            [
                "--bucket",
                self.worker.bucket_name,
                "--prefix",
                self.worker.models_prefix,
                "--destination",
                str(self.worker.local.model_dir),
                "--region",
                self.worker.region,
                "--sync" if self.worker.sync_models else "--no-sync",
            ],
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={MODEL_CACHE: Path(result["path"])},
            details={"objects": result["objects"], "downloaded": result["downloaded"]},
        )
