"""Synchronize the persistent local model cache from S3."""

from recon_pipeline.datasets.fourdanyone.artifacts import MODEL_CACHE
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import AWS_HEALTH
from ..aws import sync_model_objects
from ..config import AwsWorkerConfig


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
