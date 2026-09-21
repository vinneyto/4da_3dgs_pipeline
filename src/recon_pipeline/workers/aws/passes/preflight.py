"""Validate AWS dependencies before expensive GPU work begins."""

from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import AWS_HEALTH
from ..aws import run_health_check
from ..config import AwsWorkerConfig


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
