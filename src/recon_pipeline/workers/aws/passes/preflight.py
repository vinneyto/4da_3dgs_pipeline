"""Launch the AWS dependency check utility."""

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from ..aws import check_notification_channels
from ..artifacts import AWS_HEALTH
from ..config import AwsWorkerConfig


class AwsPreflightPass:
    id = "aws-preflight"
    name = "AWS preflight"
    requires = frozenset()
    provides = frozenset({AWS_HEALTH})

    def __init__(
        self,
        worker: AwsWorkerConfig,
        pipeline_config: FourDAnyoneConfig,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.worker = worker
        self.pipeline_config = pipeline_config
        self.runner = runner or CommandRunner()

    def run(self, context: PipelineContext) -> PassResult:
        worker = self.worker
        arguments = [
            "--bucket",
            worker.bucket_name,
            "--region",
            worker.region,
            "--input-key",
            worker.bucket.video_key,
        ]
        if not self.pipeline_config.dataset_enabled:
            arguments.append("--no-check-input")
        if worker.sync_models:
            arguments.extend(["--models-prefix", worker.models_prefix])
        if worker.upload_results:
            arguments.extend(["--write-prefix", worker.runs_prefix])
        if worker.shutdown_on != "never":
            arguments.extend(
                [
                    "--sagemaker-domain-id",
                    worker.sagemaker.domain_id,
                    "--sagemaker-space-name",
                    worker.sagemaker.space_name,
                    "--sagemaker-app-name",
                    worker.sagemaker.app_name,
                ]
            )
        health = run_utility(
            "recon_pipeline.utilities.cloud.aws.preflight",
            arguments,
            context,
            runner=self.runner,
        )
        health.update(check_notification_channels(worker))
        return PassResult(
            artifacts={AWS_HEALTH: health},
            details={
                "caller": health["caller_arn"],
                "bucket": health["bucket"],
                "email": health["email_status"],
                "telegram": health["telegram_status"],
            },
        )
