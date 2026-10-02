"""Launch the AWS dependency check utility."""

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
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
        health = run_utility(
            "recon_pipeline.workers.aws.utilities.preflight",
            ["--require-input-video"] if self.pipeline_config.dataset_enabled else [],
            context,
            runner=self.runner,
            documents={"--worker-config": self.worker.to_dict()},
        )
        return PassResult(
            artifacts={AWS_HEALTH: health},
            details={
                "caller": health["caller_arn"],
                "bucket": health["bucket"],
                "email": health["email_status"],
                "telegram": health["telegram_status"],
            },
        )
