"""Launch the experiment results upload utility."""

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from ..artifacts import RUN_RESULT, S3_RESULT
from ..config import AwsWorkerConfig


class S3UploadResultsPass:
    id = "s3-upload-results"
    name = "Upload results to S3"
    requires = frozenset({RUN_RESULT})
    provides = frozenset({S3_RESULT})

    def __init__(
        self,
        worker: AwsWorkerConfig,
        config: FourDAnyoneConfig,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.worker = worker
        self.config = config
        self.runner = runner or CommandRunner()

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.workers.aws.utilities.upload_results",
            [
                "--experiment",
                self.config.experiment_name,
                "--source",
                str(self.config.experiment_dir),
                "--replace-existing",
            ],
            context,
            runner=self.runner,
            documents={"--worker-config": self.worker.to_dict()},
        )
        return PassResult(artifacts={S3_RESULT: result["s3_uri"]}, details=result)
