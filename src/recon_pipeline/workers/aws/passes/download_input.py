"""Launch the input video download utility."""

from pathlib import Path

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import INPUT_VIDEO
from ..artifacts import AWS_HEALTH
from ..config import AwsWorkerConfig


class S3DownloadInputPass:
    id = "s3-download-input"
    name = "Download input video"
    requires = frozenset({AWS_HEALTH})
    provides = frozenset({INPUT_VIDEO})

    def __init__(
        self, worker: AwsWorkerConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.worker = worker
        self.runner = runner or CommandRunner()

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.utilities.storage.s3.download_input",
            [
                "--bucket",
                self.worker.bucket_name,
                "--key",
                self.worker.bucket.video_key,
                "--output",
                str(self.worker.local_video_path),
                "--region",
                self.worker.region,
                "--replace-existing",
            ],
            context,
            runner=self.runner,
        )
        return PassResult(artifacts={INPUT_VIDEO: Path(result["path"])}, details=result)
