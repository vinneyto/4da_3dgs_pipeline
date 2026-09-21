"""Download the configured input video from S3."""

from recon_pipeline.datasets.fourdanyone.artifacts import INPUT_VIDEO
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import AWS_HEALTH
from ..aws import download_input_video
from ..config import AwsWorkerConfig


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
