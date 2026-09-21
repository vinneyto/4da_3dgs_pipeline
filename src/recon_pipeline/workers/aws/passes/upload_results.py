"""Upload the completed experiment directory to S3."""

from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import RUN_RESULT, S3_RESULT
from ..aws import delete_experiment_results, upload_directory
from ..config import AwsWorkerConfig


class S3UploadResultsPass:
    id = "s3-upload-results"
    name = "Upload results to S3"
    requires = frozenset({RUN_RESULT})
    provides = frozenset({S3_RESULT})

    def __init__(self, worker: AwsWorkerConfig, config: FourDAnyoneConfig) -> None:
        self.worker = worker
        self.config = config

    def cleanup(self, context: PipelineContext) -> None:
        delete_experiment_results(
            self.worker,
            self.config.experiment_name,
        )

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
