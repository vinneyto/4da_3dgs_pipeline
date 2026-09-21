"""Apply the configured SageMaker App shutdown policy after pipeline execution."""

from recon_pipeline.core import PipelineContext, PipelineOutcome

from ..aws import stop_sagemaker_app
from ..config import AwsWorkerConfig


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
