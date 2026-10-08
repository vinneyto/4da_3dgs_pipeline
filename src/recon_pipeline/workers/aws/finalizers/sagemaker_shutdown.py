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
        log_session = context.values.get("cloudwatch_session")
        if log_session is not None:
            log_session.flush_before_shutdown(context)
        if context.values.get("cloudwatch_persistence_failed"):
            print("Shutdown deferred: CloudWatch logs remain in the local retry spool", flush=True)
            return
        if context.values.get("s3_persistence_failed"):
            print(
                "Shutdown deferred: artifacts or diagnostics could not be saved to S3",
                flush=True,
            )
            return
        if isinstance(outcome.error, KeyboardInterrupt):
            return
        policy = self.worker.shutdown_on
        should_stop = (
            policy == "always"
            or (policy == "success" and outcome.error is None)
            or (policy == "failure" and outcome.error is not None)
        )
        if should_stop:
            stop_sagemaker_app(self.worker.region, self.worker.sagemaker)
