"""Launch the source experiment restoration utility."""

from pathlib import Path

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import experiment_artifact
from ..artifacts import AWS_HEALTH
from ..config import AwsWorkerConfig


class S3RestoreExperimentPass:
    requires = frozenset({AWS_HEALTH})

    def __init__(
        self,
        worker: AwsWorkerConfig,
        experiment_name: str,
        destination: Path,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.worker = worker
        self.experiment_name = experiment_name
        self.destination = destination
        self.runner = runner or CommandRunner()
        self.id = f"s3-restore-experiment:{experiment_name}"
        self.name = f"Restore source experiment {experiment_name}"
        self.provides = frozenset({experiment_artifact(experiment_name)})

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.utilities.storage.s3.restore_experiment",
            [
                "--bucket",
                self.worker.bucket_name,
                "--region",
                self.worker.region,
                "--prefix",
                "/".join(
                    part
                    for part in (self.worker.runs_prefix, self.experiment_name)
                    if part
                ),
                "--destination",
                str(self.destination),
                "--replace-existing",
            ],
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={experiment_artifact(self.experiment_name): Path(result["path"])},
            details=result,
        )
