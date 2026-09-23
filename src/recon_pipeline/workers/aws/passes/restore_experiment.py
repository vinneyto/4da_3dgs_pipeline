"""Restore a source experiment needed by artifact-only passes."""

from pathlib import Path

from recon_pipeline.datasets.fourdanyone.artifacts import experiment_artifact
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import AWS_HEALTH
from ..aws import sync_experiment_results
from ..config import AwsWorkerConfig


class S3RestoreExperimentPass:
    requires = frozenset({AWS_HEALTH})

    def __init__(
        self,
        worker: AwsWorkerConfig,
        experiment_name: str,
        destination: Path,
    ) -> None:
        self.worker = worker
        self.experiment_name = experiment_name
        self.destination = destination
        self.id = f"s3-restore-experiment:{experiment_name}"
        self.name = f"Restore source experiment {experiment_name}"
        self.provides = frozenset({experiment_artifact(experiment_name)})

    def cleanup(self, context: PipelineContext) -> None:
        # A source experiment belongs to another run. Never delete its local
        # generation when retrying a derived reconstruction.
        pass

    def run(self, context: PipelineContext) -> PassResult:
        generation = self.destination / "4danyone"
        required = (generation / "metadata.json", generation / "cameras.json")
        if all(path.is_file() for path in required):
            found = downloaded = 0
            message = "Reusing source experiment from persistent storage"
        else:
            context.report_progress(0.0, "Restoring source experiment from S3")
            found, downloaded = sync_experiment_results(
                self.worker, self.experiment_name, self.destination
            )
            message = "Source experiment restored"
        context.report_progress(1.0, message)
        return PassResult(
            artifacts={experiment_artifact(self.experiment_name): generation},
            details={"objects": found, "downloaded": downloaded, "path": str(generation)},
        )
