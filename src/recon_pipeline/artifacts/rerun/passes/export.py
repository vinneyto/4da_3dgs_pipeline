"""Launch the standalone Rerun recording utility."""

from pathlib import Path

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import (
    EXPERIMENT_WORKSPACE,
    MODEL_CACHE,
    experiment_artifact,
)
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig

RERUN_RECORDING = "dataset.rerun"


class RerunExportPass:
    id = "rerun-export"
    name = "Rerun dataset recording"
    provides = frozenset({RERUN_RECORDING})

    def __init__(
        self, config: FourDAnyoneConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.requires = frozenset(
            {
                EXPERIMENT_WORKSPACE,
                MODEL_CACHE,
                experiment_artifact(config.rerun_source_experiment_name),
            }
        )

    def run(self, context: PipelineContext) -> PassResult:
        config = self.config
        result = run_utility(
            "recon_pipeline.artifacts.rerun.utilities.export",
            [
                "--generation",
                str(config.rerun_generation_dir),
                "--output",
                str(config.rerun_path),
                "--experiment",
                config.experiment_name,
                "--fourdanyone-root",
                str(config.fourdanyone_root),
                "--model-dir",
                str(config.model_dir),
                "--view-count",
                str(config.rerun.view_count),
                "--device",
                config.rerun.device,
                "--replace-existing",
            ],
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={RERUN_RECORDING: Path(result["path"])}, details=result
        )
