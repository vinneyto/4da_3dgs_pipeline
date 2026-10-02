"""Launch the standalone synchronized frame exporter."""

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import (
    EXPERIMENT_WORKSPACE,
    MODEL_CACHE,
    experiment_artifact,
)
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig

NERFSTUDIO_DATASETS = "dataset.nerfstudio"


class NerfstudioExportPass:
    id = "nerfstudio-export"
    name = "Nerfstudio dataset export"
    provides = frozenset({NERFSTUDIO_DATASETS})

    def __init__(
        self, config: FourDAnyoneConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.requires = frozenset(
            {
                EXPERIMENT_WORKSPACE,
                MODEL_CACHE,
                experiment_artifact(config.nerfstudio_source_experiment_name),
            }
        )

    def arguments(self) -> list[str]:
        config = self.config
        return [
            "--fourdanyone-root",
            str(config.fourdanyone_root),
            "--generation",
            str(config.nerfstudio_generation_dir),
            "--output",
            str(config.datasets_dir),
            "--model-dir",
            str(config.model_dir),
            "--frames",
            *(str(frame) for frame in config.nerfstudio.frames),
            "--device",
            config.nerfstudio.device,
            "--replace-existing",
        ]

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.reconstructions.nerfstudio.utilities.export",
            self.arguments(),
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={NERFSTUDIO_DATASETS: result["datasets"]},
            details={"frames": result["frames"], "count": result["count"]},
        )
