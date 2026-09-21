"""Build a Rerun recording from a recovered 4DAnyone experiment."""

from __future__ import annotations

from recon_pipeline.datasets.fourdanyone.artifacts import (
    EXPERIMENT_WORKSPACE,
    MODEL_CACHE,
    experiment_artifact,
)
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core import PassResult, PipelineContext

from ..exporter import RerunExporter


RERUN_RECORDING = "dataset.rerun"


class RerunExportPass:
    id = "rerun-export"
    name = "Rerun dataset recording"
    provides = frozenset({RERUN_RECORDING})

    def __init__(self, config: FourDAnyoneConfig) -> None:
        self.config = config
        self.requires = frozenset(
            {
                EXPERIMENT_WORKSPACE,
                MODEL_CACHE,
                experiment_artifact(config.rerun_source_experiment_name),
            }
        )

    def cleanup(self, context: PipelineContext) -> None:
        if self.config.rerun_path.exists():
            self.config.rerun_path.unlink()

    def run(self, context: PipelineContext) -> PassResult:
        path = self.config.rerun_path
        def progress(current: int, total: int, message: str) -> None:
            context.report_progress(current / total, message)

        context.report_progress(0.0, "Building camera and skeleton recording")
        path = RerunExporter(
            generation=self.config.rerun_generation_dir,
            output=path,
            experiment=self.config.experiment_name,
            fourdanyone_root=self.config.fourdanyone_root,
            model_dir=self.config.model_dir,
            view_count=self.config.rerun.view_count,
            device=self.config.rerun.device,
            on_progress=progress,
        ).export()
        return PassResult(
            artifacts={RERUN_RECORDING: path},
            details={"path": str(path)},
        )
