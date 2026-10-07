"""Launch the standalone 4DAnyone dataset utility."""

from __future__ import annotations

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import handle_progress, run_utility
from ..artifacts import (
    EXPERIMENT_WORKSPACE,
    INPUT_VIDEO,
    MODEL_CACHE,
    experiment_artifact,
)
from ..config import FourDAnyoneConfig


class FourDAnyoneInferencePass:
    id = "fourdanyone-inference"
    name = "4DAnyone inference"
    requires = frozenset({EXPERIMENT_WORKSPACE, INPUT_VIDEO, MODEL_CACHE})

    def __init__(
        self, config: FourDAnyoneConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.provides = frozenset({experiment_artifact(config.experiment_name)})

    def arguments(self) -> list[str]:
        config = self.config
        return [
            "--fourdanyone-root",
            str(config.fourdanyone_root),
            "--video",
            str(config.video_path),
            "--output",
            str(config.inference_dir),
            "--model-dir",
            str(config.model_dir),
            "--views-per-layer",
            str(config.views_per_layer),
            "--layer-pitches",
            *(str(pitch) for pitch in config.layer_pitches),
            "--start-yaw",
            str(config.start_yaw),
            "--yaw-span",
            str(config.yaw_span),
            "--target-fps",
            str(config.target_fps),
            "--seed",
            str(config.seed),
            "--turbo" if config.enable_turbo else "--no-turbo",
            "--attention-backend",
            config.attention_backend,
            "--replace-existing",
        ]

    def _handle_line(self, context: PipelineContext, line: str) -> None:
        handle_progress(context, line)

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.utilities.datasets.fourdanyone.inference",
            self.arguments(),
            context,
            runner=self.runner,
            python=self.config.python,
        )
        return PassResult(
            artifacts={
                experiment_artifact(
                    self.config.experiment_name
                ): self.config.inference_dir
            },
            details={
                "experiment": self.config.experiment_name,
                "views": self.config.num_views,
                "views_per_layer": self.config.views_per_layer,
                "layer_pitches": list(self.config.layer_pitches),
                "path": result["path"],
            },
        )
