"""Launch the standalone run manifest writer."""

import json

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.reconstructions.nerfstudio.passes import NERFSTUDIO_DATASETS
from recon_pipeline.artifacts.rerun.passes import RERUN_RECORDING
from ..artifacts import RUN_RESULT
from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import (
    splatfacto_artifact,
)
from recon_pipeline.artifacts.rerun.passes.splatfacto import (
    splatfacto_recording_artifact,
)


class WriteRunManifestPass:
    id = "write-run-manifest"
    name = "Write run manifest"
    requires = frozenset({EXPERIMENT_WORKSPACE})
    provides = frozenset({RUN_RESULT})

    def __init__(
        self, config: FourDAnyoneConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.requires = frozenset(
            {EXPERIMENT_WORKSPACE}
            | (
                {splatfacto_artifact(frame) for frame in config.reconstruction_frames}
                if config.reconstruction.enabled
                else set()
            )
            | (
                {
                    splatfacto_recording_artifact(frame)
                    for frame in config.reconstruction_frames
                }
                if config.reconstruction_rerun.enabled
                else set()
            )
        )

    def run(self, context: PipelineContext) -> PassResult:
        arguments = (
            ["--rerun-file", str(context.artifacts[RERUN_RECORDING])]
            if RERUN_RECORDING in context.artifacts
            else []
        )
        arguments.extend(
            [
                "--output",
                str(self.config.experiment_dir / "pipeline-result.json"),
                "--experiment-name",
                self.config.experiment_name,
                "--experiment-dir",
                str(self.config.experiment_dir),
                "--inference-dir",
                str(self.config.inference_dir),
                "--num-views",
                str(self.config.num_views),
                "--datasets",
                json.dumps(context.artifacts.get(NERFSTUDIO_DATASETS, [])),
                "--reconstructions",
                json.dumps(
                    [
                        value
                        for key, value in context.artifacts.items()
                        if key.startswith("reconstruction.splatfacto:")
                    ]
                ),
                "--reconstruction-recordings",
                json.dumps(
                    [
                        value
                        for key, value in context.artifacts.items()
                        if key.startswith("recording.splatfacto:")
                    ]
                ),
                "--durations",
                json.dumps(dict(context.values.get("pass_durations", {}))),
            ]
        )
        result = run_utility(
            "recon_pipeline.utilities.artefacts.manifest.write",
            arguments,
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={RUN_RESULT: result["manifest"]}, details={"path": result["path"]}
        )
