"""Launch the standalone workspace preparation utility."""

import json

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from ..artifacts import EXPERIMENT_WORKSPACE
from ..config import FourDAnyoneConfig


class PrepareExperimentPass:
    id = "prepare-experiment"
    name = "Prepare experiment workspace"
    requires = frozenset()
    provides = frozenset({EXPERIMENT_WORKSPACE})

    def __init__(
        self, config: FourDAnyoneConfig, *, runner: CommandRunner | None = None
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()

    def run(self, context: PipelineContext) -> PassResult:
        result = run_utility(
            "recon_pipeline.utilities.datasets.fourdanyone.prepare_experiment",
            [
                "--directory",
                str(self.config.experiment_dir),
                "--settings-output",
                str(self.config.experiment_dir / "pipeline-config.json"),
                "--settings",
                json.dumps(self.config.to_dict()),
            ],
            context,
            runner=self.runner,
        )
        return PassResult(
            artifacts={EXPERIMENT_WORKSPACE: self.config.experiment_dir},
            details={"path": result["path"]},
        )
