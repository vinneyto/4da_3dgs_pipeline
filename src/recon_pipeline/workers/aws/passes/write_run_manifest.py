"""Launch the standalone run manifest writer."""

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.reconstructions.nerfstudio.passes import NERFSTUDIO_DATASETS
from recon_pipeline.artifacts.rerun.passes import RERUN_RECORDING
from ..artifacts import RUN_RESULT


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

    def run(self, context: PipelineContext) -> PassResult:
        arguments = (
            ["--rerun-file", str(context.artifacts[RERUN_RECORDING])]
            if RERUN_RECORDING in context.artifacts
            else []
        )
        result = run_utility(
            "recon_pipeline.workers.aws.utilities.write_run_manifest",
            arguments,
            context,
            runner=self.runner,
            documents={
                "--pipeline-config": self.config.to_dict(),
                "--datasets": context.artifacts.get(NERFSTUDIO_DATASETS, []),
                "--durations": dict(context.values.get("pass_durations", {})),
            },
        )
        return PassResult(
            artifacts={RUN_RESULT: result["manifest"]}, details={"path": result["path"]}
        )
