"""Prepare the persistent workspace for one 4DAnyone experiment."""

from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import EXPERIMENT_WORKSPACE
from ..config import FourDAnyoneConfig


class PrepareExperimentPass:
    id = "prepare-experiment"
    name = "Prepare experiment workspace"
    requires = frozenset()
    provides = frozenset({EXPERIMENT_WORKSPACE})

    def __init__(self, config: FourDAnyoneConfig) -> None:
        self.config = config

    def cleanup(self, context: PipelineContext) -> None:
        config_path = self.config.experiment_dir / "pipeline-config.json"
        if config_path.exists():
            config_path.unlink()

    def run(self, context: PipelineContext) -> PassResult:
        self.config.experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.config.experiment_dir / "pipeline-config.json"
        self.config.write_json(config_path)
        return PassResult(
            artifacts={EXPERIMENT_WORKSPACE: self.config.experiment_dir},
            details={"path": str(self.config.experiment_dir)},
        )
