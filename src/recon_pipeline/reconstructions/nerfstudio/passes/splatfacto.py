"""Launch one independently checkpointed frame reconstruction."""

from pathlib import Path

from recon_pipeline.core import PassResult
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.utilities.reconstructions.nerfstudio._profile import (
    profile_arguments,
)
from .export import NERFSTUDIO_DATASETS


def splatfacto_artifact(frame):
    return f"reconstruction.splatfacto:frame_{frame:03d}"


class SplatfactoPass:
    requires = frozenset({EXPERIMENT_WORKSPACE, NERFSTUDIO_DATASETS})

    def __init__(self, config, frame, *, runner=None):
        self.config, self.frame, self.runner = config, frame, runner
        self.id = f"splatfacto:frame_{frame:03d}"
        self.name = f"Splatfacto reconstruction frame {frame}"
        self.provides = frozenset({splatfacto_artifact(frame)})

    def run(self, context):
        datasets = context.require(NERFSTUDIO_DATASETS)
        dataset = next(
            item["dataset_dir"] for item in datasets if item["frame"] == self.frame
        )
        settings = self.config.reconstruction
        result = run_utility(
            "recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto",
            [
                "--dataset",
                str(dataset),
                "--output",
                str(self.config.reconstruction_dir / f"frame_{self.frame:03d}"),
                "--nerfstudio-bin",
                settings.nerfstudio_bin,
                "--experiment-name",
                f"{self.config.experiment_name}_frame_{self.frame:03d}",
                "--replace-existing",
                *profile_arguments(settings),
            ],
            context,
            runner=self.runner,
            python=Path(settings.nerfstudio_bin) / "python",
        )
        result["frame"] = self.frame
        return PassResult(
            artifacts={splatfacto_artifact(self.frame): result},
            details={"frame": self.frame, "splat_ply": result["splat_ply"]},
        )
