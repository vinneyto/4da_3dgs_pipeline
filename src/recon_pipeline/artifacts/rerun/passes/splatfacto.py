"""Launch the optional recording exporter for one trained static frame."""

from recon_pipeline.core import PassResult
from recon_pipeline.core.utility import run_utility
from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import (
    splatfacto_artifact,
)


def splatfacto_recording_artifact(frame):
    return f"recording.splatfacto:frame_{frame:03d}"


class SplatfactoRerunPass:
    def __init__(self, config, frame, *, runner=None):
        self.config, self.frame, self.runner = config, frame, runner
        self.id = f"splatfacto-rerun:frame_{frame:03d}"
        self.name = f"Rerun reconstruction frame {frame}"
        self.requires = frozenset({splatfacto_artifact(frame)})
        self.provides = frozenset({splatfacto_recording_artifact(frame)})

    def run(self, context):
        trained = context.require(splatfacto_artifact(self.frame))
        settings = self.config.reconstruction_rerun
        result = run_utility(
            "recon_pipeline.utilities.artefacts.rerun.splatfacto",
            [
                "--splat",
                trained["splat_ply"],
                "--dataset",
                trained["dataset_dir"],
                "--run-root",
                trained["run_root"],
                "--output",
                str(
                    self.config.reconstruction_dir
                    / f"frame_{self.frame:03d}/rerun/reconstruction.rrd"
                ),
                "--experiment-name",
                trained["experiment_name"],
                "--run-timestamp",
                trained["run_timestamp"],
                "--max-splats",
                str(settings.max_splats),
                "--view-count",
                str(settings.view_count),
                "--replace-existing",
            ],
            context,
            runner=self.runner,
            python=settings.python,
        )
        result["frame"] = self.frame
        return PassResult(
            artifacts={splatfacto_recording_artifact(self.frame): result},
            details=result,
        )
