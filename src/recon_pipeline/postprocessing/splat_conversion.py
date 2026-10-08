"""Postprocess completed Gaussian artifacts with independent checkpoints."""

from recon_pipeline.core import PassResult
from recon_pipeline.core.utility import run_utility
from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import (
    splatfacto_artifact,
)


def converted_splat_artifact(frame):
    return f"postprocessing.splat_conversion:frame_{frame:03d}"


class SplatConversionPass:
    def __init__(self, config, frame, *, runner=None):
        self.config, self.frame, self.runner = config, frame, runner
        self.id = f"splat-convert:frame_{frame:03d}"
        self.name = f"Convert Gaussian artifacts frame {frame}"
        self.requires = frozenset({splatfacto_artifact(frame)})
        self.provides = frozenset({converted_splat_artifact(frame)})

    @property
    def output_dir(self):
        return (
            self.config.experiment_dir
            / "postprocessing/splat_conversion"
            / f"frame_{self.frame:03d}"
        )

    def run(self, context):
        source = context.require(splatfacto_artifact(self.frame))
        settings = self.config.postprocessing.splat_conversion
        result = run_utility(
            "recon_pipeline.utilities.artefacts.splats.convert",
            [
                "--input",
                source["splat_ply"],
                "--output",
                str(self.output_dir),
                "--formats",
                *settings.formats,
                "--replace-existing",
                *(
                    ["--splat-transform", settings.splat_transform]
                    if settings.splat_transform
                    else []
                ),
            ],
            context,
            runner=self.runner,
        )
        result["frame"] = self.frame
        return PassResult(
            artifacts={converted_splat_artifact(self.frame): result},
            details={"frame": self.frame},
        )
