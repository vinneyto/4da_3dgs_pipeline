"""Export the trained Gaussian splats for one timestamp."""

from __future__ import annotations

import shutil
from pathlib import Path

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig

from .train import splatfacto_environment, training_artifact


def splat_artifact(frame: int) -> str:
    return f"reconstruction.ply:frame_{frame:03d}"


def read_splat_count(path: Path) -> int:
    with path.open("rb") as source:
        first_bytes = source.read(65_536)
    if b"end_header\n" not in first_bytes:
        raise ValueError(f"PLY header is incomplete: {path}")
    header = first_bytes.split(b"end_header\n", 1)[0].decode("ascii")
    required = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    if not all(f"property float {field}" in header for field in required):
        raise ValueError(f"not a Gaussian PLY with SH colors: {path}")
    for line in header.splitlines():
        if line.startswith("element vertex "):
            count = int(line.removeprefix("element vertex "))
            if count > 0:
                return count
    raise ValueError(f"Gaussian PLY contains no vertices: {path}")


class GaussianSplatExportPass:
    name = "Export Gaussian splats"

    def __init__(self, config: FourDAnyoneConfig, frame: int, *, runner: CommandRunner | None = None):
        self.config = config
        self.frame = frame
        self.id = f"gaussian-splat-export:frame_{frame:03d}"
        self.requires = frozenset({training_artifact(frame)})
        self.provides = frozenset({splat_artifact(frame)})
        self.runner = runner or CommandRunner()

    def cleanup(self, context: PipelineContext) -> None:
        directory = self.config.splatfacto_dir(self.frame) / "export"
        if directory.exists():
            shutil.rmtree(directory)

    def run(self, context: PipelineContext) -> PassResult:
        assert self.config.splatfacto_env is not None
        executable = self.config.splatfacto_env / "bin/ns-export"
        if not executable.is_file():
            raise FileNotFoundError(executable)
        config_path = Path(context.require(training_artifact(self.frame)))
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        directory = self.config.splatfacto_dir(self.frame) / "export"
        directory.mkdir(parents=True, exist_ok=True)
        self.runner.run(
            [str(executable), "gaussian-splat", "--load-config", str(config_path),
             "--output-dir", str(directory), "--output-filename", "splat.ply",
             "--ply-color-mode", "sh_coeffs"],
            cwd=directory, env=splatfacto_environment(self.config),
        )
        ply = directory / "splat.ply"
        count = read_splat_count(ply)
        context.report_progress(1.0, f"Exported {count} Gaussians for frame {self.frame}")
        return PassResult(
            artifacts={splat_artifact(self.frame): str(ply)},
            details={"frame": self.frame, "path": str(ply), "gaussians": count, "bytes": ply.stat().st_size},
        )
