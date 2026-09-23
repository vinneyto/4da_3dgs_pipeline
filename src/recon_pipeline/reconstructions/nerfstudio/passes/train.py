"""Train stock Nerfstudio Splatfacto on one exported RGBA dataset."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PIL import Image

from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig

from .export import NERFSTUDIO_DATASETS


def training_artifact(frame: int) -> str:
    return f"reconstruction.splatfacto:frame_{frame:03d}"


def splatfacto_environment(config: FourDAnyoneConfig) -> dict[str, str]:
    assert config.splatfacto_env is not None
    environment = os.environ.copy()
    environment["PATH"] = f"{config.splatfacto_env / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


class SplatfactoTrainPass:
    name = "Train stock Splatfacto"
    requires = frozenset({EXPERIMENT_WORKSPACE, NERFSTUDIO_DATASETS})

    def __init__(self, config: FourDAnyoneConfig, frame: int, *, runner: CommandRunner | None = None):
        self.config = config
        self.frame = frame
        self.id = f"splatfacto-train:frame_{frame:03d}"
        self.provides = frozenset({training_artifact(frame)})
        self.runner = runner or CommandRunner()

    def cleanup(self, context: PipelineContext) -> None:
        directory = self.config.splatfacto_dir(self.frame) / "train"
        if directory.exists():
            shutil.rmtree(directory)

    def _dataset(self, context: PipelineContext) -> Path:
        entries = context.require(NERFSTUDIO_DATASETS)
        match = [entry for entry in entries if entry["frame"] == self.frame]
        if len(match) != 1:
            raise ValueError(f"dataset export did not produce frame {self.frame}")
        dataset = Path(match[0]["dataset_dir"])
        transforms = dataset / "transforms.json"
        payload = json.loads(transforms.read_text())
        frames = payload.get("frames", [])
        if not frames or "mask_path" in payload or any("mask_path" in item for item in frames):
            raise ValueError(f"frame {self.frame} requires RGBA images without mask_path")
        point_cloud = payload.get("ply_file_path")
        if not point_cloud or not (dataset / point_cloud).is_file():
            raise ValueError(f"frame {self.frame} requires an existing initial point cloud")
        for item in frames:
            image_path = dataset / str(item["file_path"])
            with Image.open(image_path) as image:
                if image.mode != "RGBA":
                    raise ValueError(f"Splatfacto frame {self.frame} requires RGBA: {image_path}")
        return dataset

    def build_command(self, dataset: Path) -> list[str]:
        assert self.config.splatfacto_env is not None
        settings = self.config.reconstruction.training
        assert settings is not None
        destination = self.config.splatfacto_dir(self.frame) / "train"
        command = [
            str(self.config.splatfacto_env / "bin/ns-train"), "splatfacto",
            "--data", str(dataset),
            "--output-dir", str(destination),
            "--experiment-name", self.config.experiment_name,
            "--timestamp", f"frame_{self.frame:03d}",
            "--max-num-iterations", str(settings.max_num_iterations),
            "--vis", settings.vis,
        ]
        for name in (
            "background_color", "stop_split_at", "cull_alpha_thresh",
            "densify_grad_thresh", "densify_size_thresh", "split_screen_size",
            "num_downscales", "resolution_schedule", "cull_scale_thresh",
            "stop_screen_size_at", "use_scale_regularization", "max_gauss_ratio",
            "sh_degree", "rasterize_mode",
        ):
            value = getattr(settings, name)
            command.extend((f"--pipeline.model.{name.replace('_', '-')}", str(value)))
        command.extend(("nerfstudio-data", "--eval-mode", settings.eval_mode))
        return command

    def run(self, context: PipelineContext) -> PassResult:
        dataset = self._dataset(context)
        assert self.config.splatfacto_env is not None
        executable = self.config.splatfacto_env / "bin/ns-train"
        if not executable.is_file():
            raise FileNotFoundError(
                f"{executable}; set up the separate Splatfacto environment before running"
            )
        directory = self.config.splatfacto_dir(self.frame)
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "train.log"
        context.report_progress(0.0, f"Training frame {self.frame}")
        with log.open("w", encoding="utf-8") as output:
            self.runner.run(
                self.build_command(dataset), cwd=directory,
                env=splatfacto_environment(self.config),
                on_line=lambda line: (output.write(line + "\n"), output.flush()),
            )
        config_path = (
            directory / "train" / self.config.experiment_name / "splatfacto"
            / f"frame_{self.frame:03d}" / "config.yml"
        )
        if not config_path.is_file():
            raise RuntimeError(f"Splatfacto completed without config.yml: {config_path}")
        context.report_progress(1.0, f"Frame {self.frame} trained")
        return PassResult(
            artifacts={training_artifact(self.frame): str(config_path)},
            details={"frame": self.frame, "config": str(config_path), "log": str(log)},
        )
