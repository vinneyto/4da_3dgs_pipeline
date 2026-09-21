"""Export synchronized 4DAnyone frames as Nerfstudio datasets."""

from __future__ import annotations

import shutil
import sys

from recon_pipeline.datasets.fourdanyone.artifacts import (
    EXPERIMENT_WORKSPACE,
    MODEL_CACHE,
    experiment_artifact,
)
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core import PassResult, PipelineContext


NERFSTUDIO_DATASETS = "dataset.nerfstudio"


class NerfstudioExportPass:
    id = "nerfstudio-export"
    name = "Nerfstudio dataset export"
    provides = frozenset({NERFSTUDIO_DATASETS})

    def __init__(
        self,
        config: FourDAnyoneConfig,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.requires = frozenset(
            {
                EXPERIMENT_WORKSPACE,
                MODEL_CACHE,
                experiment_artifact(config.nerfstudio_source_experiment_name),
            }
        )

    def build_command(self, frame: int) -> list[str]:
        config = self.config
        return [
            sys.executable,
            str(config.fourdanyone_root / "scripts/export_nerfstudio.py"),
            "--data_dir",
            str(config.nerfstudio_generation_dir),
            "--output_dir",
            str(config.dataset_dir(frame)),
            "--frame_index",
            str(frame),
            "--model_dir",
            str(config.model_dir),
            "--device",
            config.nerfstudio.device,
        ]

    def _validate_inputs(self) -> None:
        required = (
            (
                self.config.fourdanyone_root / "scripts/export_nerfstudio.py",
                "Nerfstudio exporter",
            ),
            (self.config.nerfstudio_generation_dir / "metadata.json", "source metadata"),
            (self.config.nerfstudio_generation_dir / "cameras.json", "source cameras"),
        )
        missing = [f"{label}: {path}" for path, label in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required paths:\n" + "\n".join(f" - {item}" for item in missing)
            )

    def cleanup(self, context: PipelineContext) -> None:
        for frame in self.config.nerfstudio.frames:
            destination = self.config.dataset_dir(frame)
            if destination.exists():
                shutil.rmtree(destination)

    def run(self, context: PipelineContext) -> PassResult:
        self._validate_inputs()
        datasets: list[dict[str, object]] = []
        frames = self.config.nerfstudio.frames
        for offset, frame in enumerate(frames):
            destination = self.config.dataset_dir(frame)
            transforms = destination / "transforms.json"
            self.runner.run(
                self.build_command(frame), cwd=self.config.fourdanyone_root
            )
            if not transforms.is_file():
                raise RuntimeError(
                    f"export completed without transforms.json: {destination}"
                )
            message = f"Exported synchronized frame {frame}"
            context.report_progress((offset + 1) / len(frames), message)
            datasets.append({"frame": frame, "dataset_dir": str(destination)})
        return PassResult(
            artifacts={NERFSTUDIO_DATASETS: datasets},
            details={"frames": list(frames), "count": len(datasets)},
        )
