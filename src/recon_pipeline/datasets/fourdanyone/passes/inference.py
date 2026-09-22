"""Run 4DAnyone inference and expose the recovered experiment artifact."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from recon_pipeline.core.command import CommandRunner
from recon_pipeline.core import PassResult, PipelineContext

from ..artifacts import (
    EXPERIMENT_WORKSPACE,
    INPUT_VIDEO,
    MODEL_CACHE,
    experiment_artifact,
)
from ..config import FourDAnyoneConfig
from ..runner import PROGRESS_PREFIX


class FourDAnyoneInferencePass:
    id = "fourdanyone-inference"
    name = "4DAnyone inference"
    requires = frozenset({EXPERIMENT_WORKSPACE, INPUT_VIDEO, MODEL_CACHE})

    def __init__(
        self,
        config: FourDAnyoneConfig,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.provides = frozenset({experiment_artifact(config.experiment_name)})

    def _request(self) -> dict[str, Any]:
        config = self.config
        return {
            "fourdanyone_root": str(config.fourdanyone_root.resolve()),
            "video_path": str(config.video_path.resolve()),
            "output_dir": str(config.inference_dir.resolve()),
            "views_per_layer": config.views_per_layer,
            "layer_pitches": list(config.layer_pitches),
            "start_yaw": config.start_yaw,
            "yaw_span": config.yaw_span,
            "enable_turbo": config.enable_turbo,
            "model_dir": str(config.model_dir.resolve()),
            "gvhmr_root": str((config.fourdanyone_root / "third_party/GVHMR").resolve()),
            "attention_backend": config.attention_backend,
            "target_fps": config.target_fps,
            "seed": config.seed,
        }

    def build_command(self, request_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "recon_pipeline.datasets.fourdanyone.runner",
            "--request",
            str(request_path),
        ]

    def _handle_line(self, context: PipelineContext, line: str) -> None:
        if not line.startswith(PROGRESS_PREFIX):
            return
        try:
            payload = json.loads(line.removeprefix(PROGRESS_PREFIX))
            fraction = float(payload["fraction"])
            message = str(payload["message"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        context.report_progress(fraction, message)

    def _validate_inputs(self) -> None:
        required = (
            (self.config.video_path, "input video"),
            (self.config.fourdanyone_root / "inference.py", "4DAnyone inference.py"),
            (
                self.config.fourdanyone_root / "third_party/GVHMR/hmr4d/__init__.py",
                "GVHMR submodule",
            ),
        )
        missing = [f"{label}: {path}" for path, label in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required paths:\n" + "\n".join(f" - {item}" for item in missing)
            )
        if not self.config.model_dir.is_dir():
            raise FileNotFoundError(
                f"model directory does not exist: {self.config.model_dir}"
            )

    def cleanup(self, context: PipelineContext) -> None:
        if self.config.inference_dir.exists():
            shutil.rmtree(self.config.inference_dir)
        request_path = self.config.experiment_dir / "inference-request.json"
        if request_path.exists():
            request_path.unlink()

    def run(self, context: PipelineContext) -> PassResult:
        self._validate_inputs()
        metadata = self.config.inference_dir / "metadata.json"
        request_path = self.config.experiment_dir / "inference-request.json"
        request_path.write_text(
            json.dumps(self._request(), indent=2, sort_keys=True) + "\n"
        )
        self.runner.run(
            self.build_command(request_path),
            cwd=self.config.fourdanyone_root,
            on_line=lambda line: self._handle_line(context, line),
        )
        if not metadata.is_file():
            raise RuntimeError(
                f"4DAnyone completed without expected metadata: {metadata}"
            )
        return PassResult(
            artifacts={
                experiment_artifact(self.config.experiment_name): self.config.inference_dir
            },
            details={
                "experiment": self.config.experiment_name,
                "views": self.config.num_views,
                "views_per_layer": self.config.views_per_layer,
                "layer_pitches": list(self.config.layer_pitches),
                "path": str(self.config.inference_dir),
            },
        )
