"""Blocking 4DAnyone -> Nerfstudio/3DGS pipeline."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .command import CommandRunner
from .config import FourDAnyoneConfig
from .fdanyone_runner import PROGRESS_PREFIX
from .progress import ProgressUpdate


ProgressCallback = Callable[[ProgressUpdate], None]


class FourDAnyonePipeline:
    """Generate multi-view video and export synchronized static 3DGS datasets.

    The default camera layout mirrors the validated Colab experiment:
    24 yaw views at pitches -15, 0 and 15 degrees (72 cameras total), a full
    360-degree orbit, 30 FPS and seed 42. By default only frame 60 is exported
    to a static Nerfstudio dataset to keep disk use bounded.
    """

    def __init__(
        self,
        config: FourDAnyoneConfig,
        *,
        on_progress: ProgressCallback | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.on_progress = on_progress or (lambda _update: None)
        self.runner = runner or CommandRunner()

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        self.on_progress(ProgressUpdate(stage=stage, fraction=fraction, message=message))

    def _inference_request(self) -> dict[str, Any]:
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

    def build_inference_command(self, request_path: Path) -> list[str]:
        return [sys.executable, "-m", "fourda_pipeline.fdanyone_runner", "--request", str(request_path)]

    def build_export_command(self, frame_index: int) -> list[str]:
        config = self.config
        return [
            sys.executable,
            str(config.fourdanyone_root / "scripts/export_nerfstudio.py"),
            "--data_dir",
            str(config.inference_dir),
            "--output_dir",
            str(config.dataset_dir(frame_index)),
            "--frame_index",
            str(frame_index),
            "--model_dir",
            str(config.model_dir),
            "--device",
            config.export_device,
        ]

    def _handle_inference_line(self, line: str) -> None:
        if not line.startswith(PROGRESS_PREFIX):
            return
        try:
            payload = json.loads(line.removeprefix(PROGRESS_PREFIX))
            native_fraction = float(payload["fraction"])
            message = str(payload["message"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        # Inference dominates runtime, so it owns 5%-80% of overall progress.
        overall = 0.05 + min(max(native_fraction, 0.0), 1.0) * 0.75
        self._progress("inference", overall, message)

    def _run_inference(self) -> None:
        config = self.config
        metadata = config.inference_dir / "metadata.json"
        if config.resume and metadata.is_file():
            self._progress("inference", 0.80, "Reusing completed 4DAnyone inference")
            return
        if config.inference_dir.exists():
            raise FileExistsError(
                f"inference output already exists: {config.inference_dir}; "
                "use --resume to reuse a completed inference"
            )
        request_path = config.experiment_dir / "inference-request.json"
        request_path.write_text(json.dumps(self._inference_request(), indent=2, sort_keys=True) + "\n")
        self.runner.run(
            self.build_inference_command(request_path),
            cwd=config.fourdanyone_root,
            on_line=self._handle_inference_line,
        )
        if not metadata.is_file():
            raise RuntimeError(f"4DAnyone completed without expected metadata: {metadata}")

    def _export_datasets(self) -> list[dict[str, Any]]:
        config = self.config
        results: list[dict[str, Any]] = []
        count = len(config.frame_indices)
        for offset, frame_index in enumerate(config.frame_indices):
            destination = config.dataset_dir(frame_index)
            transforms = destination / "transforms.json"
            if config.resume and transforms.is_file():
                message = f"Reusing exported frame {frame_index}"
            else:
                if destination.exists():
                    raise FileExistsError(
                        f"dataset output already exists: {destination}; use --resume to reuse it"
                    )
                self.runner.run(
                    self.build_export_command(frame_index),
                    cwd=config.fourdanyone_root,
                )
                if not transforms.is_file():
                    raise RuntimeError(f"export completed without transforms.json: {destination}")
                message = f"Exported synchronized frame {frame_index}"
            export_span = 0.10 if config.rerun.enabled else 0.19
            fraction = 0.80 + ((offset + 1) / count) * export_span
            self._progress("export", fraction, message)
            results.append({"frame_index": frame_index, "dataset_dir": str(destination)})
        return results

    def _export_rerun(self) -> str | None:
        config = self.config
        if not config.rerun.enabled:
            return None
        if config.resume and config.rerun_path.is_file():
            self._progress("rerun", 0.99, "Reusing existing Rerun recording")
            return str(config.rerun_path)
        if config.rerun_path.exists():
            if config.rerun.replace_existing:
                config.rerun_path.unlink()
            else:
                raise FileExistsError(
                    f"Rerun output already exists: {config.rerun_path}; "
                    "set artifacts.dataset.rerun.replace_existing=true to replace it"
                )

        from fourda_rerun.exporter import RerunExporter

        def on_rerun_progress(current: int, total: int, message: str) -> None:
            fraction = 0.90 + (current / total) * 0.09
            self._progress("rerun", fraction, message)

        self._progress("rerun", 0.90, "Building camera and skeleton recording")
        output = RerunExporter(
            generation=config.rerun_generation_dir,
            output=config.rerun_path,
            experiment=config.experiment_name,
            fourdanyone_root=config.fourdanyone_root,
            model_dir=config.model_dir,
            view_count=config.rerun.view_count,
            device=config.rerun.device,
            on_progress=on_rerun_progress,
        ).export()
        return str(output)

    def run(self) -> dict[str, Any]:
        self._progress("validation", 0.01, "Validating pipeline configuration")
        self.config.validate_paths()
        self.config.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config.write_json(self.config.experiment_dir / "pipeline-config.json")
        self._progress("validation", 0.05, f"Camera plan: {self.config.num_views} views")

        started = datetime.now(UTC)
        if self.config.dataset_enabled:
            self._run_inference()
            datasets = self._export_datasets()
        else:
            datasets = []
            self._progress("dataset", 0.80, "Dataset stage disabled")
        rerun_file = self._export_rerun()
        finished = datetime.now(UTC)

        result = {
            "experiment_name": self.config.experiment_name,
            "experiment_dir": str(self.config.experiment_dir),
            "inference_dir": str(self.config.inference_dir),
            "datasets": datasets,
            "rerun_file": rerun_file,
            "num_views": self.config.num_views,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
        }
        result_path = self.config.experiment_dir / "pipeline-result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        self._progress("complete", 1.0, "Pipeline completed")
        return result
