import json
from pathlib import Path

from recon_pipeline.datasets.fourdanyone.passes import FourDAnyoneInferencePass
from recon_pipeline.reconstructions.nerfstudio.passes import NerfstudioExportPass
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core import PipelineContext


def make_config(tmp_path: Path) -> FourDAnyoneConfig:
    return FourDAnyoneConfig(
        video_path=tmp_path / "in.mov",
        experiment_name="test",
        fourdanyone_root=tmp_path / "4DAnyone",
        model_dir=tmp_path / "data/models",
        runs_dir=tmp_path / "data/runs",
    )


def test_native_inference_progress_maps_to_overall_range(tmp_path: Path) -> None:
    updates = []
    context = PipelineContext()
    context._progress_callback = lambda fraction, message: updates.append(
        (fraction, message)
    )
    pipeline_pass = FourDAnyoneInferencePass(make_config(tmp_path))
    pipeline_pass._handle_line(
        context,
        "FOURDA_PROGRESS " + json.dumps({"fraction": 0.45, "message": "Generating target-view videos"})
    )
    assert updates == [(0.45, "Generating target-view videos")]


def test_export_command_uses_official_exporter(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    command = NerfstudioExportPass(config).build_command(60)
    assert command[1].endswith("scripts/export_nerfstudio.py")
    assert command[command.index("--frame_index") + 1] == "60"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--data_dir") + 1].endswith("runs/test/4danyone")
