import json
from pathlib import Path

from fourda_pipeline.config import FourDAnyoneConfig
from fourda_pipeline.pipeline import FourDAnyonePipeline


def test_native_inference_progress_maps_to_overall_range(tmp_path: Path) -> None:
    updates = []
    pipeline = FourDAnyonePipeline(
        FourDAnyoneConfig(video_path=tmp_path / "in.mov", experiment_name="test"),
        on_progress=updates.append,
    )
    pipeline._handle_inference_line(
        "FOURDA_PROGRESS " + json.dumps({"fraction": 0.45, "message": "Generating target-view videos"})
    )
    assert updates[0].stage == "inference"
    assert updates[0].fraction == 0.3875
    assert updates[0].message == "Generating target-view videos"


def test_export_command_uses_official_exporter(tmp_path: Path) -> None:
    config = FourDAnyoneConfig(
        video_path=tmp_path / "in.mov",
        experiment_name="test",
        fourdanyone_root=tmp_path / "4DAnyone",
        data_root=tmp_path / "data",
    )
    command = FourDAnyonePipeline(config).build_export_command(60)
    assert command[1].endswith("scripts/export_nerfstudio.py")
    assert command[command.index("--frame_index") + 1] == "60"
    assert command[command.index("--device") + 1] == "cuda:0"
