import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recon_pipeline.core import JsonPassCheckpointStore, Pipeline, PipelineContext
from recon_pipeline.core.command import CommandError
from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.reconstructions.nerfstudio.passes import NERFSTUDIO_DATASETS
from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import (
    SplatfactoPass,
    splatfacto_artifact,
)
from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import SplatfactoConfig
from recon_pipeline.utilities.reconstructions.nerfstudio._profile import TrainingProfile
from recon_pipeline.utilities.reconstructions.nerfstudio.prepare_rgba import prepare
from recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto import validate_ply
from recon_pipeline.workers.aws.passes import WriteRunManifestPass
from test_aws_pipeline_plan import make_pipeline_config


def dataset(root):
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "masks").mkdir()
    cameras = []
    for index in range(4):
        name = f"camera_{index}"
        Image.fromarray(np.full((5, 5, 3), 128, np.uint8)).save(
            root / f"images/{name}.jpg"
        )
        alpha = np.zeros((5, 5), np.uint8)
        alpha[1:4, 1:4] = 255
        Image.fromarray(alpha).save(root / f"masks/{name}.png")
        transform = np.eye(4)
        transform[:3, 3] = [np.cos(index * np.pi / 2), np.sin(index * np.pi / 2), 0]
        cameras.append(
            {
                "file_path": f"images/{name}.jpg",
                "mask_path": f"masks/{name}.png",
                "transform_matrix": transform.tolist(),
            }
        )
    payload = {
        "frames": cameras,
        "w": 5,
        "h": 5,
        "fl_x": 4,
        "fl_y": 4,
        "cx": 2.5,
        "cy": 2.5,
        "ply_file_path": "points.ply",
    }
    (root / "points.ply").write_text("seed point cloud")
    (root / "transforms.json").write_text(json.dumps(payload))
    return root


@pytest.fixture
def binaries(tmp_path):
    root = tmp_path / "external environment/bin"
    root.mkdir(parents=True)
    (root / "python").write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    (root / "python").chmod(0o755)
    script = """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def option(name): return args[args.index(name)+1]
root = Path(__file__).parent
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"tool": Path(__file__).name, "args": args}) + "\\n")
if Path(__file__).name == "ns-train":
    assert option("--logging.local-writer.max-log-size") == "0"
    assert option("--pipeline.model.background-color") == "random"
    assert option("--eval-mode") == "all"
    assert option("--pipeline.model.rasterize-mode") == "classic"
    assert args[-3] == "nerfstudio-data"
    payload = json.loads((Path(option("--data")) / "transforms.json").read_text())
    assert all("mask_path" not in camera for camera in payload["frames"])
    if os.environ.get("FAKE_FAIL_FRAME", "NONE") in option("--experiment-name"):
        print("fixture training failed", file=sys.stderr)
        sys.exit(7)
    destination = Path(option("--output-dir")) / option("--experiment-name") / "splatfacto" / option("--timestamp")
    destination.mkdir(parents=True)
    (destination / "config.yml").write_text("fixture config")
    (destination / "dataparser_transforms.json").write_text(json.dumps({"transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0]], "scale": 1}))
    (destination / "checkpoints").mkdir()
    (destination / "checkpoints/step-000059999.ckpt").write_text("fixture weights")
    print("7 (0.01%)  training progress")
    print("library diagnostic", file=sys.stderr)
else:
    import numpy as np
    from plyfile import PlyData, PlyElement
    assert option("--ply-color-mode") == "sh_coeffs"
    fields = ["x","y","z","opacity"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(24)] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    vertices = np.zeros(8, dtype=[(name, "f4") for name in fields])
    vertices["x"] = np.arange(8) * 0.01
    vertices["rot_0"] = 1
    vertices["opacity"] = 1
    for i in range(3): vertices[f"scale_{i}"] = -3
    destination = Path(option("--output-dir"))
    destination.mkdir()
    PlyData([PlyElement.describe(vertices, "vertex")]).write(destination / option("--output-filename"))
"""
    for name in ("ns-train", "ns-export"):
        path = root / name
        path.write_text(f"#!{sys.executable}\n" + script)
        path.chmod(0o755)
    return root


def test_rgba_preparation_preserves_source_and_matches_colab(tmp_path):
    source = dataset(tmp_path / "source")
    original = (source / "transforms.json").read_bytes()
    result = prepare(source, tmp_path / "training")
    assert result["camera_count"] == 4
    assert (source / "transforms.json").read_bytes() == original
    assert Image.open(source / "images/camera_0.jpg").mode == "RGB"
    prepared = json.loads((tmp_path / "training/transforms.json").read_text())
    camera = prepared["frames"][0]
    assert "mask_path" not in camera
    assert (
        camera["transform_matrix"]
        == json.loads(original)["frames"][0]["transform_matrix"]
    )
    rgba = np.asarray(Image.open(tmp_path / "training" / camera["file_path"]))
    assert rgba.shape == (5, 5, 4)
    assert np.count_nonzero(rgba[..., 3]) == 1  # Colab's 1px erosion of a 3x3 mask.
    assert (tmp_path / "training/points.ply").read_bytes() == (
        source / "points.ply"
    ).read_bytes()


@pytest.mark.parametrize("problem", ["mask", "empty", "escape"])
def test_rgba_rejects_invalid_dataset(tmp_path, problem):
    source = dataset(tmp_path / "source")
    if problem == "mask":
        (source / "masks/camera_0.png").unlink()
    elif problem == "empty":
        Image.fromarray(np.zeros((5, 5), np.uint8)).save(source / "masks/camera_0.png")
    else:
        payload = json.loads((source / "transforms.json").read_text())
        payload["frames"][0]["file_path"] = "../outside.png"
        (source / "transforms.json").write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(source, tmp_path / "training")


def test_training_cli_uses_profile_and_only_protocol_on_stdout(tmp_path, binaries):
    source = dataset(tmp_path / "source")
    command = [
        sys.executable,
        "-m",
        "recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto",
        "--dataset",
        str(source),
        "--output",
        str(tmp_path / "result"),
        "--nerfstudio-bin",
        str(binaries),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    result = events[-1]["data"]
    assert result["training_profile"] == asdict(TrainingProfile())
    assert result["full_splat_count"] == 8
    assert validate_ply(Path(result["splat_ply"]), 2) == 8
    assert not any("iteration 7" in event.get("message", "") for event in events)
    assert "library diagnostic" in completed.stderr
    assert Path(result["dataparser_transform"]).is_file()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "--replace-existing" in again.stderr
    (tmp_path / "result/stale").touch()
    replaced = subprocess.run(
        [*command, "--replace-existing"], capture_output=True, text=True
    )
    assert replaced.returncode == 0, replaced.stderr
    assert not (tmp_path / "result/stale").exists()


def test_frame_checkpoints_resume_after_failure_and_manifest_contains_models(
    tmp_path, binaries, monkeypatch
):
    config = make_pipeline_config(
        tmp_path,
        nerfstudio=NerfstudioArtifactConfig(frames=(30, 60)),
        reconstruction=SplatfactoConfig(enabled=True, nerfstudio_bin=str(binaries)),
    )
    datasets = [
        {"frame": frame, "dataset_dir": str(dataset(tmp_path / f"input/frame_{frame}"))}
        for frame in (30, 60)
    ]

    def context():
        return PipelineContext(
            artifacts={
                EXPERIMENT_WORKSPACE: config.experiment_dir,
                NERFSTUDIO_DATASETS: datasets,
            }
        )

    passes = [SplatfactoPass(config, frame) for frame in (30, 60)] + [
        WriteRunManifestPass(config)
    ]
    store = JsonPassCheckpointStore(tmp_path / "state.json", config.experiment_name)
    monkeypatch.setenv("FAKE_FAIL_FRAME", "frame_060")
    with pytest.raises(CommandError) as error:
        Pipeline(passes, checkpoint_store=store).run(context())
    assert "fixture training failed" in error.value.output_tail
    assert list(store.load()) == ["splatfacto:frame_030"]
    monkeypatch.delenv("FAKE_FAIL_FRAME")
    result = context()
    Pipeline(passes, checkpoint_store=store).run(result)
    calls = [
        json.loads(line) for line in (binaries / "calls.jsonl").read_text().splitlines()
    ]
    assert (
        sum(
            call["tool"] == "ns-train" and "frame_030" in " ".join(call["args"])
            for call in calls
        )
        == 1
    )
    manifest = json.loads((config.experiment_dir / "pipeline-result.json").read_text())
    assert [item["frame"] for item in manifest["reconstructions"]] == [30, 60]
    assert all(
        Path(item["splat_ply"]).is_file() for item in manifest["reconstructions"]
    )
    Pipeline(passes, checkpoint_store=store, force=True).run(context())
    assert len(store.load()) == 3


def test_rerun_reconstruction_exports_real_recording_and_preserves_full_ply(
    tmp_path, binaries
):
    pytest.importorskip("rerun")
    pytest.importorskip("tensorboard")
    from recon_pipeline.artifacts.rerun.passes.splatfacto import (
        SplatfactoRerunPass,
        splatfacto_recording_artifact,
    )
    from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import (
        SplatfactoRerunConfig,
    )
    from plyfile import PlyData

    config = make_pipeline_config(
        tmp_path,
        reconstruction=SplatfactoConfig(enabled=True, nerfstudio_bin=str(binaries)),
        reconstruction_rerun=SplatfactoRerunConfig(
            enabled=True, python=sys.executable, max_splats=3
        ),
    )
    source = dataset(tmp_path / "input")
    context = PipelineContext(
        artifacts={
            EXPERIMENT_WORKSPACE: config.experiment_dir,
            NERFSTUDIO_DATASETS: [{"frame": 60, "dataset_dir": str(source)}],
        }
    )
    Pipeline(
        [
            SplatfactoPass(config, 60),
            SplatfactoRerunPass(config, 60),
            WriteRunManifestPass(config),
        ]
    ).run(context)
    trained = context.artifacts[splatfacto_artifact(60)]
    recording = context.artifacts[splatfacto_recording_artifact(60)]
    assert Path(recording["rerun_path"]).stat().st_size > 0
    assert recording["visualized_splats"] == 3
    assert recording["full_splat_count"] == 8
    assert len(PlyData.read(trained["splat_ply"])["vertex"].data) == 8
    assert len(recording["selected_camera_indices"]) == 4
    manifest = json.loads((config.experiment_dir / "pipeline-result.json").read_text())
    assert manifest["reconstruction_recordings"][0]["frame"] == 60


@pytest.mark.parametrize(
    "changes",
    [
        {"max_num_iterations": 0},
        {"mask_threshold": 256},
        {"sh_degree": 4},
        {"densify_grad_thresh": float("nan")},
        {"frames": [60, 60]},
        {"nerfstudio_bin": "relative"},
    ],
)
def test_invalid_reconstruction_config_is_rejected(changes):
    with pytest.raises(ValueError):
        SplatfactoConfig(**changes)


def test_reconstruction_selection_must_be_exported_frames(tmp_path):
    with pytest.raises(ValueError, match="exported"):
        make_pipeline_config(
            tmp_path, reconstruction=SplatfactoConfig(enabled=True, frames=(30,))
        )
    with pytest.raises(ValueError, match="artifacts.dataset.nerfstudio"):
        make_pipeline_config(
            tmp_path,
            reconstruction=SplatfactoConfig(enabled=True),
            nerfstudio=NerfstudioArtifactConfig(enabled=False),
        )


def test_training_progress_is_integer_and_raw_rows_stay_only_in_file(
    tmp_path, binaries
):
    trainer = binaries / "ns-train"
    trainer.write_text(
        trainer.read_text().replace(
            'print("7 (0.01%)  training progress")',
            'for step in range(0, 60001, 60): print(f"{step} ({step / 600:.1f}%) training progress")',
        )
    )
    command = [
        sys.executable,
        "-m",
        "recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto",
        "--dataset",
        str(dataset(tmp_path / "source")),
        "--output",
        str(tmp_path / "result"),
        "--nerfstudio-bin",
        str(binaries),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    fractions = [event["fraction"] for event in events if event["event"] == "progress"]
    assert len(fractions) <= 101
    assert len(set(fractions)) == len(fractions)
    assert all(abs(f * 100 - round(f * 100)) < 1e-8 for f in fractions)
    assert fractions[0] == 0 and fractions[-1] == 1
    assert "training progress" not in completed.stderr
    assert "library diagnostic" in completed.stderr
    assert (tmp_path / "result/logs/train.log").read_text().count(
        "training progress"
    ) == 1001
