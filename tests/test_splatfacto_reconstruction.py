"""Check the RGBA input contract and actual per-frame command/output paths."""

import json
from pathlib import Path

import pytest
from PIL import Image

from recon_pipeline.core import PipelineContext
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.reconstructions.nerfstudio.passes.export import NERFSTUDIO_DATASETS
from recon_pipeline.reconstructions.nerfstudio.passes.gaussian_export import (
    GaussianSplatExportPass, read_splat_count, splat_artifact,
)
from recon_pipeline.reconstructions.nerfstudio.passes.train import (
    SplatfactoTrainPass, training_artifact,
)
from recon_pipeline.reconstructions.nerfstudio.training import (
    SplatfactoReconstructionConfig, SplatfactoTrainingConfig,
)


def sample(tmp_path: Path):
    config = FourDAnyoneConfig(
        video_path=tmp_path / "input.mov", experiment_name="derived",
        fourdanyone_root=tmp_path / "4DAnyone", model_dir=tmp_path / "models",
        runs_dir=tmp_path / "runs", splatfacto_env=tmp_path / "env",
        dataset_enabled=False,
        nerfstudio=NerfstudioArtifactConfig(enabled=True, frames=(10,)),
        reconstruction=SplatfactoReconstructionConfig(
            enabled=True, frames=(10,), training=SplatfactoTrainingConfig(),
        ),
    )
    dataset = config.dataset_dir(10)
    dataset.mkdir(parents=True)
    Image.new("RGBA", (2, 2), (20, 30, 40, 255)).save(dataset / "view.png")
    (dataset / "sparse_pcd.ply").write_bytes(b"ply\nend_header\n")
    transforms = {"ply_file_path": "sparse_pcd.ply", "frames": [{"file_path": "view.png"}]}
    (dataset / "transforms.json").write_text(json.dumps(transforms))
    context = PipelineContext()
    context._progress_callback = lambda fraction, message: None
    context.artifacts[NERFSTUDIO_DATASETS] = [{"frame": 10, "dataset_dir": str(dataset)}]
    return config, dataset, transforms, context


def test_training_uses_rgba_dataset_and_persists_independent_config(tmp_path: Path) -> None:
    config, dataset, _, context = sample(tmp_path)
    (config.splatfacto_env / "bin").mkdir(parents=True)
    (config.splatfacto_env / "bin/ns-train").touch()
    calls = []

    class Runner:
        def run(self, command, *, cwd, env, on_line):
            calls.append(command)
            config_file = cwd / "train/derived/splatfacto/frame_010/config.yml"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("training config")

    result = SplatfactoTrainPass(config, 10, runner=Runner()).run(context)
    assert Path(result.artifacts[training_artifact(10)]).is_file()
    assert calls[0][0] == str(config.splatfacto_env / "bin/ns-train")
    assert calls[0][calls[0].index("--data") + 1] == str(dataset)
    assert calls[0][calls[0].index("--pipeline.model.background-color") + 1] == "random"
    assert calls[0][-3:] == ["nerfstudio-data", "--eval-mode", "all"]


def test_training_rejects_mask_path_in_export(tmp_path: Path) -> None:
    config, dataset, transforms, context = sample(tmp_path)
    transforms["frames"][0]["mask_path"] = "mask.png"
    (dataset / "transforms.json").write_text(json.dumps(transforms))
    with pytest.raises(ValueError, match="without mask_path"):
        SplatfactoTrainPass(config, 10)._dataset(context)


def test_gaussian_export_records_count_and_file(tmp_path: Path) -> None:
    config, _, _, context = sample(tmp_path)
    (config.splatfacto_env / "bin").mkdir(parents=True)
    (config.splatfacto_env / "bin/ns-export").touch()
    config_file = config.splatfacto_dir(10) / "train/derived/splatfacto/frame_010/config.yml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("training config")
    context.artifacts[training_artifact(10)] = str(config_file)

    class Runner:
        def run(self, command, *, cwd, env):
            assert command[command.index("--ply-color-mode") + 1] == "sh_coeffs"
            fields = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
            (cwd / "splat.ply").write_bytes((
                "ply\nformat binary_little_endian 1.0\nelement vertex 23\n"
                + "".join(f"property float {field}\n" for field in fields)
                + "end_header\n"
            ).encode("ascii"))

    result = GaussianSplatExportPass(config, 10, runner=Runner()).run(context)
    assert result.details["gaussians"] == 23
    assert read_splat_count(Path(result.artifacts[splat_artifact(10)])) == 23
