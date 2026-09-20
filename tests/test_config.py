from pathlib import Path

import pytest

from fourda_nerfstudio.config import NerfstudioArtifactConfig
from fourda_pipeline.config import FourDAnyoneConfig, extract_4danyone_dataset_config


def make_config(tmp_path: Path, **changes) -> FourDAnyoneConfig:
    values = {
        "video_path": tmp_path / "leo.MOV",
        "experiment_name": "leo_72views",
        "fourdanyone_root": tmp_path / "4DAnyone",
        "model_dir": tmp_path / "data/models",
        "runs_dir": tmp_path / "data/runs",
    }
    values.update(changes)
    return FourDAnyoneConfig(**values)


def test_colab_defaults_produce_72_cameras(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.views_per_layer == 24
    assert config.layer_pitches == (-15, 0, 15)
    assert config.num_views == 72
    assert config.nerfstudio.frame_indices == (60,)
    assert config.enable_turbo is True


def test_total_views_must_be_divisible_by_six(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible by 6"):
        make_config(
            tmp_path,
            experiment_name="invalid",
            views_per_layer=5,
            layer_pitches=(-15, 0, 15),
        )


def test_config_round_trip(tmp_path: Path) -> None:
    original = make_config(
        tmp_path,
        experiment_name="round-trip",
        layer_pitches=(-15, 15),
        views_per_layer=6,
        nerfstudio=NerfstudioArtifactConfig(frame_indices=(30, 60)),
    )
    restored = FourDAnyoneConfig.from_dict(original.to_dict())
    assert restored == original


def test_rerun_can_be_enabled_with_one_flag(tmp_path: Path) -> None:
    config = FourDAnyoneConfig.from_dict(
        {
            "video_path": str(tmp_path / "leo.MOV"),
            "experiment_name": "rerun",
            "fourdanyone_root": str(tmp_path / "4DAnyone"),
            "model_dir": str(tmp_path / "models"),
            "runs_dir": str(tmp_path / "runs"),
            "rerun": True,
        }
    )
    assert config.rerun.enabled is True
    assert config.rerun.view_count == 4


def test_paths_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="all paths must be absolute"):
        FourDAnyoneConfig(
            video_path=Path("input.mov"),
            experiment_name="relative",
            fourdanyone_root=Path("4DAnyone"),
            model_dir=Path("models"),
            runs_dir=Path("runs"),
        )


def test_reconstruction_stage_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        extract_4danyone_dataset_config(
            {
                "experiment_name": "leo",
                "dataset": {"type": "4danyone", "config": {}},
                "reconstruction": {
                    "type": "nerfstudio_splatfacto",
                    "config": {},
                },
            }
        )


def test_disabled_reconstruction_stage_is_ignored_until_implemented() -> None:
    payload = extract_4danyone_dataset_config(
        {
            "experiment_name": "leo",
            "dataset": {"enabled": True, "type": "4danyone", "config": {}},
            "reconstruction": {
                "enabled": False,
                "type": "nerfstudio_splatfacto",
                "config": {"unknown_future_setting": True},
            },
        }
    )

    assert payload["experiment_name"] == "leo"


def test_disabled_dataset_cannot_run_without_an_artifact() -> None:
    with pytest.raises(ValueError, match="no pipeline stage or artifact"):
        extract_4danyone_dataset_config(
            {
                "experiment_name": "leo",
                "dataset": {"enabled": False, "type": "4danyone"},
                "reconstruction": {"enabled": False},
            }
        )


def test_dataset_rerun_can_run_without_dataset_stage() -> None:
    payload = extract_4danyone_dataset_config(
        {
            "dataset": {"enabled": False, "type": "4danyone", "config": {}},
            "reconstruction": {"enabled": False},
        },
        experiment_name="rerun-layout-v2",
        artifacts={
            "dataset": {
                "rerun": {
                    "enabled": True,
                    "source_experiment_name": "leo-original",
                    "replace_existing": True,
                }
            },
            "reconstruction": {"rerun": {"enabled": False}},
        },
    )

    assert payload["dataset_enabled"] is False
    assert payload["experiment_name"] == "rerun-layout-v2"
    assert payload["rerun"]["source_experiment_name"] == "leo-original"
    assert payload["rerun"]["replace_existing"] is True


def test_nerfstudio_artifact_can_run_without_dataset_stage() -> None:
    payload = extract_4danyone_dataset_config(
        {
            "dataset": {"enabled": False, "type": "4danyone", "config": {}},
            "reconstruction": {"enabled": False},
        },
        experiment_name="export-v2",
        artifacts={
            "dataset": {
                "nerfstudio": {
                    "enabled": True,
                    "source_experiment_name": "leo-original",
                    "frame_indices": [30, 60, 90],
                    "device": "cpu",
                    "replace_existing": True,
                },
                "rerun": {"enabled": False},
            },
            "reconstruction": {"rerun": {"enabled": False}},
        },
    )

    assert payload["dataset_enabled"] is False
    assert payload["nerfstudio"]["source_experiment_name"] == "leo-original"
    assert payload["nerfstudio"]["frame_indices"] == [30, 60, 90]


def test_reconstruction_artifact_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="reconstruction.rerun"):
        extract_4danyone_dataset_config(
            {
                "dataset": {"enabled": True, "type": "4danyone", "config": {}},
                "reconstruction": {"enabled": False},
            },
            experiment_name="leo",
            artifacts={
                "dataset": {"rerun": {"enabled": False}},
                "reconstruction": {"rerun": {"enabled": True}},
            },
        )
