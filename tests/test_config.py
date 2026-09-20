from pathlib import Path

import pytest

from fourda_pipeline.config import FourDAnyoneConfig


def test_colab_defaults_produce_72_cameras(tmp_path: Path) -> None:
    config = FourDAnyoneConfig(video_path=tmp_path / "leo.MOV", experiment_name="leo_72views")
    assert config.views_per_layer == 24
    assert config.layer_pitches == (-15, 0, 15)
    assert config.num_views == 72
    assert config.frame_indices == (60,)
    assert config.enable_turbo is True


def test_total_views_must_be_divisible_by_six(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible by 6"):
        FourDAnyoneConfig(
            video_path=tmp_path / "leo.MOV",
            experiment_name="invalid",
            views_per_layer=5,
            layer_pitches=(-15, 0, 15),
        )


def test_config_round_trip(tmp_path: Path) -> None:
    original = FourDAnyoneConfig(
        video_path=tmp_path / "leo.MOV",
        experiment_name="round-trip",
        data_root=tmp_path / "data",
        layer_pitches=(-15, 15),
        views_per_layer=6,
        frame_indices=(30, 60),
    )
    restored = FourDAnyoneConfig.from_dict(original.to_dict())
    assert restored == original
