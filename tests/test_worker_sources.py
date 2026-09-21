from pathlib import Path

from fourda_nerfstudio.config import NerfstudioArtifactConfig
from fourda_4danyone.config import FourDAnyoneConfig
from fourda_rerun.config import RerunConfig
from fourda_aws_worker.worker import required_source_experiments


def make_config(tmp_path: Path, **changes) -> FourDAnyoneConfig:
    values = {
        "video_path": tmp_path / "leo.MOV",
        "experiment_name": "leo-new",
        "fourdanyone_root": tmp_path / "4DAnyone",
        "model_dir": tmp_path / "models",
        "runs_dir": tmp_path / "runs",
    }
    values.update(changes)
    return FourDAnyoneConfig(**values)


def test_new_inference_produces_its_own_artifact_sources(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        nerfstudio=NerfstudioArtifactConfig(enabled=True),
        rerun=RerunConfig(enabled=True),
    )

    assert required_source_experiments(config) == {}


def test_artifacts_share_one_restored_source(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        dataset_enabled=False,
        nerfstudio=NerfstudioArtifactConfig(
            enabled=True,
            source_experiment_name="leo-original",
        ),
        rerun=RerunConfig(
            enabled=True,
            source_experiment_name="leo-original",
        ),
    )

    assert required_source_experiments(config) == {
        "leo-original": tmp_path / "runs/leo-original"
    }
