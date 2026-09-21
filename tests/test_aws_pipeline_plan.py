from pathlib import Path

import pytest

from fourda_aws_worker.finalizers import SageMakerShutdownFinalizer
from fourda_aws_worker.finalizers import sagemaker_shutdown
from fourda_aws_worker.pipeline import build_aws_pipeline
from fourda_aws_worker.status import JobStatus
from fourda_nerfstudio.config import NerfstudioArtifactConfig
from fourda_4danyone.config import FourDAnyoneConfig
from fourda_pipeline.core import PipelineContext, PipelineOutcome
from fourda_rerun.config import RerunConfig
from test_aws_health import make_config as make_worker_config


def make_pipeline_config(tmp_path: Path, **changes) -> FourDAnyoneConfig:
    values = {
        "video_path": tmp_path / "data/input/leo.MOV",
        "experiment_name": "leo-new",
        "fourdanyone_root": tmp_path / "4DAnyone",
        "model_dir": tmp_path / "data/models",
        "runs_dir": tmp_path / "data/runs",
    }
    values.update(changes)
    return FourDAnyoneConfig(**values)


def test_aws_builder_creates_explicit_full_execution_plan(tmp_path: Path) -> None:
    worker = make_worker_config(tmp_path)
    config = make_pipeline_config(
        tmp_path,
        nerfstudio=NerfstudioArtifactConfig(enabled=True),
        rerun=RerunConfig(enabled=True),
    )

    pipeline = build_aws_pipeline(
        worker, config, tmp_path / "job", JobStatus(job_id=worker.job_id)
    )
    plan = pipeline.prepare()

    assert [item.id for item in plan.passes] == [
        "aws-preflight",
        "s3-download-input",
        "s3-sync-models",
        "prepare-experiment",
        "fourda-inference",
        "nerfstudio-export",
        "rerun-export",
        "write-run-manifest",
        "s3-upload-results",
    ]
    assert [item.id for item in plan.finalizers] == ["sagemaker-shutdown"]


def test_artifact_only_plan_restores_shared_source_once(tmp_path: Path) -> None:
    worker = make_worker_config(tmp_path)
    config = make_pipeline_config(
        tmp_path,
        dataset_enabled=False,
        nerfstudio=NerfstudioArtifactConfig(
            enabled=True, source_experiment_name="leo-original"
        ),
        rerun=RerunConfig(enabled=True, source_experiment_name="leo-original"),
    )

    pipeline = build_aws_pipeline(
        worker, config, tmp_path / "job", JobStatus(job_id=worker.job_id)
    )
    ids = [item.id for item in pipeline.prepare().passes]

    assert ids.count("s3-restore-experiment:leo-original") == 1
    assert "s3-download-input" not in ids
    assert "fourda-inference" not in ids
    assert "nerfstudio-export" in ids
    assert "rerun-export" in ids


@pytest.mark.parametrize(
    ("policy", "failed", "expected"),
    [
        ("never", False, False),
        ("success", False, True),
        ("success", True, False),
        ("failure", True, True),
        ("failure", False, False),
        ("always", False, True),
        ("always", True, True),
    ],
)
def test_shutdown_finalizer_applies_outcome_policy(
    monkeypatch, tmp_path: Path, policy: str, failed: bool, expected: bool
) -> None:
    worker = make_worker_config(tmp_path, shutdown_on=policy)
    calls = []
    monkeypatch.setattr(
        sagemaker_shutdown,
        "stop_sagemaker_app",
        lambda region, app: calls.append((region, app)),
    )
    outcome = PipelineOutcome(
        succeeded=not failed,
        error=RuntimeError("failed") if failed else None,
    )

    SageMakerShutdownFinalizer(worker).run(PipelineContext(), outcome)

    assert bool(calls) is expected
