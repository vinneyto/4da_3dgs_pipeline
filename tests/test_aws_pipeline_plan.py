from pathlib import Path

import pytest

from recon_pipeline.workers.aws.finalizers import SageMakerShutdownFinalizer
from recon_pipeline.workers.aws.finalizers import sagemaker_shutdown
from recon_pipeline.workers.aws.pipeline import build_aws_pipeline
from recon_pipeline.workers.aws.status import JobStatus
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.core import PipelineContext, PipelineOutcome
from recon_pipeline.artifacts.rerun.config import RerunConfig
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
        "s3-upload:prepare-experiment",
        "fourdanyone-inference",
        "s3-upload:fourdanyone-inference",
        "nerfstudio-export",
        "s3-upload:nerfstudio-export",
        "rerun-export",
        "s3-upload:rerun-export",
        "write-run-manifest",
        "s3-upload:write-run-manifest",
    ]
    assert [item.id for item in plan.finalizers] == [
        "s3-save-diagnostics",
        "sagemaker-shutdown",
    ]


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
    assert "fourdanyone-inference" not in ids
    assert "nerfstudio-export" in ids
    assert "rerun-export" in ids


def test_reconstruction_plan_has_one_training_and_recording_pass_per_frame(tmp_path):
    from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import (
        SplatfactoConfig,
        SplatfactoRerunConfig,
    )

    worker = make_worker_config(tmp_path)
    config = make_pipeline_config(
        tmp_path,
        dataset_enabled=False,
        nerfstudio=NerfstudioArtifactConfig(
            frames=(30, 60, 90), source_experiment_name="original"
        ),
        reconstruction=SplatfactoConfig(enabled=True, frames=(30, 90)),
        reconstruction_rerun=SplatfactoRerunConfig(enabled=True),
    )
    pipeline = build_aws_pipeline(
        worker, config, tmp_path / "job", JobStatus(job_id=worker.job_id)
    )
    ids = [item.id for item in pipeline.prepare().passes]
    assert ids == [
        "aws-preflight",
        "s3-sync-models",
        "s3-restore-experiment:original",
        "prepare-experiment",
        "s3-upload:prepare-experiment",
        "nerfstudio-export",
        "s3-upload:nerfstudio-export",
        "splatfacto:frame_030",
        "s3-upload:splatfacto:frame_030",
        "splatfacto-rerun:frame_030",
        "s3-upload:splatfacto-rerun:frame_030",
        "splatfacto:frame_090",
        "s3-upload:splatfacto:frame_090",
        "splatfacto-rerun:frame_090",
        "s3-upload:splatfacto-rerun:frame_090",
        "write-run-manifest",
        "s3-upload:write-run-manifest",
    ]


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
