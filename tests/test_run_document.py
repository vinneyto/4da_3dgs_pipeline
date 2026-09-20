import json
from pathlib import Path

from fourda_pipeline.config import load_pipeline_config
from fourda_pipeline.job_config import load_job_config


def document(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "pipeline": {
            "video_path": str(tmp_path / "input.mov"),
            "experiment_name": "leo",
            "fourdanyone_root": str(tmp_path / "4DAnyone"),
            "model_dir": str(tmp_path / "models"),
            "runs_dir": str(tmp_path / "runs"),
        },
        "job": {
            "job_id": "leo",
            "jobs_dir": str(tmp_path / "jobs"),
            "shutdown_on": "success",
        },
        "aws": {
            "region": "us-east-1",
            "bucket": "cp-4da-test",
            "models_prefix": "models",
            "runs_prefix": "runs",
            "upload_results": True,
            "sns": {"topic_name": "cp-4da-test", "email": "user@example.com"},
            "sagemaker": {"domain_id": "d-test", "space_name": "space", "app_name": "default"},
        },
    }


def test_local_pipeline_reads_only_pipeline_section(tmp_path: Path) -> None:
    payload = document(tmp_path)
    del payload["job"]
    del payload["aws"]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    config = load_pipeline_config(path)
    assert config.experiment_name == "leo"
    assert "s3" not in config.to_dict()


def test_job_reads_aws_boundary_from_same_document(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document(tmp_path)))

    pipeline, job = load_job_config(path)
    assert pipeline.experiment_name == "leo"
    assert job.jobs_dir == tmp_path / "jobs"
    assert job.aws.bucket == "cp-4da-test"
    assert job.shutdown_on == "success"
