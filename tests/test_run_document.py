import json
from pathlib import Path

from fourda_aws_worker.config import load_aws_worker_config
from fourda_pipeline.config import load_pipeline_config


def document(tmp_path: Path) -> dict:
    return {
        "schema_version": 2,
        "pipeline": {
            "experiment_name": "leo",
            "views_per_layer": 24,
            "layer_pitches": [0],
            "frame": 60,
            "turbo": True,
        },
        "aws_worker": {
            "job_id": "leo",
            "shutdown_on": "success",
            "region": "us-east-1",
            "bucket": "cp-4da-test",
            "s3_video_path": "s3://cp-4da-test/input/leo.MOV",
            "input_prefix": "input",
            "models_prefix": "models",
            "runs_prefix": "runs",
            "sync_models": True,
            "upload_results": True,
            "sns_topic_name": "cp-4da-test",
            "notification_email": "user@example.com",
            "sagemaker_domain_id": "d-test",
            "sagemaker_space_name": "space",
            "sagemaker_app_name": "default",
            "local": {
                "data_root": str(tmp_path / "data"),
                "fourdanyone_root": str(tmp_path / "4DAnyone"),
            },
        },
    }


def test_local_pipeline_reads_only_pipeline_section(tmp_path: Path) -> None:
    payload = document(tmp_path)
    del payload["aws_worker"]
    payload["pipeline"].update(
        video_path=str(tmp_path / "input.mov"),
        fourdanyone_root=str(tmp_path / "4DAnyone"),
        model_dir=str(tmp_path / "models"),
        runs_dir=str(tmp_path / "runs"),
    )
    payload["pipeline"]["frame_indices"] = [payload["pipeline"].pop("frame")]
    payload["pipeline"]["enable_turbo"] = payload["pipeline"].pop("turbo")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    config = load_pipeline_config(path)
    assert config.experiment_name == "leo"
    assert "s3" not in config.to_dict()


def test_aws_worker_reads_its_boundary_from_same_document(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document(tmp_path)))

    pipeline, worker = load_aws_worker_config(path)
    assert pipeline.experiment_name == "leo"
    assert pipeline.video_path == tmp_path / "data/input/leo.MOV"
    assert pipeline.model_dir == tmp_path / "data/models"
    assert pipeline.nerfstudio.frame_indices == (60,)
    assert worker.jobs_dir == tmp_path / "data/jobs"
    assert worker.bucket.name == "cp-4da-test"
    assert worker.video_s3_uri == "s3://cp-4da-test/input/leo.MOV"
    assert worker.shutdown_on == "success"


def test_checked_in_examples_are_parseable() -> None:
    root = Path(__file__).parents[1]

    local = load_pipeline_config(root / "config/local-run.example.json")
    pipeline, worker = load_aws_worker_config(root / "config/run.example.json")

    assert local.num_views == 24
    assert pipeline.num_views == 24
    assert worker.video_s3_uri.endswith("/input/leo.MOV")


def test_schema_v4_materializes_typed_dataset_stage(tmp_path: Path) -> None:
    payload = document(tmp_path)
    payload["schema_version"] = 4
    payload["pipeline"] = {
        "experiment_name": "leo-v4",
        "dataset": {
            "type": "4danyone",
            "config": {
                "views_per_layer": 24,
                "layer_pitches": [0],
                "frame": 60,
            },
            "artifacts": {"rerun": True},
        },
    }
    path = tmp_path / "run-v4.json"
    path.write_text(json.dumps(payload))

    pipeline, _ = load_aws_worker_config(path)

    assert pipeline.experiment_name == "leo-v4"
    assert pipeline.rerun.enabled is True
    assert pipeline.video_path == tmp_path / "data/input/leo.MOV"
