import json
from pathlib import Path

import pytest

from fourda_aws_worker.config import load_aws_worker_config
from fourda_aws_worker.config_cli import main


def arguments(output: Path) -> list[str]:
    return [
        "--output",
        str(output),
        "--experiment-name",
        "leo_one_layer_24views_01",
        "--s3-video-path",
        "s3://cp-4da-test/input/leo.MOV",
        "--layer-pitches",
        "0",
        "--sns-topic-name",
        "cp-4da-pipeline-test",
        "--notification-email",
        "owner@example.com",
        "--sagemaker-domain-id",
        "d-test",
        "--sagemaker-space-name",
        "cp-4da-jupyter-test",
    ]


def test_generator_writes_a_valid_worker_document(tmp_path: Path) -> None:
    output = tmp_path / "run.json"

    main(arguments(output))

    raw = json.loads(output.read_text())
    pipeline, worker = load_aws_worker_config(output)
    assert raw["schema_version"] == 6
    assert raw["environment"]["python_version"] == "3.11"
    assert raw["environment"]["torch_version"] == "2.8.0"
    assert raw["pipeline"]["dataset"]["enabled"] is True
    assert raw["pipeline"]["dataset"]["type"] == "4danyone"
    assert raw["pipeline"]["dataset"]["config"]["layer_pitches"] == [0]
    assert "frame" not in raw["pipeline"]["dataset"]["config"]
    assert "export_device" not in raw["pipeline"]["dataset"]["config"]
    assert raw["artifacts"]["dataset"]["nerfstudio"] == {
        "enabled": True,
        "source_experiment_name": "leo_one_layer_24views_01",
        "frames": [60],
        "device": "cuda:0",
        "replace_existing": False,
    }
    assert raw["pipeline"]["reconstruction"] == {
        "enabled": False,
        "type": "nerfstudio_splatfacto",
        "config": {},
    }
    assert raw["artifacts"]["reconstruction"] == {
        "rerun": {
            "enabled": False,
            "source_experiment_name": "leo_one_layer_24views_01",
            "view_count": 4,
            "device": "auto",
            "replace_existing": False,
        }
    }
    assert pipeline.experiment_name == "leo_one_layer_24views_01"
    assert pipeline.num_views == 24
    assert pipeline.layer_pitches == (0,)
    assert pipeline.nerfstudio.enabled is True
    assert pipeline.nerfstudio.frames == (60,)
    assert worker.job_id == pipeline.experiment_name
    assert worker.bucket.name == "cp-4da-test"
    assert raw["aws_worker"]["sagemaker"]["app_name"] == "default"
    assert raw["aws_worker"]["bucket"]["video"] == "leo.MOV"
    assert worker.local_video_path == Path("/home/sagemaker-user/4danyone-data/input/leo.MOV")


def test_generator_refuses_to_replace_a_document_without_force(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.write_text("do not replace")

    with pytest.raises(SystemExit):
        main(arguments(output))

    assert output.read_text() == "do not replace"


def test_generator_can_replace_a_document_with_force(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.write_text("replace me")

    main([*arguments(output), "--force", "--layer-pitches", "-15", "0", "15"])

    pipeline, _ = load_aws_worker_config(output)
    assert pipeline.layer_pitches == (-15, 0, 15)
    assert pipeline.num_views == 72


def test_generator_writes_optional_telegram_monitoring(tmp_path: Path) -> None:
    output = tmp_path / "run.json"

    main(
        [
            *arguments(output),
            "--telegram-chat-id",
            "123456",
            "--telegram-stream-logs",
            "--telegram-resource-status-interval-seconds",
            "60",
        ]
    )

    _, worker = load_aws_worker_config(output)
    assert worker.telegram is not None
    assert worker.telegram.stream_logs is True
    assert worker.telegram.resource_status_interval_seconds == 60
    assert worker.telegram.shutdown_command is True
    assert worker.telegram.shutdown_user_id == "123456"


def test_generator_writes_new_relative_bucket_shape_and_rerun(tmp_path: Path) -> None:
    output = tmp_path / "run.json"

    main(
        [
            "--output",
            str(output),
            "--experiment-name",
            "leo",
            "--bucket",
            "cp-4da-test",
            "--video",
            "people/leo.MOV",
            "--rerun",
            "--sagemaker-domain-id",
            "d-test",
            "--sagemaker-space-name",
            "space-test",
        ]
    )

    raw = json.loads(output.read_text())
    pipeline, worker = load_aws_worker_config(output)
    assert raw["aws_worker"]["bucket"] == {
        "name": "cp-4da-test",
        "video": "people/leo.MOV",
        "input_prefix": "input",
        "models_prefix": "models",
        "runs_prefix": "runs",
    }
    assert worker.video_s3_uri == "s3://cp-4da-test/input/people/leo.MOV"
    assert pipeline.rerun.enabled is True
    assert pipeline.rerun.view_count == 4
    assert raw["artifacts"]["dataset"]["rerun"]["enabled"] is True
    assert raw["artifacts"]["dataset"]["rerun"]["source_experiment_name"] == "leo"


def test_generator_supports_artifact_only_run(tmp_path: Path) -> None:
    output = tmp_path / "artifact-only.json"

    main(
        [
            "--output",
            str(output),
            "--experiment-name",
            "leo-rerun-layout-v2",
            "--no-dataset",
            "--rerun",
            "--no-nerfstudio",
            "--rerun-source-experiment-name",
            "leo-original",
            "--rerun-replace-existing",
            "--bucket",
            "cp-4da-test",
            "--video",
            "leo.MOV",
            "--sagemaker-domain-id",
            "d-test",
            "--sagemaker-space-name",
            "space-test",
        ]
    )

    raw = json.loads(output.read_text())
    pipeline, _ = load_aws_worker_config(output)
    assert raw["pipeline"]["dataset"]["enabled"] is False
    assert pipeline.dataset_enabled is False
    assert pipeline.rerun.enabled is True
    assert pipeline.rerun.source_experiment_name == "leo-original"
    assert pipeline.rerun.replace_existing is True


def test_generator_supports_nerfstudio_only_run(tmp_path: Path) -> None:
    output = tmp_path / "nerfstudio-only.json"

    main(
        [
            "--output",
            str(output),
            "--experiment-name",
            "leo-static-exports-v2",
            "--no-dataset",
            "--nerfstudio",
            "--nerfstudio-source-experiment-name",
            "leo-original",
            "--nerfstudio-frames",
            "30",
            "60",
            "90",
            "--nerfstudio-device",
            "cpu",
            "--nerfstudio-replace-existing",
            "--bucket",
            "cp-4da-test",
            "--video",
            "leo.MOV",
            "--sagemaker-domain-id",
            "d-test",
            "--sagemaker-space-name",
            "space-test",
        ]
    )

    raw = json.loads(output.read_text())
    pipeline, _ = load_aws_worker_config(output)
    artifact = raw["artifacts"]["dataset"]["nerfstudio"]
    assert raw["pipeline"]["dataset"]["enabled"] is False
    assert artifact["source_experiment_name"] == "leo-original"
    assert artifact["frames"] == [30, 60, 90]
    assert artifact["device"] == "cpu"
    assert artifact["replace_existing"] is True
    assert pipeline.dataset_enabled is False
    assert pipeline.nerfstudio.enabled is True
