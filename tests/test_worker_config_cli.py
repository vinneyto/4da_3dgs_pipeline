import json
from pathlib import Path

import pytest

from recon_pipeline.workers.aws.config import load_aws_worker_config
from recon_pipeline.workers.aws.config_cli import main


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


def test_generator_clones_template_with_new_identity_and_video(tmp_path: Path) -> None:
    template = tmp_path / "leo.json"
    output = tmp_path / "alex.json"
    main(
        [
            *arguments(template),
            "--layer-pitches",
            "-15",
            "0",
            "15",
            "--rerun",
            "--telegram-chat-id",
            "123456",
            "--shutdown-on",
            "always",
        ]
    )

    main(
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--experiment-name",
            "alex_three_layers_72views_01",
            "--video",
            "alex.MOV",
        ]
    )

    source = json.loads(template.read_text())
    cloned = json.loads(output.read_text())
    pipeline, worker = load_aws_worker_config(output)
    assert cloned["experiment_name"] == "alex_three_layers_72views_01"
    assert cloned["aws_worker"]["job_id"] == "alex_three_layers_72views_01"
    assert cloned["aws_worker"]["bucket"]["video"] == "alex.MOV"
    assert cloned["aws_worker"]["bucket"]["name"] == "cp-4da-test"
    assert cloned["pipeline"] == source["pipeline"]
    assert cloned["environment"] == source["environment"]
    assert cloned["aws_worker"]["notifications"] == source["aws_worker"]["notifications"]
    assert cloned["aws_worker"]["shutdown_on"] == "always"
    assert cloned["artifacts"]["dataset"]["nerfstudio"]["source_experiment_name"] == (
        "alex_three_layers_72views_01"
    )
    assert cloned["artifacts"]["dataset"]["rerun"]["source_experiment_name"] == (
        "alex_three_layers_72views_01"
    )
    assert pipeline.layer_pitches == (-15, 0, 15)
    assert worker.video_s3_uri == "s3://cp-4da-test/input/alex.MOV"


def test_template_clone_preserves_external_artifact_source(tmp_path: Path) -> None:
    template = tmp_path / "artifact.json"
    output = tmp_path / "clone.json"
    main(arguments(template))
    document = json.loads(template.read_text())
    document["artifacts"]["dataset"]["rerun"]["source_experiment_name"] = "shared-run"
    template.write_text(json.dumps(document))

    main(
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--experiment-name",
            "alex",
            "--video",
            "alex.MOV",
        ]
    )

    cloned = json.loads(output.read_text())
    assert cloned["artifacts"]["dataset"]["rerun"]["source_experiment_name"] == (
        "shared-run"
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--video", "alex.MOV"], "--template requires --experiment-name"),
        (["--experiment-name", "alex"], "--template requires --video"),
    ],
)
def test_template_clone_requires_identity_and_video(
    tmp_path: Path, extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    template = tmp_path / "template.json"
    main(arguments(template))

    with pytest.raises(SystemExit):
        main(["--template", str(template), "--output", str(tmp_path / "out.json"), *extra])

    assert message in capsys.readouterr().err


def test_generator_writes_concise_telegram_configuration(tmp_path: Path) -> None:
    output = tmp_path / "run.json"

    main(
        [
            *arguments(output),
            "--telegram-chat-id",
            "123456",
        ]
    )

    raw = json.loads(output.read_text())
    _, worker = load_aws_worker_config(output)
    telegram = raw["aws_worker"]["notifications"]["telegram"]
    assert worker.telegram is not None
    assert worker.telegram.shutdown_command is True
    assert worker.telegram.shutdown_user_id == "123456"
    assert "stream_logs" not in telegram
    assert "resource_status_interval_seconds" not in telegram


def test_loader_ignores_legacy_noisy_telegram_options(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    main([*arguments(output), "--telegram-chat-id", "123456"])
    raw = json.loads(output.read_text())
    raw["aws_worker"]["notifications"]["telegram"]["stream_logs"] = True
    raw["aws_worker"]["notifications"]["telegram"][
        "resource_status_interval_seconds"
    ] = 60
    output.write_text(json.dumps(raw))

    _, worker = load_aws_worker_config(output)

    assert worker.telegram is not None
    assert worker.telegram.chat_id == "123456"

def test_generator_supports_explicit_notification_switches(tmp_path: Path) -> None:
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
            "leo.MOV",
            "--sagemaker-domain-id",
            "d-test",
            "--sagemaker-space-name",
            "space-test",
            "--no-email",
            "--telegram",
            "--telegram-chat-id",
            "123456",
        ]
    )

    raw = json.loads(output.read_text())
    _, worker = load_aws_worker_config(output)
    assert raw["aws_worker"]["notifications"]["email"] is None
    assert worker.sns is None
    assert worker.telegram is not None
    assert worker.telegram.chat_id == "123456"


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
