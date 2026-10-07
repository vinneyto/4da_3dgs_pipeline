import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from recon_pipeline.environment import ENV_NAMES, PipelineEnvironment
from recon_pipeline.environment_cli import check_environment, configure
from recon_pipeline.workers.aws.config import load_aws_worker_config, load_document
from recon_pipeline.workers.aws.config_cli import main as config_main

ROOT = Path(__file__).parents[1]


def test_runtime_never_fills_missing_variables_with_installation_defaults():
    with pytest.raises(ValueError) as error:
        PipelineEnvironment.from_environ({})
    assert "RECON_DATA_ROOT" in str(error.value)
    assert "RECON_NERFSTUDIO_BIN" in str(error.value)
    assert "check_environment.sh" in str(error.value)


def test_relative_machine_path_is_rejected(pipeline_environment, monkeypatch):
    monkeypatch.setenv("RECON_DATA_ROOT", "data")
    with pytest.raises(ValueError, match="RECON_DATA_ROOT must be absolute"):
        PipelineEnvironment.from_environ()


def test_same_json_uses_current_machine_for_every_execution_path(
    pipeline_environment,
    monkeypatch,
    tmp_path,
):
    path = ROOT / "config/splatfacto-run.example.json"
    first, _ = load_aws_worker_config(path)
    monkeypatch.setenv("RECON_DATA_ROOT", str(tmp_path / "other/data"))
    monkeypatch.setenv("RECON_FOURDANYONE_ROOT", str(tmp_path / "other/4DAnyone"))
    monkeypatch.setenv("RECON_CONDA_ENV", str(tmp_path / "other/4danyone"))
    monkeypatch.setenv("RECON_NERFSTUDIO_BIN", str(tmp_path / "other/splatfacto/bin"))
    monkeypatch.setenv(
        "RECON_SPLATFACTO_RERUN_PYTHON", str(tmp_path / "other/rerun/bin/python")
    )
    second, worker = load_aws_worker_config(path)
    assert first.experiment_name == second.experiment_name
    assert second.model_dir == tmp_path / "other/data/models"
    assert second.fourdanyone_root == tmp_path / "other/4DAnyone"
    assert second.python == str(tmp_path / "other/4danyone/bin/python")
    assert second.reconstruction.nerfstudio_bin == str(
        tmp_path / "other/splatfacto/bin"
    )
    assert second.reconstruction_rerun.python == str(
        tmp_path / "other/rerun/bin/python"
    )
    assert worker.jobs_dir == tmp_path / "other/data/jobs"
    assert "local" not in worker.to_dict()


@pytest.mark.parametrize(
    "location", ["environment", "local", "nerfstudio_bin", "python", "video_path"]
)
def test_v7_refuses_environment_values_even_for_disabled_stages(tmp_path, location):
    document = json.loads((ROOT / "config/run.example.json").read_text())
    if location == "environment":
        document[location] = {}
    elif location == "local":
        document["aws_worker"][location] = {"data_root": "/old"}
    elif location == "nerfstudio_bin":
        document["pipeline"]["reconstruction"]["config"][location] = "/old/bin"
    elif location == "python":
        document["artifacts"]["reconstruction"]["rerun"][location] = "/old/python"
    else:
        document["pipeline"]["dataset"]["config"][location] = "/old/video.mov"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="machine environment settings"):
        load_document(path)


def test_old_template_is_migrated_without_env_or_modifying_source(
    tmp_path, monkeypatch
):
    # Configuration creation/validation must work without any machine env.
    for name in ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    document = json.loads((ROOT / "config/splatfacto-run.example.json").read_text())
    document["schema_version"] = 6
    document["environment"] = {"conda_env": "/old/conda"}
    document["aws_worker"]["local"] = {
        "data_root": "/old/data",
        "fourdanyone_root": "/old/upstream",
    }
    document["pipeline"]["reconstruction"]["config"]["nerfstudio_bin"] = "/old/ns/bin"
    document["artifacts"]["reconstruction"]["rerun"]["python"] = "/old/rerun/python"
    source = tmp_path / "old.json"
    source.write_text(json.dumps(document))
    original = source.read_bytes()
    output = tmp_path / "new.json"
    config_main(
        [
            "--template",
            str(source),
            "--output",
            str(output),
            "--experiment-name",
            "new",
            "--video",
            "new.mov",
        ]
    )
    migrated = json.loads(output.read_text())
    assert source.read_bytes() == original
    assert migrated["schema_version"] == 7
    assert "environment" not in migrated
    assert "local" not in migrated["aws_worker"]
    assert "nerfstudio_bin" not in migrated["pipeline"]["reconstruction"]["config"]
    assert "python" not in migrated["artifacts"]["reconstruction"]["rerun"]
    assert (
        migrated["pipeline"]["reconstruction"]["config"]["max_num_iterations"] == 60000
    )
    assert (
        migrated["artifacts"]["dataset"]["nerfstudio"]["source_experiment_name"]
        == "leo_one_layer_24views_01"
    )


def test_config_generator_runs_on_unconfigured_machine(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recon_pipeline.workers.aws.config_cli",
            "--output",
            str(tmp_path / "run.json"),
            "--experiment-name",
            "test",
            "--video",
            "input.mov",
            "--bucket",
            "test-bucket",
            "--sagemaker-domain-id",
            "d-test",
            "--sagemaker-space-name",
            "test",
        ],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("RECON_")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "environment" not in json.loads((tmp_path / "run.json").read_text())


def test_checker_reports_all_missing_vars_and_does_not_dump_secrets(tmp_path):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("RECON_")}
    environment["CP_4DA_TELEGRAM_BOT_TOKEN"] = "private-fixture-token"
    environment["AWS_SECRET_ACCESS_KEY"] = "private-fixture-key"
    before = sorted(tmp_path.iterdir())
    completed = subprocess.run(
        [str(ROOT / "scripts/check_environment.sh"), "--json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    failed = {item["check"] for item in report["checks"] if item["status"] == "FAIL"}
    assert set(ENV_NAMES.values()) <= failed
    assert "private-fixture" not in completed.stdout + completed.stderr
    assert sorted(tmp_path.iterdir()) == before


def test_configure_preserves_exports_and_saved_settings_and_is_idempotent(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECON_DATA_ROOT", str(tmp_path / "existing data"))
    monkeypatch.setenv("RECON_NS_BIN", str(tmp_path / "existing ns/bin"))
    env_file = tmp_path / "env.sh"
    configure(env_file, ROOT)
    text = env_file.read_text()
    assert "existing data" in text
    assert "existing ns/bin" in text
    monkeypatch.delenv("RECON_DATA_ROOT")
    monkeypatch.delenv("RECON_NS_BIN")
    configure(env_file, ROOT)
    assert env_file.read_text() == text
    for file in (".bashrc", ".bash_profile"):
        assert (tmp_path / file).read_text().count(
            "# Reconstruction pipeline environment"
        ) == 1
    monkeypatch.setenv("RECON_DATA_ROOT", str(tmp_path / "new data"))
    configure(env_file, ROOT)
    assert "new data" in env_file.read_text()
    assert not (
        tmp_path / "new data"
    ).exists()  # configure-only does not install/create workspace
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; python3 -c "from recon_pipeline.environment import PipelineEnvironment; print(PipelineEnvironment.from_environ().data_root)"',
            "bash",
            str(env_file),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "new data")


def test_old_standalone_ns_bin_alias(pipeline_environment, monkeypatch):
    monkeypatch.delenv("RECON_NERFSTUDIO_BIN")
    monkeypatch.setenv("RECON_NS_BIN", "/external/ns/bin")
    assert PipelineEnvironment.from_environ().nerfstudio_bin == Path("/external/ns/bin")


def test_pass_uses_env_interpreter_and_explicit_paths(pipeline_environment):
    from recon_pipeline.core import PipelineContext
    from recon_pipeline.datasets.fourdanyone.passes import FourDAnyoneInferencePass

    pipeline, _ = load_aws_worker_config(ROOT / "config/run.example.json")
    commands = []

    class Runner:
        def run(self, command, *, on_line, **options):
            commands.append(command)
            on_line(
                json.dumps(
                    {"event": "result", "data": {"path": str(pipeline.inference_dir)}}
                )
            )

    FourDAnyoneInferencePass(pipeline, runner=Runner()).run(PipelineContext())
    command = commands[0]
    assert command[0] == str(pipeline_environment.python)
    assert command[command.index("--fourdanyone-root") + 1] == str(
        pipeline_environment.fourdanyone_root
    )
    assert command[command.index("--model-dir") + 1] == str(
        pipeline_environment.data_root / "models"
    )
    assert "--config" not in command
    settings = pipeline.settings_dict()
    assert "python" not in settings and "fourdanyone_root" not in settings
    assert "nerfstudio_bin" not in settings["reconstruction"]


def test_checker_does_not_hide_broken_torch_as_cpu_warning(
    pipeline_environment, monkeypatch
):
    import recon_pipeline.environment_cli as cli

    python = pipeline_environment.python
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexit 1\n")
    python.chmod(0o755)
    report = cli.check_environment()
    cuda = next(c for c in report["checks"] if c["check"] == "4DAnyone CUDA")
    assert cuda["status"] == "FAIL"


def test_start_snapshot_is_portable_and_background_worker_inherits_env(
    pipeline_environment,
    monkeypatch,
    tmp_path,
):
    from recon_pipeline.workers.aws import cli
    from recon_pipeline.workers.aws.job import AwsBackgroundJob
    from recon_pipeline.workers.aws.status import JobStatus

    captured = []

    def start(job, request):
        captured.append(request)
        assert (
            job.root == pipeline_environment.data_root / "jobs/leo_one_layer_24views_01"
        )
        return JobStatus(job_id=job.job_id)

    monkeypatch.setattr(AwsBackgroundJob, "start", start)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recon-aws-worker",
            "start",
            "--config",
            str(ROOT / "config/run.example.json"),
        ],
    )
    cli.main()
    request = captured[0]
    assert "local" not in request["aws_worker"]
    assert "environment" not in request
    assert "nerfstudio_bin" not in request["pipeline"]["reconstruction"]["config"]
    assert "python" not in request["artifacts"]["reconstruction"]["rerun"]
    # The process loader takes paths from its inherited environment again.
    from recon_pipeline.workers.aws.config import (
        AwsWorkerConfig,
        materialize_pipeline_config,
    )

    worker = AwsWorkerConfig.from_document(request)
    config = materialize_pipeline_config(request, worker)
    assert config.python == str(pipeline_environment.python)


def test_configure_does_not_mask_an_existing_login_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    profile = tmp_path / ".profile"
    profile.write_text("export EXISTING_SETTING=preserved\n")
    configure(tmp_path / "env.sh", ROOT)
    assert not (tmp_path / ".bash_profile").exists()
    assert profile.read_text().startswith("export EXISTING_SETTING=preserved\n")
    assert "Reconstruction pipeline environment" in profile.read_text()


def test_checker_reports_detected_cuda_without_unset_version_comparison(
    monkeypatch, tmp_path
):
    toolkit = tmp_path / "existing-cuda"
    binary = toolkit / "bin/nvcc"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        '#!/bin/sh\necho "Cuda compilation tools, release 12.6, V12.6.85"\n'
    )
    binary.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.delenv("RECON_CUDA_VERSION", raising=False)
    report = check_environment()
    version = next(c for c in report["checks"] if c["check"] == "CUDA toolkit version")
    assert version["status"] == "WARN"
    assert "detected 12.6" in version["detail"]
    assert "comparison skipped" in version["detail"]
    monkeypatch.setenv("RECON_CUDA_VERSION", "12.1.1")
    report = check_environment()
    version = next(c for c in report["checks"] if c["check"] == "CUDA toolkit version")
    assert version["status"] == "FAIL"
    assert version["detail"] == "detected 12.6, expected 12.1.1"


def test_configure_only_preserves_existing_cuda_toolkit(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CUDA_HOME", "/opt/conda")
    env_file = tmp_path / "env.sh"
    configure(env_file, ROOT)
    assert "export CUDA_HOME=/opt/conda" in env_file.read_text()
    monkeypatch.delenv("CUDA_HOME")
    configure(env_file, ROOT)
    assert "export CUDA_HOME=/opt/conda" in env_file.read_text()
