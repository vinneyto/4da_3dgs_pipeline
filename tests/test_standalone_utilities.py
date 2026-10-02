"""Exercise CLI boundaries with tiny upstream fixtures, without AWS or a GPU."""

import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from recon_pipeline.core import (
    JsonPassCheckpointStore,
    Pipeline,
    PipelineContext,
    PipelineObserver,
)
from recon_pipeline.core.command import CommandError
from recon_pipeline.core.events import PassProgress
from recon_pipeline.core.utility import run_utility
from recon_pipeline.datasets.fourdanyone.artifacts import INPUT_VIDEO, MODEL_CACHE
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.datasets.fourdanyone.passes import (
    FourDAnyoneInferencePass,
    PrepareExperimentPass,
)
from recon_pipeline.reconstructions.nerfstudio.config import NerfstudioArtifactConfig
from recon_pipeline.reconstructions.nerfstudio.passes import NerfstudioExportPass
from recon_pipeline.workers.aws.passes import WriteRunManifestPass

UTILITY_MODULES = [
    "utilities.datasets.fourdanyone.prepare_experiment",
    "utilities.datasets.fourdanyone.inference",
    "utilities.reconstructions.nerfstudio.export",
    "utilities.artefacts.rerun.export",
    "utilities.cloud.aws.preflight",
    "utilities.storage.s3.download_input",
    "utilities.storage.s3.sync_models",
    "utilities.storage.s3.restore_experiment",
    "utilities.storage.s3.upload_results",
    "utilities.artefacts.manifest.write",
]


@pytest.mark.parametrize("module", UTILITY_MODULES)
def test_utility_help_requires_no_gpu_or_aws(module):
    completed = subprocess.run(
        [sys.executable, "-m", "recon_pipeline." + module, "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--help" in completed.stdout
    assert "--worker-config" not in completed.stdout
    assert "--pipeline-config" not in completed.stdout
    assert "--result-file" not in completed.stdout


def test_each_console_script_has_a_separate_utility_file():
    project = Path(__file__).parents[1]
    scripts = tomllib.loads((project / "pyproject.toml").read_text())["project"][
        "scripts"
    ]
    targets = [target for target in scripts.values() if ".utilities." in target]
    assert len(targets) == len(UTILITY_MODULES)
    assert len(set(targets)) == len(targets)
    for target in targets:
        module = importlib.import_module(target.split(":")[0])
        assert Path(module.__file__).is_file()
        assert Path(module.__file__).is_relative_to(
            project / "src/recon_pipeline/utilities"
        )
        assert callable(module.main)


def fake_upstream(tmp_path):
    root = tmp_path / "upstream"
    (root / "third_party/GVHMR/hmr4d").mkdir(parents=True)
    (root / "third_party/GVHMR/hmr4d/__init__.py").touch()
    (root / "inference.py").write_text("""
import json, logging
from pathlib import Path
def inference(**request):
    assert Path.cwd() == Path(__file__).parent
    assert request["layer_pitches"] == [-15, 0, 15]
    assert request["enable_turbo"] is False
    assert request["seed"] == 123
    logging.getLogger("fdanyone.progress").info("Generating fixture", extra={"fraction": 0.45})
    output = Path(request["output_dir"])
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(json.dumps(request))
    (output / "cameras.json").write_text("{}")
""")
    (root / "scripts").mkdir()
    (root / "scripts/export_nerfstudio.py").write_text("""
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("data_dir", "output_dir", "frame_index", "model_dir", "device"):
    p.add_argument("--" + name, required=True)
a = p.parse_args()
assert Path.cwd() == Path(__file__).parents[1]
output = Path(a.output_dir)
output.mkdir(parents=True)
(output / "transforms.json").write_text(json.dumps(vars(a)))
""")
    video = tmp_path / "input.mov"
    video.touch()
    models = tmp_path / "models"
    models.mkdir()
    return FourDAnyoneConfig(
        video_path=video,
        experiment_name="fixture",
        fourdanyone_root=root,
        model_dir=models,
        runs_dir=tmp_path / "runs",
        seed=123,
        enable_turbo=False,
        nerfstudio=NerfstudioArtifactConfig(enabled=True, frames=(1, 2)),
    )


class ProgressObserver(PipelineObserver):
    def __init__(self):
        self.events = []

    def handle(self, event, context):
        if isinstance(event, PassProgress):
            self.events.append(event)


def test_passes_launch_real_clis_and_checkpoint_results(tmp_path):
    config = fake_upstream(tmp_path)
    passes = [
        PrepareExperimentPass(config),
        FourDAnyoneInferencePass(config),
        NerfstudioExportPass(config),
        WriteRunManifestPass(config),
    ]
    store = JsonPassCheckpointStore(
        config.experiment_dir / ".recon-pipeline/state.json", "fixture"
    )
    observer = ProgressObserver()

    def context():
        return PipelineContext(
            artifacts={INPUT_VIDEO: config.video_path, MODEL_CACHE: config.model_dir}
        )

    Pipeline(passes, checkpoint_store=store, observers=[observer]).run(context())
    manifest = json.loads((config.experiment_dir / "pipeline-result.json").read_text())
    assert [item["frame"] for item in manifest["datasets"]] == [1, 2]
    assert set(manifest["pass_durations_seconds"]) == {
        "prepare-experiment",
        "fourdanyone-inference",
        "nerfstudio-export",
    }
    assert any(event.fraction == 0.45 for event in observer.events)
    assert (config.experiment_dir / "inference-request.json").is_file()
    assert (
        json.loads((config.dataset_dir(2) / "transforms.json").read_text())[
            "frame_index"
        ]
        == "2"
    )
    # A completed prefix skips utilities entirely; force recreates their owned outputs.
    marker = config.inference_dir / "obsolete.txt"
    marker.touch()
    Pipeline(passes, checkpoint_store=store).run(context())
    assert marker.exists()
    Pipeline(passes, checkpoint_store=store, force=True).run(context())
    assert not marker.exists()
    assert len(store.load()) == 4


def test_failed_utility_preserves_diagnostics_and_has_no_checkpoint(tmp_path):
    config = fake_upstream(tmp_path)
    (config.fourdanyone_root / "inference.py").write_text(
        'def inference(**request):\n    raise RuntimeError("fixture model failed")\n'
    )
    store = JsonPassCheckpointStore(config.experiment_dir / "state.json", "fixture")
    context = PipelineContext(
        artifacts={INPUT_VIDEO: config.video_path, MODEL_CACHE: config.model_dir}
    )
    with pytest.raises(CommandError) as failure:
        Pipeline(
            [PrepareExperimentPass(config), FourDAnyoneInferencePass(config)],
            checkpoint_store=store,
        ).run(context)
    assert "fixture model failed" in failure.value.output_tail
    assert list(store.load()) == ["prepare-experiment"]


def test_direct_inference_cli_works_with_relative_paths(tmp_path):
    config = fake_upstream(tmp_path)
    command = [
        sys.executable,
        "-m",
        "recon_pipeline.utilities.datasets.fourdanyone.inference",
        "--fourdanyone-root",
        "upstream",
        "--video",
        "input.mov",
        "--output",
        "standalone",
        "--model-dir",
        "models",
        "--no-turbo",
        "--seed",
        "123",
    ]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    events = [json.loads(line) for line in first.stdout.splitlines()]
    assert events[-1] == {
        "event": "result",
        "data": {"path": str(tmp_path / "standalone")},
    }
    assert any(event["event"] == "progress" for event in events)
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode != 0
    assert "--replace-existing" in again.stderr
    (tmp_path / "standalone/stale.txt").touch()
    replaced = subprocess.run(
        [*command, "--replace-existing"], cwd=tmp_path, capture_output=True, text=True
    )
    assert replaced.returncode == 0, replaced.stderr
    assert not (tmp_path / "standalone/stale.txt").exists()


def test_subprocess_bridge_rejects_missing_result(tmp_path):
    class NoResultRunner:
        def run(self, command, **kwargs):
            pass

    with pytest.raises(RuntimeError, match="without a stdout result"):
        run_utility("unused", [], PipelineContext(), runner=NoResultRunner())


def test_rerun_utility_exports_and_reports_progress(monkeypatch, tmp_path, capsys):
    from recon_pipeline.utilities.artefacts.rerun import export

    generation = tmp_path / "generation"
    generation.mkdir()
    for name in ("metadata.json", "cameras.json"):
        (generation / name).write_text("{}")
    output = tmp_path / "recording.rrd"
    calls = []

    class FakeExporter:
        def __init__(self, **options):
            self.options = options
            calls.append(options)

        def export(self):
            self.options["on_progress"](1, 2, "Fixture frame")
            self.options["output"].touch()
            return self.options["output"]

    monkeypatch.setattr(export, "RerunExporter", FakeExporter)
    arguments = [
        "--generation",
        str(generation),
        "--output",
        str(output),
        "--experiment",
        "fixture",
        "--fourdanyone-root",
        str(tmp_path),
        "--model-dir",
        str(tmp_path),
        "--view-count",
        "2",
        "--device",
        "cpu",
    ]
    export.main(arguments)
    assert calls[0]["view_count"] == 2
    assert calls[0]["device"] == "cpu"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1] == {"event": "result", "data": {"path": str(output)}}
    assert events[1]["fraction"] == 0.5
    with pytest.raises(FileExistsError):
        export.main(arguments)
    assert len(calls) == 1
    export.main([*arguments, "--replace-existing"])
    assert len(calls) == 2


def test_aws_passes_forward_operation_arguments_and_read_stdout(monkeypatch, tmp_path):
    from contextlib import redirect_stdout
    from dataclasses import asdict
    from io import StringIO
    from recon_pipeline.utilities.cloud.aws.access import AwsHealthCheckResult
    from recon_pipeline.utilities.cloud.aws import preflight
    from recon_pipeline.utilities.storage.s3 import (
        download_input,
        restore_experiment,
        sync_models,
        upload_results,
    )
    from recon_pipeline.workers.aws.passes import (
        AwsPreflightPass,
        S3DownloadInputPass,
        S3RestoreExperimentPass,
        S3SyncModelsPass,
        S3UploadResultsPass,
    )
    from test_aws_health import make_config

    worker = make_config(tmp_path)
    config = fake_upstream(tmp_path)
    health = AwsHealthCheckResult(
        "account",
        "caller",
        worker.bucket_name,
        (),
        None,
        None,
        "disabled",
        "disabled",
        None,
        worker.video_s3_uri,
        2,
    )
    observed = []

    def check(**options):
        assert options["bucket"] == worker.bucket_name
        assert options["input_key"] == "input/leo.MOV"
        assert options["check_input"]
        assert options["sns_topic_name"] == worker.sns.topic_name
        assert options["sagemaker_domain_id"] == worker.sagemaker.domain_id
        return health

    monkeypatch.setattr(preflight, "check_access", check)

    def download(**options):
        assert options == {
            "bucket": worker.bucket_name,
            "key": "input/leo.MOV",
            "destination": worker.local_video_path,
            "region": worker.region,
        }
        assert not worker.local_video_path.exists()
        worker.local_video_path.parent.mkdir(parents=True, exist_ok=True)
        worker.local_video_path.write_text("new")
        return worker.local_video_path

    monkeypatch.setattr(download_input, "download_file", download)
    monkeypatch.setattr(sync_models, "sync_prefix", lambda **options: (2, 1))

    def restore(**options):
        assert options["bucket"] == worker.bucket_name
        assert options["prefix"] == "runs/source"
        assert options["require_objects"]
        generation = options["destination"] / "4danyone"
        assert not generation.exists()
        generation.mkdir(parents=True)
        for name in ("metadata.json", "cameras.json"):
            (generation / name).touch()
        return 3, 3

    monkeypatch.setattr(restore_experiment, "sync_prefix", restore)
    monkeypatch.setattr(
        upload_results,
        "delete_prefix",
        lambda **options: observed.append(("delete", options["prefix"])),
    )
    monkeypatch.setattr(
        upload_results,
        "upload_directory",
        lambda **options: observed.append(("upload", options["prefix"]))
        or "s3://fixture/runs/fixture/",
    )

    class LocalCliRunner:
        def run(self, command, *, cwd, on_line):
            assert command[:3] == [sys.executable, "-u", "-m"]
            assert "--worker-config" not in command and "--result-file" not in command
            module = importlib.import_module(command[3])
            stream = StringIO()
            with redirect_stdout(stream):
                module.main(command[4:])
            for line in stream.getvalue().splitlines():
                on_line(line)

    runner = LocalCliRunner()
    updates = []
    context = PipelineContext()
    context._progress_callback = lambda fraction, message: updates.append(
        (fraction, message)
    )
    result = AwsPreflightPass(worker, config, runner=runner).run(context)
    assert next(iter(result.artifacts.values())) == json.loads(
        json.dumps(asdict(health))
    )
    worker.local_video_path.parent.mkdir(parents=True)
    worker.local_video_path.touch()
    assert (
        S3DownloadInputPass(worker, runner=runner)
        .run(context)
        .artifacts[INPUT_VIDEO]
        .read_text()
        == "new"
    )
    assert S3SyncModelsPass(worker, runner=runner).run(context).details == {
        "objects": 2,
        "downloaded": 1,
    }
    destination = tmp_path / "restored"
    (destination / "4danyone").mkdir(parents=True)
    (destination / "4danyone/stale.txt").touch()
    assert (
        S3RestoreExperimentPass(worker, "source", destination, runner=runner)
        .run(context)
        .details["downloaded"]
        == 3
    )
    config.experiment_dir.mkdir(parents=True)
    assert (
        S3UploadResultsPass(worker, config, runner=runner)
        .run(context)
        .details["s3_uri"]
        == "s3://fixture/runs/fixture/"
    )
    assert observed == [("delete", "runs/fixture"), ("upload", "runs/fixture")]
    assert updates


def test_prepare_treats_settings_as_an_arbitrary_json_object(tmp_path):
    directory = tmp_path / "workspace"
    document = {"example": 123, "anything": {"hello": "world"}}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recon_pipeline.utilities.datasets.fourdanyone.prepare_experiment",
            "--directory",
            str(directory),
            "--settings",
            json.dumps(document),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((directory / "config.json").read_text()) == document
    assert json.loads(completed.stdout)["event"] == "result"


@pytest.mark.parametrize("module", UTILITY_MODULES)
def test_importing_utilities_does_not_load_workers_or_pipeline(module):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib,sys; importlib.import_module(sys.argv[1]); "
            "assert not [m for m in sys.modules if m.startswith(('recon_pipeline.workers', 'recon_pipeline.core')) "
            "or '.passes' in m]",
            "recon_pipeline." + module,
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_library_output_is_diagnostic_not_protocol(tmp_path):
    config = fake_upstream(tmp_path)
    upstream = config.fourdanyone_root / "inference.py"
    upstream.write_text(
        upstream.read_text().replace(
            '    output = Path(request["output_dir"])',
            '    print("library diagnostic")\n    output = Path(request["output_dir"])',
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recon_pipeline.utilities.datasets.fourdanyone.inference",
            *FourDAnyoneInferencePass(config).arguments(),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "library diagnostic" in completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert {event["event"] for event in events} == {"progress", "result"}


@pytest.mark.parametrize(
    "events,exception",
    [
        ([{"event": "result", "data": []}], TypeError),
        (
            [{"event": "result", "data": {}}, {"event": "result", "data": {}}],
            RuntimeError,
        ),
    ],
)
def test_pass_rejects_invalid_or_duplicate_stdout_results(events, exception):
    class Runner:
        def run(self, command, *, on_line, **options):
            for event in events:
                on_line(json.dumps(event))

    with pytest.raises(exception):
        run_utility("unused", [], PipelineContext(), runner=Runner())


def test_result_on_stderr_cannot_replace_the_stdout_result(tmp_path):
    (tmp_path / "fixture_cli.py").write_text(
        "import sys,json\n"
        'print(json.dumps({"event":"result","data":{"path":"wrong"}}),file=sys.stderr)\n'
        'print(json.dumps({"event":"result","data":{"path":"correct"}}))\n'
    )
    result = run_utility(
        "fixture_cli", [], PipelineContext(), env={"PYTHONPATH": str(tmp_path)}
    )
    assert result == {"path": "correct"}


def test_threaded_progress_and_library_diagnostics_use_separate_streams(capsys):
    import threading
    from recon_pipeline.utilities._output import report_progress, run_operation

    def operation():
        print("ordinary library diagnostic")
        thread = threading.Thread(
            target=lambda: report_progress(0.5, "Background progress")
        )
        thread.start()
        thread.join()
        return {"path": "artifact"}

    run_operation(operation)
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events == [
        {"event": "progress", "fraction": 0.5, "message": "Background progress"},
        {"event": "result", "data": {"path": "artifact"}},
    ]
    assert "ordinary library diagnostic" in captured.err


def test_preflight_pass_transmits_inline_token_only_through_environment(tmp_path):
    from dataclasses import replace
    from recon_pipeline.workers.aws.config import TelegramConfig
    from recon_pipeline.workers.aws.passes import AwsPreflightPass
    from test_aws_health import make_config

    worker = replace(
        make_config(tmp_path),
        telegram=TelegramConfig(chat_id="123", bot_token="fixture-secret"),
    )
    config = fake_upstream(tmp_path)

    class Runner:
        def run(self, command, *, env, on_line, **options):
            assert "fixture-secret" not in command
            assert env == {"RECON_PREFLIGHT_TELEGRAM_TOKEN": "fixture-secret"}
            on_line(
                json.dumps(
                    {
                        "event": "result",
                        "data": {
                            "caller_arn": "caller",
                            "bucket": "bucket",
                            "email_status": "disabled",
                            "telegram_status": "ready",
                        },
                    }
                )
            )

    AwsPreflightPass(worker, config, runner=Runner()).run(PipelineContext())
