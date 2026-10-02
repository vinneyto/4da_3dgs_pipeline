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
    "datasets.fourdanyone.utilities.prepare_experiment",
    "datasets.fourdanyone.utilities.inference",
    "reconstructions.nerfstudio.utilities.export",
    "artifacts.rerun.utilities.export",
    "workers.aws.utilities.preflight",
    "workers.aws.utilities.download_input",
    "workers.aws.utilities.sync_models",
    "workers.aws.utilities.restore_experiment",
    "workers.aws.utilities.upload_results",
    "workers.aws.utilities.write_run_manifest",
]


@pytest.mark.parametrize("module", UTILITY_MODULES)
def test_utility_help_requires_no_gpu_or_aws(module):
    completed = subprocess.run(
        [sys.executable, "-m", "recon_pipeline." + module, "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--result-file" in completed.stdout


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
        "recon_pipeline.datasets.fourdanyone.utilities.inference",
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
        "--result-file",
        "result.json",
    ]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads((tmp_path / "result.json").read_text())["path"] == str(
        tmp_path / "standalone"
    )
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

    with pytest.raises(RuntimeError, match="without a JSON result"):
        run_utility("unused", [], PipelineContext(), runner=NoResultRunner())


def test_rerun_utility_exports_and_reports_progress(monkeypatch, tmp_path):
    from recon_pipeline.artifacts.rerun.utilities import export

    generation = tmp_path / "generation"
    generation.mkdir()
    for name in ("metadata.json", "cameras.json"):
        (generation / name).write_text("{}")
    output = tmp_path / "recording.rrd"
    result_file = tmp_path / "result.json"
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
        "--result-file",
        str(result_file),
    ]
    export.main(arguments)
    assert calls[0]["view_count"] == 2
    assert calls[0]["device"] == "cpu"
    assert json.loads(result_file.read_text()) == {"path": str(output)}
    with pytest.raises(FileExistsError):
        export.main(arguments)
    assert len(calls) == 1
    export.main([*arguments, "--replace-existing"])
    assert len(calls) == 2


def test_aws_passes_call_their_utilities_with_serializable_worker_config(
    monkeypatch, tmp_path
):
    """Use real CLI parsing/config readback while faking only AWS service calls."""
    from contextlib import redirect_stdout
    from dataclasses import asdict
    from io import StringIO

    from recon_pipeline.workers.aws.aws import AwsHealthCheckResult
    from recon_pipeline.workers.aws.passes import (
        AwsPreflightPass,
        S3DownloadInputPass,
        S3RestoreExperimentPass,
        S3SyncModelsPass,
        S3UploadResultsPass,
    )
    from recon_pipeline.workers.aws.utilities import (
        preflight,
        download_input,
        restore_experiment,
        sync_models,
        upload_results,
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

    def check(loaded, job_id, *, require_input_video):
        assert loaded == worker
        assert job_id == worker.job_id
        assert require_input_video
        return health

    monkeypatch.setattr(preflight, "run_health_check", check)

    def download(loaded):
        assert loaded == worker
        assert not worker.local_video_path.exists()
        worker.local_video_path.parent.mkdir(parents=True, exist_ok=True)
        worker.local_video_path.write_text("new")
        return worker.local_video_path

    monkeypatch.setattr(download_input, "download_input_video", download)
    monkeypatch.setattr(sync_models, "sync_model_objects", lambda loaded: (2, 1))

    def restore(loaded, name, destination):
        assert loaded == worker
        assert name == "source"
        generation = destination / "4danyone"
        assert not generation.exists()
        generation.mkdir(parents=True)
        for name in ("metadata.json", "cameras.json"):
            (generation / name).touch()
        return 3, 3

    monkeypatch.setattr(restore_experiment, "sync_experiment_results", restore)
    monkeypatch.setattr(
        upload_results,
        "delete_experiment_results",
        lambda loaded, name: observed.append(("delete", name)),
    )
    monkeypatch.setattr(
        upload_results,
        "upload_directory",
        lambda loaded, source, name: observed.append(("upload", name))
        or "s3://fixture/runs/fixture/",
    )

    class LocalCliRunner:
        def run(self, command, *, cwd, on_line):
            assert command[:3] == [sys.executable, "-u", "-m"]
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
    preflight_result = AwsPreflightPass(worker, config, runner=runner).run(context)
    assert next(iter(preflight_result.artifacts.values())) == json.loads(
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
    assert observed == [("delete", "fixture"), ("upload", "fixture")]
    assert updates


def test_prepare_accepts_a_local_run_document(tmp_path):
    config = fake_upstream(tmp_path)
    document = {
        "schema_version": 6,
        "experiment_name": config.experiment_name,
        "pipeline": {
            "dataset": {"enabled": True, "type": "4danyone", "config": config.to_dict()}
        },
    }
    source = tmp_path / "local-run.json"
    source.write_text(json.dumps(document))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recon_pipeline.datasets.fourdanyone.utilities.prepare_experiment",
            "--pipeline-config",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (config.experiment_dir / "pipeline-config.json").is_file()
