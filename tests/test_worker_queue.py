import copy
import json
from pathlib import Path

import pytest

from recon_pipeline.workers.aws import queue
from recon_pipeline.workers.aws.cli import build_parser
from recon_pipeline.workers.aws.status import JobStatus


@pytest.fixture
def configs(pipeline_environment, tmp_path):
    source = json.loads(
        (Path(__file__).parents[1] / "config/splatfacto-run.example.json").read_text()
    )
    paths = []
    for index in range(3):
        document = copy.deepcopy(source)
        document["experiment_name"] = document["aws_worker"]["job_id"] = f"run-{index}"
        document["artifacts"]["reconstruction"]["rerun"][
            "source_experiment_name"
        ] = f"run-{index}"
        document["aws_worker"]["shutdown_on"] = "always"
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(document))
        paths.append(path)
    return paths


def test_queue_snapshot_and_single_override_preserve_source_files(configs):
    before = [p.read_text() for p in configs]
    job, request = queue.build_start_request(configs, "always", "my-queue")
    assert job.job_id == "my-queue"
    assert request.kind == "queue"
    assert request.shutdown_on == "always"
    assert [d.aws_worker.job_id for d in request.configs] == [
        "run-0",
        "run-1",
        "run-2",
    ]
    assert all(
        "environment" not in d.to_dict() and "local" not in d.to_dict()["aws_worker"]
        for d in request.configs
    )
    single, snapshot = queue.build_start_request(configs[:1], "never")
    assert single.job_id == "run-0"
    assert snapshot.aws_worker.shutdown_on == "never"
    assert before == [p.read_text() for p in configs]


@pytest.mark.parametrize(
    "mismatch", ["space_name", "region", "job_id", "experiment_name"]
)
def test_queue_rejects_incompatible_or_duplicate_configs(configs, mismatch):
    document = json.loads(configs[1].read_text())
    if mismatch == "space_name":
        document["aws_worker"]["sagemaker"][mismatch] = "another-space"
    elif mismatch == "region":
        document["aws_worker"][mismatch] = "us-west-2"
    elif mismatch == "job_id":
        document["aws_worker"][mismatch] = "run-0"
    else:
        document[mismatch] = "run-0"
        document["artifacts"]["reconstruction"]["rerun"][
            "source_experiment_name"
        ] = "run-0"
    configs[1].write_text(json.dumps(document))
    with pytest.raises(ValueError):
        queue.build_start_request(configs, "always")


@pytest.mark.parametrize(
    "policy,failed,stops",
    [
        ("always", True, True),
        ("always", False, True),
        ("never", True, False),
        ("never", False, False),
        ("success", True, False),
        ("success", False, True),
        ("failure", True, True),
        ("failure", False, False),
    ],
)
def test_queue_runs_sequentially_continues_after_failure_and_shuts_down_last(
    configs, monkeypatch, policy, failed, stops
):
    job, request = queue.build_start_request(configs, policy)
    job.prepare(request)
    events = []

    def run_child(child, document):
        index = len(events)
        assert index == int(child.job_id[-1])
        assert document.aws_worker.shutdown_on == "never"
        child.prepare(document)
        state = "failed" if failed and index == 0 else "succeeded"
        JobStatus(job_id=child.job_id, state=state).write(child.status_path)
        events.append(child.job_id)
        return int(state == "failed")

    def shutdown(region, app):
        assert events == ["run-0", "run-1", "run-2"]
        events.append("shutdown")

    monkeypatch.setattr(queue, "_run_child", run_child)
    monkeypatch.setattr(queue, "stop_sagemaker_app", shutdown)
    assert queue.run_queue(job.root, request) == int(failed)
    assert events == ["run-0", "run-1", "run-2"] + (["shutdown"] if stops else [])
    status = job.status()
    assert status.state == ("failed" if failed else "succeeded")
    assert status.progress == 1
    assert len(json.loads(Path(status.result_path).read_text())["jobs"]) == 3


def test_queue_cancellation_does_not_start_next_job_or_shutdown(configs, monkeypatch):
    job, request = queue.build_start_request(configs, "always")
    job.prepare(request)
    calls = []

    def cancel(child, document):
        calls.append(child.job_id)
        raise KeyboardInterrupt()

    monkeypatch.setattr(queue, "_run_child", cancel)
    monkeypatch.setattr(
        queue, "stop_sagemaker_app", lambda *a: pytest.fail("Unexpected shutdown")
    )
    assert queue.run_queue(job.root, request) == 130
    assert calls == ["run-0"]
    assert job.status().state == "cancelled"


def test_queue_initialization_error_skips_to_next_config(configs, monkeypatch):
    job, request = queue.build_start_request(configs, "never")
    job.prepare(request)
    calls = []

    def run_child(child, document):
        calls.append(child.job_id)
        if child.job_id == "run-0":
            raise RuntimeError("launch failed")
        child.prepare(document)
        JobStatus(job_id=child.job_id, state="succeeded").write(child.status_path)
        return 0

    monkeypatch.setattr(queue, "_run_child", run_child)
    assert queue.run_queue(job.root, request) == 1
    assert calls == ["run-0", "run-1", "run-2"]


def test_queue_child_launch_waits_and_tees_logs(configs, monkeypatch):
    job, request = queue.build_start_request(configs[:1], "never")
    calls = []

    class Process:
        pid = 12345
        stdout = __import__("io").StringIO("training log\n")

        def wait(self):
            calls.append("wait")
            return 1

        def poll(self):
            return 1

    def launch(command, **kwargs):
        assert "start_new_session" not in kwargs
        calls.append("launch")
        return Process()

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    assert queue._run_child(job, request) == 1
    assert calls[0] == "launch" and calls[1] == "wait"
    assert job.log_path.read_text() == "training log\n"


def test_queue_cli_accepts_multiple_configs_and_management_id():
    parser = build_parser()
    args = parser.parse_args(
        [
            "start",
            "--config",
            "a.json",
            "--config",
            "b.json",
            "--queue-id",
            "batch",
            "--shutdown-on",
            "always",
        ]
    )
    assert args.config == [Path("a.json"), Path("b.json")]
    assert args.shutdown_on == "always"
    for name in ("logs", "status", "stop"):
        args = parser.parse_args([name, "--queue-id", "batch"])
        assert args.queue_id == "batch" and args.config is None


@pytest.mark.parametrize(
    "marker", ["s3-persistence-failed", "cloudwatch-persistence-failed"]
)
def test_queue_continues_but_defers_shutdown_after_failed_persistence(
    configs, monkeypatch, marker
):
    job, request = queue.build_start_request(configs, "always")
    job.prepare(request)
    calls = []

    def run_child(child, document):
        child.prepare(document)
        calls.append(child.job_id)
        state = "failed" if child.job_id == "run-0" else "succeeded"
        JobStatus(job_id=child.job_id, state=state).write(child.status_path)
        if state == "failed":
            (child.root / marker).touch()
        return int(state == "failed")

    monkeypatch.setattr(queue, "_run_child", run_child)
    monkeypatch.setattr(
        queue,
        "stop_sagemaker_app",
        lambda *args: pytest.fail("Unsaved data must prevent shutdown"),
    )
    assert queue.run_queue(job.root, request) == 1
    assert calls == ["run-0", "run-1", "run-2"]


def test_cloudwatch_cli_override_applies_to_all_configs_without_editing_files(configs):
    before = [p.read_text() for p in configs]
    args = build_parser().parse_args(
        [
            "start",
            "--config",
            str(configs[0]),
            "--config",
            str(configs[1]),
            "--cloudwatch-log-group",
            "/recon-pipeline/test",
        ]
    )
    _, request = queue.build_start_request(
        args.config, cloudwatch_log_group=args.cloudwatch_log_group
    )
    assert all(
        d.aws_worker.cloudwatch.log_group == "/recon-pipeline/test"
        for d in request.configs
    )
    assert before == [p.read_text() for p in configs]
