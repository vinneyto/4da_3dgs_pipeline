import json
from dataclasses import replace
from recon_pipeline.workers.aws.requests import ExperimentRequest
from pathlib import Path

import pytest

from recon_pipeline.workers.aws.cli import build_parser
from recon_pipeline.workers.aws.job import AwsBackgroundJob
from recon_pipeline.workers.aws.status import JobStatus


@pytest.fixture
def experiment_request():
    document = json.loads(
        (Path(__file__).parents[1] / "config/run.example.json").read_text()
    )
    return ExperimentRequest.from_dict(document)


class FakeProcess:
    pid = 4321


def test_terminal_job_is_archived_and_restarted(
    monkeypatch, tmp_path: Path, experiment_request
) -> None:
    job = AwsBackgroundJob("leo", tmp_path)
    job.root.mkdir(parents=True)
    job.request_path.write_text('{"attempt": 1}\n')
    job.log_path.write_text("old log\n")
    JobStatus(job_id="leo", state="failed", pid=123).write(job.status_path)
    monkeypatch.setattr(
        "recon_pipeline.workers.aws.job.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    status = job.start(replace(experiment_request, force=True))

    attempts = list((job.root / "attempts").iterdir())
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "request.json").read_text()) == {"attempt": 1}
    assert (attempts[0] / "pipeline.log").read_text() == "old log\n"
    assert json.loads(job.request_path.read_text())["force"] is True
    assert status.pid == FakeProcess.pid


def test_live_job_cannot_be_restarted(
    monkeypatch, tmp_path: Path, experiment_request
) -> None:
    job = AwsBackgroundJob("leo", tmp_path)
    job.root.mkdir(parents=True)
    JobStatus(job_id="leo", state="running", pid=123).write(job.status_path)
    monkeypatch.setattr(job, "_process_is_running", lambda pid: True)

    with pytest.raises(RuntimeError, match="already exists and is running"):
        job.start(replace(experiment_request, force=True))


def test_start_command_accepts_force() -> None:
    args = build_parser().parse_args(["start", "--config", "/tmp/run.json", "--force"])

    assert args.command == "start"
    assert args.force is True
