from pathlib import Path

from fourda_aws_worker import observers
from fourda_aws_worker.observers import AwsNotificationObserver, JobStatusObserver
from fourda_aws_worker.status import JobStatus
from fourda_pipeline.core import PipelineContext
from fourda_pipeline.command import CommandError
from fourda_pipeline.events import PassCompleted, PassFailed, PassProgress, PassStarted
from test_aws_health import make_config


def test_notification_observer_reports_pass_duration_and_next_pass(
    monkeypatch, tmp_path: Path
) -> None:
    published = []
    monkeypatch.setattr(
        observers,
        "publish_notification",
        lambda config, subject, body: published.append((subject, body)) or {"telegram": "sent"},
    )
    observer = AwsNotificationObserver(make_config(tmp_path), "leo")
    event = PassCompleted(
        pass_id="fourda-inference",
        pass_name="4DAnyone inference",
        pass_index=4,
        pass_count=7,
        duration_seconds=125.0,
        next_pass_name="Nerfstudio dataset export",
    )

    observer.process(event, PipelineContext())

    assert published[0][0] == "Pass 4/7 completed"
    assert "Duration: 2m 5s" in published[0][1]
    assert "Next: Nerfstudio dataset export" in published[0][1]


def test_notification_observer_reports_pass_start(monkeypatch, tmp_path: Path) -> None:
    published = []
    monkeypatch.setattr(
        observers,
        "publish_notification",
        lambda config, subject, body: published.append((subject, body)) or {"telegram": "sent"},
    )
    observer = AwsNotificationObserver(make_config(tmp_path), "leo")

    observer.process(
        PassStarted("fourda-inference", "4DAnyone inference", 4, 7),
        PipelineContext(),
    )

    assert published == [("Pass 4/7 started", "4DAnyone inference")]


def test_notification_observer_includes_failed_command_output(
    monkeypatch, tmp_path: Path
) -> None:
    published = []
    monkeypatch.setattr(
        observers,
        "publish_notification",
        lambda config, subject, body: published.append((subject, body)) or {"telegram": "sent"},
    )
    observer = AwsNotificationObserver(make_config(tmp_path), "leo")
    error = CommandError(["python", "inference.py"], 1, output_tail="model output\nCUDA OOM")

    observer.process(
        PassFailed("fourda-inference", "4DAnyone inference", 4, 7, 125.0, error),
        PipelineContext(),
    )

    assert published[0][0] == "Pass 4/7 failed"
    assert "Duration: 2m 5s" in published[0][1]
    assert "Output tail:\nmodel output\nCUDA OOM" in published[0][1]


def test_status_observer_maps_local_pass_progress_to_whole_plan(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    status = JobStatus(job_id="job")
    observer = JobStatusObserver(status, path)

    observer.handle(
        PassProgress(
            pass_id="fourda-inference",
            pass_name="4DAnyone inference",
            pass_index=3,
            pass_count=5,
            fraction=0.5,
            message="Generating views",
        ),
        PipelineContext(),
    )

    loaded = JobStatus.read(path)
    assert loaded.progress == 0.5
    assert loaded.stage == "fourda-inference"
    assert loaded.message == "Generating views"
