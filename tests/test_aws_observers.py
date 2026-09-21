from dataclasses import replace
from pathlib import Path

from recon_pipeline.core import PipelineContext
from recon_pipeline.core.command import CommandError
from recon_pipeline.core.events import (
    FinalizerFailed,
    PassCompleted,
    PassFailed,
    PassProgress,
    PassStarted,
    PipelineStarted,
    PipelineSucceeded,
)
from recon_pipeline.workers.aws import observers
from recon_pipeline.workers.aws.config import TelegramConfig
from recon_pipeline.workers.aws.observers import AwsNotificationObserver, JobStatusObserver
from recon_pipeline.workers.aws.status import JobStatus
from test_aws_health import make_config


def notification_observer(monkeypatch, tmp_path: Path):
    published: list[tuple[str, str]] = []
    telegram_sent: list[tuple[str, str]] = []
    telegram_edits: list[tuple[int, str, str]] = []
    config = replace(
        make_config(tmp_path),
        sns=None,
        telegram=TelegramConfig(chat_id="123456", bot_token="test-token"),
    )
    monkeypatch.setattr(
        observers,
        "publish_notification",
        lambda config, subject, body: (
            published.append((subject, body)) or {"telegram": "sent"}
        ),
    )
    monkeypatch.setattr(
        observers,
        "publish_telegram",
        lambda _config, subject, body: (
            telegram_sent.append((subject, body)) or 101
        ),
    )
    monkeypatch.setattr(
        observers,
        "edit_telegram",
        lambda _config, message_id, subject, body: telegram_edits.append(
            (message_id, subject, body)
        ),
    )
    monkeypatch.setattr(
        observers,
        "collect_machine_resource_status",
        lambda _config: "CPU: test · RAM: test · Disk: test\nGPU: test",
    )
    return (
        AwsNotificationObserver(config, "leo"),
        published,
        telegram_sent,
        telegram_edits,
    )


def test_notification_observer_starts_with_experiment_and_pass_plan(
    monkeypatch, tmp_path: Path
) -> None:
    observer, published, _, _ = notification_observer(monkeypatch, tmp_path)

    observer.process(
        PipelineStarted(("AWS preflight", "4DAnyone inference")),
        PipelineContext(),
    )

    assert published == [
        (
            "Pipeline started: leo",
            "Experiment: leo\nPasses:\n1. AWS preflight\n2. 4DAnyone inference",
        )
    ]


def test_notification_observer_reports_pass_start_with_resources(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, telegram_sent, telegram_edits = notification_observer(
        monkeypatch, tmp_path
    )

    observer.process(
        PassStarted("fourdanyone-inference", "4DAnyone inference", 4, 7),
        PipelineContext(),
    )

    assert telegram_sent[0][0] == "🔵 Pass 4/7 started"
    assert "4DAnyone inference" in telegram_sent[0][1]
    assert "CPU: test" in telegram_sent[0][1]
    assert "GPU: test" in telegram_sent[0][1]
    assert telegram_edits == []


def test_notification_observer_reports_pass_completion_duration_and_resources(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, telegram_sent, telegram_edits = notification_observer(
        monkeypatch, tmp_path
    )
    observer.process(
        PassStarted("fourdanyone-inference", "4DAnyone inference", 4, 7),
        PipelineContext(),
    )
    event = PassCompleted(
        pass_id="fourdanyone-inference",
        pass_name="4DAnyone inference",
        pass_index=4,
        pass_count=7,
        duration_seconds=125.0,
        next_pass_name="Nerfstudio dataset export",
    )

    observer.process(event, PipelineContext())

    assert len(telegram_sent) == 1
    assert telegram_edits[0][0] == 101
    assert telegram_edits[0][1] == "✅ Pass 4/7 completed"
    assert "Duration: 2m 5s" in telegram_edits[0][2]
    assert "CPU: test" in telegram_edits[0][2]
    assert "Next:" not in telegram_edits[0][2]


def test_notification_observer_includes_failed_command_output_and_resources(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, telegram_sent, telegram_edits = notification_observer(
        monkeypatch, tmp_path
    )
    error = CommandError(["python", "inference.py"], 1, output_tail="model output\nCUDA OOM")

    observer.process(
        PassStarted("fourdanyone-inference", "4DAnyone inference", 4, 7),
        PipelineContext(),
    )

    observer.process(
        PassFailed("fourdanyone-inference", "4DAnyone inference", 4, 7, 125.0, error),
        PipelineContext(),
    )

    assert len(telegram_sent) == 1
    assert telegram_edits[0][0] == 101
    assert telegram_edits[0][1] == "❌ Pass 4/7 failed"
    assert "Duration: 2m 5s" in telegram_edits[0][2]
    assert "CPU: test" in telegram_edits[0][2]
    assert "Output tail:\nmodel output\nCUDA OOM" in telegram_edits[0][2]


def test_notification_observer_falls_back_to_new_message_when_edit_fails(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, telegram_sent, _ = notification_observer(monkeypatch, tmp_path)
    monkeypatch.setattr(
        observers,
        "edit_telegram",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("message cannot be edited")),
    )
    context = PipelineContext()
    observer.process(PassStarted("inference", "Inference", 1, 1), context)
    observer.process(
        PassCompleted("inference", "Inference", 1, 1, 2.0, None),
        context,
    )

    assert [subject for subject, _ in telegram_sent] == [
        "🔵 Pass 1/1 started",
        "✅ Pass 1/1 completed",
    ]


def test_notification_observer_updates_running_pass_with_coarse_progress(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, telegram_sent, telegram_edits = notification_observer(
        monkeypatch, tmp_path
    )
    context = PipelineContext()
    observer.process(PassStarted("inference", "4DAnyone inference", 5, 9), context)
    observer._telegram_progress_edited_at["inference"] -= 6
    observer._telegram_pass_started_at["inference"] -= 125

    observer.process(
        PassProgress(
            "inference",
            "4DAnyone inference",
            5,
            9,
            0.45,
            "Generating target-view videos",
        ),
        context,
    )

    assert len(telegram_sent) == 1
    assert telegram_edits[0][0] == 101
    assert telegram_edits[0][1] == "🔵 Pass 5/9 running · 45%"
    assert "Stage: Generating target-view videos" in telegram_edits[0][2]
    assert "Elapsed: 2m 5s" in telegram_edits[0][2]
    assert "CPU: test" in telegram_edits[0][2]


def test_notification_observer_throttles_duplicate_or_fast_progress_updates(
    monkeypatch, tmp_path: Path
) -> None:
    observer, _, _, telegram_edits = notification_observer(monkeypatch, tmp_path)
    context = PipelineContext()
    observer.process(PassStarted("inference", "Inference", 1, 1), context)

    event = PassProgress("inference", "Inference", 1, 1, 0.1, "Preparing")
    observer.process(event, context)
    assert telegram_edits == []

    observer._telegram_progress_edited_at["inference"] -= 6
    observer.process(event, context)
    observer._telegram_progress_edited_at["inference"] -= 6
    observer.process(event, context)

    assert len(telegram_edits) == 1


def test_notification_observer_ignores_summary_and_finalizer_events(
    monkeypatch, tmp_path: Path
) -> None:
    observer, published, telegram_sent, telegram_edits = notification_observer(
        monkeypatch, tmp_path
    )
    context = PipelineContext()

    observer.process(PipelineSucceeded(10.0), context)
    observer.process(
        FinalizerFailed("shutdown", "Shutdown", 1.0, RuntimeError("denied")),
        context,
    )

    assert published == []
    assert telegram_sent == []
    assert telegram_edits == []


def test_status_observer_maps_local_pass_progress_to_whole_plan(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    status = JobStatus(job_id="job")
    observer = JobStatusObserver(status, path)

    observer.handle(
        PassProgress(
            pass_id="fourdanyone-inference",
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
    assert loaded.stage == "fourdanyone-inference"
    assert loaded.message == "Generating views"
