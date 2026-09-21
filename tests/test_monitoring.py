from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from fourda_aws_worker.config import TelegramConfig
from fourda_aws_worker.monitoring import (
    TelegramLogTailer,
    TelegramCommandMonitor,
    TelegramRuntimeMonitoring,
    collect_resource_status,
)
from test_aws_health import make_config


def test_telegram_resource_interval_has_a_safe_minimum() -> None:
    with pytest.raises(ValueError, match="at least 60"):
        TelegramConfig(
            chat_id="123",
            bot_token="token",
            resource_status_interval_seconds=30,
        )


def test_log_tailer_publishes_new_log_content(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("")
    messages: list[tuple[str, str]] = []
    tailer = TelegramLogTailer(
        log_path,
        "job-01",
        lambda subject, body: messages.append((subject, body)),
        poll_interval_seconds=0.01,
        flush_interval_seconds=0.02,
    )

    tailer.start()
    with log_path.open("a") as handle:
        handle.write("first line\nsecond line\n")
        handle.flush()
    time.sleep(0.08)
    tailer.stop()

    assert messages
    assert messages[0][0] == "4DAnyone log: job-01"
    assert "first line\nsecond line" in "".join(body for _, body in messages)


def test_runtime_monitoring_is_opt_in(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), sns=None)
    disabled = TelegramRuntimeMonitoring(config, tmp_path, lambda *_args: None)
    assert disabled.monitors == []

    enabled = replace(
        config,
        telegram=TelegramConfig(
            chat_id="123",
            bot_token="token",
            stream_logs=True,
            resource_status_interval_seconds=60,
        ),
    )
    monitoring = TelegramRuntimeMonitoring(enabled, tmp_path, lambda *_args: None)
    assert len(monitoring.monitors) == 2


def test_resource_status_contains_job_and_machine_sections(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.local.data_root.mkdir(parents=True)
    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"progress": 0.2381, "stage": "inference", "message": "Recovering motion"}'
    )

    report = collect_resource_status(config, status_path)

    assert "Job: 23.8% · inference · Recovering motion" in report
    assert "CPU:" in report
    assert "RAM:" in report
    assert "Disk:" in report
    assert "GPU:" in report


def test_shutdown_command_stops_only_the_authorized_app(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        telegram=TelegramConfig(chat_id="123", bot_token="token"),
    )
    messages: list[tuple[str, str]] = []
    shutdowns: list[bool] = []
    monitor = TelegramCommandMonitor(
        config,
        lambda subject, body: messages.append((subject, body)),
        lambda _offset, _timeout: [],
        lambda: shutdowns.append(True),
    )

    monitor._handle(
        {
            "update_id": 1,
            "message": {
                "chat": {"id": 123},
                "from": {"id": 123},
                "text": "/shutdown",
            },
        }
    )

    assert shutdowns == [True]
    assert messages[0][0] == f"Stopping SageMaker App: {config.job_id}"


def test_shutdown_command_rejects_another_chat_or_user(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        telegram=TelegramConfig(
            chat_id="-1000",
            allowed_user_id="123",
            bot_token="token",
        ),
    )
    shutdowns: list[bool] = []
    monitor = TelegramCommandMonitor(
        config,
        lambda *_args: None,
        lambda _offset, _timeout: [],
        lambda: shutdowns.append(True),
    )

    monitor._handle(
        {
            "update_id": 1,
            "message": {
                "chat": {"id": -1000},
                "from": {"id": 999},
                "text": "/shutdown",
            },
        }
    )

    assert shutdowns == []
