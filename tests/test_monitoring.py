from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from recon_pipeline.workers.aws.config import TelegramConfig
from recon_pipeline.workers.aws.monitoring import (
    TelegramCommandMonitor,
    TelegramRuntimeMonitoring,
    collect_machine_resource_status,
)
from test_aws_health import make_config


def test_runtime_monitoring_only_runs_the_shutdown_command_listener(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), sns=None)
    disabled = TelegramRuntimeMonitoring(config, lambda *_args: None)
    assert disabled.monitors == []

    enabled = replace(
        config,
        telegram=TelegramConfig(chat_id="123", bot_token="token"),
    )
    monitoring = TelegramRuntimeMonitoring(
        enabled,
        lambda *_args: None,
        lambda _offset, _timeout: [],
        lambda: None,
    )
    assert len(monitoring.monitors) == 1
    assert isinstance(monitoring.monitors[0], TelegramCommandMonitor)


def test_machine_resource_status_is_compact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.local.data_root.mkdir(parents=True)

    report = collect_machine_resource_status(config)

    assert "CPU:" in report
    assert "RAM:" in report
    assert "Disk:" in report
    assert "GPU:" in report
    assert "Job:" not in report


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
