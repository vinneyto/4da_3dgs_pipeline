"""Event-scoped resource snapshots and Telegram command handling."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import AwsWorkerConfig


TelegramPublisher = Callable[[str, str], None]
TelegramUpdateFetcher = Callable[[int | None, int], list[dict[str, Any]]]
SageMakerShutdown = Callable[[], None]


def _memory_status() -> str:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError):
        return "unavailable"
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = max(total - available, 0)
    return f"{used / 2**30:.1f}/{total / 2**30:.1f} GiB"


def _gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"

    formatted: list[str] = []
    for row in result.stdout.splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) == 5:
            name, utilization, used, total, temperature = values
            formatted.append(
                f"{name} · {utilization}% · {used}/{total} MiB · {temperature}°C"
            )
        elif row.strip():
            formatted.append(row.strip())
    return "; ".join(formatted) or "unavailable"


def collect_machine_resource_status(config: AwsWorkerConfig) -> str:
    """Collect a compact machine snapshot for a pass lifecycle notification."""
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        cpu = (
            f"{load_1m:.2f}/{load_5m:.2f}/{load_15m:.2f} "
            f"on {os.cpu_count()} CPUs"
        )
    except OSError:
        cpu = "unavailable"

    try:
        disk = shutil.disk_usage(config.local.data_root)
        disk_status = f"{disk.used / 2**30:.1f}/{disk.total / 2**30:.1f} GiB"
    except OSError:
        disk_status = "unavailable"

    return "\n".join(
        [
            f"CPU: {cpu} · RAM: {_memory_status()} · Disk: {disk_status}",
            f"GPU: {_gpu_status()}",
        ]
    )


class TelegramCommandMonitor:
    """Accept authenticated control commands for exactly one worker App."""

    def __init__(
        self,
        config: AwsWorkerConfig,
        publish: TelegramPublisher,
        get_updates: TelegramUpdateFetcher,
        shutdown: SageMakerShutdown,
        *,
        poll_timeout_seconds: int = 20,
    ) -> None:
        self.config = config
        self.publish = publish
        self.get_updates = get_updates
        self.shutdown = shutdown
        self.poll_timeout_seconds = poll_timeout_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"telegram-commands-{config.job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.poll_timeout_seconds + 5)

    def _authorized(self, message: dict[str, Any]) -> bool:
        telegram = self.config.telegram
        assert telegram is not None
        chat_id = str(message.get("chat", {}).get("id", ""))
        user_id = str(message.get("from", {}).get("id", ""))
        return chat_id == telegram.chat_id and user_id == telegram.shutdown_user_id

    def _handle(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        text = str(message.get("text", "")).strip()
        command, _, argument = text.partition(" ")
        command = command.split("@", 1)[0].casefold()
        if command != "/shutdown" or not self._authorized(message):
            return
        requested_job = argument.strip()
        if requested_job and requested_job != self.config.job_id:
            self.publish(
                f"Shutdown ignored: {self.config.job_id}",
                f"Command targeted a different job: {requested_job}",
            )
            return
        self.publish(
            f"Stopping SageMaker App: {self.config.job_id}",
            "Authorized /shutdown command received. The worker and its current pass "
            "will be terminated when SageMaker stops the App.",
        )
        try:
            self.shutdown()
        except Exception as error:
            self.publish(
                f"SageMaker shutdown failed: {self.config.job_id}",
                f"{type(error).__name__}: {error}",
            )
        else:
            self._stop.set()

    def _run(self) -> None:
        offset: int | None = None
        try:
            # Never execute a command that was queued before this worker started.
            pending = self.get_updates(None, 0)
            if pending:
                offset = max(int(item["update_id"]) for item in pending) + 1
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                updates = self.get_updates(offset, self.poll_timeout_seconds)
                for update in updates:
                    offset = int(update["update_id"]) + 1
                    self._handle(update)
                    if self._stop.is_set():
                        break
            except Exception:
                # Bot polling is control-plane convenience, never job-critical.
                self._stop.wait(5)


class TelegramRuntimeMonitoring:
    """Own the optional Telegram command monitor for one worker process."""

    def __init__(
        self,
        config: AwsWorkerConfig,
        publish: TelegramPublisher,
        get_updates: TelegramUpdateFetcher | None = None,
        shutdown: SageMakerShutdown | None = None,
    ) -> None:
        self.monitors: list[TelegramCommandMonitor] = []
        telegram = config.telegram
        if (
            telegram is not None
            and telegram.enabled
            and telegram.shutdown_command
            and get_updates is not None
            and shutdown is not None
        ):
            self.monitors.append(
                TelegramCommandMonitor(config, publish, get_updates, shutdown)
            )

    def start(self) -> None:
        for monitor in self.monitors:
            monitor.start()

    def stop(self) -> None:
        for monitor in reversed(self.monitors):
            monitor.stop()
