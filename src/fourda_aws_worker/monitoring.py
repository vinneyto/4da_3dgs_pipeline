"""Optional Telegram log streaming and resource monitoring for a running worker."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import AwsWorkerConfig


TelegramPublisher = Callable[[str, str], None]
MAX_TELEGRAM_BODY = 3500
MAX_LOG_BACKLOG = 32000


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
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return "; ".join(rows) or "unavailable"


def collect_resource_status(
    config: AwsWorkerConfig,
    status_path: Path,
) -> str:
    """Collect a compact, dependency-free SageMaker resource snapshot."""
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        cpu = f"load {load_1m:.2f}/{load_5m:.2f}/{load_15m:.2f} on {os.cpu_count()} CPUs"
    except OSError:
        cpu = "unavailable"

    disk = shutil.disk_usage(config.local.data_root)
    disk_status = f"{disk.used / 2**30:.1f}/{disk.total / 2**30:.1f} GiB"

    job = "unavailable"
    try:
        status = json.loads(status_path.read_text())
        job = (
            f"{status.get('progress', 0.0) * 100:.1f}% · "
            f"{status.get('stage', 'unknown')} · {status.get('message', '')}"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    return "\n".join(
        [
            f"Job: {job}",
            f"CPU: {cpu}",
            f"RAM: {_memory_status()}",
            f"Disk: {disk_status}",
            f"GPU: {_gpu_status()}",
        ]
    )


class TelegramLogTailer:
    """Tail the durable worker log and send rate-limited Telegram batches."""

    def __init__(
        self,
        log_path: Path,
        job_id: str,
        publish: TelegramPublisher,
        *,
        poll_interval_seconds: float = 1.0,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self.log_path = log_path
        self.job_id = job_id
        self.publish = publish
        self.poll_interval_seconds = poll_interval_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"telegram-log-{job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.flush_interval_seconds + 5)

    def _send(self, buffer: str) -> str:
        if not buffer:
            return ""
        chunk = buffer[:MAX_TELEGRAM_BODY]
        try:
            self.publish(f"4DAnyone log: {self.job_id}", chunk)
        except Exception:
            # Notification delivery is deliberately never job-critical.
            pass
        return buffer[len(chunk) :]

    def _run(self) -> None:
        offset = 0
        buffer = ""
        last_flush = time.monotonic()
        while not self._stop.wait(self.poll_interval_seconds):
            offset, buffer = self._read(offset, buffer)
            if len(buffer) > MAX_LOG_BACKLOG:
                buffer = "[... older log output dropped ...]\n" + buffer[-MAX_LOG_BACKLOG:]
            if buffer and time.monotonic() - last_flush >= self.flush_interval_seconds:
                buffer = self._send(buffer)
                last_flush = time.monotonic()

        offset, buffer = self._read(offset, buffer)
        # Send a bounded final tail without delaying App shutdown indefinitely.
        for _ in range(3):
            if not buffer:
                break
            buffer = self._send(buffer)

    def _read(self, offset: int, buffer: str) -> tuple[int, str]:
        try:
            with self.log_path.open(errors="replace") as handle:
                handle.seek(offset)
                data = handle.read()
                return handle.tell(), buffer + data
        except OSError:
            return offset, buffer


class TelegramResourceMonitor:
    """Publish periodic job and machine resource snapshots."""

    def __init__(
        self,
        config: AwsWorkerConfig,
        status_path: Path,
        interval_seconds: int,
        publish: TelegramPublisher,
    ) -> None:
        self.config = config
        self.status_path = status_path
        self.interval_seconds = interval_seconds
        self.publish = publish
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"telegram-resources-{config.job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish(
                    f"4DAnyone resources: {self.config.job_id}",
                    collect_resource_status(self.config, self.status_path),
                )
            except Exception:
                # A Telegram outage must not affect reconstruction.
                pass


class TelegramRuntimeMonitoring:
    """Own the optional monitoring threads for one worker process."""

    def __init__(
        self,
        config: AwsWorkerConfig,
        job_dir: Path,
        publish: TelegramPublisher,
    ) -> None:
        self.monitors: list[TelegramLogTailer | TelegramResourceMonitor] = []
        telegram = config.telegram
        if telegram is None or not telegram.enabled:
            return
        if telegram.stream_logs:
            self.monitors.append(
                TelegramLogTailer(job_dir / "pipeline.log", config.job_id, publish)
            )
        if telegram.resource_status_interval_seconds is not None:
            self.monitors.append(
                TelegramResourceMonitor(
                    config,
                    job_dir / "status.json",
                    telegram.resource_status_interval_seconds,
                    publish,
                )
            )

    def start(self) -> None:
        for monitor in self.monitors:
            monitor.start()

    def stop(self) -> None:
        for monitor in reversed(self.monitors):
            monitor.stop()
