"""Persistent job process managed by the AWS worker."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .status import JobStatus, utc_now


def validate_job_id(job_id: str) -> str:
    if not job_id or any(part in job_id for part in ("/", "\\", "..")):
        raise ValueError("job_id must be a simple directory name")
    return job_id


class AwsBackgroundJob:
    def __init__(self, job_id: str, jobs_root: Path) -> None:
        self.job_id = validate_job_id(job_id)
        if not jobs_root.is_absolute():
            raise ValueError(f"jobs_root must be absolute: {jobs_root}")
        self.root = jobs_root / self.job_id
        self.request_path = self.root / "request.json"
        self.status_path = self.root / "status.json"
        self.log_path = self.root / "pipeline.log"

    def _archive_previous_attempt(self) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.root / "attempts" / timestamp
        destination.mkdir(parents=True, exist_ok=False)
        for path in (self.request_path, self.status_path, self.log_path):
            if path.exists():
                path.replace(destination / path.name)

    @staticmethod
    def _process_is_running(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def start(self, request: dict[str, Any]) -> JobStatus:
        if self.status_path.is_file():
            existing = JobStatus.read(self.status_path)
            if not existing.terminal and self._process_is_running(existing.pid):
                raise RuntimeError(f"job already exists and is {existing.state}: {self.job_id}")
            self._archive_previous_attempt()

        self.root.mkdir(parents=True, exist_ok=True)
        self.request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        status = JobStatus(job_id=self.job_id)
        status.write(self.status_path)

        log_handle = self.log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            [sys.executable, "-m", "recon_pipeline.workers.aws.worker", "--job-dir", str(self.root)],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        log_handle.close()
        # The worker can reach "running" before Popen returns. Re-read the
        # file before recording its PID to preserve any newer worker state.
        status = JobStatus.read(self.status_path)
        status.update(pid=process.pid)
        if status.state == "queued":
            status.message = "AWS background worker started"
        status.write(self.status_path)
        return status

    def status(self) -> JobStatus:
        if not self.status_path.is_file():
            raise FileNotFoundError(f"unknown AWS job: {self.job_id}")
        return JobStatus.read(self.status_path)

    def stop(self) -> JobStatus:
        status = self.status()
        if status.terminal:
            return status
        if not status.pid:
            raise RuntimeError("AWS job has no worker PID")
        try:
            os.killpg(status.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        status.update(
            state="cancelled",
            stage="cancelled",
            message="Cancellation requested",
            finished_at=utc_now(),
        )
        status.write(self.status_path)
        return status
