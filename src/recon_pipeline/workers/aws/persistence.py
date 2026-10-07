"""Periodic cloud diagnostics and a blocking report before managed shutdown."""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from recon_pipeline.core import PipelineContext, PipelineObserver
from recon_pipeline.core.events import (
    PassStarted,
    PassFailed,
    PassCompleted,
    PassSkipped,
)
from recon_pipeline.core.utility import run_utility


def exception_report(error):
    return {
        "type": type(error).__name__,
        "message": str(error) or repr(error),
        "repr": repr(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "command": list(getattr(error, "command", [])),
        "return_code": getattr(error, "return_code", None),
        "output_tail": getattr(error, "output_tail", None),
    }


class S3PersistenceObserver(PipelineObserver):
    interval_seconds = 60

    def __init__(self, worker, config, job_dir):
        self.worker, self.config, self.job_dir = worker, config, Path(job_dir)
        self.attempt = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        )
        self.prefix = "/".join(
            filter(None, (worker.runs_prefix, config.experiment_name))
        )
        self.attempt_prefix = self.prefix + "/.recon-pipeline/attempts/" + self.attempt
        self.report_path = self.job_dir / "attempt-report.json"
        self.document = {
            "schema_version": 1,
            "experiment_name": config.experiment_name,
            "job_id": worker.job_id,
            "attempt_id": self.attempt,
            "run_prefix": self.prefix,
            "state": "running",
            "pass_id": None,
            "completed_passes": [],
        }
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self, context):
        self.thread.start()

    def handle(self, event, context):
        with self.lock:
            if isinstance(event, PassStarted):
                self.document["pass_id"] = event.pass_id
            elif isinstance(event, PassFailed):
                self.document.update(
                    state="failed",
                    pass_id=event.pass_id,
                    error=exception_report(event.error),
                )
            elif isinstance(event, (PassCompleted, PassSkipped)):
                self.document["completed_passes"].append(event.pass_id)

    def _save_report(self):
        with self.lock:
            self.document["updated_at"] = datetime.now(UTC).isoformat()
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.document, indent=2) + "\n")
            temporary.replace(self.report_path)

    def upload(self, full=False):
        self._save_report()
        context = PipelineContext(_progress_callback=lambda fraction, message: None)
        return run_utility(
            "recon_pipeline.utilities.storage.s3.upload_diagnostics",
            [
                "--bucket",
                self.worker.bucket_name,
                "--region",
                self.worker.region,
                "--prefix",
                self.attempt_prefix,
                "--root",
                str(self.config.experiment_dir),
                "--job-dir",
                str(self.job_dir),
                "--report",
                str(self.report_path),
                *(["--full"] if full else []),
            ],
            context,
        )

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.upload()
            except Exception as error:
                print(f"Periodic S3 diagnostics failed: {error!r}", flush=True)
            self.stop.wait(self.interval_seconds)

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()

    def finish(self, context, outcome):
        self.close()
        with self.lock:
            self.document["state"] = (
                "cancelled"
                if isinstance(outcome.error, KeyboardInterrupt)
                else "failed" if outcome.error else "succeeded"
            )
            self.document["duration_seconds"] = outcome.duration_seconds
            self.document["pass_durations_seconds"] = dict(
                context.values.get("pass_durations", {})
            )
            if outcome.error:
                self.document["error"] = exception_report(outcome.error)
                self.document["pass_id"] = context.values.get(
                    "failed_pass", self.document["pass_id"]
                )
        try:
            self.upload(full=True)
            if str(context.values.get("failed_pass", "")).startswith("s3-upload:"):
                # A saved error report does not replace the unpublished artifact.
                context.values["s3_persistence_failed"] = True
                self.job_dir.mkdir(parents=True, exist_ok=True)
                (self.job_dir / "s3-persistence-failed").touch()
            else:
                (self.job_dir / "s3-persistence-failed").unlink(missing_ok=True)
        except BaseException:
            context.values["s3_persistence_failed"] = True
            self.job_dir.mkdir(parents=True, exist_ok=True)
            (self.job_dir / "s3-persistence-failed").touch()
            raise


class S3DiagnosticsFinalizer:
    id = "s3-save-diagnostics"
    name = "Save experiment status and diagnostics to S3"

    def __init__(self, persistence):
        self.persistence = persistence

    def run(self, context, outcome):
        self.persistence.finish(context, outcome)
