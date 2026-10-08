"""Stream worker output to CloudWatch, with an acknowledged on-disk retry spool."""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from recon_pipeline.core import PipelineObserver
from recon_pipeline.core.events import FinalizerStarted, PassStarted, PassSkipped


def new_attempt_id():
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]


def _json_default(value):
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value) or repr(value),
            "traceback": "".join(traceback.format_exception(value)),
        }
    return str(value)


class _LogStream:
    """Preserve the original terminal/file output; only the cloud copy is buffered."""

    chunk_characters = 16_000

    def __init__(self, original, session, channel):
        self.original, self.session, self.channel = original, session, channel
        self.buffer = ""
        self.lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, text):
        with self.lock:
            result = self.original.write(text)
            self.buffer += text
            while "\n" in self.buffer or len(self.buffer) >= self.chunk_characters:
                newline = self.buffer.find("\n")
                length = min(
                    newline + 1 if newline >= 0 else len(self.buffer),
                    self.chunk_characters,
                )
                part, self.buffer = self.buffer[:length], self.buffer[length:]
                self.session.record(
                    "output", channel=self.channel, text=part.rstrip("\n")
                )
            return result

    def flush(self):
        with self.lock:
            self.original.flush()

    def flush_partial(self):
        with self.lock:
            if self.buffer:
                self.session.record("output", channel=self.channel, text=self.buffer)
                self.buffer = ""


class CloudWatchLogSession(PipelineObserver):
    """One stream per attempt; old unacknowledged attempts retry on the same disk."""

    interval_seconds = 5
    batch_bytes = 1024 * 1024
    batch_events = 10_000

    def __init__(self, worker, config, job_dir, *, client=None, attempt_id=None):
        self.worker, self.config, self.job_dir = worker, config, Path(job_dir)
        self.attempt = attempt_id or new_attempt_id()
        self.log_group = worker.cloudwatch.log_group
        self.log_stream = f"{config.experiment_name}/{self.attempt}"
        self.spool_dir = self.job_dir / "cloudwatch"
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.spool = self.spool_dir / (self.attempt + ".jsonl")
        self.metadata_path = self.spool.with_suffix(".json")
        self.metadata_path.write_text(
            json.dumps(
                {
                    "region": worker.region,
                    "log_group": self.log_group,
                    "log_stream": self.log_stream,
                    "offset": 0,
                }
            )
            + "\n"
        )
        self.spool.touch()
        self.client = client
        self.clients = {}
        self.created = set()
        self.pass_id = "planning"
        self.last_error = None
        self.capture_error = None
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._loop, name="cloudwatch-logs", daemon=True
        )
        self.stdout = _LogStream(sys.stdout, self, "stdout")
        self.stderr = _LogStream(sys.stderr, self, "stderr")
        token = worker.telegram
        self.secrets = []
        if token:
            import os

            secret = token.bot_token or os.environ.get(token.bot_token_env or "")
            if secret:
                self.secrets.append(secret)

    def describe(self):
        encode = lambda value: quote(quote(value, safe=""), safe="").replace("%", "$")
        return {
            "region": self.worker.region,
            "log_group": self.log_group,
            "log_stream": self.log_stream,
            "console_url": (
                f"https://console.aws.amazon.com/cloudwatch/home?region={self.worker.region}"
                f"#logsV2:log-groups/log-group/{encode(self.log_group)}"
                f"/log-events/{encode(self.log_stream)}"
            ),
            "last_error": self.capture_error or self.last_error,
        }

    def record(self, event, **fields):
        message = json.dumps(
            {
                "event": event,
                "experiment_name": self.config.experiment_name,
                "job_id": self.worker.job_id,
                "attempt_id": self.attempt,
                "pass_id": self.pass_id,
                **fields,
            },
            ensure_ascii=False,
            default=_json_default,
        )
        for secret in self.secrets:
            message = message.replace(secret, "[REDACTED]")
        # Output is chunked by _LogStream. Structured tracebacks may be larger.
        # Preserve valid JSON for Insights even for oversized structured events.
        for start in range(0, len(message), 200_000):
            part = message[start : start + 200_000]
            if len(message) > 200_000:
                part = json.dumps(
                    {
                        "event": "event_fragment",
                        "original_event": event,
                        "experiment_name": self.config.experiment_name,
                        "job_id": self.worker.job_id,
                        "attempt_id": self.attempt,
                        "pass_id": self.pass_id,
                        "fragment_index": start // 200_000,
                        "fragment_count": (len(message) + 199_999) // 200_000,
                        "payload_fragment": part,
                    },
                    ensure_ascii=False,
                )
            event_record = {
                "timestamp": time.time_ns() // 1_000_000,
                "message": part,
            }
            try:
                with self.lock, self.spool.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event_record, ensure_ascii=False) + "\n")
            except OSError as error:
                self.capture_error = repr(error)
                self.stderr.original.write(
                    f"CloudWatch spool write failed: {error!r}\n"
                )

    def handle(self, event, context):
        if isinstance(event, (PassStarted, PassSkipped)):
            self.pass_id = event.pass_id
        elif isinstance(event, FinalizerStarted):
            self.pass_id = event.finalizer_id
        self.record(
            type(event).__name__,
            **{item.name: getattr(event, item.name) for item in fields(event)},
        )

    def start(self, context):
        context.values["cloudwatch_session"] = self

    def _client(self, region):
        if self.client is not None:
            return self.client
        if region not in self.clients:
            import boto3
            from botocore.config import Config

            self.clients[region] = boto3.client(
                "logs",
                region_name=region,
                config=Config(
                    connect_timeout=3,
                    read_timeout=5,
                    retries={"mode": "standard", "total_max_attempts": 2},
                ),
            )
        return self.clients[region]

    def _send(self, deadline):
        """Advance a byte cursor only after AWS accepts every event in its batch."""
        for spool in sorted(self.spool_dir.glob("*.jsonl")):
            path = spool.with_suffix(".json")
            metadata = json.loads(path.read_text())
            client = self._client(metadata["region"])
            target = (metadata["region"], metadata["log_group"], metadata["log_stream"])
            with spool.open("rb") as stream:
                stream.seek(metadata["offset"])
                while time.monotonic() < deadline:
                    events, size = [], 0
                    offset = stream.tell()
                    while len(events) < self.batch_events:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            break
                        record = json.loads(line)
                        event_size = len(record["message"].encode("utf-8")) + 26
                        if size + event_size > self.batch_bytes or (
                            events
                            and abs(record["timestamp"] - events[0]["timestamp"])
                            > 24 * 60 * 60 * 1000
                        ):
                            stream.seek(start)
                            break
                        events.append(record)
                        size += event_size
                        offset = stream.tell()
                    if not events:
                        break
                    if target not in self.created:
                        try:
                            client.create_log_stream(
                                logGroupName=target[1], logStreamName=target[2]
                            )
                        except Exception as error:
                            if (
                                getattr(error, "response", {})
                                .get("Error", {})
                                .get("Code")
                                != "ResourceAlreadyExistsException"
                            ):
                                raise
                        self.created.add(target)
                    response = client.put_log_events(
                        logGroupName=target[1],
                        logStreamName=target[2],
                        logEvents=sorted(events, key=lambda event: event["timestamp"]),
                    )
                    if response.get("rejectedLogEventsInfo") or response.get(
                        "rejectedEntityInfo"
                    ):
                        raise RuntimeError(
                            f"CloudWatch rejected log events: {response!r}"
                        )
                    metadata["offset"] = offset
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(metadata) + "\n")
                    temporary.replace(path)
                if metadata["offset"] == spool.stat().st_size:
                    # Keep only unacknowledged bytes between attempts.
                    if spool != self.spool:
                        spool.unlink()
                        path.unlink()

    def _pending(self):
        return any(
            path.stat().st_size
            > json.loads(path.with_suffix(".json").read_text())["offset"]
            for path in self.spool_dir.glob("*.jsonl")
        )

    def flush(self, timeout=20):
        self.stdout.flush_partial()
        self.stderr.flush_partial()
        deadline = time.monotonic() + timeout
        if not self.send_lock.acquire(timeout=timeout):
            raise TimeoutError("CloudWatch sender is still busy")
        try:
            self._send(deadline)
            if self.capture_error:
                raise RuntimeError(
                    f"CloudWatch log capture failed: {self.capture_error}"
                )
            if self._pending():
                raise TimeoutError("CloudWatch logs remain in the local retry spool")
            self.last_error = None
        except Exception as error:
            self.last_error = repr(error)
            raise
        finally:
            self.send_lock.release()

    def flush_before_shutdown(self, context):
        marker = self.job_dir / "cloudwatch-persistence-failed"
        try:
            self.flush()
        except Exception as error:
            context.values["cloudwatch_persistence_failed"] = True
            marker.touch()
            self.stderr.original.write(
                f"CloudWatch delivery failed; shutdown deferred: {error!r}\n"
            )
        else:
            context.values.pop("cloudwatch_persistence_failed", None)
            marker.unlink(missing_ok=True)

    def _loop(self):
        while not self.stop.wait(self.interval_seconds):
            try:
                self.flush()
            except Exception as error:
                # Do not log sender errors through the sender itself.
                self.stderr.original.write(
                    f"CloudWatch delivery retry pending: {error!r}\n"
                )
                self.stderr.original.flush()

    def __enter__(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.record("worker_started")
        self.thread.start()
        return self

    def __exit__(self, error_type, error, tb):
        try:
            self.stop.set()
            self.thread.join(timeout=30)
            if error:
                self.record("worker_error", error=error)
            self.record("worker_finished")
            try:
                self.flush()
            except Exception as flush_error:
                (self.job_dir / "cloudwatch-persistence-failed").touch()
                self.stderr.original.write(
                    f"CloudWatch logs retained locally: {flush_error!r}\n"
                )
            else:
                (self.job_dir / "cloudwatch-persistence-failed").unlink(missing_ok=True)
        finally:
            sys.stdout, sys.stderr = self.stdout.original, self.stderr.original


class CloudWatchFlushFinalizer:
    id = "cloudwatch-flush"
    name = "Flush CloudWatch logs before shutdown"

    def __init__(self, session):
        self.session = session

    def run(self, context, outcome):
        self.session.flush_before_shutdown(context)
