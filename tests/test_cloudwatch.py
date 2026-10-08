import json
import sys
from types import SimpleNamespace

import pytest

from recon_pipeline.core import PipelineContext, PipelineOutcome
from recon_pipeline.core.command import CommandError, CommandRunner
from recon_pipeline.core.events import PassFailed, PassStarted
from recon_pipeline.workers.aws.cloudwatch import CloudWatchLogSession
from recon_pipeline.workers.aws.config import CloudWatchConfig
from recon_pipeline.workers.aws.finalizers import sagemaker_shutdown
from recon_pipeline.workers.aws.persistence import S3PersistenceObserver
from recon_pipeline.workers.aws.pipeline import build_aws_pipeline
from recon_pipeline.workers.aws.status import JobStatus
from test_aws_health import make_config as make_worker
from test_aws_pipeline_plan import make_pipeline_config


class Logs:
    def __init__(self):
        self.streams = []
        self.batches = []
        self.failure = None
        self.response = {}

    def create_log_stream(self, **kwargs):
        self.streams.append(kwargs)

    def put_log_events(self, **kwargs):
        if self.failure:
            raise self.failure
        self.batches.append(kwargs)
        return self.response

    def messages(self):
        return [json.loads(e["message"]) for b in self.batches for e in b["logEvents"]]


def session(tmp_path, client=None, attempt="attempt-one"):
    worker = SimpleNamespace(
        cloudwatch=CloudWatchConfig("/recon-pipeline/test"),
        region="us-east-1",
        job_id="job",
        telegram=None,
    )
    return CloudWatchLogSession(
        worker,
        SimpleNamespace(experiment_name="experiment"),
        tmp_path / "job",
        client=client or Logs(),
        attempt_id=attempt,
    )


def test_worker_and_subprocess_output_and_failure_are_streamed(tmp_path, capsys):
    logs = Logs()
    with session(tmp_path, logs) as capture:
        context = PipelineContext()
        capture.start(context)
        capture.handle(PassStarted("frame:0", "Train", 1, 1), context)
        print("worker output")
        with pytest.raises(CommandError) as failure:
            CommandRunner().run(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('training output'); print('compiler failure', file=sys.stderr); sys.exit(1)",
                ],
                cwd=tmp_path,
            )
        capture.handle(PassFailed("frame:0", "Train", 1, 1, 2, failure.value), context)
        print("partial", end="")
    messages = logs.messages()
    output = [m for m in messages if m["event"] == "output"]
    assert {m["text"] for m in output} >= {
        "worker output",
        "training output",
        "compiler failure",
        "partial",
    }
    assert all(m["pass_id"] == "frame:0" for m in output)
    error = next(m["error"] for m in messages if m["event"] == "PassFailed")
    assert error["type"] == "CommandError" and "CommandError" in error["traceback"]
    assert "worker output" in capsys.readouterr().out
    assert not (capture.job_dir / "cloudwatch-persistence-failed").exists()


def test_retry_acknowledges_only_accepted_events_and_replays_old_attempt(tmp_path):
    logs = Logs()
    first = session(tmp_path, logs)
    first.record("old-output", text="first attempt")
    logs.failure = OSError("offline")
    with pytest.raises(OSError):
        first.flush()
    assert json.loads(first.metadata_path.read_text())["offset"] == 0
    second = session(tmp_path, logs, attempt="attempt-two")
    second.record("new-output", text="second attempt")
    logs.failure = None
    second.flush()
    assert [b["logStreamName"] for b in logs.batches] == [
        "experiment/attempt-one",
        "experiment/attempt-two",
    ]
    assert not first.spool.exists()
    count = len(logs.batches)
    second.flush()
    assert len(logs.batches) == count
    assert second.describe()["last_error"] is None


def test_partial_rejection_keeps_cursor_and_prevents_shutdown(tmp_path, monkeypatch):
    logs = Logs()
    capture = session(tmp_path, logs)
    capture.record("output", text="important failure")
    logs.response = {"rejectedLogEventsInfo": {"tooOldLogEventEndIndex": 0}}
    context = PipelineContext(values={"cloudwatch_session": capture})
    worker = SimpleNamespace(shutdown_on="always", region="region", sagemaker=None)
    stopped = []
    monkeypatch.setattr(
        sagemaker_shutdown, "stop_sagemaker_app", lambda *args: stopped.append(args)
    )
    sagemaker_shutdown.SageMakerShutdownFinalizer(worker).run(
        context, PipelineOutcome(succeeded=True)
    )
    assert not stopped
    assert context.values["cloudwatch_persistence_failed"]
    assert (capture.job_dir / "cloudwatch-persistence-failed").is_file()
    assert json.loads(capture.metadata_path.read_text())["offset"] == 0
    logs.response = {}
    sagemaker_shutdown.SageMakerShutdownFinalizer(worker).run(
        context, PipelineOutcome(succeeded=True)
    )
    assert stopped and not (capture.job_dir / "cloudwatch-persistence-failed").exists()


def test_utf8_batch_limits_and_no_sequence_token(tmp_path):
    logs = Logs()
    capture = session(tmp_path, logs)
    for _ in range(100):
        capture.stdout.write("я" * 16000 + "\n")
    capture.flush()
    assert len(logs.batches) > 1
    assert sum(len(b["logEvents"]) for b in logs.batches) == 200
    for batch in logs.batches:
        assert "sequenceToken" not in batch
        assert (
            sum(len(e["message"].encode("utf-8")) + 26 for e in batch["logEvents"])
            <= 1024 * 1024
        )
        assert len(batch["logEvents"]) <= 10000
        assert batch["logEvents"] == sorted(
            batch["logEvents"], key=lambda e: e["timestamp"]
        )


def test_backlog_batches_span_no_more_than_24_hours(tmp_path):
    logs = Logs()
    capture = session(tmp_path, logs)
    capture.spool.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": 1000 + day * 25 * 60 * 60 * 1000,
                    "message": "{}",
                }
            )
            + "\n"
            for day in range(3)
        )
    )
    capture.flush()
    assert len(logs.batches) == 3


def test_oversized_structured_event_remains_valid_json_and_reassembles(tmp_path):
    capture = session(tmp_path)
    text = "я" * 300000
    capture.record("failure", text=text)
    capture.flush()
    fragments = capture.client.messages()
    assert all(m["event"] == "event_fragment" for m in fragments)
    assert [m["fragment_index"] for m in fragments] == list(range(len(fragments)))
    restored = json.loads("".join(m["payload_fragment"] for m in fragments))
    assert restored["event"] == "failure" and restored["text"] == text


def test_configured_token_is_redacted_and_blank_error_has_a_type(tmp_path):
    capture = session(tmp_path)
    capture.secrets = ["secret-token"]
    capture.record("failure", text="secret-token", error=AssertionError())
    capture.flush()
    message = capture.client.messages()[0]
    assert message["text"] == "[REDACTED]"
    assert message["error"]["type"] == "AssertionError"
    assert message["error"]["message"] == "AssertionError()"


def test_spool_failure_is_not_silently_ignored(tmp_path):
    capture = session(tmp_path)
    capture.spool.unlink()
    capture.spool.mkdir()
    capture.record("output", text="unsaved")
    context = PipelineContext()
    capture.flush_before_shutdown(context)
    assert context.values["cloudwatch_persistence_failed"]
    assert capture.describe()["last_error"]


def test_report_points_to_attempt_stream_and_falls_back_to_s3_logs(
    tmp_path, monkeypatch
):
    worker, config = make_worker(tmp_path), make_pipeline_config(tmp_path)
    capture = session(tmp_path)
    persistence = S3PersistenceObserver(
        worker, config, tmp_path / "job", log_session=capture
    )
    arguments = []
    monkeypatch.setattr(
        "recon_pipeline.workers.aws.persistence.run_utility",
        lambda module, args, context: arguments.append(args),
    )
    persistence.upload(full=True)
    report = json.loads(persistence.report_path.read_text())
    assert report["attempt_id"] == capture.attempt
    assert report["cloudwatch_logs"]["log_stream"] == capture.log_stream
    assert report["cloudwatch_logs"]["console_url"].startswith(
        "https://console.aws.amazon.com/"
    )
    assert "--skip-logs" in arguments[-1]
    capture.last_error = "offline"
    persistence.upload(full=True)
    assert "--skip-logs" not in arguments[-1] and "--full" in arguments[-1]


def test_finalizers_save_diagnostics_and_flush_before_shutdown(tmp_path):
    worker, config = make_worker(tmp_path), make_pipeline_config(tmp_path)
    capture = session(tmp_path)
    pipeline = build_aws_pipeline(
        worker,
        config,
        tmp_path / "job",
        JobStatus(job_id=worker.job_id),
        log_session=capture,
    )
    assert [f.id for f in pipeline.prepare().finalizers] == [
        "s3-save-diagnostics",
        "cloudwatch-flush",
        "sagemaker-shutdown",
    ]


@pytest.mark.parametrize("delivery_fails", [False, True])
def test_worker_validates_delivery_before_computation_and_captures_failure(
    tmp_path, monkeypatch, delivery_fails
):
    from recon_pipeline.core import Pipeline
    from recon_pipeline.workers.aws.worker import _run_experiment
    from test_pipeline_core import FakePass

    logs = Logs()
    capture = session(tmp_path, logs)
    JobStatus(job_id="job").write(capture.job_dir / "status.json")
    computation = FakePass("frame:0", "Train", fail=True)
    builds = []

    def build(worker, config, job_dir, status, **options):
        builds.append(options)
        return Pipeline([computation], observers=[options["log_session"]])

    monkeypatch.setattr("recon_pipeline.workers.aws.worker.build_aws_pipeline", build)
    worker = SimpleNamespace(upload_results=False, shutdown_on="never")
    if delivery_fails:
        logs.failure = OSError("CloudWatch access denied")
    with capture:
        assert (
            _run_experiment(capture.job_dir, {}, worker, capture.config, capture) == 1
        )
    if delivery_fails:
        assert not builds and not computation.calls
        assert JobStatus.read(capture.job_dir / "status.json").stage == "planning"
        assert (capture.job_dir / "cloudwatch-persistence-failed").exists()
    else:
        assert computation.calls == ["frame:0"]
        assert any(m["event"] == "PassFailed" for m in logs.messages())
        assert any(
            "RuntimeError: frame:0 failed" in m.get("text", "") for m in logs.messages()
        )


@pytest.mark.parametrize("group", ["", "bad:group", "x" * 513, None])
def test_invalid_group_is_rejected(group):
    with pytest.raises(ValueError):
        CloudWatchConfig(group)
