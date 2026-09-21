"""AWS worker observers for status, console output, notifications, and monitoring."""

from __future__ import annotations

from pathlib import Path

from recon_pipeline.core import (
    PipelineContext,
    PipelineObserver,
    QueuedPipelineObserver,
)
from recon_pipeline.core.events import (
    FinalizerFailed,
    FinalizerStarted,
    PassCompleted,
    PassFailed,
    PassProgress,
    PassSkipped,
    PassStarted,
    PipelineEvent,
    PipelineFailed,
    PipelineFinalized,
    PipelineStarted,
    PipelineSucceeded,
)

from .aws import (
    get_telegram_updates,
    publish_notification,
    publish_telegram,
    stop_sagemaker_app,
)
from .artifacts import RUN_RESULT
from .config import AwsWorkerConfig
from .monitoring import TelegramRuntimeMonitoring, collect_machine_resource_status
from .status import JobStatus, utc_now


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(max(seconds, 0.0), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {seconds:.0f}s"
    if minutes:
        return f"{minutes}m {seconds:.0f}s"
    return f"{seconds:.1f}s"


def _failure_output(error: BaseException, limit: int = 3000) -> str | None:
    output = str(getattr(error, "output_tail", "")).strip()
    if not output:
        return None
    if len(output) > limit:
        output = "[... earlier output omitted ...]\n" + output[-limit:]
    return output


class ConsoleObserver(PipelineObserver):
    def handle(self, event: PipelineEvent, context: PipelineContext) -> None:
        if isinstance(event, PipelineStarted):
            print(f"Execution plan: {len(event.pass_names)} passes", flush=True)
        elif isinstance(event, PassStarted):
            print(
                f"[{event.pass_index}/{event.pass_count}] Starting {event.pass_name}",
                flush=True,
            )
        elif isinstance(event, PassSkipped):
            print(
                f"[{event.pass_index}/{event.pass_count}] Skipping {event.pass_name}: "
                f"{event.reason}",
                flush=True,
            )
        elif isinstance(event, PassProgress):
            overall = ((event.pass_index - 1) + event.fraction) / event.pass_count
            print(
                f"[{overall * 100:6.2f}%] {event.pass_id}: {event.message}",
                flush=True,
            )
        elif isinstance(event, PassCompleted):
            print(
                f"[{event.pass_index}/{event.pass_count}] {event.pass_name} completed "
                f"in {_format_duration(event.duration_seconds)}",
                flush=True,
            )
        elif isinstance(event, PassFailed):
            print(
                f"[{event.pass_index}/{event.pass_count}] {event.pass_name} failed "
                f"after {_format_duration(event.duration_seconds)}: {event.error}",
                flush=True,
            )
        elif isinstance(event, FinalizerStarted):
            print(f"Finalizer: {event.finalizer_name}", flush=True)
        elif isinstance(event, FinalizerFailed):
            print(
                f"Finalizer {event.finalizer_name} failed: {event.error}", flush=True
            )


class JobStatusObserver(PipelineObserver):
    def __init__(self, status: JobStatus, path: Path) -> None:
        self.status = status
        self.path = path

    def _write(self, **changes: object) -> None:
        self.status.update(**changes)
        self.status.write(self.path)

    def start(self, context: PipelineContext) -> None:
        self._write(
            state="running",
            stage="planning",
            message="Pipeline execution plan prepared",
            started_at=utc_now(),
        )

    def handle(self, event: PipelineEvent, context: PipelineContext) -> None:
        if isinstance(event, PassStarted):
            self._write(
                state="running",
                stage=event.pass_id,
                progress=(event.pass_index - 1) / event.pass_count,
                message=f"Starting {event.pass_name}",
            )
        elif isinstance(event, PassSkipped):
            self._write(
                state="running",
                stage=event.pass_id,
                progress=event.pass_index / event.pass_count,
                message=f"Skipping completed {event.pass_name}",
            )
        elif isinstance(event, PassProgress):
            self._write(
                state="running",
                stage=event.pass_id,
                progress=((event.pass_index - 1) + event.fraction) / event.pass_count,
                message=event.message,
            )
        elif isinstance(event, PassCompleted):
            self._write(
                state="running",
                stage=event.pass_id,
                progress=event.pass_index / event.pass_count,
                message=(
                    f"{event.pass_name} completed in "
                    f"{_format_duration(event.duration_seconds)}"
                ),
            )
        elif isinstance(event, PassFailed):
            self._write(
                state="failed",
                stage=event.pass_id,
                message=f"{event.pass_name} failed: {event.error}",
                error=repr(event.error),
            )
        elif isinstance(event, PipelineFailed):
            self._write(
                state="failed",
                stage="failed",
                message=f"Pipeline failed: {event.error}",
                error=repr(event.error),
            )
        elif isinstance(event, PipelineSucceeded):
            self._write(
                state="running",
                stage="finalizing",
                progress=1.0,
                message="Pipeline passes completed; running finalizers",
            )
        elif isinstance(event, FinalizerFailed):
            self._write(
                state="failed",
                stage=event.finalizer_id,
                message=f"Finalizer failed: {event.error}",
                error=repr(event.error),
            )
        elif isinstance(event, PipelineFinalized):
            result_path = None
            if RUN_RESULT in context.artifacts:
                result_path = str(
                    Path(context.artifacts[RUN_RESULT]["experiment_dir"])
                    / "pipeline-result.json"
                )
            error = event.error or (
                event.finalizer_errors[0] if event.finalizer_errors else None
            )
            self._write(
                state="succeeded" if event.succeeded else "failed",
                stage="complete" if event.succeeded else "failed",
                progress=1.0,
                message=(
                    "AWS job completed successfully"
                    if event.succeeded
                    else f"AWS job failed: {error}"
                ),
                error=repr(error) if error else None,
                finished_at=utc_now(),
                result_path=result_path,
            )


class AwsNotificationObserver(QueuedPipelineObserver):
    """Send concise pass lifecycle notifications; delivery remains non-critical."""

    def __init__(
        self, worker: AwsWorkerConfig, experiment_name: str
    ) -> None:
        super().__init__(thread_name=f"aws-notifications-{worker.job_id}")
        self.worker = worker
        self.experiment_name = experiment_name

    def _publish(self, subject: str, body: str) -> None:
        outcomes = publish_notification(self.worker, subject, body)
        print(f"Notifications: {outcomes or 'disabled'}", flush=True)

    def _resources(self) -> str:
        return collect_machine_resource_status(self.worker)

    def process(self, event: PipelineEvent, context: PipelineContext) -> None:
        if isinstance(event, PipelineStarted):
            passes = "\n".join(
                f"{index}. {name}"
                for index, name in enumerate(event.pass_names, start=1)
            ) or "No passes pending; all checkpoints are complete"
            self._publish(
                f"Pipeline started: {self.experiment_name}",
                f"Experiment: {self.experiment_name}\nPasses:\n{passes}",
            )
        elif isinstance(event, PassStarted):
            self._publish(
                f"Pass {event.pass_index}/{event.pass_count} started",
                f"{event.pass_name}\n{self._resources()}",
            )
        elif isinstance(event, PassCompleted):
            self._publish(
                f"Pass {event.pass_index}/{event.pass_count} completed",
                "\n".join(
                    [
                        event.pass_name,
                        f"Duration: {_format_duration(event.duration_seconds)}",
                        self._resources(),
                    ]
                ),
            )
        elif isinstance(event, PassFailed):
            output = _failure_output(event.error)
            lines = [
                event.pass_name,
                f"Duration: {_format_duration(event.duration_seconds)}",
                f"Error: {event.error}",
                self._resources(),
            ]
            if output:
                lines.extend(["", "Output tail:", output])
            self._publish(
                f"Pass {event.pass_index}/{event.pass_count} failed",
                "\n".join(lines),
            )


class RuntimeMonitoringObserver(PipelineObserver):
    def __init__(self, worker: AwsWorkerConfig) -> None:
        self.monitoring = TelegramRuntimeMonitoring(
            worker,
            lambda subject, message: publish_telegram(worker, subject, message),
            lambda offset, timeout: get_telegram_updates(worker, offset, timeout),
            lambda: stop_sagemaker_app(worker.region, worker.sagemaker),
        )

    def start(self, context: PipelineContext) -> None:
        self.monitoring.start()

    def close(self) -> None:
        self.monitoring.stop()
