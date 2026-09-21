"""Synchronous pass pipeline with observable execution and guaranteed finalizers."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .events import (
    FinalizerCompleted,
    FinalizerFailed,
    FinalizerStarted,
    PassCompleted,
    PassFailed,
    PassProgress,
    PassStarted,
    PipelineEvent,
    PipelineFailed,
    PipelineFinalized,
    PipelineStarted,
    PipelineSucceeded,
)


ArtifactKey = str


@dataclass(slots=True)
class PassResult:
    artifacts: dict[ArtifactKey, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    """Mutable data shared only through explicit artifacts and run metadata."""

    artifacts: dict[ArtifactKey, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    pass_results: dict[str, PassResult] = field(default_factory=dict)
    _progress_callback: Any = field(default=None, repr=False)

    def require(self, key: ArtifactKey) -> Any:
        try:
            return self.artifacts[key]
        except KeyError as error:
            raise KeyError(f"required pipeline artifact is unavailable: {key}") from error

    def report_progress(self, fraction: float, message: str) -> None:
        if self._progress_callback is None:
            raise RuntimeError("progress can only be reported while a pass is running")
        self._progress_callback(min(max(float(fraction), 0.0), 1.0), str(message))


class PipelinePass(Protocol):
    id: str
    name: str
    requires: frozenset[ArtifactKey]
    provides: frozenset[ArtifactKey]

    def run(self, context: PipelineContext) -> PassResult: ...


class PipelineFinalizer(Protocol):
    id: str
    name: str

    def run(self, context: PipelineContext, outcome: "PipelineOutcome") -> None: ...


class PipelineObserver:
    """No-op lifecycle base for synchronous or internally asynchronous observers."""

    def start(self, context: PipelineContext) -> None:
        pass

    def handle(self, event: PipelineEvent, context: PipelineContext) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class QueuedPipelineObserver(PipelineObserver):
    """Run observer side effects on one ordered background thread."""

    def __init__(self, *, thread_name: str) -> None:
        self._queue: queue.Queue[tuple[PipelineEvent, PipelineContext] | None] = (
            queue.Queue()
        )
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self.errors: list[Exception] = []
        self._started = False

    def start(self, context: PipelineContext) -> None:
        self._thread.start()
        self._started = True

    def handle(self, event: PipelineEvent, context: PipelineContext) -> None:
        if self._started:
            self._queue.put((event, context))
        else:
            self.process(event, context)

    def process(self, event: PipelineEvent, context: PipelineContext) -> None:
        raise NotImplementedError

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                event, context = item
                self.process(event, context)
            except Exception as error:
                self.errors.append(error)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        if self._started:
            self._queue.join()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._queue.join()
            self._thread.join(timeout=5)


class PipelineEventBus:
    """Best-effort fan-out: observer failures never fail computation."""

    def __init__(self, observers: list[PipelineObserver] | None = None) -> None:
        self.observers = list(observers or [])
        self.errors: list[Exception] = []

    def _invoke(self, method: str, *arguments: Any) -> None:
        for observer in self.observers:
            try:
                getattr(observer, method)(*arguments)
            except Exception as error:
                self.errors.append(error)

    def start(self, context: PipelineContext) -> None:
        self._invoke("start", context)

    def publish(self, event: PipelineEvent, context: PipelineContext) -> None:
        self._invoke("handle", event, context)

    def flush(self) -> None:
        self._invoke("flush")

    def close(self) -> None:
        for observer in reversed(self.observers):
            try:
                observer.close()
            except Exception as error:
                self.errors.append(error)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    passes: tuple[PipelinePass, ...]
    finalizers: tuple[PipelineFinalizer, ...]
    initial_artifacts: frozenset[ArtifactKey]
    produced_artifacts: frozenset[ArtifactKey]


@dataclass(slots=True)
class PipelineOutcome:
    succeeded: bool
    duration_seconds: float = 0.0
    error: BaseException | None = None
    finalizer_errors: list[BaseException] = field(default_factory=list)


class PipelineFinalizationError(RuntimeError):
    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"{len(errors)} pipeline finalizer(s) failed")


class Pipeline:
    """Execute an explicit pass list; events only observe and never schedule work."""

    def __init__(
        self,
        passes: list[PipelinePass],
        *,
        finalizers: list[PipelineFinalizer] | None = None,
        observers: list[PipelineObserver] | None = None,
    ) -> None:
        self.passes = list(passes)
        self.finalizers = list(finalizers or [])
        self.events = PipelineEventBus(observers)

    def prepare(
        self, initial_artifacts: set[ArtifactKey] | frozenset[ArtifactKey] = frozenset()
    ) -> ExecutionPlan:
        """Validate the ordered artifact contract without executing side effects."""
        available = set(initial_artifacts)
        ids: set[str] = set()
        for pipeline_pass in self.passes:
            if not pipeline_pass.id or pipeline_pass.id in ids:
                raise ValueError(f"pipeline pass id must be unique: {pipeline_pass.id!r}")
            ids.add(pipeline_pass.id)
            missing = set(pipeline_pass.requires) - available
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"pass {pipeline_pass.id!r} requires unavailable artifacts: {names}"
                )
            duplicates = set(pipeline_pass.provides) & available
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(
                    f"pass {pipeline_pass.id!r} would replace existing artifacts: {names}"
                )
            available.update(pipeline_pass.provides)
        finalizer_ids: set[str] = set()
        for finalizer in self.finalizers:
            if not finalizer.id or finalizer.id in finalizer_ids:
                raise ValueError(f"finalizer id must be unique: {finalizer.id!r}")
            finalizer_ids.add(finalizer.id)
        return ExecutionPlan(
            passes=tuple(self.passes),
            finalizers=tuple(self.finalizers),
            initial_artifacts=frozenset(initial_artifacts),
            produced_artifacts=frozenset(available),
        )

    def run(self, context: PipelineContext | None = None) -> PipelineOutcome:
        context = context or PipelineContext()
        plan = self.prepare(set(context.artifacts))
        pass_count = len(plan.passes)
        pipeline_started = time.monotonic()
        context.values["pipeline_started_monotonic"] = pipeline_started
        context.values.setdefault("pass_durations", {})
        outcome = PipelineOutcome(succeeded=False)
        self.events.start(context)
        self.events.publish(
            PipelineStarted(tuple(item.name for item in plan.passes)), context
        )
        try:
            for index, pipeline_pass in enumerate(plan.passes):
                pass_index = index + 1
                self.events.publish(
                    PassStarted(
                        pipeline_pass.id,
                        pipeline_pass.name,
                        pass_index,
                        pass_count,
                    ),
                    context,
                )
                pass_started = time.monotonic()

                def report(fraction: float, message: str) -> None:
                    self.events.publish(
                        PassProgress(
                            pipeline_pass.id,
                            pipeline_pass.name,
                            pass_index,
                            pass_count,
                            fraction,
                            message,
                        ),
                        context,
                    )

                context._progress_callback = report
                try:
                    result = pipeline_pass.run(context)
                    if not isinstance(result, PassResult):
                        raise TypeError(
                            f"pass {pipeline_pass.id!r} must return PassResult"
                        )
                    missing = set(pipeline_pass.provides) - set(result.artifacts)
                    unexpected = set(result.artifacts) - set(pipeline_pass.provides)
                    if missing or unexpected:
                        raise ValueError(
                            f"pass {pipeline_pass.id!r} returned an invalid artifact set; "
                            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                        )
                    context.artifacts.update(result.artifacts)
                    context.pass_results[pipeline_pass.id] = result
                except BaseException as error:
                    self.events.publish(
                        PassFailed(
                            pipeline_pass.id,
                            pipeline_pass.name,
                            pass_index,
                            pass_count,
                            time.monotonic() - pass_started,
                            error,
                        ),
                        context,
                    )
                    raise
                finally:
                    context._progress_callback = None
                next_name = (
                    plan.passes[index + 1].name if index + 1 < pass_count else None
                )
                duration = time.monotonic() - pass_started
                context.values["pass_durations"][pipeline_pass.id] = duration
                self.events.publish(
                    PassCompleted(
                        pipeline_pass.id,
                        pipeline_pass.name,
                        pass_index,
                        pass_count,
                        duration,
                        next_name,
                        dict(result.details),
                    ),
                    context,
                )
            outcome.succeeded = True
            outcome.duration_seconds = time.monotonic() - pipeline_started
            self.events.publish(PipelineSucceeded(outcome.duration_seconds), context)
        except BaseException as error:
            outcome.error = error
            outcome.duration_seconds = time.monotonic() - pipeline_started
            self.events.publish(
                PipelineFailed(outcome.duration_seconds, error), context
            )
        finally:
            # Ensure queued pass/failure notifications leave before a shutdown finalizer.
            self.events.flush()
            for finalizer in plan.finalizers:
                self.events.publish(
                    FinalizerStarted(finalizer.id, finalizer.name), context
                )
                finalizer_started = time.monotonic()
                try:
                    finalizer.run(context, outcome)
                except BaseException as error:
                    outcome.finalizer_errors.append(error)
                    self.events.publish(
                        FinalizerFailed(
                            finalizer.id,
                            finalizer.name,
                            time.monotonic() - finalizer_started,
                            error,
                        ),
                        context,
                    )
                else:
                    self.events.publish(
                        FinalizerCompleted(
                            finalizer.id,
                            finalizer.name,
                            time.monotonic() - finalizer_started,
                        ),
                        context,
                    )
            outcome.duration_seconds = time.monotonic() - pipeline_started
            finalized_success = outcome.error is None and not outcome.finalizer_errors
            self.events.publish(
                PipelineFinalized(
                    finalized_success,
                    outcome.duration_seconds,
                    outcome.error,
                    tuple(outcome.finalizer_errors),
                ),
                context,
            )
            self.events.flush()
            self.events.close()

        if outcome.error is not None:
            raise outcome.error
        if outcome.finalizer_errors:
            raise PipelineFinalizationError(outcome.finalizer_errors)
        return outcome
