from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from recon_pipeline.core import (
    PassResult,
    Pipeline,
    PipelineContext,
    PipelineObserver,
)
from recon_pipeline.core.events import (
    FinalizerCompleted,
    FinalizerFailed,
    PassCompleted,
    PassFailed,
    PipelineFinalized,
)


@dataclass
class FakePass:
    id: str
    name: str
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list)
    fail: bool = False

    def run(self, context: PipelineContext) -> PassResult:
        self.calls.append(self.id)
        if self.fail:
            raise RuntimeError(f"{self.id} failed")
        context.report_progress(0.5, f"running {self.id}")
        return PassResult(
            artifacts={key: f"{self.id}:{key}" for key in self.provides}
        )


@dataclass
class FakeFinalizer:
    id: str
    name: str
    calls: list[str]
    fail: bool = False

    def run(self, context, outcome) -> None:
        self.calls.append(self.id)
        if self.fail:
            raise RuntimeError(f"{self.id} failed")


class RecordingObserver(PipelineObserver):
    def __init__(self, *, fail: bool = False) -> None:
        self.events = []
        self.fail = fail
        self.closed = False

    def handle(self, event, context) -> None:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("observer failed")

    def close(self) -> None:
        self.closed = True


def test_prepare_validates_ordered_artifact_dependencies() -> None:
    pipeline = Pipeline(
        [
            FakePass(
                id="export",
                name="Export",
                requires=frozenset({"experiment"}),
            )
        ]
    )

    with pytest.raises(ValueError, match="requires unavailable artifacts: experiment"):
        pipeline.prepare()


def test_passes_run_synchronously_and_publish_completion() -> None:
    calls: list[str] = []
    observer = RecordingObserver()
    first = FakePass("first", "First", provides=frozenset({"one"}), calls=calls)
    second = FakePass(
        "second",
        "Second",
        requires=frozenset({"one"}),
        provides=frozenset({"two"}),
        calls=calls,
    )
    context = PipelineContext()

    outcome = Pipeline([first, second], observers=[observer]).run(context)

    assert outcome.succeeded is True
    assert calls == ["first", "second"]
    assert context.artifacts == {"one": "first:one", "two": "second:two"}
    completed = [event for event in observer.events if isinstance(event, PassCompleted)]
    assert [event.pass_id for event in completed] == ["first", "second"]
    assert completed[0].next_pass_name == "Second"
    assert completed[1].next_pass_name is None
    assert observer.closed is True


def test_failure_stops_regular_passes_but_runs_every_finalizer() -> None:
    calls: list[str] = []
    observer = RecordingObserver()
    pipeline = Pipeline(
        [
            FakePass("before", "Before", calls=calls),
            FakePass("broken", "Broken", calls=calls, fail=True),
            FakePass("after", "After", calls=calls),
        ],
        finalizers=[
            FakeFinalizer("cleanup-a", "Cleanup A", calls, fail=True),
            FakeFinalizer("cleanup-b", "Cleanup B", calls),
        ],
        observers=[observer],
    )

    with pytest.raises(RuntimeError, match="broken failed"):
        pipeline.run()

    assert calls == ["before", "broken", "cleanup-a", "cleanup-b"]
    assert any(isinstance(event, PassFailed) for event in observer.events)
    assert any(isinstance(event, FinalizerFailed) for event in observer.events)
    assert any(isinstance(event, FinalizerCompleted) for event in observer.events)
    finalized = [
        event for event in observer.events if isinstance(event, PipelineFinalized)
    ][0]
    assert finalized.succeeded is False
    assert len(finalized.finalizer_errors) == 1


def test_observer_failure_is_non_critical() -> None:
    observer = RecordingObserver(fail=True)
    outcome = Pipeline(
        [FakePass("work", "Work")], observers=[observer]
    ).run()

    assert outcome.succeeded is True
    assert observer.closed is True
