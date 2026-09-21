from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from recon_pipeline.core import (
    JsonPassCheckpointStore,
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
    PassSkipped,
    PipelineFinalized,
)


@dataclass
class FakePass:
    id: str
    name: str
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list)
    cleanup_calls: list[str] = field(default_factory=list)
    fail: bool = False

    def cleanup(self, context: PipelineContext) -> None:
        self.cleanup_calls.append(self.id)

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


def checkpoint_store(tmp_path) -> JsonPassCheckpointStore:
    return JsonPassCheckpointStore(tmp_path / "pass-state.json", "experiment")


def test_retry_starts_at_first_incomplete_pass_and_cleans_rerun_suffix(
    tmp_path,
) -> None:
    store = checkpoint_store(tmp_path)
    first_calls: list[str] = []
    first = FakePass("first", "First", provides=frozenset({"one"}), calls=first_calls)
    broken = FakePass(
        "broken",
        "Broken",
        requires=frozenset({"one"}),
        provides=frozenset({"two"}),
        calls=first_calls,
        fail=True,
    )
    after = FakePass(
        "after",
        "After",
        requires=frozenset({"two"}),
        provides=frozenset({"three"}),
        calls=first_calls,
    )

    with pytest.raises(RuntimeError, match="broken failed"):
        Pipeline([first, broken, after], checkpoint_store=store).run()

    assert list(store.load()) == ["first"]
    assert first_calls == ["first", "broken"]

    retry_calls: list[str] = []
    retry_first = FakePass(
        "first", "First", provides=frozenset({"one"}), calls=retry_calls
    )
    retry_broken = FakePass(
        "broken",
        "Broken",
        requires=frozenset({"one"}),
        provides=frozenset({"two"}),
        calls=retry_calls,
    )
    retry_after = FakePass(
        "after",
        "After",
        requires=frozenset({"two"}),
        provides=frozenset({"three"}),
        calls=retry_calls,
    )
    context = PipelineContext()

    Pipeline(
        [retry_first, retry_broken, retry_after], checkpoint_store=store
    ).run(context)

    assert retry_calls == ["broken", "after"]
    assert retry_first.cleanup_calls == []
    assert retry_broken.cleanup_calls == ["broken"]
    assert retry_after.cleanup_calls == ["after"]
    assert context.artifacts == {
        "one": "first:one",
        "two": "broken:two",
        "three": "after:three",
    }


def test_inserted_pass_invalidates_every_later_checkpoint(tmp_path) -> None:
    store = checkpoint_store(tmp_path)
    Pipeline(
        [
            FakePass("first", "First", provides=frozenset({"one"})),
            FakePass(
                "last",
                "Last",
                requires=frozenset({"one"}),
                provides=frozenset({"last"}),
            ),
        ],
        checkpoint_store=store,
    ).run()

    calls: list[str] = []
    observer = RecordingObserver()
    passes = [
        FakePass("first", "First", provides=frozenset({"one"}), calls=calls),
        FakePass(
            "inserted",
            "Inserted",
            requires=frozenset({"one"}),
            provides=frozenset({"middle"}),
            calls=calls,
        ),
        FakePass(
            "last",
            "Last",
            requires=frozenset({"middle"}),
            provides=frozenset({"last"}),
            calls=calls,
        ),
    ]

    Pipeline(passes, checkpoint_store=store, observers=[observer]).run()

    assert calls == ["inserted", "last"]
    assert list(store.load()) == ["first", "inserted", "last"]
    skipped = [event for event in observer.events if isinstance(event, PassSkipped)]
    assert [event.pass_id for event in skipped] == ["first"]


def test_removed_pass_invalidates_every_later_checkpoint(tmp_path) -> None:
    store = checkpoint_store(tmp_path)
    Pipeline(
        [
            FakePass("first", "First", provides=frozenset({"one"})),
            FakePass(
                "removed",
                "Removed",
                requires=frozenset({"one"}),
                provides=frozenset({"middle"}),
            ),
            FakePass(
                "last",
                "Last",
                requires=frozenset({"middle"}),
                provides=frozenset({"last"}),
            ),
        ],
        checkpoint_store=store,
    ).run()

    calls: list[str] = []
    Pipeline(
        [
            FakePass("first", "First", provides=frozenset({"one"}), calls=calls),
            FakePass(
                "last",
                "Last",
                requires=frozenset({"one"}),
                provides=frozenset({"last"}),
                calls=calls,
            ),
        ],
        checkpoint_store=store,
    ).run()

    assert calls == ["last"]
    assert list(store.load()) == ["first", "last"]


def test_completed_pipeline_skips_every_pass_unless_forced(tmp_path) -> None:
    store = checkpoint_store(tmp_path)
    original_calls: list[str] = []
    original = FakePass("only", "Only", calls=original_calls)
    Pipeline([original], checkpoint_store=store).run()

    skipped_calls: list[str] = []
    skipped = FakePass("only", "Only", calls=skipped_calls)
    Pipeline([skipped], checkpoint_store=store).run()
    assert skipped_calls == []
    assert skipped.cleanup_calls == []

    forced_calls: list[str] = []
    forced = FakePass("only", "Only", calls=forced_calls)
    Pipeline([forced], checkpoint_store=store, force=True).run()
    assert forced_calls == ["only"]
    assert forced.cleanup_calls == ["only"]
