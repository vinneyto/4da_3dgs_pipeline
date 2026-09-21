"""Cloud-independent execution events emitted by the pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineStarted:
    pass_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PassStarted:
    pass_id: str
    pass_name: str
    pass_index: int
    pass_count: int


@dataclass(frozen=True, slots=True)
class PassSkipped:
    pass_id: str
    pass_name: str
    pass_index: int
    pass_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class PassProgress:
    pass_id: str
    pass_name: str
    pass_index: int
    pass_count: int
    fraction: float
    message: str


@dataclass(frozen=True, slots=True)
class PassCompleted:
    pass_id: str
    pass_name: str
    pass_index: int
    pass_count: int
    duration_seconds: float
    next_pass_name: str | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PassFailed:
    pass_id: str
    pass_name: str
    pass_index: int
    pass_count: int
    duration_seconds: float
    error: BaseException


@dataclass(frozen=True, slots=True)
class PipelineSucceeded:
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PipelineFailed:
    duration_seconds: float
    error: BaseException


@dataclass(frozen=True, slots=True)
class FinalizerStarted:
    finalizer_id: str
    finalizer_name: str


@dataclass(frozen=True, slots=True)
class FinalizerCompleted:
    finalizer_id: str
    finalizer_name: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class FinalizerFailed:
    finalizer_id: str
    finalizer_name: str
    duration_seconds: float
    error: BaseException


@dataclass(frozen=True, slots=True)
class PipelineFinalized:
    succeeded: bool
    duration_seconds: float
    error: BaseException | None
    finalizer_errors: tuple[BaseException, ...]


PipelineEvent = (
    PipelineStarted
    | PassStarted
    | PassSkipped
    | PassProgress
    | PassCompleted
    | PassFailed
    | PipelineSucceeded
    | PipelineFailed
    | FinalizerStarted
    | FinalizerCompleted
    | FinalizerFailed
    | PipelineFinalized
)
