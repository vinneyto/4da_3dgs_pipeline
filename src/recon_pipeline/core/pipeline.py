"""Public pipeline API.

Domain packages provide passes; this module only exposes the synchronous core.
"""

from .core import (
    ExecutionPlan,
    PassResult,
    Pipeline,
    PipelineContext,
    PipelineFinalizationError,
    PipelineFinalizer,
    PipelineObserver,
    PipelineOutcome,
    PipelinePass,
    QueuedPipelineObserver,
)
from .checkpoints import JsonPassCheckpointStore, PassCheckpoint, PassCheckpointStore

__all__ = [
    "ExecutionPlan",
    "PassResult",
    "Pipeline",
    "PipelineContext",
    "PipelineFinalizationError",
    "PipelineFinalizer",
    "PipelineObserver",
    "PipelineOutcome",
    "PipelinePass",
    "QueuedPipelineObserver",
    "JsonPassCheckpointStore",
    "PassCheckpoint",
    "PassCheckpointStore",
]
