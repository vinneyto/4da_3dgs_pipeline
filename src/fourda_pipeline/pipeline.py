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
]
