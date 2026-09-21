"""AWS-owned pipeline finalizers."""

from .sagemaker_shutdown import SageMakerShutdownFinalizer

__all__ = ["SageMakerShutdownFinalizer"]
