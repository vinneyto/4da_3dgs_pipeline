"""Composable reconstruction operations and pipelines."""

__all__ = ["Pipeline"]


def __getattr__(name):
    if name == "Pipeline":
        from .core import Pipeline

        return Pipeline
    raise AttributeError(name)
