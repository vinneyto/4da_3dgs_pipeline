"""Operation-level CLI output, with no dependency on a pipeline or worker."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, TextIO

_output: TextIO | None = None


def _emit(event: dict[str, Any]) -> None:
    print(
        json.dumps(event, sort_keys=True),
        file=_output if _output is not None else sys.stdout,
        flush=True,
    )


def report_progress(fraction: float, message: str) -> None:
    _emit({"event": "progress", "fraction": fraction, "message": message})


def run_operation(operation: Callable[[], dict[str, Any]]) -> None:
    """Keep stdout machine-readable while libraries print diagnostics to stderr."""
    global _output
    previous = _output
    _output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            result = operation()
        _emit({"event": "result", "data": result})
    finally:
        _output = previous
