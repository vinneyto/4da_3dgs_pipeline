"""Launch a utility and interpret its stdout events inside a pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .command import CommandRunner
from .core import PipelineContext


def _event(line: str) -> dict[str, Any] | None:
    if line.startswith("FOURDA_PROGRESS "):
        line = line.removeprefix("FOURDA_PROGRESS ")
        try:
            return {"event": "progress", **json.loads(line)}
        except (TypeError, ValueError):
            return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def handle_progress(context: PipelineContext, line: str) -> None:
    event = _event(line)
    if event is None or event.get("event") != "progress":
        return
    try:
        fraction = float(event["fraction"])
        message = str(event["message"])
    except (KeyError, TypeError, ValueError):
        return
    context.report_progress(fraction, message)


def run_utility(
    module: str,
    arguments: Sequence[str],
    context: PipelineContext,
    *,
    runner: CommandRunner | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = None

    def on_stdout(line: str) -> None:
        nonlocal result
        event = _event(line)
        if event is not None and event.get("event") == "result":
            if result is not None:
                raise RuntimeError(f"utility {module} emitted multiple results")
            if not isinstance(event.get("data"), dict):
                raise TypeError(f"utility {module} must return a JSON object")
            result = event["data"]
        handle_progress(context, line)

    options = {"env": env} if env is not None else {}
    (runner or CommandRunner()).run(
        [sys.executable, "-u", "-m", module, *arguments],
        cwd=Path.cwd(),
        on_line=on_stdout,
        **options,
    )
    if result is None:
        raise RuntimeError(f"utility {module} completed without a stdout result")
    return result
