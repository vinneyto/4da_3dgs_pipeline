"""JSON results and progress for independently runnable Python utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .command import CommandRunner
from .core import PipelineContext

PROGRESS_PREFIX = "RECON_PROGRESS "


def report_progress(fraction: float, message: str) -> None:
    print(
        PROGRESS_PREFIX + json.dumps({"fraction": fraction, "message": message}),
        flush=True,
    )


def handle_progress(context: PipelineContext, line: str) -> None:
    # Keep compatibility with native progress emitted by the older runner.
    prefix = next(
        (p for p in (PROGRESS_PREFIX, "FOURDA_PROGRESS ") if line.startswith(p)), None
    )
    if prefix is None:
        return
    try:
        payload = json.loads(line[len(prefix) :])
        fraction = float(payload["fraction"])
        message = str(payload["message"])
    except (KeyError, TypeError, ValueError):
        return
    context.report_progress(fraction, message)


def add_result_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Write the JSON result to this file instead of stdout",
    )


def write_result(result: dict[str, Any], path: Path | None) -> None:
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(content, end="", flush=True)
    else:
        path.write_text(content)


def run_utility(
    module: str,
    arguments: Sequence[str],
    context: PipelineContext,
    *,
    runner: CommandRunner | None = None,
    documents: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Launch one CLI, bridging only config files, results and progress."""
    with TemporaryDirectory(prefix="recon-utility-") as directory:
        command = [sys.executable, "-u", "-m", module, *arguments]
        for option, document in (documents or {}).items():
            path = Path(directory) / (option.removeprefix("--") + ".json")
            path.write_text(json.dumps(document) + "\n")
            command.extend([option, str(path)])
        result_path = Path(directory) / "result.json"
        command.extend(["--result-file", str(result_path)])
        (runner or CommandRunner()).run(
            command,
            cwd=Path.cwd(),
            on_line=lambda line: handle_progress(context, line),
        )
        if not result_path.is_file():
            raise RuntimeError(f"utility {module} completed without a JSON result")
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise TypeError(f"utility {module} must return a JSON object")
        return result
