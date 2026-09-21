"""Streaming subprocess execution."""

from __future__ import annotations

import subprocess
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path


LineCallback = Callable[[str], None]


class CommandError(RuntimeError):
    def __init__(
        self,
        command: Sequence[str],
        return_code: int,
        *,
        output_tail: str = "",
    ) -> None:
        super().__init__(f"command failed with exit code {return_code}: {' '.join(command)}")
        self.command = tuple(command)
        self.return_code = return_code
        self.output_tail = output_tail


class CommandRunner:
    """Run a command while preserving live log output."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        on_line: LineCallback | None = None,
    ) -> None:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_tail: deque[str] = deque(maxlen=200)
        for line in process.stdout:
            print(line, end="", flush=True)
            output_tail.append(line.rstrip("\n"))
            if on_line is not None:
                on_line(line.rstrip("\n"))
        return_code = process.wait()
        if return_code:
            raise CommandError(
                command,
                return_code,
                output_tail="\n".join(output_tail),
            )
