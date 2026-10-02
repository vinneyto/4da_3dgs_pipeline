"""Streaming subprocess execution."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
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
        super().__init__(
            f"command failed with exit code {return_code}: {' '.join(command)}"
        )
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
        env: Mapping[str, str] | None = None,
    ) -> None:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, **env} if env is not None else None,
        )
        assert process.stdout is not None and process.stderr is not None
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        output_tail: deque[str] = deque(maxlen=200)

        def read(stream, channel):
            try:
                for line in stream:
                    lines.put((channel, line))
            finally:
                stream.close()
                lines.put((channel, None))

        threads = [
            threading.Thread(target=read, args=(stream, channel), daemon=True)
            for stream, channel in (
                (process.stdout, "stdout"),
                (process.stderr, "stderr"),
            )
        ]
        for thread in threads:
            thread.start()
        try:
            closed = 0
            while closed < 2:
                channel, line = lines.get()
                if line is None:
                    closed += 1
                    continue
                print(
                    line,
                    end="",
                    file=sys.stdout if channel == "stdout" else sys.stderr,
                    flush=True,
                )
                output_tail.append(line.rstrip("\n"))
                if channel == "stdout" and on_line is not None:
                    on_line(line.rstrip("\n"))
        except BaseException:
            if process.poll() is None:
                process.kill()
            raise
        finally:
            process.wait()
            for thread in threads:
                thread.join(timeout=5)
        if process.returncode:
            raise CommandError(
                command, process.returncode, output_tail="\n".join(output_tail)
            )
