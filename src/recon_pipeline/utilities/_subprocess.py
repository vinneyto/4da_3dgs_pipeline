"""Stream external command diagnostics while retaining complete operation logs."""

import subprocess
import sys
from pathlib import Path


def run_logged(command, log: Path, env, progress=None):
    print("$ " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                # A callback may consume noisy training rows after reporting progress.
                if progress is None or progress(line) is not False:
                    print(line, end="", file=sys.stderr, flush=True)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
