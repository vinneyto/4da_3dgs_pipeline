"""4DAnyone subprocess entry point that exposes its native progress events."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROGRESS_PREFIX = "FOURDA_PROGRESS "


class JsonProgressHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "fraction": float(getattr(record, "fraction", 0.0)),
            "message": record.getMessage(),
        }
        print(PROGRESS_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    request = json.loads(args.request.read_text())
    root = Path(request.pop("fourdanyone_root")).expanduser().resolve()
    sys.path.insert(0, str(root))

    progress = logging.getLogger("fdanyone.progress")
    progress.setLevel(logging.INFO)
    progress.addHandler(JsonProgressHandler())

    # Importing the public entry point preserves 4DAnyone's allocator and
    # attention-backend bootstrap before the heavy model modules are loaded.
    from inference import inference

    result = inference(**request)
    print("FOURDA_RESULT " + json.dumps(result, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
