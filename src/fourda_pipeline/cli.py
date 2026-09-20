"""Blocking command-line entry point."""

from __future__ import annotations

import argparse
import json

from .cli_common import add_pipeline_arguments, config_from_args
from .pipeline import FourDAnyonePipeline
from .progress import ProgressUpdate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fourda-pipeline",
        description="Run 4DAnyone and export synchronized Nerfstudio/3DGS input",
    )
    add_pipeline_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    def print_progress(update: ProgressUpdate) -> None:
        print(f"[{update.fraction * 100:6.2f}%] {update.stage}: {update.message}", flush=True)

    result = FourDAnyonePipeline(config, on_progress=print_progress).run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
