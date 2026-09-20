"""Blocking command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_pipeline_config
from .pipeline import FourDAnyonePipeline
from .progress import ProgressUpdate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fourda-pipeline",
        description="Run 4DAnyone and export synchronized Nerfstudio/3DGS input",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the run JSON document")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)

    def print_progress(update: ProgressUpdate) -> None:
        print(f"[{update.fraction * 100:6.2f}%] {update.stage}: {update.message}", flush=True)

    result = FourDAnyonePipeline(config, on_progress=print_progress).run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
