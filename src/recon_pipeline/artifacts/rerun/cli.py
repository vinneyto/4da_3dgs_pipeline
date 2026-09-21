"""Standalone CLI for rebuilding a Rerun artifact without rerunning inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .exporter import RerunExporter


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recon-rerun")
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--fourdanyone-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    output = RerunExporter(
        generation=args.generation,
        output=args.output,
        experiment=args.experiment,
        fourdanyone_root=args.fourdanyone_root,
        model_dir=args.model_dir,
        view_count=args.view_count,
        device=args.device,
        on_progress=lambda current, total, message: print(message, flush=True),
    ).export()
    print(f"Rerun recording written: {output}", flush=True)


if __name__ == "__main__":
    main()
