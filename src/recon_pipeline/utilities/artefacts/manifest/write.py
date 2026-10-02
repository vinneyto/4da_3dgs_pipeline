"""Write an experiment manifest from explicit metadata and artifact paths."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from recon_pipeline.utilities._output import run_operation, report_progress


def write_manifest(args: argparse.Namespace) -> dict:
    if (
        not isinstance(args.datasets, list)
        or not isinstance(args.durations, dict)
        or not isinstance(args.reconstructions, list)
        or not isinstance(args.reconstruction_recordings, list)
    ):
        raise ValueError("datasets must be a JSON list and durations a JSON object")
    result = {
        "experiment_name": args.experiment_name,
        "experiment_dir": str(args.experiment_dir),
        "inference_dir": str(args.inference_dir),
        "datasets": args.datasets,
        "reconstructions": args.reconstructions,
        "reconstruction_recordings": args.reconstruction_recordings,
        "rerun_file": str(args.rerun_file) if args.rerun_file else None,
        "num_views": args.num_views,
        "pass_durations_seconds": args.durations,
        "elapsed_seconds": sum(args.durations.values()),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_progress(1.0, "Run manifest written")
    return {"manifest": result, "path": str(args.output)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument(
        "--datasets",
        type=json.loads,
        default=[],
        help="JSON list of frame/dataset_dir objects",
    )
    parser.add_argument(
        "--durations",
        type=json.loads,
        default={},
        help="JSON object mapping operation names to seconds",
    )
    parser.add_argument(
        "--reconstructions",
        type=json.loads,
        default=[],
        help="JSON list of trained frame results",
    )
    parser.add_argument("--reconstruction-recordings", type=json.loads, default=[])
    parser.add_argument("--rerun-file", type=Path)
    args = parser.parse_args(argv)
    run_operation(lambda: write_manifest(args))


if __name__ == "__main__":
    main()
