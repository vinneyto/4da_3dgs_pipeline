"""Write a local run manifest from explicit config, artifacts and durations."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from recon_pipeline.core.utility import (
    add_result_argument,
    report_progress,
    write_result,
)
from recon_pipeline.datasets.fourdanyone.config import (
    FourDAnyoneConfig,
    load_pipeline_config,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        required=True,
        help="Local run JSON or materialized FourDAnyoneConfig JSON, including local paths",
    )
    parser.add_argument(
        "--datasets", type=Path, help="JSON list of frame/dataset_dir objects"
    )
    parser.add_argument(
        "--durations", type=Path, help="JSON object mapping pass IDs to seconds"
    )
    parser.add_argument("--rerun-file", type=Path)
    add_result_argument(parser)
    args = parser.parse_args(argv)
    document = json.loads(args.pipeline_config.read_text())
    config = (
        load_pipeline_config(args.pipeline_config)
        if "schema_version" in document
        else FourDAnyoneConfig.from_dict(document)
    )
    datasets = json.loads(args.datasets.read_text()) if args.datasets else []
    durations = json.loads(args.durations.read_text()) if args.durations else {}
    if not isinstance(datasets, list) or not isinstance(durations, dict):
        parser.error("datasets must be a JSON list and durations a JSON object")
    result = {
        "experiment_name": config.experiment_name,
        "experiment_dir": str(config.experiment_dir),
        "inference_dir": str(config.inference_dir),
        "datasets": datasets,
        "rerun_file": str(args.rerun_file) if args.rerun_file else None,
        "num_views": config.num_views,
        "pass_durations_seconds": durations,
        "elapsed_seconds": sum(durations.values()),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    path = config.experiment_dir / "pipeline-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_progress(1.0, "Run manifest written")
    write_result({"manifest": result, "path": str(path)}, args.result_file)


if __name__ == "__main__":
    main()
