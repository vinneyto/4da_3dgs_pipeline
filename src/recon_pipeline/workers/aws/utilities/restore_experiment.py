"""Restore one experiment from S3 for independent artifact generation."""

import shutil
from pathlib import Path
from typing import Sequence

from recon_pipeline.core.utility import report_progress, write_result
from ..aws import sync_experiment_results
from ._common import experiment_name, load_worker, worker_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = worker_parser(__doc__)
    parser.add_argument("--experiment", type=experiment_name, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    worker = load_worker(args.worker_config)
    generation = args.destination / "4danyone"
    required = (generation / "metadata.json", generation / "cameras.json")
    if args.replace_existing and generation.exists():
        shutil.rmtree(generation)
    if all(path.is_file() for path in required):
        found = downloaded = 0
        message = "Reusing source experiment from persistent storage"
    else:
        report_progress(0.0, "Restoring source experiment from S3")
        found, downloaded = sync_experiment_results(
            worker, args.experiment, args.destination
        )
        message = "Source experiment restored"
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            f"restored experiment requires metadata.json and cameras.json: {generation}"
        )
    report_progress(1.0, message)
    write_result(
        {"path": str(generation), "objects": found, "downloaded": downloaded},
        args.result_file,
    )


if __name__ == "__main__":
    main()
