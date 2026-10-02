"""Upload one experiment directory to its S3 results prefix."""

from pathlib import Path
from typing import Sequence

from recon_pipeline.core.utility import report_progress, write_result
from ..aws import delete_experiment_results, upload_directory
from ._common import experiment_name, load_worker, worker_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = worker_parser(__doc__)
    parser.add_argument("--experiment", type=experiment_name, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete this experiment's S3 prefix before uploading",
    )
    args = parser.parse_args(argv)
    worker = load_worker(args.worker_config)
    if not args.source.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {args.source}")
    if args.replace_existing:
        delete_experiment_results(worker, args.experiment)
    report_progress(0.0, "Uploading experiment results")
    uri = upload_directory(worker, args.source, args.experiment)
    report_progress(1.0, f"Results uploaded to {uri}")
    write_result({"s3_uri": uri}, args.result_file)


if __name__ == "__main__":
    main()
