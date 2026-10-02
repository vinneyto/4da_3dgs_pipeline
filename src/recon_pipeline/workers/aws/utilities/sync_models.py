"""Synchronize a persistent model cache from S3."""

from typing import Sequence

from recon_pipeline.core.utility import report_progress, write_result
from ..aws import sync_model_objects
from ._common import load_worker, worker_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = worker_parser(__doc__)
    args = parser.parse_args(argv)
    worker = load_worker(args.worker_config)
    report_progress(0.0, "Synchronizing model objects")
    found, downloaded = sync_model_objects(worker)
    report_progress(1.0, "Model cache is ready")
    write_result(
        {
            "path": str(worker.local.model_dir),
            "objects": found,
            "downloaded": downloaded,
        },
        args.result_file,
    )


if __name__ == "__main__":
    main()
