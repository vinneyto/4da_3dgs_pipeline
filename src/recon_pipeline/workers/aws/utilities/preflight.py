"""Check AWS identity, storage and configured notification dependencies."""

from dataclasses import asdict
from typing import Sequence

from recon_pipeline.core.utility import report_progress, write_result
from ..aws import run_health_check
from ._common import load_worker, worker_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = worker_parser(__doc__)
    parser.add_argument("--require-input-video", action="store_true")
    args = parser.parse_args(argv)
    worker = load_worker(args.worker_config)
    report_progress(0.1, "Validating AWS identity and S3 access")
    health = run_health_check(
        worker, worker.job_id, require_input_video=args.require_input_video
    )
    report_progress(1.0, "AWS dependencies are ready")
    write_result(asdict(health), args.result_file)


if __name__ == "__main__":
    main()
