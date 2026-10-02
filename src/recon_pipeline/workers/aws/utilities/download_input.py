"""Download one configured input video from S3."""

from typing import Sequence

from recon_pipeline.core.utility import report_progress, write_result
from ..aws import download_input_video
from ._common import load_worker, worker_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = worker_parser(__doc__)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    worker = load_worker(args.worker_config)
    if args.replace_existing:
        path = worker.local_video_path
        for target in (path, path.with_suffix(path.suffix + ".download")):
            if target.exists():
                target.unlink()
    report_progress(0.0, f"Downloading {worker.video_s3_uri}")
    path = download_input_video(worker)
    report_progress(1.0, f"Input video ready at {path}")
    write_result({"path": str(path), "s3_uri": worker.video_s3_uri}, args.result_file)


if __name__ == "__main__":
    main()
