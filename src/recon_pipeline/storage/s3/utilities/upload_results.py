"""Upload a local directory into an explicit S3 prefix."""

from pathlib import Path
from typing import Sequence
from recon_pipeline.cli import report_progress, run_operation
from ..operations import delete_prefix, upload_directory
from ._arguments import parser


def upload(args) -> dict:
    if not args.source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {args.source}")
    if args.replace_existing:
        delete_prefix(bucket=args.bucket, prefix=args.prefix, region=args.region)
    report_progress(0.0, "Uploading experiment results")
    uri = upload_directory(
        bucket=args.bucket, prefix=args.prefix, source=args.source, region=args.region
    )
    report_progress(1.0, f"Results uploaded to {uri}")
    return {"s3_uri": uri}


def main(argv: Sequence[str] | None = None) -> None:
    p = parser(__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete only the selected S3 prefix before uploading",
    )
    args = p.parse_args(argv)
    run_operation(lambda: upload(args))


if __name__ == "__main__":
    main()
