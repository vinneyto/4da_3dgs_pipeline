"""Download an S3 object to an explicit local file."""

from pathlib import Path
from typing import Sequence
from recon_pipeline.cli import report_progress, run_operation
from ..operations import download_file
from ._arguments import parser


def download(args) -> dict:
    if args.replace_existing:
        for target in (
            args.output,
            args.output.with_suffix(args.output.suffix + ".download"),
        ):
            if target.exists():
                target.unlink()
    uri = f"s3://{args.bucket}/{args.key}"
    report_progress(0.0, f"Downloading {uri}")
    path = download_file(
        bucket=args.bucket, key=args.key, destination=args.output, region=args.region
    )
    report_progress(1.0, f"Input video ready at {path}")
    return {"path": str(path), "s3_uri": uri}


def main(argv: Sequence[str] | None = None) -> None:
    p = parser(__doc__)
    p.add_argument("--key", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replace-existing", action="store_true")
    args = p.parse_args(argv)
    run_operation(lambda: download(args))


if __name__ == "__main__":
    main()
