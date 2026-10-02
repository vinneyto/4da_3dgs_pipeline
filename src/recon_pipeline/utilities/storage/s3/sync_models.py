"""Synchronize model objects from an S3 prefix into a local cache."""

import argparse
from pathlib import Path
from typing import Sequence
from recon_pipeline.utilities._output import report_progress, run_operation
from .operations import sync_prefix
from ._arguments import parser


def synchronize(args) -> dict:
    report_progress(0.0, "Synchronizing model objects")
    found, downloaded = (
        sync_prefix(
            bucket=args.bucket,
            prefix=args.prefix,
            destination=args.destination,
            region=args.region,
        )
        if args.sync
        else (0, 0)
    )
    report_progress(1.0, "Model cache is ready")
    return {"path": str(args.destination), "objects": found, "downloaded": downloaded}


def main(argv: Sequence[str] | None = None) -> None:
    p = parser(__doc__)
    p.add_argument("--prefix", default="models")
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    run_operation(lambda: synchronize(args))


if __name__ == "__main__":
    main()
