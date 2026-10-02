"""Check access to explicitly selected AWS resources."""

import argparse
from dataclasses import asdict
from typing import Sequence

from recon_pipeline.utilities._output import report_progress, run_operation
from .access import check_access


def check(args) -> dict:
    report_progress(0.1, "Validating AWS identity and S3 access")
    health = check_access(**vars(args))
    report_progress(1.0, "AWS dependencies are ready")
    return asdict(health)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--input-key")
    parser.add_argument(
        "--check-input", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--models-prefix")
    parser.add_argument("--write-prefix")
    parser.add_argument("--sagemaker-domain-id")
    parser.add_argument("--sagemaker-space-name")
    parser.add_argument("--sagemaker-app-name", default="default")
    args = parser.parse_args(argv)
    run_operation(lambda: check(args))


if __name__ == "__main__":
    main()
