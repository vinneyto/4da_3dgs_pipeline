"""Shared argument and config loading for AWS utilities; no operations."""

import argparse
import json
from pathlib import Path

from recon_pipeline.core.utility import add_result_argument
from ..config import AwsWorkerConfig


def worker_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--worker-config",
        type=Path,
        required=True,
        help="AWS worker JSON object or a run JSON containing aws_worker",
    )
    add_result_argument(parser)
    return parser


def load_worker(path: Path) -> AwsWorkerConfig:
    document = json.loads(path.read_text())
    if "aws_worker" in document:
        return AwsWorkerConfig.from_document(document)
    return AwsWorkerConfig.from_dict(document)


def experiment_name(value: str) -> str:
    if not value or any(part in value for part in ("/", "\\", "..")):
        raise argparse.ArgumentTypeError(
            "experiment name must be a simple directory name"
        )
    return value
