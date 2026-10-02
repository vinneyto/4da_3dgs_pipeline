"""Shared S3 command arguments; no knowledge of jobs or experiments."""

import argparse


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--bucket", required=True)
    result.add_argument("--region", default="us-east-1")
    return result
