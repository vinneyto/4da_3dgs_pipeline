"""Manage detached 4DAnyone jobs and their AWS integrations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .aws_integration import configure_email
from .job import BackgroundJob
from .job_config import JobConfig, load_document, load_job_config
from .status import JobStatus


def _print_status(status: JobStatus, as_json: bool = False) -> None:
    payload = asdict(status)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"job:      {status.job_id}")
    print(f"state:    {status.state}")
    print(f"stage:    {status.stage}")
    print(f"progress: {status.progress * 100:.2f}%")
    print(f"message:  {status.message}")
    print(f"updated:  {status.updated_at}")
    if status.result_path:
        print(f"result:   {status.result_path}")
    if status.error:
        print(f"error:    {status.error}")


def _logs(job: BackgroundJob, lines: int, follow: bool) -> None:
    if not job.log_path.is_file():
        raise FileNotFoundError(f"job log does not exist yet: {job.log_path}")
    initial = job.log_path.read_text(errors="replace").splitlines()
    for line in initial[-lines:]:
        print(line)
    if not follow:
        return
    with job.log_path.open(errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                print(line, end="", flush=True)
                continue
            if job.status().terminal:
                return
            time.sleep(1)


def _job_from_config(path: Path) -> tuple[BackgroundJob, JobConfig]:
    document = load_document(path)
    job_config = JobConfig.from_document(document)
    return BackgroundJob(job_config.job_id, job_config.jobs_dir), job_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fourda-job")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("configure-email", "Create the configured SNS topic and email subscription"),
        ("start", "Start a detached background job"),
        ("status", "Show durable job state"),
        ("logs", "Show or follow job output"),
        ("stop", "Send SIGTERM to the worker process group"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        if name == "status":
            command.add_argument("--json", action="store_true")
        if name == "logs":
            command.add_argument("--lines", type=int, default=100)
            command.add_argument("--follow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "configure-email":
        _, job_config = _job_from_config(args.config)
        result = configure_email(job_config.aws)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Confirm the AWS Subscription Confirmation email before relying on notifications.")
        return

    job, job_config = _job_from_config(args.config)
    if args.command == "start":
        pipeline_config, _ = load_job_config(args.config)
        request = {
            "pipeline": pipeline_config.to_dict(),
            "job": {
                "job_id": job_config.job_id,
                "jobs_dir": str(job_config.jobs_dir),
                "shutdown_on": job_config.shutdown_on,
            },
            "aws": asdict(job_config.aws),
        }
        status = job.start(request)
        _print_status(status)
        print(f"log:      {job.log_path}")
    elif args.command == "status":
        _print_status(job.status(), args.json)
    elif args.command == "logs":
        _logs(job, args.lines, args.follow)
    elif args.command == "stop":
        _print_status(job.stop())
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
