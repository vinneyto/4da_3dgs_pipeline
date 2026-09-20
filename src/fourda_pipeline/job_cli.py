"""Manage detached 4DAnyone jobs inside a SageMaker JupyterLab App."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from .aws_integration import configure_email
from .cli_common import add_pipeline_arguments, config_from_args
from .job import BackgroundJob
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fourda-job")
    commands = parser.add_subparsers(dest="command", required=True)

    email = commands.add_parser("configure-email", help="Create SNS topic and email subscription")
    email.add_argument("--email", required=True)
    default_id = os.environ.get("CP_DEPLOYMENT_ID", "default")
    email.add_argument("--topic-name", default=f"cp-4da-pipeline-{default_id}")
    email.add_argument("--region", default=os.environ.get("CP_AWS_REGION", "us-east-1"))

    start = commands.add_parser("start", help="Start a detached background job")
    add_pipeline_arguments(start)
    start.add_argument("--job-id", help="Defaults to experiment name")
    start.add_argument("--shutdown-on", choices=("never", "success", "always"), default="never")
    start.add_argument("--sns-topic-arn")

    status = commands.add_parser("status", help="Show durable job state")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true")

    logs = commands.add_parser("logs", help="Show or follow job output")
    logs.add_argument("job_id")
    logs.add_argument("--lines", type=int, default=100)
    logs.add_argument("--follow", action="store_true")

    stop = commands.add_parser("stop", help="Send SIGTERM to the worker process group")
    stop.add_argument("job_id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "configure-email":
        result = configure_email(args.email, args.topic_name, args.region)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Confirm the AWS Subscription Confirmation email before relying on notifications.")
        return

    if args.command == "start":
        config = config_from_args(args)
        job = BackgroundJob(args.job_id or config.experiment_name)
        request = {
            "pipeline": config.to_dict(),
            "shutdown_on": args.shutdown_on,
            "sns_topic_arn": args.sns_topic_arn,
            "region": os.environ.get("CP_AWS_REGION", "us-east-1"),
        }
        status = job.start(request)
        _print_status(status)
        print(f"log:      {job.log_path}")
        return

    job = BackgroundJob(args.job_id)
    if args.command == "status":
        _print_status(job.status(), args.json)
    elif args.command == "logs":
        _logs(job, args.lines, args.follow)
    elif args.command == "stop":
        _print_status(job.stop())
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
