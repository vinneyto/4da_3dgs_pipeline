"""Manage detached reconstruction pipeline jobs on AWS."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .aws import configure_email
from .config import AwsWorkerConfig, load_aws_worker_config, load_document
from .job import AwsBackgroundJob
from .pipeline import build_aws_pipeline
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


def _logs(job: AwsBackgroundJob, lines: int, follow: bool) -> None:
    if not job.log_path.is_file():
        raise FileNotFoundError(f"AWS job log does not exist yet: {job.log_path}")
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


def _job_from_config(path: Path) -> tuple[AwsBackgroundJob, AwsWorkerConfig]:
    config = AwsWorkerConfig.from_document(load_document(path))
    return AwsBackgroundJob(config.job_id, config.jobs_dir), config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recon-aws-worker")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("configure-email", "Create the configured SNS topic and email subscription"),
        ("plan", "Print the ordered pass plan without calling AWS"),
        ("start", "Start a detached AWS background job"),
        ("status", "Show durable AWS job state"),
        ("logs", "Show or follow AWS job output"),
        ("stop", "Send SIGTERM to the AWS worker process group"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        if name == "start":
            command.add_argument(
                "--force",
                action="store_true",
                help="Ignore all completed pass checkpoints and rerun from the beginning",
            )
        if name == "status":
            command.add_argument("--json", action="store_true")
        if name == "logs":
            command.add_argument("--lines", type=int, default=100)
            command.add_argument("--follow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "configure-email":
        _, config = _job_from_config(args.config)
        result = configure_email(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Confirm the AWS Subscription Confirmation email before relying on notifications.")
        return

    if args.command == "plan":
        pipeline_config, config = load_aws_worker_config(args.config)
        job = AwsBackgroundJob(config.job_id, config.jobs_dir)
        pipeline = build_aws_pipeline(
            config,
            pipeline_config,
            job.root,
            JobStatus(job_id=config.job_id),
        )
        plan = pipeline.prepare()
        for index, pipeline_pass in enumerate(plan.passes, start=1):
            requires = ", ".join(sorted(pipeline_pass.requires)) or "-"
            provides = ", ".join(sorted(pipeline_pass.provides)) or "-"
            print(f"{index}. {pipeline_pass.id} — {pipeline_pass.name}")
            print(f"   requires: {requires}")
            print(f"   provides: {provides}")
        for finalizer in plan.finalizers:
            print(f"finalizer. {finalizer.id} — {finalizer.name}")
        return

    job, config = _job_from_config(args.config)
    if args.command == "start":
        load_aws_worker_config(args.config)
        document = load_document(args.config)
        request = {
            "schema_version": document["schema_version"],
            "experiment_name": document.get("experiment_name"),
            "pipeline": document["pipeline"],
            "artifacts": document.get("artifacts"),
            "aws_worker": config.to_dict(),
            "force": args.force,
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
