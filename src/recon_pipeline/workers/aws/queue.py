"""Sequential background runs with one shutdown policy for the whole queue."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from recon_pipeline.run_document import strip_environment
from .aws import stop_sagemaker_app
from .config import (
    CloudWatchConfig,
    load_aws_worker_config,
    load_document,
    SageMakerAppConfig,
)
from .job import AwsBackgroundJob
from .status import JobStatus, utc_now


def build_start_request(
    paths, shutdown_on=None, queue_id=None, force=False, *, cloudwatch_log_group=None
):
    if not paths:
        raise ValueError("At least one config is required")
    if cloudwatch_log_group is not None:
        CloudWatchConfig(cloudwatch_log_group)
    requests, workers = [], []
    for path in paths:
        _, worker = load_aws_worker_config(path)
        document = strip_environment(load_document(path))
        if shutdown_on is not None:
            document["aws_worker"]["shutdown_on"] = shutdown_on
        if cloudwatch_log_group is not None:
            document["aws_worker"]["cloudwatch"] = {"log_group": cloudwatch_log_group}
        document["force"] = force
        requests.append(document)
        workers.append(worker)
    if len(requests) == 1 and queue_id is None:
        return AwsBackgroundJob(workers[0].job_id, workers[0].jobs_dir), requests[0]
    first = workers[0]
    if any((w.region, w.sagemaker) != (first.region, first.sagemaker) for w in workers):
        raise ValueError(
            "All queued configs must target the same SageMaker App and region"
        )
    ids = [w.job_id for w in workers]
    experiments = [d["experiment_name"] for d in requests]
    if len(set(ids)) != len(ids) or len(set(experiments)) != len(experiments):
        raise ValueError(
            "Queued configs must have distinct job IDs and experiment names"
        )
    job = AwsBackgroundJob(queue_id or "queue-" + first.job_id, first.jobs_dir)
    if job.job_id in ids:
        raise ValueError("Queue ID must differ from experiment job IDs")
    for worker in workers:
        child = AwsBackgroundJob(worker.job_id, worker.jobs_dir)
        if child.status_path.exists():
            status = child.status()
            if not status.terminal and child._process_is_running(status.pid):
                raise RuntimeError(f"Queued job already running: {worker.job_id}")
    # Preserve the configs' policies in the snapshot; disable them only for children.
    return job, {
        "kind": "queue",
        "configs": requests,
        "shutdown_on": shutdown_on or workers[-1].shutdown_on,
        "region": first.region,
        "sagemaker": {
            "domain_id": first.sagemaker.domain_id,
            "space_name": first.sagemaker.space_name,
            "app_name": first.sagemaker.app_name,
        },
    }


def _run_child(job, request):
    job.prepare(request)
    with job.log_path.open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "recon_pipeline.workers.aws.worker",
                "--job-dir",
                str(job.root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            status = job.status()
            status.update(pid=process.pid)
            status.write(job.status_path)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def run_queue(job_dir: Path, request: dict) -> int:
    status_path = job_dir / "status.json"
    status = JobStatus.read(status_path)
    status.update(state="running", pid=os.getpid(), started_at=utc_now(), stage="queue")
    status.write(status_path)
    results = []
    result_path = job_dir / "queue-result.json"
    configs = request["configs"]
    child = None

    def save_results():
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"jobs": results}, indent=2) + "\n")
        temporary.replace(result_path)

    def cancel(signum, frame):
        raise KeyboardInterrupt("Queue cancellation requested")

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for index, document in enumerate(configs):
            child_request = {
                **document,
                "aws_worker": {**document["aws_worker"], "shutdown_on": "never"},
            }
            child = AwsBackgroundJob(document["aws_worker"]["job_id"], job_dir.parent)
            status.update(
                stage=child.job_id,
                progress=index / len(configs),
                message=f"Experiment {index + 1}/{len(configs)}: {child.job_id}",
            )
            status.write(status_path)
            print(status.message, flush=True)
            try:
                code = _run_child(child, child_request)
                child_status = child.status()
                if not child_status.terminal:
                    child_status.update(
                        state="failed" if code else "succeeded",
                        finished_at=utc_now(),
                        error=f"Worker exited with code {code}" if code else None,
                    )
                    child_status.write(child.status_path)
                results.append(
                    {
                        "job_id": child.job_id,
                        "state": child_status.state,
                        "exit_code": code,
                    }
                )
            except Exception as error:
                traceback.print_exc()
                if child.status_path.exists():
                    child_status = child.status()
                    if not child_status.terminal and not child._process_is_running(
                        child_status.pid
                    ):
                        child_status.update(
                            state="failed",
                            error=repr(error),
                            message=str(error),
                            finished_at=utc_now(),
                        )
                        child_status.write(child.status_path)
                results.append(
                    {"job_id": child.job_id, "state": "failed", "error": str(error)}
                )
            save_results()
        failed = any(
            r["state"] != "succeeded" or r.get("exit_code", 0) != 0 for r in results
        )
        status.update(
            state="failed" if failed else "succeeded",
            stage="completed",
            progress=1.0,
            message=f'Queue finished: {len(results)} experiments; {sum(r["state"] == "succeeded" for r in results)} succeeded',
            finished_at=utc_now(),
            result_path=str(result_path),
        )
        status.write(status_path)
        persistence_failed = any(
            (job_dir.parent / document["aws_worker"]["job_id"] / marker).exists()
            for document in configs
            for marker in ("s3-persistence-failed", "cloudwatch-persistence-failed")
        )
        policy = "never" if persistence_failed else request["shutdown_on"]
        if persistence_failed:
            print(
                "Queue shutdown deferred: artifacts, diagnostics or CloudWatch logs remain unsaved",
                flush=True,
            )
        if (
            policy == "always"
            or (policy == "success" and not failed)
            or (policy == "failure" and failed)
        ):
            stop_sagemaker_app(
                request["region"], SageMakerAppConfig(**request["sagemaker"])
            )
        return int(failed)
    except KeyboardInterrupt:
        if child is not None and child.status_path.exists():
            child_status = child.status()
            if not child_status.terminal and not child._process_is_running(
                child_status.pid
            ):
                child_status.update(
                    state="cancelled", message="Queue cancelled", finished_at=utc_now()
                )
                child_status.write(child.status_path)
        status.update(
            state="cancelled",
            stage="cancelled",
            message="Queue cancelled",
            finished_at=utc_now(),
        )
        status.write(status_path)
        return 130
    except Exception as error:
        status.update(
            state="failed",
            stage="queue",
            error=repr(error),
            message=str(error),
            finished_at=utc_now(),
        )
        status.write(status_path)
        traceback.print_exc()
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
