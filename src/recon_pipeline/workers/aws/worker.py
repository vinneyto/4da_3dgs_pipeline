"""Worker process for a durable AWS pass pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from contextlib import nullcontext
from pathlib import Path

from recon_pipeline.environment import PipelineEnvironment
from recon_pipeline.core import (
    PipelineContext,
    PipelineFinalizationError,
    PipelineOutcome,
)
from .persistence import S3PersistenceObserver
from .finalizers import SageMakerShutdownFinalizer

from .config import AwsWorkerConfig, materialize_pipeline_config
from .pipeline import build_aws_pipeline, required_source_experiments
from .status import JobStatus, utc_now
from .cloudwatch import CloudWatchLogSession


def run_worker(job_dir: Path) -> int:
    request = json.loads((job_dir / "request.json").read_text())
    if request.get("kind") == "queue":
        from .queue import run_queue

        return run_queue(job_dir, request)
    environment = PipelineEnvironment.from_environ()
    worker = AwsWorkerConfig.from_document(request, environment)
    config = materialize_pipeline_config(request, worker, environment)
    log_session = (
        CloudWatchLogSession(worker, config, job_dir)
        if worker.cloudwatch is not None and worker.cloudwatch.enabled
        else None
    )
    with log_session if log_session is not None else nullcontext():
        return _run_experiment(job_dir, request, worker, config, log_session)


def _run_experiment(job_dir, request, worker, config, log_session=None):
    status_path = job_dir / "status.json"
    status = JobStatus.read(status_path)
    status.update(pid=os.getpid())
    status.write(status_path)

    try:
        if log_session is not None:
            # Validate delivery before expensive GPU work, replaying any old backlog.
            log_session.flush()
        pipeline = build_aws_pipeline(
            worker,
            config,
            job_dir,
            status,
            force=bool(request.get("force", False)),
            log_session=log_session,
        )
        plan = pipeline.prepare()
    except BaseException as error:
        status.update(
            state="failed",
            stage="planning",
            message=f"Pipeline preparation failed: {error}",
            error=repr(error),
            finished_at=utc_now(),
        )
        status.write(status_path)
        traceback.print_exc()
        context = PipelineContext(values={"failed_pass": "planning"})
        if log_session is not None:
            context.values["cloudwatch_session"] = log_session
            log_session.record("planning_failed", error=error)
        outcome = PipelineOutcome(succeeded=False, error=error)
        if worker.upload_results:
            try:
                S3PersistenceObserver(
                    worker, config, job_dir, log_session=log_session
                ).finish(context, outcome)
            except BaseException:
                traceback.print_exc()
        SageMakerShutdownFinalizer(worker).run(context, outcome)
        return 1
    print("Prepared pipeline:", flush=True)
    for index, pipeline_pass in enumerate(plan.passes, start=1):
        print(f"  {index}. {pipeline_pass.id} — {pipeline_pass.name}", flush=True)
    for finalizer in plan.finalizers:
        print(f"  finalizer: {finalizer.id} — {finalizer.name}", flush=True)

    def cancel(signum, frame):
        raise KeyboardInterrupt("Worker cancellation requested")

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        pipeline.run(PipelineContext())
    except PipelineFinalizationError:
        traceback.print_exc()
        return 2
    except KeyboardInterrupt:
        return 130
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.job_dir))


if __name__ == "__main__":
    main()
