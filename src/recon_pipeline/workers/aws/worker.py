"""Worker process for a durable AWS pass pipeline."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from recon_pipeline.core import PipelineContext, PipelineFinalizationError

from .config import AwsWorkerConfig, materialize_pipeline_config
from .pipeline import build_aws_pipeline, required_source_experiments
from .status import JobStatus, utc_now


def run_worker(job_dir: Path) -> int:
    request = json.loads((job_dir / "request.json").read_text())
    worker = AwsWorkerConfig.from_dict(request["aws_worker"])
    config = materialize_pipeline_config(request, worker)
    status_path = job_dir / "status.json"
    status = JobStatus.read(status_path)
    status.update(pid=os.getpid())
    status.write(status_path)

    pipeline = build_aws_pipeline(
        worker,
        config,
        job_dir,
        status,
        force=bool(request.get("force", False)),
    )
    try:
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
        return 1
    print("Prepared pipeline:", flush=True)
    for index, pipeline_pass in enumerate(plan.passes, start=1):
        print(f"  {index}. {pipeline_pass.id} — {pipeline_pass.name}", flush=True)
    for finalizer in plan.finalizers:
        print(f"  finalizer: {finalizer.id} — {finalizer.name}", flush=True)

    try:
        pipeline.run(PipelineContext())
    except PipelineFinalizationError:
        traceback.print_exc()
        return 2
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.job_dir))


if __name__ == "__main__":
    main()
