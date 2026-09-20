"""Worker process for a durable background pipeline job."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from .aws_integration import load_aws_config, publish_completion, stop_current_sagemaker_app
from .config import FourDAnyoneConfig
from .pipeline import FourDAnyonePipeline
from .progress import ProgressUpdate
from .status import JobStatus, utc_now


def _notification_text(status: JobStatus, request: dict, result: dict | None) -> str:
    lines = [
        f"4DAnyone job: {status.job_id}",
        f"State: {status.state}",
        f"Message: {status.message}",
    ]
    if result:
        lines.extend(
            [
                f"Experiment: {result['experiment_name']}",
                f"Result: {result['experiment_dir']}",
                f"S3: {result.get('s3_output_uri') or 'disabled'}",
                f"Elapsed seconds: {result['elapsed_seconds']:.1f}",
            ]
        )
    elif status.error:
        lines.append(f"Error: {status.error}")
    lines.append(f"Shutdown policy: {request.get('shutdown_on', 'never')}")
    return "\n".join(lines)


def run_worker(job_dir: Path) -> int:
    request = json.loads((job_dir / "request.json").read_text())
    status_path = job_dir / "status.json"
    status = JobStatus.read(status_path)
    status.update(
        state="running",
        stage="validation",
        message="Pipeline worker is running",
        pid=os.getpid(),
        started_at=utc_now(),
    )
    status.write(status_path)

    def on_progress(update: ProgressUpdate) -> None:
        status.update(
            state="running",
            stage=update.stage,
            progress=update.fraction,
            message=update.message,
        )
        status.write(status_path)
        print(f"[{update.fraction * 100:6.2f}%] {update.stage}: {update.message}", flush=True)

    result: dict | None = None
    exit_code = 0
    try:
        config = FourDAnyoneConfig.from_dict(request["pipeline"])
        result = FourDAnyonePipeline(config, on_progress=on_progress).run()
        status.update(
            state="succeeded",
            stage="complete",
            progress=1.0,
            message="Pipeline completed successfully",
            finished_at=utc_now(),
            result_path=str(config.experiment_dir / "pipeline-result.json"),
        )
    except BaseException as error:
        exit_code = 1
        traceback.print_exc()
        status.update(
            state="failed",
            stage="failed",
            message=f"Pipeline failed: {error}",
            error=repr(error),
            finished_at=utc_now(),
        )
    finally:
        status.write(status_path)

    aws_config = load_aws_config()
    region = request.get("region") or aws_config.get("region") or os.environ.get("CP_AWS_REGION", "us-east-1")
    topic_arn = request.get("sns_topic_arn") or aws_config.get("sns_topic_arn")
    if topic_arn:
        try:
            publish_completion(
                topic_arn,
                f"4DAnyone {status.state}: {status.job_id}",
                _notification_text(status, request, result),
                region,
            )
            print("SNS completion notification sent", flush=True)
        except Exception:
            print("SNS completion notification failed:", flush=True)
            traceback.print_exc()

    shutdown_on = request.get("shutdown_on", "never")
    should_stop = shutdown_on == "always" or (shutdown_on == "success" and status.state == "succeeded")
    if should_stop:
        try:
            print("Requesting SageMaker JupyterLab App shutdown", flush=True)
            stop_current_sagemaker_app(region)
        except Exception:
            print("SageMaker App shutdown request failed:", flush=True)
            traceback.print_exc()
            return 2 if exit_code == 0 else exit_code
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.job_dir))


if __name__ == "__main__":
    main()
