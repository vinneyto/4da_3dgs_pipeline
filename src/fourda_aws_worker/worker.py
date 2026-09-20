"""Worker process for a durable AWS background job."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from fourda_pipeline.pipeline import FourDAnyonePipeline
from fourda_pipeline.progress import ProgressUpdate

from .aws import (
    download_input_video,
    publish_completion,
    publish_started,
    publish_telegram,
    run_health_check,
    stop_sagemaker_app,
    sync_model_objects,
    upload_directory,
)
from .config import AwsWorkerConfig, materialize_pipeline_config
from .monitoring import TelegramRuntimeMonitoring
from .status import JobStatus, utc_now


def _notification_text(
    status: JobStatus,
    config: AwsWorkerConfig,
    result: dict | None,
    input_s3_uri: str | None,
    result_s3_uri: str | None,
) -> str:
    lines = [
        f"4DAnyone AWS job: {status.job_id}",
        f"State: {status.state}",
        f"Message: {status.message}",
    ]
    if result:
        lines.extend(
            [
                f"Experiment: {result['experiment_name']}",
                f"S3 input: {input_s3_uri}",
                f"Local result: {result['experiment_dir']}",
                f"S3 result: {result_s3_uri or 'upload disabled'}",
                f"Elapsed seconds: {result['elapsed_seconds']:.1f}",
            ]
        )
    elif status.error:
        lines.append(f"Error: {status.error}")
    lines.append(f"Shutdown policy: {config.shutdown_on}")
    return "\n".join(lines)


def run_worker(job_dir: Path) -> int:
    request = json.loads((job_dir / "request.json").read_text())
    aws_worker_config = AwsWorkerConfig.from_dict(request["aws_worker"])
    pipeline_config = materialize_pipeline_config(request["pipeline"], aws_worker_config)
    status_path = job_dir / "status.json"
    monitoring = TelegramRuntimeMonitoring(
        aws_worker_config,
        job_dir,
        lambda subject, message: publish_telegram(
            aws_worker_config, subject, message
        ),
    )
    monitoring.start()
    status = JobStatus.read(status_path)
    status.update(
        state="running",
        stage="validation",
        message="AWS job worker is running",
        pid=os.getpid(),
        started_at=utc_now(),
    )
    status.write(status_path)

    def on_progress(update: ProgressUpdate) -> None:
        # AWS validation/staging owns 0%-10%, the local core owns 10%-95%,
        # and result upload owns 95%-100%.
        fraction = 0.10 + update.fraction * 0.85
        status.update(
            state="running",
            stage=update.stage,
            progress=fraction,
            message=update.message,
        )
        status.write(status_path)
        print(f"[{fraction * 100:6.2f}%] {update.stage}: {update.message}", flush=True)

    result: dict | None = None
    input_s3_uri: str | None = aws_worker_config.video_s3_uri
    result_s3_uri: str | None = None
    exit_code = 0
    try:
        status.update(
            state="running",
            stage="aws-health-check",
            progress=0.0,
            message="Validating AWS identity, S3, optional notifications, and SageMaker App",
        )
        status.write(status_path)
        health = run_health_check(aws_worker_config, status.job_id)
        print(
            "AWS health check passed: "
            f"caller={health.caller_arn}, bucket={health.bucket}, "
            f"email={health.email_status}, telegram={health.telegram_status}",
            flush=True,
        )
        status.update(
            state="running",
            stage="input-download",
            progress=0.02,
            message=f"Downloading input video from {aws_worker_config.video_s3_uri}",
        )
        status.write(status_path)
        video_path = download_input_video(aws_worker_config)
        print(f"Input video ready at {video_path}", flush=True)

        status.update(
            state="running",
            stage="model-sync",
            progress=0.06,
            message=(
                "Synchronizing models from "
                f"s3://{aws_worker_config.bucket}/{aws_worker_config.models_prefix}/"
            ),
        )
        status.write(status_path)
        model_count, downloaded_count = sync_model_objects(aws_worker_config)
        print(
            f"Model cache ready: {model_count} S3 objects, {downloaded_count} downloaded",
            flush=True,
        )

        pipeline_config.validate_paths()
        notification_results = publish_started(
            aws_worker_config, health, pipeline_config.experiment_name
        )
        print(f"Start notifications: {notification_results or 'disabled'}", flush=True)

        result = FourDAnyonePipeline(pipeline_config, on_progress=on_progress).run()

        if aws_worker_config.upload_results:
            status.update(
                state="running",
                stage="result-upload",
                progress=0.96,
                message=f"Uploading result to s3://{aws_worker_config.bucket}",
            )
            status.write(status_path)
            result_s3_uri = upload_directory(
                aws_worker_config,
                pipeline_config.experiment_dir,
                pipeline_config.experiment_name,
            )
            print(f"Uploaded result to {result_s3_uri}", flush=True)

        status.update(
            state="succeeded",
            stage="complete",
            progress=1.0,
            message="AWS job completed successfully",
            finished_at=utc_now(),
            result_path=str(pipeline_config.experiment_dir / "pipeline-result.json"),
        )
    except BaseException as error:
        exit_code = 1
        traceback.print_exc()
        status.update(
            state="failed",
            stage="failed",
            message=f"AWS job failed: {error}",
            error=repr(error),
            finished_at=utc_now(),
        )
    finally:
        status.write(status_path)

    try:
        notification_results = publish_completion(
            aws_worker_config,
            f"4DAnyone AWS job {status.state}: {status.job_id}",
            _notification_text(status, aws_worker_config, result, input_s3_uri, result_s3_uri),
        )
        print(f"Completion notifications: {notification_results or 'disabled'}", flush=True)
    except Exception:
        print("Completion notification dispatch failed:", flush=True)
        traceback.print_exc()

    monitoring.stop()

    should_stop = aws_worker_config.shutdown_on == "always" or (
        aws_worker_config.shutdown_on == "success" and status.state == "succeeded"
    )
    if should_stop:
        try:
            print("Requesting SageMaker JupyterLab App shutdown", flush=True)
            stop_sagemaker_app(aws_worker_config.region, aws_worker_config.sagemaker)
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
