"""Worker process for a durable AWS background job."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from fourda_pipeline.config import FourDAnyoneConfig
from fourda_pipeline.pipeline import FourDAnyonePipeline
from fourda_pipeline.progress import ProgressUpdate

from .aws import publish_completion, stop_sagemaker_app, upload_directory, upload_input_video
from .config import AwsJobConfig
from .status import JobStatus, utc_now


def _notification_text(
    status: JobStatus,
    config: AwsJobConfig,
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
                f"S3 input: {input_s3_uri or 'upload disabled'}",
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
    pipeline_config = FourDAnyoneConfig.from_dict(request["pipeline"])
    aws_job_config = AwsJobConfig.from_dict(request["aws_job"])
    status_path = job_dir / "status.json"
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
        # Input upload owns 0%-5%, the local core owns 5%-95%, and result
        # upload owns 95%-100%.
        fraction = 0.05 + update.fraction * 0.90
        status.update(
            state="running",
            stage=update.stage,
            progress=fraction,
            message=update.message,
        )
        status.write(status_path)
        print(f"[{fraction * 100:6.2f}%] {update.stage}: {update.message}", flush=True)

    result: dict | None = None
    input_s3_uri: str | None = None
    result_s3_uri: str | None = None
    exit_code = 0
    try:
        if aws_job_config.upload_input:
            status.update(
                state="running",
                stage="input-upload",
                progress=0.01,
                message=f"Uploading input video to s3://{aws_job_config.bucket}",
            )
            status.write(status_path)
            input_s3_uri = upload_input_video(aws_job_config, pipeline_config.video_path)
            print(f"Uploaded input video to {input_s3_uri}", flush=True)

        result = FourDAnyonePipeline(pipeline_config, on_progress=on_progress).run()

        if aws_job_config.upload_results:
            status.update(
                state="running",
                stage="result-upload",
                progress=0.96,
                message=f"Uploading result to s3://{aws_job_config.bucket}",
            )
            status.write(status_path)
            result_s3_uri = upload_directory(
                aws_job_config,
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
        publish_completion(
            aws_job_config,
            f"4DAnyone AWS job {status.state}: {status.job_id}",
            _notification_text(status, aws_job_config, result, input_s3_uri, result_s3_uri),
        )
        print("SNS completion notification sent", flush=True)
    except Exception:
        print("SNS completion notification failed:", flush=True)
        traceback.print_exc()

    should_stop = aws_job_config.shutdown_on == "always" or (
        aws_job_config.shutdown_on == "success" and status.state == "succeeded"
    )
    if should_stop:
        try:
            print("Requesting SageMaker JupyterLab App shutdown", flush=True)
            stop_sagemaker_app(aws_job_config.region, aws_job_config.sagemaker)
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
