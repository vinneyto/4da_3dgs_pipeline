"""Generate a validated AWS worker run document from CLI arguments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .config import AwsWorkerConfig, materialize_pipeline_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon-config",
        description="Generate a validated schema-v6 JSON document for recon-aws-worker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")

    environment = parser.add_argument_group("persistent execution environment")
    environment.add_argument(
        "--conda-bootstrap",
        type=Path,
        default=Path("/opt/conda/etc/profile.d/conda.sh"),
    )
    environment.add_argument(
        "--conda-env",
        type=Path,
        default=Path("/home/sagemaker-user/.conda/envs/4danyone"),
    )
    environment.add_argument(
        "--pipeline-repo-root",
        type=Path,
        default=Path("/home/sagemaker-user/work/4da_3dgs_pipeline"),
    )
    environment.add_argument(
        "--fourdanyone-git-url",
        default="https://github.com/ant-research/4DAnyone.git",
    )
    environment.add_argument(
        "--fourdanyone-git-ref",
        default="e38f210827f7b3effbe5b573ea07cfcf17e72dca",
    )
    environment.add_argument("--python-version", default="3.11")
    environment.add_argument("--torch-version", default="2.8.0")
    environment.add_argument("--torchvision-version", default="0.23.0")
    environment.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu126",
    )
    environment.add_argument("--opencv-fallback-version", default="4.14.0.94")
    environment.add_argument(
        "--lock-file",
        type=Path,
        default=Path(
            "/home/sagemaker-user/4danyone-data/environment/requirements-lock.txt"
        ),
    )

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--experiment-name", required=True)
    pipeline.add_argument("--job-id", help="Defaults to --experiment-name")
    pipeline.add_argument(
        "--dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 4DAnyone dataset stage",
    )
    pipeline.add_argument("--views-per-layer", type=int, default=24)
    pipeline.add_argument("--layer-pitches", type=int, nargs="+", default=[0])
    pipeline.add_argument("--start-yaw", type=int, default=0)
    pipeline.add_argument("--yaw-span", type=int, default=360)
    pipeline.add_argument("--target-fps", type=float, default=30.0)
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=True)
    pipeline.add_argument("--attention-backend", default="auto")
    pipeline.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    nerfstudio = parser.add_argument_group("Nerfstudio dataset artifact")
    nerfstudio.add_argument(
        "--nerfstudio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export synchronized static datasets from the completed 4DAnyone run",
    )
    nerfstudio.add_argument(
        "--nerfstudio-frames",
        type=int,
        nargs="+",
        default=[60],
    )
    nerfstudio.add_argument("--nerfstudio-device", default="cuda:0")
    nerfstudio.add_argument("--nerfstudio-source-experiment-name")
    nerfstudio.add_argument(
        "--nerfstudio-replace-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # Preserve commands generated before schema v6 while keeping the new JSON
    # vocabulary focused on artifacts.
    nerfstudio.add_argument(
        "--nerfstudio-frame-indices",
        dest="nerfstudio_frames",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    nerfstudio.add_argument("--frame", type=int, help=argparse.SUPPRESS)
    nerfstudio.add_argument("--export-device", help=argparse.SUPPRESS)
    pipeline.add_argument(
        "--rerun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create an interactive Rerun recording after reconstruction",
    )
    pipeline.add_argument("--rerun-view-count", type=int, default=4)
    pipeline.add_argument("--rerun-device", default="auto")
    pipeline.add_argument(
        "--rerun-source-experiment-name",
        help="Defaults to --experiment-name",
    )
    pipeline.add_argument(
        "--rerun-replace-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    aws = parser.add_argument_group("AWS worker")
    video = aws.add_mutually_exclusive_group(required=True)
    video.add_argument("--video", help="Path relative to the bucket input prefix")
    video.add_argument(
        "--s3-video-path",
        help="Deprecated full S3 input URI; use --bucket and --video",
    )
    aws.add_argument("--bucket", help="Required with --video; inferred for legacy S3 URIs")
    aws.add_argument("--region", default="us-east-1")
    aws.add_argument("--input-prefix", default="input")
    aws.add_argument("--models-prefix", default="models")
    aws.add_argument("--runs-prefix", default="runs")
    aws.add_argument("--sync-models", action=argparse.BooleanOptionalAction, default=True)
    aws.add_argument("--upload-results", action=argparse.BooleanOptionalAction, default=True)
    aws.add_argument(
        "--shutdown-on",
        choices=("never", "success", "failure", "always"),
        default="never",
    )

    notifications = parser.add_argument_group("notifications")
    notifications.add_argument(
        "--email",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or explicitly disable SNS email notifications",
    )
    notifications.add_argument("--sns-topic-name")
    notifications.add_argument("--notification-email")
    notifications.add_argument(
        "--telegram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or explicitly disable Telegram notifications",
    )
    notifications.add_argument("--telegram-chat-id")
    notifications.add_argument(
        "--telegram-bot-token-env",
        default="CP_4DA_TELEGRAM_BOT_TOKEN",
        help=(
            "Environment variable containing the bot token "
            "(the token is never written to JSON)"
        ),
    )
    notifications.add_argument(
        "--telegram-shutdown-command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the authorized chat/user to stop this SageMaker App with /shutdown",
    )
    notifications.add_argument(
        "--telegram-allowed-user-id",
        help="Telegram user ID authorized for /shutdown (defaults to chat ID)",
    )

    sagemaker = parser.add_argument_group("SageMaker")
    sagemaker.add_argument("--sagemaker-domain-id", required=True)
    sagemaker.add_argument("--sagemaker-space-name", required=True)
    sagemaker.add_argument("--sagemaker-app-name", default="default")

    local = parser.add_argument_group("persistent local workspace")
    local.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/sagemaker-user/4danyone-data"),
    )
    local.add_argument(
        "--fourdanyone-root",
        type=Path,
        default=Path("/home/sagemaker-user/work/4DAnyone"),
    )
    return parser


def _resolve_bucket(s3_video_path: str, explicit_bucket: str | None) -> str:
    if explicit_bucket:
        return explicit_bucket
    parsed = urlparse(s3_video_path)
    if parsed.scheme == "s3" and parsed.netloc:
        return parsed.netloc
    raise ValueError("--bucket is required when --s3-video-path is not a full s3:// URI")


def _resolve_video(args: argparse.Namespace) -> tuple[str, str]:
    input_prefix = args.input_prefix.strip("/")
    if args.video:
        if not args.bucket:
            raise ValueError("--bucket is required with --video")
        return args.bucket, args.video

    bucket = _resolve_bucket(args.s3_video_path, args.bucket)
    parsed = urlparse(args.s3_video_path)
    key = parsed.path.lstrip("/") if parsed.scheme == "s3" else args.s3_video_path.strip("/")
    prefix = f"{input_prefix}/" if input_prefix else ""
    if prefix and key.startswith(prefix):
        key = key[len(prefix) :]
    elif "/" in key and input_prefix:
        raise ValueError("--s3-video-path must be inside --input-prefix")
    return bucket, key


def build_document(args: argparse.Namespace) -> dict[str, Any]:
    email_enabled = (
        args.email
        if args.email is not None
        else bool(args.sns_topic_name or args.notification_email)
    )
    if bool(args.sns_topic_name) != bool(args.notification_email):
        raise ValueError(
            "--sns-topic-name and --notification-email must be provided together"
        )
    if email_enabled and not args.notification_email:
        raise ValueError(
            "--email requires --sns-topic-name and --notification-email"
        )
    if not email_enabled and (args.sns_topic_name or args.notification_email):
        raise ValueError(
            "--no-email cannot be combined with --sns-topic-name or --notification-email"
        )
    telegram_enabled = (
        args.telegram
        if args.telegram is not None
        else bool(args.telegram_chat_id)
    )
    if telegram_enabled and not args.telegram_chat_id:
        raise ValueError("--telegram requires --telegram-chat-id")
    if not telegram_enabled and args.telegram_chat_id:
        raise ValueError("--no-telegram cannot be combined with --telegram-chat-id")
    bucket, video = _resolve_video(args)
    job_id = args.job_id or args.experiment_name
    document: dict[str, Any] = {
        "schema_version": 6,
        "environment": {
            "conda_bootstrap": str(args.conda_bootstrap),
            "conda_env": str(args.conda_env),
            "pipeline_repo_root": str(args.pipeline_repo_root),
            "fourdanyone_git_url": args.fourdanyone_git_url,
            "fourdanyone_git_ref": args.fourdanyone_git_ref,
            "python_version": args.python_version,
            "torch_version": args.torch_version,
            "torchvision_version": args.torchvision_version,
            "torch_index_url": args.torch_index_url,
            "opencv_fallback_version": args.opencv_fallback_version,
            "lock_file": str(args.lock_file),
        },
        "experiment_name": args.experiment_name,
        "pipeline": {
            "dataset": {
                "enabled": args.dataset,
                "type": "4danyone",
                "config": {
                    "views_per_layer": args.views_per_layer,
                    "layer_pitches": args.layer_pitches,
                    "start_yaw": args.start_yaw,
                    "yaw_span": args.yaw_span,
                    "target_fps": args.target_fps,
                    "seed": args.seed,
                    "turbo": args.turbo,
                    "attention_backend": args.attention_backend,
                    "resume": args.resume,
                },
            },
            "reconstruction": {
                "enabled": False,
                "type": "nerfstudio_splatfacto",
                "config": {},
            },
        },
        "artifacts": {
            "dataset": {
                "nerfstudio": {
                    "enabled": args.nerfstudio,
                    "source_experiment_name": (
                        args.nerfstudio_source_experiment_name
                        or args.experiment_name
                    ),
                    "frames": (
                        [args.frame]
                        if args.frame is not None
                        else args.nerfstudio_frames
                    ),
                    "device": args.export_device or args.nerfstudio_device,
                    "replace_existing": args.nerfstudio_replace_existing,
                },
                "rerun": {
                    "enabled": args.rerun,
                    "source_experiment_name": (
                        args.rerun_source_experiment_name or args.experiment_name
                    ),
                    "view_count": args.rerun_view_count,
                    "device": args.rerun_device,
                    "replace_existing": args.rerun_replace_existing,
                }
            },
            "reconstruction": {
                "rerun": {
                    "enabled": False,
                    "source_experiment_name": args.experiment_name,
                    "view_count": 4,
                    "device": "auto",
                    "replace_existing": False,
                }
            },
        },
        "aws_worker": {
            "job_id": job_id,
            "region": args.region,
            "bucket": {
                "name": bucket,
                "video": video,
                "input_prefix": args.input_prefix,
                "models_prefix": args.models_prefix,
                "runs_prefix": args.runs_prefix,
            },
            "sync_models": args.sync_models,
            "upload_results": args.upload_results,
            "shutdown_on": args.shutdown_on,
            "notifications": {
                "email": (
                    {
                        "enabled": True,
                        "topic_name": args.sns_topic_name,
                        "email": args.notification_email,
                    }
                    if email_enabled
                    else None
                ),
                "telegram": (
                    {
                        "enabled": True,
                        "chat_id": args.telegram_chat_id,
                        "bot_token_env": args.telegram_bot_token_env,
                        "shutdown_command": args.telegram_shutdown_command,
                        "allowed_user_id": args.telegram_allowed_user_id,
                    }
                    if telegram_enabled
                    else None
                ),
            },
            "sagemaker": {
                "domain_id": args.sagemaker_domain_id,
                "space_name": args.sagemaker_space_name,
                "app_name": args.sagemaker_app_name,
            },
            "local": {
                "data_root": str(args.data_root),
                "fourdanyone_root": str(args.fourdanyone_root),
            },
        },
    }
    worker = AwsWorkerConfig.from_document(document)
    materialize_pipeline_config(document, worker)
    return document


def write_document(document: dict[str, Any], output: Path, force: bool = False) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"output already exists (use --force to replace it): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = build_document(args)
        write_document(document, args.output, args.force)
    except (FileExistsError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    total_views = args.views_per_layer * len(args.layer_pitches)
    bucket_config = document["aws_worker"]["bucket"]
    video_key = "/".join(
        part
        for part in (bucket_config["input_prefix"].strip("/"), bucket_config["video"])
        if part
    )
    print(f"Wrote AWS worker config: {args.output}")
    print(f"Job: {args.job_id or args.experiment_name}")
    print(f"Input: s3://{bucket_config['name']}/{video_key}")
    print(f"Target views: {total_views}")
    print(f"Dataset stage: {'enabled' if args.dataset else 'disabled'}")
    print(f"Nerfstudio artifact: {'enabled' if args.nerfstudio else 'disabled'}")
    print(f"Dataset Rerun artifact: {'enabled' if args.rerun else 'disabled'}")


if __name__ == "__main__":
    main()
