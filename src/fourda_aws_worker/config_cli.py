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
        prog="fourda-worker-config",
        description="Generate a validated schema-v2 JSON document for fourda-aws-worker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--experiment-name", required=True)
    pipeline.add_argument("--job-id", help="Defaults to --experiment-name")
    pipeline.add_argument("--views-per-layer", type=int, default=24)
    pipeline.add_argument("--layer-pitches", type=int, nargs="+", default=[0])
    pipeline.add_argument("--start-yaw", type=int, default=0)
    pipeline.add_argument("--yaw-span", type=int, default=360)
    pipeline.add_argument("--target-fps", type=float, default=30.0)
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--frame", type=int, default=60)
    pipeline.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=True)
    pipeline.add_argument("--attention-backend", default="auto")
    pipeline.add_argument("--export-device", default="cuda:0")
    pipeline.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    aws = parser.add_argument_group("AWS worker")
    aws.add_argument("--s3-video-path", required=True)
    aws.add_argument("--bucket", help="Inferred from an s3:// video URI when omitted")
    aws.add_argument("--region", default="us-east-1")
    aws.add_argument("--input-prefix", default="input")
    aws.add_argument("--models-prefix", default="models")
    aws.add_argument("--runs-prefix", default="runs")
    aws.add_argument("--sync-models", action=argparse.BooleanOptionalAction, default=True)
    aws.add_argument("--upload-results", action=argparse.BooleanOptionalAction, default=True)
    aws.add_argument(
        "--shutdown-on",
        choices=("never", "success", "always"),
        default="never",
    )

    notifications = parser.add_argument_group("notifications")
    notifications.add_argument("--sns-topic-name")
    notifications.add_argument("--notification-email")
    notifications.add_argument("--telegram-chat-id")
    notifications.add_argument(
        "--telegram-stream-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    notifications.add_argument(
        "--telegram-resource-status-interval-seconds",
        type=int,
    )
    notifications.add_argument(
        "--telegram-bot-token-env",
        default="CP_4DA_TELEGRAM_BOT_TOKEN",
        help=(
            "Environment variable containing the bot token "
            "(the token is never written to JSON)"
        ),
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


def build_document(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.sns_topic_name) != bool(args.notification_email):
        raise ValueError(
            "--sns-topic-name and --notification-email must be provided together"
        )
    if (
        args.telegram_stream_logs
        or args.telegram_resource_status_interval_seconds is not None
    ) and not args.telegram_chat_id:
        raise ValueError(
            "Telegram log/resource options require --telegram-chat-id"
        )
    bucket = _resolve_bucket(args.s3_video_path, args.bucket)
    job_id = args.job_id or args.experiment_name
    document: dict[str, Any] = {
        "schema_version": 2,
        "pipeline": {
            "experiment_name": args.experiment_name,
            "views_per_layer": args.views_per_layer,
            "layer_pitches": args.layer_pitches,
            "start_yaw": args.start_yaw,
            "yaw_span": args.yaw_span,
            "target_fps": args.target_fps,
            "seed": args.seed,
            "turbo": args.turbo,
            "attention_backend": args.attention_backend,
            "frame": args.frame,
            "export_device": args.export_device,
            "resume": args.resume,
        },
        "aws_worker": {
            "job_id": job_id,
            "region": args.region,
            "bucket": bucket,
            "s3_video_path": args.s3_video_path,
            "input_prefix": args.input_prefix,
            "models_prefix": args.models_prefix,
            "runs_prefix": args.runs_prefix,
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
                    if args.notification_email
                    else None
                ),
                "telegram": (
                    {
                        "enabled": True,
                        "chat_id": args.telegram_chat_id,
                        "bot_token_env": args.telegram_bot_token_env,
                        "stream_logs": args.telegram_stream_logs,
                        "resource_status_interval_seconds": (
                            args.telegram_resource_status_interval_seconds
                        ),
                    }
                    if args.telegram_chat_id
                    else None
                ),
            },
            "sagemaker_domain_id": args.sagemaker_domain_id,
            "sagemaker_space_name": args.sagemaker_space_name,
            "sagemaker_app_name": args.sagemaker_app_name,
            "local": {
                "data_root": str(args.data_root),
                "fourdanyone_root": str(args.fourdanyone_root),
            },
        },
    }
    worker = AwsWorkerConfig.from_document(document)
    materialize_pipeline_config(document["pipeline"], worker)
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
    print(f"Wrote AWS worker config: {args.output}")
    print(f"Job: {args.job_id or args.experiment_name}")
    print(f"Input: {args.s3_video_path}")
    print(f"Target views: {total_views}")


if __name__ == "__main__":
    main()
