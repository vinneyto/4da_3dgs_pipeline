"""Generate a validated AWS worker run document from CLI arguments."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .config import AwsWorkerConfig, load_document, materialize_pipeline_config
from recon_pipeline.reconstructions.nerfstudio.training import SplatfactoTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon-config",
        description="Generate a validated AWS worker run document.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    parser.add_argument(
        "--template",
        type=Path,
        help=(
            "Copy an existing run document and override its run identity and input. "
            "Use with --experiment-name and --video"
        ),
    )

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
        "--splatfacto-env", type=Path,
        default=Path("/home/sagemaker-user/.conda/envs/splatfacto"),
        help="Separate persistent Nerfstudio 1.1.5 environment",
    )
    environment.add_argument(
        "--lock-file",
        type=Path,
        default=Path(
            "/home/sagemaker-user/4danyone-data/environment/requirements-lock.txt"
        ),
    )

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--experiment-name")
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
    pipeline.add_argument(
        "--reuse-4danyone-experiment",
        help="Clone an existing run, skip 4DAnyone inference, and export selected frames from it",
    )

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
    reconstruction = parser.add_argument_group("3DGS reconstruction")
    reconstruction.add_argument(
        "--splatfacto", action=argparse.BooleanOptionalAction, default=None,
    )
    reconstruction.add_argument("--splatfacto-frames", type=int, nargs="+")
    reconstruction.add_argument(
        "--splatfacto-profile", choices=("4danyone_rgba_compact_v1",),
        default="4danyone_rgba_compact_v1",
        help="Expanded into explicit training parameters in the saved JSON",
    )
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
    video = aws.add_mutually_exclusive_group()
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
    sagemaker.add_argument("--sagemaker-domain-id")
    sagemaker.add_argument("--sagemaker-space-name")
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


def _replace_artifact_source_experiment(
    value: Any,
    *,
    previous_experiment_name: str,
    experiment_name: str,
) -> None:
    """Update artifact references that followed the template's run identity."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source_experiment_name" and child == previous_experiment_name:
                value[key] = experiment_name
            else:
                _replace_artifact_source_experiment(
                    child,
                    previous_experiment_name=previous_experiment_name,
                    experiment_name=experiment_name,
                )
    elif isinstance(value, list):
        for child in value:
            _replace_artifact_source_experiment(
                child,
                previous_experiment_name=previous_experiment_name,
                experiment_name=experiment_name,
            )


def build_document_from_template(args: argparse.Namespace) -> dict[str, Any]:
    if not args.experiment_name:
        raise ValueError("--template requires --experiment-name")
    if not args.video and not args.reuse_4danyone_experiment:
        raise ValueError("--template requires --video")
    if args.reuse_4danyone_experiment and not args.splatfacto_frames:
        raise ValueError("--reuse-4danyone-experiment requires --splatfacto-frames")
    if args.reuse_4danyone_experiment and args.splatfacto is False:
        raise ValueError("--reuse-4danyone-experiment cannot use --no-splatfacto")
    if args.splatfacto_frames and not args.reuse_4danyone_experiment and args.splatfacto is not True:
        raise ValueError("--splatfacto-frames requires --splatfacto")
    if args.reuse_4danyone_experiment == args.experiment_name:
        raise ValueError("source and target experiment names must differ")

    document = load_document(args.template)
    previous_experiment_name = document.get("experiment_name")
    if not isinstance(previous_experiment_name, str) or not previous_experiment_name:
        raise ValueError("template must contain a non-empty experiment_name")

    document = copy.deepcopy(document)
    document["experiment_name"] = args.experiment_name
    worker_payload = document.get("aws_worker")
    if not isinstance(worker_payload, dict):
        raise ValueError("template must contain an aws_worker object")
    bucket_payload = worker_payload.get("bucket")
    if not isinstance(bucket_payload, dict):
        raise ValueError("template must use the schema-v6 aws_worker.bucket object")

    worker_payload["job_id"] = args.job_id or args.experiment_name
    if args.video:
        bucket_payload["video"] = args.video
    if args.bucket:
        bucket_payload["name"] = args.bucket
    _replace_artifact_source_experiment(
        document.get("artifacts"),
        previous_experiment_name=previous_experiment_name,
        experiment_name=args.experiment_name,
    )
    if args.reuse_4danyone_experiment:
        source = args.reuse_4danyone_experiment
        document["pipeline"]["dataset"]["enabled"] = False
        export = document["artifacts"]["dataset"]["nerfstudio"]
        export.update(enabled=True, source_experiment_name=source,
                      frames=args.splatfacto_frames, replace_existing=False)
        document["artifacts"]["dataset"]["rerun"]["enabled"] = False
    if args.splatfacto_frames and not args.reuse_4danyone_experiment:
        document["artifacts"]["dataset"]["nerfstudio"]["frames"] = args.splatfacto_frames
    if args.splatfacto is True or args.reuse_4danyone_experiment:
        frames = args.splatfacto_frames or document["artifacts"]["dataset"]["nerfstudio"]["frames"]
        document["pipeline"]["reconstruction"] = {
            "enabled": True, "type": "nerfstudio_splatfacto",
            "config": {"frames": frames, "training": asdict(SplatfactoTrainingConfig())},
        }
        document.setdefault("environment", {})["splatfacto_env"] = str(args.splatfacto_env)
    elif args.splatfacto is False:
        document["pipeline"]["reconstruction"] = {
            "enabled": False, "type": "nerfstudio_splatfacto", "config": {},
        }
    document["schema_version"] = 7 if document["pipeline"]["reconstruction"]["enabled"] else document["schema_version"]

    worker = AwsWorkerConfig.from_document(document)
    materialize_pipeline_config(document, worker)
    return document


def build_document(args: argparse.Namespace) -> dict[str, Any]:
    if args.template:
        return build_document_from_template(args)
    if args.reuse_4danyone_experiment:
        raise ValueError("--reuse-4danyone-experiment requires --template")
    if args.splatfacto_frames and args.splatfacto is not True:
        raise ValueError("--splatfacto-frames requires --splatfacto")
    if not args.experiment_name:
        raise ValueError("--experiment-name is required")
    if not args.video and not args.s3_video_path:
        raise ValueError("one of --video or --s3-video-path is required")
    if not args.sagemaker_domain_id:
        raise ValueError("--sagemaker-domain-id is required")
    if not args.sagemaker_space_name:
        raise ValueError("--sagemaker-space-name is required")
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
        "schema_version": 7 if args.splatfacto else 6,
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
            **({"splatfacto_env": str(args.splatfacto_env)} if args.splatfacto else {}),
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
                "enabled": bool(args.splatfacto),
                "type": "nerfstudio_splatfacto",
                "config": (
                    {"frames": args.splatfacto_frames or args.nerfstudio_frames,
                     "training": asdict(SplatfactoTrainingConfig())}
                    if args.splatfacto else {}
                ),
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
                        else (args.splatfacto_frames or args.nerfstudio_frames)
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
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    dataset_config = document["pipeline"]["dataset"]["config"]
    total_views = dataset_config["views_per_layer"] * len(
        dataset_config["layer_pitches"]
    )
    bucket_config = document["aws_worker"]["bucket"]
    video_key = "/".join(
        part
        for part in (bucket_config["input_prefix"].strip("/"), bucket_config["video"])
        if part
    )
    print(f"Wrote AWS worker config: {args.output}")
    print(f"Job: {document['aws_worker']['job_id']}")
    if document["pipeline"]["dataset"]["enabled"]:
        print(f"Input: s3://{bucket_config['name']}/{video_key}")
    else:
        print("Input video: skipped (dataset stage disabled)")
    print(f"Target views: {total_views}")
    print(
        "Dataset stage: "
        f"{'enabled' if document['pipeline']['dataset']['enabled'] else 'disabled'}"
    )
    print(
        "Nerfstudio artifact: "
        f"{'enabled' if document['artifacts']['dataset']['nerfstudio']['enabled'] else 'disabled'}"
    )
    print(
        "Dataset Rerun artifact: "
        f"{'enabled' if document['artifacts']['dataset']['rerun']['enabled'] else 'disabled'}"
    )
    reconstruction = document["pipeline"]["reconstruction"]
    if reconstruction["enabled"]:
        print(f"Splatfacto frames: {reconstruction['config']['frames']}")


if __name__ == "__main__":
    main()
