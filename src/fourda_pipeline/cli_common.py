"""Arguments shared by blocking and background CLIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import FourDAnyoneConfig


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, required=True, help="Input monocular video")
    parser.add_argument("--experiment-name", required=True, help="Simple output/job name")
    parser.add_argument("--views-per-layer", type=int, default=24)
    parser.add_argument("--layer-pitches", type=comma_separated_ints, default=(-15, 0, 15))
    parser.add_argument("--start-yaw", type=int, default=0)
    parser.add_argument("--yaw-span", type=int, default=360)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-indices", type=comma_separated_ints, default=(60,))
    parser.add_argument("--base-model", action="store_true", help="Disable the default turbo model")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--export-device", default="cuda:0")
    parser.add_argument("--fourdanyone-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--s3-output-uri", help="Destination such as s3://bucket/runs/name/")
    parser.add_argument("--no-s3-upload", action="store_true")


def config_from_args(args: argparse.Namespace) -> FourDAnyoneConfig:
    optional: dict = {}
    for argument in ("fourdanyone_root", "data_root", "model_dir", "runs_dir"):
        value = getattr(args, argument, None)
        if value is not None:
            optional[argument] = value

    s3_uri = None if args.no_s3_upload else args.s3_output_uri
    if s3_uri is None and not args.no_s3_upload:
        bucket = os.environ.get("CP_4DA_BUCKET")
        if bucket:
            s3_uri = f"s3://{bucket}/runs/{args.experiment_name}/"

    return FourDAnyoneConfig(
        video_path=args.video,
        experiment_name=args.experiment_name,
        views_per_layer=args.views_per_layer,
        layer_pitches=args.layer_pitches,
        start_yaw=args.start_yaw,
        yaw_span=args.yaw_span,
        target_fps=args.target_fps,
        seed=args.seed,
        enable_turbo=not args.base_model,
        attention_backend=args.attention_backend,
        frame_indices=args.frame_indices,
        export_device=args.export_device,
        s3_output_uri=s3_uri,
        resume=args.resume,
        **optional,
    )
