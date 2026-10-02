"""Generate a dataset with 4DAnyone, independently of the pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from recon_pipeline.cli import report_progress, run_operation


class JsonProgressHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        report_progress(float(getattr(record, "fraction", 0.0)), record.getMessage())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fourdanyone-root", "video", "output", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--gvhmr-root",
        type=Path,
        help="Defaults to third_party/GVHMR under the upstream root",
    )
    parser.add_argument("--views-per-layer", type=int, default=24)
    parser.add_argument("--layer-pitches", type=int, nargs="+", default=[-15, 0, 15])
    parser.add_argument("--start-yaw", type=int, default=0)
    parser.add_argument("--yaw-span", type=int, default=360)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def generate(args: argparse.Namespace) -> dict:
    root, video, output, models = (
        path.expanduser().resolve()
        for path in (args.fourdanyone_root, args.video, args.output, args.model_dir)
    )
    gvhmr = (
        args.gvhmr_root.expanduser().resolve()
        if args.gvhmr_root
        else root / "third_party/GVHMR"
    )
    for path in (video, root / "inference.py", gvhmr / "hmr4d/__init__.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required path: {path}")
    if not models.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {models}")
    if (
        args.views_per_layer < 1
        or not args.layer_pitches
        or args.views_per_layer * len(args.layer_pitches) % 6
    ):
        raise ValueError("total target views must be positive and divisible by 6")
    if any(p < -15 or p > 45 for p in args.layer_pitches):
        raise ValueError("layer pitches must be between -15 and 45")
    if not 1 <= args.yaw_span <= 360 or args.target_fps <= 0:
        raise ValueError(
            "yaw span must be between 1 and 360 and target FPS must be positive"
        )
    if output.exists():
        if not args.replace_existing:
            raise FileExistsError(
                f"output already exists (use --replace-existing): {output}"
            )
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "video_path": str(video),
        "output_dir": str(output),
        "views_per_layer": args.views_per_layer,
        "layer_pitches": args.layer_pitches,
        "start_yaw": args.start_yaw,
        "yaw_span": args.yaw_span,
        "enable_turbo": args.turbo,
        "model_dir": str(models),
        "gvhmr_root": str(gvhmr),
        "attention_backend": args.attention_backend,
        "target_fps": args.target_fps,
        "seed": args.seed,
    }
    (output.parent / "inference-request.json").write_text(
        json.dumps({"fourdanyone_root": str(root), **request}, indent=2, sort_keys=True)
        + "\n"
    )
    progress = logging.getLogger("fdanyone.progress")
    progress.setLevel(logging.INFO)
    progress.addHandler(JsonProgressHandler())
    sys.path.insert(0, str(root))
    os.chdir(root)
    # Preserve upstream allocator/attention bootstrap before model imports.
    from inference import inference

    inference(**request)
    if not (output / "metadata.json").is_file():
        raise RuntimeError(f"4DAnyone completed without expected metadata: {output}")
    return {"path": str(output)}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_operation(lambda: generate(args))


if __name__ == "__main__":
    main()
