"""Compatibility adapter for old --request inference commands."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .utilities.inference import main as inference_main

PROGRESS_PREFIX = "FOURDA_PROGRESS "


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    arguments = []
    options = {
        "fourdanyone_root": "--fourdanyone-root",
        "video_path": "--video",
        "output_dir": "--output",
        "model_dir": "--model-dir",
        "gvhmr_root": "--gvhmr-root",
        "views_per_layer": "--views-per-layer",
        "layer_pitches": "--layer-pitches",
        "start_yaw": "--start-yaw",
        "yaw_span": "--yaw-span",
        "target_fps": "--target-fps",
        "seed": "--seed",
        "attention_backend": "--attention-backend",
    }
    for key, option in options.items():
        if key in request:
            value = request[key]
            arguments.extend(
                [option, *(str(item) for item in value)]
                if isinstance(value, list)
                else [option, str(value)]
            )
    arguments.append("--turbo" if request.get("enable_turbo", True) else "--no-turbo")
    inference_main(arguments)


if __name__ == "__main__":
    main()
