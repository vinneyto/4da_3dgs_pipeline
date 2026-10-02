"""Prepare a private RGBA training copy of a static multicamera dataset."""

import argparse
import json
import shutil
from pathlib import Path

from recon_pipeline.utilities._output import report_progress, run_operation


def dataset_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its directory: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare(
    dataset: Path, output: Path, threshold: int = 127, erosion: int = 1, progress=None
) -> dict:
    import cv2
    import numpy as np
    from PIL import Image

    dataset, output = dataset.resolve(), output.resolve()
    if (
        output == dataset
        or dataset.is_relative_to(output)
        or output.is_relative_to(dataset)
    ):
        raise ValueError(
            "Training copy and source dataset must be separate directories"
        )
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= threshold <= 255 or erosion < 0:
        raise ValueError("Invalid mask threshold or erosion radius")
    payload = json.loads((dataset / "transforms.json").read_text())
    cameras = payload.get("frames", [])
    if len(cameras) < 4:
        raise ValueError("A static dataset must contain at least four cameras")
    for camera in cameras:
        image = dataset_file(dataset, camera["file_path"])
        dataset_file(dataset, camera.get("mask_path") or f"masks/{image.stem}.png")
    if payload.get("ply_file_path"):
        dataset_file(dataset, payload["ply_file_path"])
    shutil.copytree(dataset, output)
    mask_dir = output / "masks_training"
    mask_dir.mkdir(exist_ok=True)
    ratios = []
    kernel = np.ones((2 * erosion + 1, 2 * erosion + 1), np.uint8) if erosion else None
    for index, camera in enumerate(cameras):
        source = dataset_file(dataset, camera["file_path"])
        mask = dataset_file(
            dataset, camera.get("mask_path") or f"masks/{source.stem}.png"
        )
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(mask) as image:
            alpha = np.asarray(image.convert("L"))
        if alpha.shape != rgb.shape[:2]:
            raise ValueError(f"Mask dimensions do not match image: {source}")
        alpha = (alpha >= threshold).astype(np.uint8) * 255
        if kernel is not None:
            alpha = cv2.erode(alpha, kernel, iterations=1)
        if not np.any(alpha):
            raise ValueError(f"Empty foreground after mask processing: {mask}")
        destination = output / source.relative_to(dataset).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate((rgb, alpha[..., None]), axis=-1)).save(
            destination
        )
        Image.fromarray(alpha).save(mask_dir / f"{source.stem}.png")
        camera["file_path"] = destination.relative_to(output).as_posix()
        camera.pop("mask_path", None)
        ratios.append(float((alpha > 0).mean()))
        (progress or report_progress)(
            (index + 1) / len(cameras),
            f"Prepared RGBA camera {index + 1}/{len(cameras)}",
        )
    payload.pop("mask_path", None)
    (output / "transforms.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "dataset_dir": str(output),
        "camera_count": len(cameras),
        "foreground_min": min(ratios),
        "foreground_max": max(ratios),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--mask-erosion-pixels", type=int, default=1)
    args = parser.parse_args(argv)
    run_operation(
        lambda: prepare(
            args.dataset, args.output, args.mask_threshold, args.mask_erosion_pixels
        )
    )


if __name__ == "__main__":
    main()
