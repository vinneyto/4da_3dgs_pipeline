"""Export synchronized frames from a completed 4DAnyone generation."""

import argparse
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Sequence

from recon_pipeline.utilities._output import report_progress, run_operation


def export_frames(
    *,
    root: Path,
    generation: Path,
    output: Path,
    model_dir: Path,
    frames: Sequence[int],
    device: str,
    replace_existing: bool,
) -> dict:
    for path in (
        root / "scripts/export_nerfstudio.py",
        generation / "metadata.json",
        generation / "cameras.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required path: {path}")
    if (
        not frames
        or any(frame < 0 for frame in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ValueError("frames must be distinct nonnegative indices")
    destinations = [output / f"frame_{frame:03d}" for frame in frames]
    if not replace_existing and any(path.exists() for path in destinations):
        raise FileExistsError("dataset output exists (use --replace-existing)")
    if replace_existing:
        for path in destinations:
            if path.exists():
                shutil.rmtree(path)
    datasets = []
    for offset, (frame, destination) in enumerate(zip(frames, destinations)):
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/export_nerfstudio.py"),
                "--data_dir",
                str(generation),
                "--output_dir",
                str(destination),
                "--frame_index",
                str(frame),
                "--model_dir",
                str(model_dir),
                "--device",
                device,
            ],
            cwd=root,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=True,
        )
        if not (destination / "transforms.json").is_file():
            raise RuntimeError(
                f"export completed without transforms.json: {destination}"
            )
        datasets.append({"frame": frame, "dataset_dir": str(destination)})
        report_progress(
            (offset + 1) / len(frames), f"Exported synchronized frame {frame}"
        )
    return {"datasets": datasets, "frames": list(frames), "count": len(datasets)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fourdanyone-root", "generation", "output", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[60])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    run_operation(
        lambda: export_frames(
            root=args.fourdanyone_root.expanduser().resolve(),
            generation=args.generation.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            model_dir=args.model_dir.expanduser().resolve(),
            frames=args.frames,
            device=args.device,
            replace_existing=args.replace_existing,
        )
    )


if __name__ == "__main__":
    main()
