"""Build a Rerun recording without running a pipeline."""

import argparse
from pathlib import Path
from typing import Sequence

from recon_pipeline.utilities._output import report_progress, run_operation
from .exporter import RerunExporter


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation", "output", "fourdanyone-root", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    run_operation(lambda: export_recording(args))


def export_recording(args: argparse.Namespace) -> dict:
    if args.view_count < 1:
        raise ValueError("view count must be positive")
    for path in (args.generation / "metadata.json", args.generation / "cameras.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required path: {path}")
    if args.output.exists():
        if not args.replace_existing:
            raise FileExistsError("recording exists (use --replace-existing)")
        args.output.unlink()
    report_progress(0.0, "Building camera and skeleton recording")
    output = RerunExporter(
        generation=args.generation,
        output=args.output,
        experiment=args.experiment,
        fourdanyone_root=args.fourdanyone_root,
        model_dir=args.model_dir,
        view_count=args.view_count,
        device=args.device,
        on_progress=lambda current, total, message: report_progress(
            current / total, message
        ),
    ).export()
    if not output.is_file():
        raise RuntimeError(
            f"Rerun export completed without expected recording: {output}"
        )
    return {"path": str(output)}


if __name__ == "__main__":
    main()
