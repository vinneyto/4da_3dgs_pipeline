"""Convert a completed Gaussian PLY artifact without running reconstruction."""

import argparse
import json
import os
import shutil
from pathlib import Path

from recon_pipeline.utilities._output import report_progress, run_operation
from recon_pipeline.utilities._subprocess import run_logged
from ._formats import EXPORT_FILENAMES, validate_formats


def convert(args):
    source, output = args.input.resolve(), args.output.resolve()
    formats = validate_formats(args.formats)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty source Gaussian PLY: {source}")
    if source.is_relative_to(output):
        raise ValueError("Postprocessing output must not contain the source PLY")
    converter = None
    if any(fmt != "ply" for fmt in formats):
        converter = shutil.which(args.splat_transform)
        if converter is None:
            raise FileNotFoundError(
                f"Install splat-transform with scripts/setup_environment.sh; missing {args.splat_transform}"
            )
    if output.exists():
        if not args.replace_existing:
            raise FileExistsError(f"Output exists: {output} (use --replace-existing)")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            [
                *([str(Path(converter).parent)] if converter else []),
                os.environ.get("PATH", ""),
            ]
        ),
    }
    exported = {}
    for index, fmt in enumerate(formats):
        report_progress(index / len(formats), f"Converting Gaussian artifact to {fmt}")
        destination = output / EXPORT_FILENAMES[fmt]
        if fmt == "ply":
            shutil.copyfile(source, destination)
        else:
            run_logged(
                [converter, "--quiet", "--gpu", "cpu", str(source), str(destination)],
                logs / f"convert-{fmt}.log",
                env,
            )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ValueError(
                f"splat-transform did not produce a non-empty artifact: {destination}"
            )
        exported[fmt] = str(destination)
    result = {
        "source_ply": str(source),
        "output_dir": str(output),
        "formats": list(formats),
        "exported_artifacts": exported,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    report_progress(1.0, "Gaussian artifact postprocessing complete")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=EXPORT_FILENAMES, required=True)
    parser.add_argument("--splat-transform", default="splat-transform")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    run_operation(lambda: convert(args))


if __name__ == "__main__":
    main()
