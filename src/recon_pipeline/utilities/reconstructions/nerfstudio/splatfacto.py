"""Train one static frame and export a full SH Gaussian PLY using installed Nerfstudio."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from recon_pipeline.utilities._output import (
    report_progress as emit_progress,
    run_operation,
)
from ._profile import TrainingProfile, add_profile_arguments
from ._exports import EXPORT_FILENAMES, validate_export_formats
from .prepare_rgba import prepare


def validate_ply(path: Path, sh_degree: int) -> int:
    with path.open("rb") as stream:
        header = stream.read(65_536).split(b"end_header\n", 1)
    if len(header) != 2:
        raise ValueError(f"Gaussian PLY has no header terminator: {path}")
    text = header[0].decode("ascii")
    fields = {
        match.group(1) for match in re.finditer(r"^property float (\S+)$", text, re.M)
    }
    required = {
        "x",
        "y",
        "z",
        "opacity",
        *(f"f_dc_{i}" for i in range(3)),
        *(f"scale_{i}" for i in range(3)),
        *(f"rot_{i}" for i in range(4)),
        *(f"f_rest_{i}" for i in range(3 * ((sh_degree + 1) ** 2 - 1))),
    }
    if required - fields:
        raise ValueError(f"Gaussian PLY is missing fields: {sorted(required - fields)}")
    if all(f"property uchar {channel}" in text for channel in ("red", "green", "blue")):
        raise ValueError("Expected SH Gaussian PLY, got RGB point cloud")
    match = re.search(r"^element vertex (\d+)$", text, re.M)
    if not match or int(match.group(1)) < 1:
        raise ValueError("Gaussian PLY contains no splats")
    return int(match.group(1))


def run_logged(command, log: Path, env, progress=None):
    print("$ " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                # A callback may consume noisy training rows after reporting progress.
                if progress is None or progress(line) is not False:
                    print(line, end="", file=sys.stderr, flush=True)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def reconstruct(args) -> dict:
    last_progress = -1

    def report_progress(fraction, message):
        nonlocal last_progress
        percent = int(fraction * 100 + 1e-8)
        if percent > last_progress:
            last_progress = percent
            emit_progress(percent / 100, message)

    formats = validate_export_formats(args.export_formats)
    converter = None
    if any(fmt != "ply" for fmt in formats):
        converter = shutil.which(args.splat_transform)
        if converter is None:
            raise FileNotFoundError(
                f"Install splat-transform with scripts/setup_environment.sh; missing {args.splat_transform}"
            )
    profile = TrainingProfile(
        **{name: getattr(args, name) for name in TrainingProfile.__dataclass_fields__}
    )
    dataset, output, binaries = (
        args.dataset.resolve(),
        args.output.resolve(),
        args.nerfstudio_bin.resolve(),
    )
    for name in ("ns-train", "ns-export"):
        if not (binaries / name).is_file():
            raise FileNotFoundError(
                f"Install Nerfstudio manually; missing {binaries / name}"
            )
    if (
        output == dataset
        or output.is_relative_to(dataset)
        or dataset.is_relative_to(output)
    ):
        raise ValueError(
            "Source dataset and reconstruction output must be separate directories"
        )
    if not (dataset / "transforms.json").is_file():
        raise FileNotFoundError(dataset / "transforms.json")
    name = args.experiment_name
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError("experiment name must be a simple directory name")
    if output.exists():
        if not args.replace_existing:
            raise FileExistsError(f"Output exists: {output} (use --replace-existing)")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    report_progress(0.0, "Preparing RGBA training copy")
    prepared = prepare(
        dataset,
        output / "dataset",
        profile.mask_threshold,
        profile.mask_erosion_pixels,
        progress=lambda fraction, message: report_progress(0.05 * fraction, message),
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S_%f")
    logs = output / "logs"
    logs.mkdir()
    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            [
                str(binaries),
                *([str(Path(converter).parent)] if converter else []),
                os.environ.get("PATH", ""),
            ]
        ),
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        str(binaries / "ns-train"),
        "splatfacto",
        "--data",
        prepared["dataset_dir"],
        "--output-dir",
        str(output / "nerfstudio_outputs"),
        "--experiment-name",
        name,
        "--timestamp",
        timestamp,
        "--max-num-iterations",
        str(profile.max_num_iterations),
        "--vis",
        "tensorboard",
        "--logging.local-writer.max-log-size",
        "0",  # Avoid terminal redraws, repeated headers and historical rows in logs.
        "--pipeline.model.background-color",
        "random",
        "--pipeline.model.rasterize-mode",
        "classic",
    ]
    for setting, value in asdict(profile).items():
        if setting not in (
            "max_num_iterations",
            "mask_threshold",
            "mask_erosion_pixels",
        ):
            command.extend(
                ["--pipeline.model." + setting.replace("_", "-"), str(value)]
            )
    command.extend(["nerfstudio-data", "--eval-mode", "all"])
    report_progress(0.05, "Training Splatfacto")

    def training_progress(line):
        plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        match = re.match(r"^(\d+)\s+\([\d.]+%\)", plain)
        if match:
            step = min(int(match[1]), profile.max_num_iterations)
            percent = 5 + 85 * step // profile.max_num_iterations
            report_progress(
                percent / 100,
                f"Splatfacto iteration {step}/{profile.max_num_iterations}",
            )
            return False  # Full rows stay in train.log; stdout carries coarse progress.
        return True

    run_logged(command, logs / "train.log", env, training_progress)
    run_root = output / "nerfstudio_outputs" / name / "splatfacto" / timestamp
    for file in (run_root / "config.yml", run_root / "dataparser_transforms.json"):
        if not file.is_file():
            raise FileNotFoundError(
                f"Training completed without required output: {file}"
            )
    report_progress(0.9, "Exporting full SH Gaussian PLY")
    run_logged(
        [
            str(binaries / "ns-export"),
            "gaussian-splat",
            "--load-config",
            str(run_root / "config.yml"),
            "--output-dir",
            str(output / "exports"),
            "--output-filename",
            "splat.ply",
            "--ply-color-mode",
            "sh_coeffs",
        ],
        logs / "export.log",
        env,
    )
    splat = output / "exports/splat.ply"
    count = validate_ply(splat, profile.sh_degree)
    exported = {}
    for index, fmt in enumerate(formats):
        destination = output / "exports" / EXPORT_FILENAMES[fmt]
        if fmt != "ply":
            report_progress(
                0.9 + (index + 1) / len(formats) * 0.09,
                f"Converting Gaussian export to {fmt}",
            )
            run_logged(
                [converter, "--quiet", "--gpu", "cpu", str(splat), str(destination)],
                logs / f"convert-{fmt}.log",
                env,
            )
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError(
                    f"splat-transform did not produce a non-empty export: {destination}"
                )
        exported[fmt] = str(destination)
    result = {
        "export_formats": list(formats),
        "exported_artifacts": exported,
        "dataset_dir": prepared["dataset_dir"],
        "source_dataset": str(dataset),
        "output_dir": str(output),
        "config": str(run_root / "config.yml"),
        "run_root": str(run_root),
        "dataparser_transform": str(run_root / "dataparser_transforms.json"),
        "splat_ply": str(splat),
        "full_splat_count": count,
        "train_log": str(logs / "train.log"),
        "camera_count": prepared["camera_count"],
        "run_timestamp": timestamp,
        "experiment_name": name,
        "training_profile": asdict(profile),
        "mask_mode": "rgba_alpha_random_background",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    report_progress(1.0, "Splatfacto reconstruction and full PLY ready")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "nerfstudio-bin"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment-name", default="splatfacto")
    parser.add_argument(
        "--export-formats", nargs="+", choices=EXPORT_FILENAMES, default=["ply"]
    )
    parser.add_argument(
        "--splat-transform",
        default="splat-transform",
        help="Installed converter executable",
    )
    parser.add_argument("--replace-existing", action="store_true")
    add_profile_arguments(parser)
    args = parser.parse_args(argv)
    run_operation(lambda: reconstruct(args))


if __name__ == "__main__":
    main()
