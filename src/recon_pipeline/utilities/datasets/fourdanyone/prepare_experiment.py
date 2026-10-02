"""Create a workspace directory and save an arbitrary JSON settings snapshot."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from recon_pipeline.utilities._output import run_operation


def prepare(directory: Path, settings: dict, settings_output: Path | None) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON object")
    directory.mkdir(parents=True, exist_ok=True)
    output = settings_output or directory / "config.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return {"path": str(directory), "settings_path": str(output)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--settings", type=json.loads, default={}, help="Arbitrary JSON object"
    )
    parser.add_argument(
        "--settings-output", type=Path, help="Defaults to DIRECTORY/config.json"
    )
    args = parser.parse_args(argv)
    run_operation(lambda: prepare(args.directory, args.settings, args.settings_output))


if __name__ == "__main__":
    main()
