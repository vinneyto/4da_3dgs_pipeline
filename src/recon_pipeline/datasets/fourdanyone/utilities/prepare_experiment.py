"""Write a materialized dataset config into its experiment workspace."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from recon_pipeline.core.utility import add_result_argument, write_result
from ..config import FourDAnyoneConfig, load_pipeline_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        required=True,
        help="Local run JSON or materialized FourDAnyoneConfig JSON, including local paths",
    )
    add_result_argument(parser)
    args = parser.parse_args(argv)
    document = json.loads(args.pipeline_config.read_text())
    config = (
        load_pipeline_config(args.pipeline_config)
        if "schema_version" in document
        else FourDAnyoneConfig.from_dict(document)
    )
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(config.experiment_dir / "pipeline-config.json")
    write_result({"path": str(config.experiment_dir)}, args.result_file)


if __name__ == "__main__":
    main()
