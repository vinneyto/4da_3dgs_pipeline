"""Restore a saved 4DAnyone experiment from an S3 prefix."""

import shutil
from pathlib import Path
from typing import Sequence
from recon_pipeline.utilities._output import report_progress, run_operation
from .operations import sync_prefix, _client
from .bundles import marker_key, read_json, restore_files
from ._arguments import parser


def restore(args) -> dict:
    generation = args.destination / "4danyone"
    required = (generation / "metadata.json", generation / "cameras.json")
    if args.replace_existing and generation.exists():
        shutil.rmtree(generation)
    if all(path.is_file() for path in required):
        found = downloaded = 0
        message = "Reusing source experiment from persistent storage"
    else:
        report_progress(0.0, "Restoring source experiment from S3")
        client = _client(args.region)
        record = read_json(
            client, args.bucket, marker_key(args.prefix, "fourdanyone-inference")
        )
        if record is not None:
            if record.get("experiment_name") != args.destination.name:
                raise ValueError("Source artifact commit belongs to another experiment")
            downloaded = restore_files(
                client, args.bucket, args.prefix, args.destination, record
            )
            found = len(record["files"])
        else:
            # Legacy experiments have no commits; fetch only the required generation.
            found, downloaded = sync_prefix(
                bucket=args.bucket,
                prefix=args.prefix.rstrip("/") + "/4danyone",
                destination=generation,
                region=args.region,
                require_objects=True,
            )
        message = "Source experiment restored"
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            f"restored experiment requires metadata.json and cameras.json: {generation}"
        )
    report_progress(1.0, message)
    return {"path": str(generation), "objects": found, "downloaded": downloaded}


def main(argv: Sequence[str] | None = None) -> None:
    p = parser(__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--replace-existing", action="store_true")
    args = p.parse_args(argv)
    run_operation(lambda: restore(args))


if __name__ == "__main__":
    main()
