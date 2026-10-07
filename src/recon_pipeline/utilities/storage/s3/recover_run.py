"""Restore matching committed artifacts and portable completion records."""

import json
from pathlib import Path
from ._arguments import parser
from .bundles import recover
from recon_pipeline.utilities._output import run_operation


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--plan", required=True, help="JSON list of expected artifact signatures"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reset-commits", action="store_true")
    args = p.parse_args(argv)
    run_operation(
        lambda: recover(
            bucket=args.bucket,
            prefix=args.prefix,
            root=args.root,
            data_root=args.data_root,
            plan=json.loads(args.plan),
            checkpoint_path=args.checkpoint,
            region=args.region,
            reset_commits=args.reset_commits,
        )
    )


if __name__ == "__main__":
    main()
