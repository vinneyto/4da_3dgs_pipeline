"""Upload explicit artifact paths and commit a portable completion record."""

import json
from pathlib import Path
from ._arguments import parser
from .bundles import publish
from recon_pipeline.utilities._output import run_operation


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pass-id", required=True)
    p.add_argument("--upload-id", required=True)
    p.add_argument("--signature", required=True)
    args = p.parse_args(argv)

    def operation():
        document = json.loads(args.checkpoint.read_text())
        checkpoint = next(
            item for item in document["passes"] if item["id"] == args.pass_id
        )
        return publish(
            bucket=args.bucket,
            prefix=args.prefix,
            root=args.root,
            data_root=args.data_root,
            paths=args.paths,
            checkpoint=checkpoint,
            upload_id=args.upload_id,
            signature=args.signature,
            region=args.region,
        )

    run_operation(operation)


if __name__ == "__main__":
    main()
