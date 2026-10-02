"""Compatibility entry point for legacy --request commands."""

from recon_pipeline.utilities.datasets.fourdanyone.legacy_request import main

PROGRESS_PREFIX = "FOURDA_PROGRESS "


if __name__ == "__main__":
    main()
