#!/usr/bin/env python3
"""Read one scalar value from the shared run JSON document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("key", help="Dot-separated key, for example pipeline.model_dir")
    args = parser.parse_args()

    value = json.loads(args.config.read_text())
    for component in args.key.split("."):
        value = value[component]
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, (str, int, float)):
        print(value)
    else:
        raise TypeError(f"{args.key} is not a scalar value")


if __name__ == "__main__":
    main()
