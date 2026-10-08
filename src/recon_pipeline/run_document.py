"""Keep portable run documents separate from machine environment settings."""

import copy

SCHEMA_VERSION = 7
SUPPORTED_SCHEMA_VERSIONS = frozenset(range(1, SCHEMA_VERSION + 1))


def strip_environment(document):
    """Upgrade a previous run/template without retaining machine-specific values."""
    result = copy.deepcopy(document)
    result.pop("environment", None)
    worker = result.get("aws_worker", {})
    worker.pop("local", None)
    worker.pop("jobs_dir", None)
    pipeline = result.get("pipeline", {})
    dataset = pipeline.get("dataset", {}).get("config", pipeline)
    for key in ("fourdanyone_root", "model_dir", "runs_dir", "python"):
        dataset.pop(key, None)
    if "video_path" in dataset:
        # Local inputs are relative to RECON_DATA_ROOT/input in portable configs.
        from pathlib import Path

        dataset.setdefault("video", Path(dataset.pop("video_path")).name)
    for key in ("nerfstudio_bin", "splat_transform"):
        pipeline.get("reconstruction", {}).get("config", {}).pop(key, None)
    rerun = (result.get("artifacts") or {}).get("reconstruction", {}).get("rerun")
    if isinstance(rerun, dict):
        rerun.pop("python", None)
    return result


def validate_environment_free(document):
    if strip_environment(document) != document:
        raise ValueError(
            "Run config contains machine environment settings. Use RECON_* variables; "
            "migrate an older config with recon-config --template OLD --output NEW "
            "--experiment-name NAME --video VIDEO."
        )
