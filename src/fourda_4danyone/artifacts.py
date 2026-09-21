"""Artifact contracts consumed and produced by 4DAnyone passes."""

INPUT_VIDEO = "input.video"
MODEL_CACHE = "models.cache"
EXPERIMENT_WORKSPACE = "experiment.workspace"


def experiment_artifact(experiment_name: str) -> str:
    return f"experiment.4danyone:{experiment_name}"
