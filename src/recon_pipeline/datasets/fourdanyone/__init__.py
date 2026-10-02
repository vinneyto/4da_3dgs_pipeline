"""4DAnyone configuration and standalone operations."""

__all__ = ["FourDAnyoneConfig", "FourDAnyoneInferencePass", "PrepareExperimentPass"]


def __getattr__(name):
    if name == "FourDAnyoneConfig":
        from .config import FourDAnyoneConfig

        return FourDAnyoneConfig
    if name in {"FourDAnyoneInferencePass", "PrepareExperimentPass"}:
        from .passes import FourDAnyoneInferencePass, PrepareExperimentPass

        return {
            "FourDAnyoneInferencePass": FourDAnyoneInferencePass,
            "PrepareExperimentPass": PrepareExperimentPass,
        }[name]
    raise AttributeError(name)
