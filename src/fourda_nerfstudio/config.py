"""Configuration for exporting synchronized Nerfstudio datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_FRAMES = (60,)


@dataclass(frozen=True, slots=True)
class NerfstudioArtifactConfig:
    """One or more static temporal slices exported from a 4DAnyone run."""

    enabled: bool = True
    source_experiment_name: str | None = None
    frames: tuple[int, ...] = DEFAULT_FRAMES
    device: str = "cuda:0"
    replace_existing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frames",
            tuple(int(value) for value in self.frames),
        )
        if self.source_experiment_name is not None and (
            not self.source_experiment_name
            or any(
                part in self.source_experiment_name
                for part in ("/", "\\", "..")
            )
        ):
            raise ValueError(
                "nerfstudio.source_experiment_name must be a simple directory name"
            )
        if not self.frames:
            raise ValueError("nerfstudio.frames must not be empty")
        if any(index < 0 or index > 120 for index in self.frames):
            raise ValueError(
                "nerfstudio frame indices must be in the inclusive range 0..120"
            )
        if len(set(self.frames)) != len(self.frames):
            raise ValueError("nerfstudio.frames must be unique")
        if not self.device:
            raise ValueError("nerfstudio.device must not be empty")

    @classmethod
    def from_dict(cls, payload: Any) -> "NerfstudioArtifactConfig":
        if payload is None:
            return cls(enabled=False)
        if isinstance(payload, bool):
            return cls(enabled=payload)
        if not isinstance(payload, dict):
            raise TypeError("artifacts.dataset.nerfstudio must be an object or boolean")
        values = dict(payload)
        if "frame_indices" in values:
            if "frames" in values:
                raise ValueError(
                    "nerfstudio must use either frames or legacy frame_indices, not both"
                )
            values["frames"] = values.pop("frame_indices")
        return cls(**values)
