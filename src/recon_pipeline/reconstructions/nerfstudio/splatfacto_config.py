"""Pipeline selection and execution environment for frame reconstruction."""

from dataclasses import dataclass
from pathlib import Path

from recon_pipeline.utilities.reconstructions.nerfstudio._profile import TrainingProfile


@dataclass(frozen=True, slots=True)
class SplatfactoConfig(TrainingProfile):
    enabled: bool = False
    frames: tuple[int, ...] | None = None
    nerfstudio_bin: str | None = None

    def __post_init__(self):
        TrainingProfile.__post_init__(self)
        if not isinstance(self.enabled, bool):
            raise TypeError("reconstruction.enabled must be a boolean")
        if (
            self.nerfstudio_bin is not None
            and not Path(self.nerfstudio_bin).is_absolute()
        ):
            raise ValueError("reconstruction.nerfstudio_bin must be absolute")
        if self.frames is not None:
            object.__setattr__(self, "frames", tuple(self.frames))
            if (
                not self.frames
                or len(set(self.frames)) != len(self.frames)
                or any(
                    not isinstance(frame, int) or frame < 0 or frame > 120
                    for frame in self.frames
                )
            ):
                raise ValueError(
                    "reconstruction.frames must be unique indices in 0..120"
                )

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("reconstruction configuration must be an object")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SplatfactoRerunConfig:
    enabled: bool = False
    python: str | None = None
    max_splats: int = 100_000
    view_count: int = 4
    source_experiment_name: str | None = None
    device: str = "auto"
    replace_existing: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("reconstruction.rerun.enabled must be a boolean")
        if self.python is not None and not Path(self.python).is_absolute():
            raise ValueError("reconstruction.rerun.python must be absolute")
        if self.max_splats < 1 or self.view_count < 1:
            raise ValueError("reconstruction Rerun limits must be positive")
        if self.device not in ("auto", "cpu"):
            raise ValueError("reconstruction Rerun uses CPU PLY import")

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return cls()
        if isinstance(payload, bool):
            return cls(enabled=payload)
        return cls(**payload)
