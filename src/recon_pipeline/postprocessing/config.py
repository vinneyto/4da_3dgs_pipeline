"""Portable postprocessing selection, with machine paths supplied at runtime."""

from dataclasses import dataclass
from pathlib import Path

from recon_pipeline.utilities.artefacts.splats._formats import validate_formats


@dataclass(frozen=True, slots=True)
class SplatConversionConfig:
    enabled: bool = False
    formats: tuple[str, ...] = ("compressed_ply", "spz", "sog")
    splat_transform: str | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("postprocessing.splat_conversion.enabled must be a boolean")
        object.__setattr__(self, "formats", validate_formats(self.formats))
        if (
            self.splat_transform is not None
            and not Path(self.splat_transform).is_absolute()
        ):
            raise ValueError(
                "postprocessing.splat_conversion.splat_transform must be absolute"
            )

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("postprocessing.splat_conversion must be an object")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PostprocessingConfig:
    splat_conversion: SplatConversionConfig = SplatConversionConfig()

    def __post_init__(self):
        if not isinstance(self.splat_conversion, SplatConversionConfig):
            object.__setattr__(
                self,
                "splat_conversion",
                SplatConversionConfig.from_dict(self.splat_conversion),
            )

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("postprocessing must be an object")
        return cls(**payload)
