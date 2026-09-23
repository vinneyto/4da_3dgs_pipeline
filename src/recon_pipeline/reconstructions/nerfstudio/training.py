"""Resolved, source-independent Splatfacto training configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class SplatfactoTrainingConfig:
    # Values from the successful Colab _05 run. The current 4DAnyone exporter
    # already supplies RGBA images and deliberately omits mask_path.
    max_num_iterations: int = 60_000
    background_color: str = "random"
    stop_split_at: int = 32_000
    cull_alpha_thresh: float = 0.02
    densify_grad_thresh: float = 0.0003
    densify_size_thresh: float = 0.0075
    split_screen_size: float = 0.03
    num_downscales: int = 1
    resolution_schedule: int = 2_000
    cull_scale_thresh: float = 0.10
    stop_screen_size_at: int = 32_000
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 10.0
    sh_degree: int = 2
    rasterize_mode: str = "classic"
    eval_mode: str = "all"
    vis: str = "tensorboard"

    def __post_init__(self) -> None:
        if self.max_num_iterations < 1 or self.stop_split_at < 1:
            raise ValueError("Splatfacto iteration counts must be positive")
        if self.background_color not in {"random", "black", "white"}:
            raise ValueError("invalid Splatfacto background_color")
        if self.eval_mode not in {"all", "fraction", "filename", "interval"}:
            raise ValueError("invalid Nerfstudio eval_mode")

    @classmethod
    def from_dict(cls, payload: Any) -> "SplatfactoTrainingConfig":
        if not isinstance(payload, dict):
            raise TypeError("pipeline.reconstruction.config.training must be an object")
        expected = {field.name for field in fields(cls)}
        missing = expected - set(payload)
        unexpected = set(payload) - expected
        if missing or unexpected:
            raise ValueError(
                f"Splatfacto training parameters: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SplatfactoReconstructionConfig:
    enabled: bool = False
    frames: tuple[int, ...] = ()
    training: SplatfactoTrainingConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        if self.enabled:
            if not self.frames or any(not isinstance(i, int) or i < 0 or i > 120 for i in self.frames):
                raise ValueError("Splatfacto frames must be distinct indices in 0..120")
            if len(set(self.frames)) != len(self.frames):
                raise ValueError("Splatfacto frames must be unique")
            if self.training is None:
                raise ValueError("enabled reconstruction needs explicit training parameters")

    @classmethod
    def from_dict(cls, payload: Any) -> "SplatfactoReconstructionConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("pipeline.reconstruction must be an object")
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError("pipeline.reconstruction.enabled must be boolean")
        if enabled and payload.get("type") != "nerfstudio_splatfacto":
            raise ValueError("only nerfstudio_splatfacto reconstruction is supported")
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise TypeError("pipeline.reconstruction.config must be an object")
        if not enabled:
            return cls()
        unexpected = set(config) - {"frames", "training"}
        if unexpected:
            raise ValueError(f"unsupported reconstruction settings: {sorted(unexpected)}")
        if "training" not in config:
            raise ValueError("enabled reconstruction needs explicit training parameters")
        return cls(
            enabled=True,
            frames=tuple(config.get("frames", ())),
            training=SplatfactoTrainingConfig.from_dict(config.get("training")),
        )
