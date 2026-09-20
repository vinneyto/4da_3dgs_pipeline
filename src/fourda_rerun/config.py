"""Configuration for the optional Rerun recording artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RerunConfig:
    """Control creation of an interactive camera and skeleton recording."""

    enabled: bool = False
    view_count: int = 4
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.view_count < 1:
            raise ValueError("rerun.view_count must be positive")
        if not self.device:
            raise ValueError("rerun.device must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | bool | None) -> "RerunConfig":
        if payload is None:
            return cls()
        if isinstance(payload, bool):
            return cls(enabled=payload)
        if not isinstance(payload, dict):
            raise TypeError("pipeline.rerun must be an object or boolean")
        return cls(**payload)
