"""Progress records shared by blocking and detached runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    stage: str
    fraction: float
    message: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("progress fraction must be in [0, 1]")

    @property
    def percent(self) -> int:
        return round(self.fraction * 100)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"percent": self.percent}

