"""Durable job status stored on the SageMaker Space volume."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class JobStatus:
    job_id: str
    state: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Job queued"
    pid: int | None = None
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    result_path: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = utc_now()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            if not hasattr(self, key):
                raise AttributeError(key)
            setattr(self, key, value)
        self.updated_at = utc_now()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> "JobStatus":
        return cls(**json.loads(path.read_text()))
