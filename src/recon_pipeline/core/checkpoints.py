"""Durable pass checkpoints for restartable pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .core import PassResult


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=str)]
    raise TypeError(f"checkpoint value is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class PassCheckpoint:
    pass_id: str
    result: PassResult
    duration_seconds: float
    completed_at: str


class PassCheckpointStore(Protocol):
    def load(self) -> dict[str, PassCheckpoint]: ...

    def retain(self, pass_ids: list[str]) -> None: ...

    def complete(
        self, pass_id: str, result: PassResult, duration_seconds: float
    ) -> None: ...

    def clear(self) -> None: ...


class JsonPassCheckpointStore:
    """Atomically persist completed pass results inside one experiment."""

    schema_version = 1

    def __init__(self, path: Path, experiment_name: str) -> None:
        self.path = Path(path)
        self.experiment_name = experiment_name

    def _read_document(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": self.schema_version,
                "experiment_name": self.experiment_name,
                "passes": [],
            }
        document = json.loads(self.path.read_text())
        if document.get("schema_version") != self.schema_version:
            raise ValueError(f"unsupported checkpoint schema: {self.path}")
        if document.get("experiment_name") != self.experiment_name:
            raise ValueError(
                f"checkpoint belongs to another experiment: {self.path}"
            )
        if not isinstance(document.get("passes"), list):
            raise ValueError(f"checkpoint passes must be a list: {self.path}")
        return document

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def load(self) -> dict[str, PassCheckpoint]:
        checkpoints: dict[str, PassCheckpoint] = {}
        for item in self._read_document()["passes"]:
            pass_id = str(item["id"])
            if pass_id in checkpoints:
                raise ValueError(f"duplicate pass checkpoint: {pass_id}")
            result = PassResult(
                artifacts=dict(item.get("artifacts", {})),
                details=dict(item.get("details", {})),
            )
            checkpoints[pass_id] = PassCheckpoint(
                pass_id=pass_id,
                result=result,
                duration_seconds=float(item.get("duration_seconds", 0.0)),
                completed_at=str(item["completed_at"]),
            )
        return checkpoints

    def retain(self, pass_ids: list[str]) -> None:
        document = self._read_document()
        by_id = {str(item["id"]): item for item in document["passes"]}
        retained = [by_id[pass_id] for pass_id in pass_ids if pass_id in by_id]
        if retained != document["passes"]:
            document["passes"] = retained
            self._write_document(document)

    def complete(
        self, pass_id: str, result: PassResult, duration_seconds: float
    ) -> None:
        document = self._read_document()
        document["passes"] = [
            item for item in document["passes"] if item.get("id") != pass_id
        ]
        document["passes"].append(
            {
                "id": pass_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": float(duration_seconds),
                "artifacts": _json_value(result.artifacts),
                "details": _json_value(result.details),
            }
        )
        self._write_document(document)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
