"""Small strict JSON-to-dataclass boundary, shared by portable settings."""

from __future__ import annotations

import math
from dataclasses import fields
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def object_value(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object")
    return value


def checked_value(value: Any, annotation: Any, location: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for alternative in args:
            try:
                return checked_value(value, alternative, location)
            except TypeError:
                pass
        raise TypeError(f"{location} must have type {annotation}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{location} must be an array")
        return tuple(
            checked_value(v, args[0], f"{location}[{i}]") for i, v in enumerate(value)
        )
    if annotation is float:
        valid = type(value) in (int, float) and math.isfinite(value)
    else:
        valid = type(value) is annotation
    if not valid:
        raise TypeError(f"{location} must have type {annotation.__name__}")
    return value


def parse_settings(cls: type[T], payload: Any, location: str) -> T:
    values = object_value(payload, location)
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = values.keys() - names
    if unknown:
        raise ValueError(f"{location}: unsupported fields {sorted(unknown)}")
    return cls(
        **{k: checked_value(v, hints[k], f"{location}.{k}") for k, v in values.items()}
    )
