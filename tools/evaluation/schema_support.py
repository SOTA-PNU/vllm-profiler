"""Schema primitives owned by repository-only comparison evaluation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, TypeVar

from perfetto_hetero_profiler.overview.schema import OverviewSchemaError
from perfetto_hetero_profiler.schema import Availability


ALIGNMENT_STATUSES = {
    "canonical", "aligned", "partial", "unaligned", "not_applicable",
    "not_available", "unknown",
}
OBSERVATION_LAYERS = {
    "request_facing_client", "hybrid_pipeline", "normalized_resource_metric",
    "gpu_only", "npu_only", "run",
}
PROFILE_MODES = {"monitor", "detailed_profile"}
RUN_MODES = {"gpu_only", "npu_only", "hybrid"}
_EMBEDDED_POSIX_PATH_RE = re.compile(r"""(?:^|[\s="'(])/(?!/)[A-Za-z0-9._-]""")
_ModelT = TypeVar("_ModelT")


def fail(path: str, message: str) -> None:
    raise OverviewSchemaError(path, message)


def _path_free_string(value: str, path: str) -> None:
    if (
        value.startswith("file://")
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or _EMBEDDED_POSIX_PATH_RE.search(value) is not None
    ):
        fail(path, "must not contain a host absolute path")


def nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(path, "must be a non-empty string")
    _path_free_string(value, path)
    return value


def integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        fail(path, "must be an integer, not bool")
    if minimum is not None and value < minimum:
        fail(path, f"must be >= {minimum}")
    return value


def _number(value: object, path: str) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        fail(path, "must be a finite number, not bool")
    return value


def json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            fail(path, "must not contain NaN or Infinity")
        return
    if isinstance(value, str):
        _path_free_string(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                fail(path, "object keys must be non-empty strings")
            _path_free_string(key, f"{path}.{key}")
            json_value(item, f"{path}.{key}")
        return
    fail(path, f"contains non-JSON value {type(value).__name__}")


def raw_primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: raw_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): raw_primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [raw_primitive(item) for item in value]
    if isinstance(value, list):
        return [raw_primitive(item) for item in value]
    return value


def require_type(value: object, expected: type[_ModelT], path: str) -> _ModelT:
    if not isinstance(value, expected):
        fail(path, f"must be {expected.__name__}")
    return value


def sorted_unique_strings(
    values: object, path: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        fail(path, "must be an immutable tuple")
    result = tuple(
        nonempty(value, f"{path}[{index}]") for index, value in enumerate(values)
    )
    if not allow_empty and not result:
        fail(path, "must not be empty")
    if result != tuple(sorted(set(result))):
        fail(path, "must be sorted and contain no duplicates")
    return result


def sorted_models(
    values: object, path: str, *, key, allow_empty: bool = True
) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        fail(path, "must be an immutable tuple")
    if not allow_empty and not values:
        fail(path, "must not be empty")
    keys = [key(value) for value in values]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        fail(path, "must be deterministically sorted with unique keys")
    return values


def strict_object(value: object, cls: type[Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(path, "must be an object")
    expected = {item.name for item in fields(cls)}
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail = f"missing fields {missing}" if missing else f"unknown fields {unknown}"
        fail(path, detail)
    return dict(value)


def tuple_of(value: object, parser, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        fail(path, "must be an array")
    return tuple(parser(item, f"{path}[{index}]") for index, item in enumerate(value))


def string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        fail(path, "must be an array")
    return tuple(value)


def availability(value: object, path: str) -> Availability:
    try:
        return Availability(value)
    except (TypeError, ValueError) as error:
        fail(path, "is not a valid availability")
        raise AssertionError from error


def validate_available_scalar(
    availability_value: Availability,
    value: object,
    reason: object,
    path: str,
) -> None:
    if not isinstance(availability_value, Availability):
        fail(f"{path}.availability", "must be an Availability enum")
    if availability_value is Availability.AVAILABLE:
        _number(value, f"{path}.value")
        if reason is not None:
            fail(f"{path}.unavailable_reason", "must be null when available")
    else:
        if value is not None:
            fail(f"{path}.value", "must be null when unavailable")
        nonempty(reason, f"{path}.unavailable_reason")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


__all__ = [
    "ALIGNMENT_STATUSES", "OBSERVATION_LAYERS", "PROFILE_MODES", "RUN_MODES",
    "availability", "fail", "integer", "json_value", "nonempty",
    "raw_primitive", "reject_duplicate_pairs", "require_type", "sorted_models",
    "sorted_unique_strings", "strict_object", "string_tuple", "tuple_of",
    "validate_available_scalar",
]
