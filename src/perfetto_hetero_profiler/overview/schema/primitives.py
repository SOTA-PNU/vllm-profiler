"""Primitive value rules and the bundled Draft 2020-12 contract.

The document stays JSON-safe, finite, deterministically ordered and free of
host paths.  The JSON Schema here carries all *structural* validation.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, TypeVar

from ...schema.constants import JSON_SCHEMA_DRAFT
from ...schema.jsonschema_runtime import (
    JsonSchemaFailure,
    compile_schema,
    validate_schema_document,
)
from ..model import OVERVIEW_REPORT_RECORD_TYPE, OverviewReport


_EMBEDDED_POSIX_PATH_RE = re.compile(
    r"""(?:^|[\s="'(])/(?!/)[A-Za-z0-9._-]"""
)

_ModelT = TypeVar("_ModelT")


class OverviewSchemaError(ValueError):
    """Stable field-path error for Overview model and JSON validation."""

    def __init__(self, field_path: str, message: str):
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


def _fail(path: str, message: str) -> None:
    raise OverviewSchemaError(path, message)


def _require_type(value: object, expected: type[_ModelT], path: str) -> _ModelT:
    if not isinstance(value, expected):
        _fail(path, f"must be {expected.__name__}")
    return value


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    _path_free_string(value, path)
    return value


def _integer(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    nullable: bool = False,
) -> int | None:
    if value is None and nullable:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer, not bool")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    nullable: bool = False,
) -> int | float | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a finite number, not bool")
    if not math.isfinite(value):
        _fail(path, "must be finite; NaN and Infinity are not allowed")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _path_free_string(value: str, path: str) -> None:
    if (
        value.startswith("file://")
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or _EMBEDDED_POSIX_PATH_RE.search(value) is not None
    ):
        _fail(path, "must not contain a host absolute path")


def _json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "must not contain NaN or Infinity")
        return
    if isinstance(value, str):
        _path_free_string(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _fail(path, "object keys must be non-empty strings")
            _path_free_string(key, f"{path}.{key}")
            _json_value(item, f"{path}.{key}")
        return
    _fail(path, f"contains non-JSON value {type(value).__name__}")


def _json_object(value: object, path: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    copied = dict(value)
    if nonempty and not copied:
        _fail(path, "must not be empty")
    _json_value(copied, path)


def _sorted_unique_strings(
    values: object,
    path: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        _fail(path, "must be an immutable tuple")
    result = tuple(_nonempty(value, f"{path}[{index}]") for index, value in enumerate(values))
    if not allow_empty and not result:
        _fail(path, "must not be empty")
    if result != tuple(sorted(set(result))):
        _fail(path, "must be sorted and contain no duplicates")
    return result


def _raw_primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _raw_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _raw_primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_raw_primitive(item) for item in value]
    if isinstance(value, list):
        return [_raw_primitive(item) for item in value]
    return value


def _canonical_sort_key(value: object) -> bytes:
    return json.dumps(
        _raw_primitive(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _deterministic_object_tuple(values: object, path: str) -> None:
    """Require a tuple of JSON objects in a stable, duplicate-free order."""

    assert isinstance(values, tuple)
    keys = [_canonical_sort_key(value) for value in values]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted without duplicates")


def _safe_relative_path(value: object, path: str) -> str:
    text = _nonempty(value, path)
    if "\\" in text:
        _fail(path, "must use POSIX separators")
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != text
        or text == "."
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        _fail(path, "must be a normalized relative path")
    return text


def _sorted_json_array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    keys = [_canonical_sort_key(item) for item in value]
    if keys != sorted(keys):
        _fail(path, "must be deterministically sorted")
    for index, item in enumerate(value):
        _json_value(item, f"{path}[{index}]")
    return value


OVERVIEW_REPORT_SCHEMA_NAME = "overview_report.schema.json"

# The JSON contracts are package data of ``perfetto_hetero_profiler.overview``,
# one level above this schema package.
_SCHEMA_PACKAGE = __package__.rpartition(".")[0]


def load_json_schema(record_type: str) -> dict[str, Any]:
    """Load one bundled Draft 2020-12 contract by record type."""

    names = {
        OVERVIEW_REPORT_RECORD_TYPE: OVERVIEW_REPORT_SCHEMA_NAME,
    }
    try:
        name = names[record_type]
    except KeyError as error:
        raise OverviewSchemaError(
            "record_type",
            f"unsupported schema record type {record_type!r}",
        ) from error
    resource = resources.files(_SCHEMA_PACKAGE) / "json" / "v1" / name
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OverviewSchemaError(
            "json_schema",
            f"cannot load {name}: {error}",
        ) from error
    if not isinstance(value, dict):
        _fail("json_schema", "must be an object")
    return value


@lru_cache(maxsize=1)
def _overview_validator():
    return compile_schema(load_json_schema(OVERVIEW_REPORT_RECORD_TYPE))


def _validate_report_structure(value: object) -> None:
    try:
        validate_schema_document(
            value,
            _overview_validator(),
            root_path="overview",
        )
    except JsonSchemaFailure as error:
        raise OverviewSchemaError(error.field_path, error.message) from error


def validate_json_schema_contract() -> None:
    """Verify checked-in schema identity and top-level model field parity."""

    contracts = (
        (
            OVERVIEW_REPORT_RECORD_TYPE,
            OverviewReport,
            OVERVIEW_REPORT_SCHEMA_NAME,
        ),
    )
    for record_type, cls, filename in contracts:
        schema = load_json_schema(record_type)
        if schema.get("$schema") != JSON_SCHEMA_DRAFT:
            _fail(filename, f"$schema must be {JSON_SCHEMA_DRAFT}")
        if schema.get("type") != "object":
            _fail(filename, "top-level type must be object")
        if schema.get("additionalProperties") is not False:
            _fail(filename, "must reject additional properties")
        properties = schema.get("properties")
        required = schema.get("required")
        expected = {item.name for item in fields(cls)}
        if not isinstance(properties, dict) or set(properties) != expected:
            _fail(filename, "top-level properties differ from dataclass fields")
        if not isinstance(required, list) or set(required) != expected:
            _fail(filename, "all top-level properties must be required")
        record_schema = properties.get("record_type")
        if (
            not isinstance(record_schema, dict)
            or record_schema.get("const") != record_type
        ):
            _fail(filename, "record_type const differs from model")


__all__ = [
    "OVERVIEW_REPORT_SCHEMA_NAME",
    "OverviewSchemaError",
    "_canonical_sort_key",
    "_deterministic_object_tuple",
    "_fail",
    "_integer",
    "_json_object",
    "_json_value",
    "_nonempty",
    "_number",
    "_path_free_string",
    "_raw_primitive",
    "_require_type",
    "_safe_relative_path",
    "_sorted_json_array",
    "_sorted_unique_strings",
    "_validate_report_structure",
    "load_json_schema",
    "validate_json_schema_contract",
]
