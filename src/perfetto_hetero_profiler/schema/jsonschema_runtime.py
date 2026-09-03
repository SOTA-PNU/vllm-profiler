"""Cached validation against the packaged core record schemas."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


_SCHEMA_PACKAGE = "perfetto_hetero_profiler.schema"
_SCHEMA_BY_RECORD_TYPE = {
    "run_manifest": ("run_manifest.schema.json", "run_manifest"),
    "event": ("event_record.schema.json", "event"),
    "metric": ("metric_sample.schema.json", "metric"),
    "artifact": ("artifact_reference.schema.json", "artifact"),
    "clock_domain": ("clock_domain.schema.json", "clock_domain"),
    "sync_point": ("sync_point.schema.json", "sync_point"),
    "clock_transform": ("clock_transform.schema.json", "clock_transform"),
}


class JsonSchemaFailure(ValueError):
    """Internal, version-independent structural validation failure."""

    def __init__(self, field_path: str, message: str):
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


def _walk_references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for keyword in ("$ref", "$dynamicRef"):
            reference = value.get(keyword)
            if isinstance(reference, str):
                yield reference
        for item in value.values():
            yield from _walk_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_references(item)


@lru_cache(maxsize=len(_SCHEMA_BY_RECORD_TYPE))
def _validator(record_type: str) -> Draft202012Validator:
    try:
        filename, _ = _SCHEMA_BY_RECORD_TYPE[record_type]
    except KeyError as error:
        raise JsonSchemaFailure("record_type", "is not a supported record type") from error
    resource = files(_SCHEMA_PACKAGE)
    for component in ("json", "v1", filename):
        resource = resource.joinpath(component)
    schema = json.loads(resource.read_text(encoding="utf-8"))
    external = sorted(ref for ref in _walk_references(schema) if not ref.startswith("#"))
    if external:
        raise RuntimeError(f"external schema reference is not allowed: {external[0]}")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _path_key(path: Iterable[object]) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, item) if isinstance(item, int) else (1, str(item))
        for item in path
    )


def _error_key(error: ValidationError) -> tuple[object, ...]:
    return (
        _path_key(error.absolute_path),
        _path_key(error.absolute_schema_path),
        str(error.validator or ""),
        _message(error),
    )


def _field_path(record_type: str, error: ValidationError) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required":
        missing = sorted(set(error.validator_value) - set(error.instance))
        if missing:
            parts.append(missing[0])
    elif error.validator == "additionalProperties" and isinstance(error.instance, dict):
        properties = error.schema.get("properties", {})
        extras = sorted(str(key) for key in set(error.instance) - set(properties))
        if extras:
            parts.append(extras[0])
    path = _SCHEMA_BY_RECORD_TYPE[record_type][1]
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _message(error: ValidationError) -> str:
    validator = error.validator
    if validator == "required":
        return "required field is missing"
    if validator == "additionalProperties":
        return "unknown field"
    if validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            expected = " or ".join(str(item) for item in expected)
        article = "an" if str(expected) in {"array", "integer", "object"} else "a"
        return f"must be {article} {expected}"
    if validator == "enum":
        return f"must be one of {list(error.validator_value)!r}"
    if validator == "const":
        return f"must be {error.validator_value!r}"
    if validator == "minimum":
        return f"must be >= {error.validator_value}"
    if validator == "exclusiveMinimum":
        return f"must be > {error.validator_value}"
    if validator == "maximum":
        return f"must be <= {error.validator_value}"
    if validator == "minLength":
        return "must be a non-empty string"
    if validator == "minItems":
        return f"must contain at least {error.validator_value} item(s)"
    if validator == "pattern":
        return f"must match pattern {error.validator_value!r}"
    return f"failed {validator or 'schema'} validation"


def validate_structure(instance: Any, record_type: str) -> None:
    """Validate a primitive record without external reference resolution."""
    errors = list(_validator(record_type).iter_errors(instance))
    if errors:
        error = min(errors, key=_error_key)
        raise JsonSchemaFailure(_field_path(record_type, error), _message(error))


def schema_validator_cache_info():
    """Return cache statistics for tests and diagnostics."""
    return _validator.cache_info()
