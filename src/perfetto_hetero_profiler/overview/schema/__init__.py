"""JSON Schema structure and semantic validation for Overview model v1.

``primitives`` holds the value rules and the structural contract, ``semantics``
the cross-field rules JSON Schema cannot express; this module turns a validated
document into canonical bytes and strictly back again.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from typing import Any

from ...schema import Availability
from ...support.json_io import compact_json_bytes
from ..model import (
    OVERVIEW_REPORT_RECORD_TYPE,
    DisplayRule,
    KpiCalculation,
    KpiClock,
    KpiSections,
    KpiScope,
    KpiSource,
    KpiValue,
    OverviewReport,
    ResourceSummary,
)
from .primitives import (
    OVERVIEW_REPORT_SCHEMA_NAME,
    OverviewSchemaError,
    _fail,
    _json_value,
    _raw_primitive,
    _validate_report_structure,
    load_json_schema,
    validate_json_schema_contract,
)
from .semantics import (
    validate_display_rule,
    validate_kpi,
    validate_kpi_clock,
    validate_kpi_scope,
    validate_kpi_sections,
    validate_kpi_source,
    validate_overview_report,
    validate_resource_summary,
)


def overview_to_dict(document: OverviewReport) -> dict[str, Any]:
    """Return a validated JSON-compatible object with no host paths."""

    validate_overview_report(document)
    value = _raw_primitive(document)
    assert isinstance(value, dict)
    _json_value(value, "overview")
    return value


def canonical_json_bytes(document: OverviewReport) -> bytes:
    """Serialize a validated document to stable, path-free canonical bytes."""

    value = overview_to_dict(document)
    try:
        return compact_json_bytes(value)
    except (TypeError, ValueError) as error:  # pragma: no cover - guarded above
        raise OverviewSchemaError(
            "overview",
            f"cannot be serialized as finite canonical JSON: {error}",
        ) from error


def canonical_sha256(document: OverviewReport) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


_KPI_SECTION_NAMES = (
    "request_facing_latency",
    "pipeline_latency",
    "throughput_and_tokens",
    "transfer",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _strict_object(value: object, cls: type[Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    expected = {item.name for item in fields(cls)}
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail = (
            f"missing fields {missing}" if missing else f"unknown fields {unknown}"
        )
        _fail(path, detail)
    return dict(value)


def _tuple_of(value: object, parser, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return tuple(parser(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return tuple(value)


def _plain_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return dict(value)


def _availability(value: object, path: str) -> Availability:
    try:
        return Availability(value)
    except (TypeError, ValueError) as error:
        _fail(path, "is not a valid availability")
        raise AssertionError from error


# Ordered per-dataclass field conversions.  Order is part of the contract: a
# document with several problems must still report the same first failure.
_FIELD_PARSERS: dict[type, tuple[tuple[str, str, Any], ...]] = {
    KpiSource: (
        ("record_ids", "strings", None),
        ("metric_names", "strings", None),
    ),
    KpiClock: (("domain_ids", "strings", None),),
    KpiValue: (
        ("availability", "availability", None),
        ("sources", "models", KpiSource),
        ("scope", "model", KpiScope),
        ("calculation", "model", KpiCalculation),
        ("clock", "model", KpiClock),
        ("quality_warnings", "strings", None),
        ("display", "model", DisplayRule),
    ),
    ResourceSummary: (
        ("scope", "model", KpiScope),
        ("clock", "model", KpiClock),
        ("aggregates", "models", KpiValue),
        ("quality_warnings", "strings", None),
    ),
    KpiSections: tuple(
        (name, "models", KpiValue) for name in _KPI_SECTION_NAMES
    ),
}


def _model_from_dict(value: object, cls: type[Any], path: str) -> Any:
    """Parse one strict dataclass, converting its fields in contract order."""

    data = _strict_object(value, cls, path)
    for name, kind, item_cls in _FIELD_PARSERS.get(cls, ()):
        field_path = f"{path}.{name}"
        if kind == "strings":
            data[name] = _string_tuple(data[name], field_path)
        elif kind == "model":
            data[name] = _model_from_dict(data[name], item_cls, field_path)
        elif kind == "models":
            data[name] = _tuple_of(
                data[name],
                lambda item, item_path, target=item_cls: _model_from_dict(
                    item, target, item_path
                ),
                field_path,
            )
        else:
            data[name] = _availability(data[name], field_path)
    return cls(**data)


def overview_report_from_dict(value: object) -> OverviewReport:
    """Parse and semantically validate a strict Overview report object."""

    _json_value(value, "overview")
    _validate_report_structure(value)
    data = _strict_object(value, OverviewReport, "overview")
    data["models"] = _tuple_of(data["models"], _plain_object, "overview.models")
    data["hardware"] = _tuple_of(
        data["hardware"], _plain_object, "overview.hardware"
    )
    data["kpis"] = _model_from_dict(data["kpis"], KpiSections, "overview.kpis")
    data["resources"] = _tuple_of(
        data["resources"],
        lambda item, item_path: _model_from_dict(item, ResourceSummary, item_path),
        "overview.resources",
    )
    data["native_profiles"] = _tuple_of(
        data["native_profiles"], _plain_object, "overview.native_profiles"
    )
    report = OverviewReport(**data)
    validate_overview_report(report)
    return report


def overview_document_from_dict(value: object) -> OverviewReport:
    """Parse the installed package's single-run Overview model."""

    if not isinstance(value, dict):
        _fail("overview", "must be an object")
    record_type = value.get("record_type")
    if record_type != OVERVIEW_REPORT_RECORD_TYPE:
        _fail("overview.record_type", "is not the Overview report record type")
    return overview_report_from_dict(value)


def overview_document_from_json(payload: str | bytes) -> OverviewReport:
    """Decode strict finite JSON and return a validated Overview report."""

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise OverviewSchemaError("overview", f"invalid JSON: {error}") from error
    return overview_document_from_dict(value)


__all__ = [
    "OVERVIEW_REPORT_SCHEMA_NAME",
    "OverviewSchemaError",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_json_schema",
    "overview_report_from_dict",
    "overview_to_dict",
    "validate_display_rule",
    "validate_json_schema_contract",
    "validate_kpi",
    "validate_kpi_clock",
    "validate_kpi_sections",
    "validate_kpi_scope",
    "validate_kpi_source",
    "validate_overview_report",
    "validate_resource_summary",
]
