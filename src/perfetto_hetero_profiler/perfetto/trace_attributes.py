"""Allowlisted performance summaries for official Perfetto Trace Attributes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Final

from ..schema import Availability
from .model import TraceAttributeSpec


TRACE_ATTRIBUTE_NAMESPACE: Final = "kr.ac.pusan.sota.vllm_profiler."
TRACE_ATTRIBUTE_SCHEMA_VERSION: Final = "1.1.0"
_LEGACY_TRACE_ATTRIBUTE_SCHEMA_VERSION: Final = "1.0.0"
TRACE_ATTRIBUTE_RECORD_TYPE: Final = "perfetto_trace_attribute_validation"

_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")

_LATENCY_EXPORTS: Final = (
    ("request_facing_latency", "latency.e2e", "kpi.latency.e2e"),
    ("request_facing_latency", "latency.ttft", "kpi.latency.ttft"),
    ("request_facing_latency", "latency.tpot", "kpi.latency.tpot"),
    ("pipeline_latency", "latency.prefill", "kpi.latency.prefill"),
    ("pipeline_latency", "latency.decode", "kpi.latency.decode"),
)
_THROUGHPUT_EXPORTS: Final = (
    (
        "throughput.requests",
        "kpi.throughput.requests",
        "requests/s",
        "value_requests_milli_per_second",
    ),
    (
        "throughput.output_tokens",
        "kpi.throughput.output_tokens",
        "tokens/s",
        "value_output_tokens_milli_per_second",
    ),
)
_TRANSFER_EXPORTS: Final = (
    ("pipeline_latency", "latency.kv_export", "transfer.kv_export_duration", "ns", "value_ns", 1),
    (
        "pipeline_latency",
        "latency.kv_transfer",
        "transfer.kv_transfer_duration",
        "ns",
        "value_ns",
        1,
    ),
    (
        "pipeline_latency",
        "latency.kv_transform",
        "transfer.kv_transform_duration",
        "ns",
        "value_ns",
        1,
    ),
    ("transfer", "transfer.bytes", "transfer.bytes", "bytes", "value_bytes", 1),
    (
        "transfer",
        "transfer.effective_bandwidth",
        "transfer.effective_bandwidth",
        "bytes/s",
        "value_bytes_per_second",
        1,
    ),
    (
        "transfer",
        "transfer.e2e_share",
        "transfer.e2e_share",
        "ratio",
        "value_milli_percent",
        100_000,
    ),
)


class TraceAttributeExportError(ValueError):
    """A canonical KPI cannot be represented by the trace attribute contract."""


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _safe_string(value: object, *, field: str, maximum: int = 240) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise TraceAttributeExportError(f"{field} must be a short non-empty string")
    if "\n" in value or "\r" in value or "\x00" in value:
        raise TraceAttributeExportError(f"{field} contains a control character")
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("file://")
        or "://" in value
        or "@" in value
        or _SHA256_RE.search(value) is not None
    ):
        raise TraceAttributeExportError(f"{field} contains private provenance")
    return value


def _int64(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceAttributeExportError(f"{field} must be an integer")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise TraceAttributeExportError(f"{field} does not fit signed int64")
    return value


def _finite(value: object, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceAttributeExportError(f"{field} must be numeric")
    if not math.isfinite(value):
        raise TraceAttributeExportError(f"{field} must be finite")
    return value


def fixed_point_half_even(
    value: object,
    *,
    multiplier: int,
    field: str,
) -> int:
    """Quantize a finite canonical number with decimal half-even rounding."""

    number = _finite(value, field=field)
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
        raise TraceAttributeExportError("fixed-point multiplier must be positive")
    try:
        quantized = (Decimal(str(number)) * Decimal(multiplier)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_EVEN,
        )
    except (InvalidOperation, ValueError) as error:
        raise TraceAttributeExportError(f"{field} cannot be quantized") from error
    return _int64(int(quantized), field=field)


def _section(calculated: Mapping[str, object], name: str) -> Sequence[Mapping[str, object]]:
    value = calculated.get(name)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TraceAttributeExportError(f"canonical KPI section {name!r} is invalid")
    return value


def _find_kpi(
    calculated: Mapping[str, object],
    section: str,
    name: str,
) -> Mapping[str, object]:
    matches = [item for item in _section(calculated, section) if item.get("name") == name]
    if len(matches) != 1:
        raise TraceAttributeExportError(
            f"canonical KPI {section}:{name} must occur exactly once"
        )
    return matches[0]


def _metric_contract(
    value: Mapping[str, object],
    *,
    expected_name: str,
    expected_unit: str,
) -> tuple[str, int, str, object, str | None]:
    if value.get("name") != expected_name or value.get("canonical_unit") != expected_unit:
        raise TraceAttributeExportError(f"canonical KPI contract differs for {expected_name}")
    availability = value.get("availability")
    if availability not in {Availability.AVAILABLE.value, Availability.NOT_AVAILABLE.value}:
        raise TraceAttributeExportError(f"{expected_name} has invalid availability")
    sample_count = _int64(value.get("sample_count"), field=f"{expected_name} sample_count")
    if sample_count < 0:
        raise TraceAttributeExportError(f"{expected_name} sample_count is negative")
    aggregation = _safe_string(
        value.get("aggregation_method"), field=f"{expected_name} aggregation"
    )
    raw_value = value.get("value")
    raw_reason = value.get("unavailable_reason")
    if availability == Availability.AVAILABLE.value:
        _finite(raw_value, field=f"{expected_name} value")
        if raw_reason is not None:
            raise TraceAttributeExportError(f"{expected_name} has value and reason")
        reason = None
    else:
        if raw_value is not None:
            raise TraceAttributeExportError(f"{expected_name} unavailable value is not null")
        reason = _safe_string(raw_reason, field=f"{expected_name} reason")
    return availability, sample_count, aggregation, raw_value, reason


def _add(values: dict[str, int | str], suffix: str, value: int | str) -> None:
    key = f"{TRACE_ATTRIBUTE_NAMESPACE}{suffix}"
    if key in values:
        raise TraceAttributeExportError(f"duplicate trace attribute key: {key}")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TraceAttributeExportError(f"unsupported trace attribute value for {key}")
    if isinstance(value, int):
        _int64(value, field=key)
    else:
        _safe_string(value, field=key)
    values[key] = value


def _emit_metric(
    values: dict[str, int | str],
    *,
    base: str,
    metric: Mapping[str, object],
    expected_name: str,
    expected_unit: str,
    value_suffix: str,
    multiplier: int,
) -> None:
    availability, count, aggregation, raw_value, reason = _metric_contract(
        metric,
        expected_name=expected_name,
        expected_unit=expected_unit,
    )
    _add(values, f"{base}.sample_count", count)
    _add(values, f"{base}.aggregation", aggregation)
    if availability == Availability.AVAILABLE.value:
        _add(
            values,
            f"{base}.{value_suffix}",
            fixed_point_half_even(
                raw_value,
                multiplier=multiplier,
                field=f"{expected_name} {value_suffix}",
            ),
        )
    else:
        assert reason is not None
        _add(values, f"{base}.{value_suffix}", Availability.NOT_AVAILABLE.value)
        _add(values, f"{base}.reason", reason)


def _model_names(loaded: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for model in getattr(getattr(loaded, "manifest", None), "models", ()):
        role = _safe_string(getattr(model, "role", None), field="model role")
        raw_id = getattr(model, "model_id", None)
        if not isinstance(raw_id, str) or not raw_id:
            raise TraceAttributeExportError("model id must be a non-empty string")
        display = raw_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        display = _safe_string(display, field=f"{role} model display name")
        if role in result:
            raise TraceAttributeExportError(f"duplicate model role: {role}")
        result[role] = display
    return result


def _device_aliases(loaded: object) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[str]] = {"gpu": [], "npu": []}
    for device in getattr(getattr(loaded, "manifest", None), "devices", ()):
        device_type = _enum_value(getattr(device, "device_type", None))
        device_id = getattr(device, "device_id", None)
        if device_type in grouped:
            grouped[str(device_type)].append(
                _safe_string(device_id, field=f"{device_type} device id")
            )
    aliases: dict[str, dict[str, str]] = {}
    for device_type, identifiers in grouped.items():
        if len(identifiers) != len(set(identifiers)):
            raise TraceAttributeExportError(f"duplicate {device_type} device id")
        aliases[device_type] = {
            identifier: f"{device_type}_{index}"
            for index, identifier in enumerate(sorted(identifiers))
        }
    return aliases


def _resource_summaries(calculated: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    value = calculated.get("resource_summaries")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TraceAttributeExportError("canonical resource summaries are invalid")
    return value


def _stage_resource_aggregate(
    calculated: Mapping[str, object],
    *,
    stage: str,
    metric_name: str,
    device_type: str | None,
    device_id: str | None,
    statistic: str,
) -> Mapping[str, object] | None:
    matches: list[Mapping[str, object]] = []
    for summary in _resource_summaries(calculated):
        if summary.get("metric_name") != metric_name:
            continue
        scope = summary.get("scope")
        if not isinstance(scope, dict):
            raise TraceAttributeExportError("canonical resource scope is invalid")
        # Only summaries already proven by the canonical calculation layer to
        # represent the named stage window are eligible. Capture-wide streams
        # are deliberately not sliced or copied in this presentation layer.
        if scope.get("phase") != stage or scope.get("window") != stage:
            continue
        if scope.get("device_type") != device_type or scope.get("device_id") != device_id:
            continue
        aggregates = summary.get("aggregates")
        if not isinstance(aggregates, list):
            raise TraceAttributeExportError("canonical resource aggregates are invalid")
        expected = f"{metric_name}.{statistic}"
        matches.extend(
            item
            for item in aggregates
            if isinstance(item, dict) and item.get("name") == expected
        )
    if len(matches) > 1:
        raise TraceAttributeExportError(
            f"ambiguous stage resource aggregate: {stage}:{metric_name}:{device_id}"
        )
    return matches[0] if matches else None


def _emit_resource(
    values: dict[str, int | str],
    calculated: Mapping[str, object],
    *,
    base: str,
    stage: str,
    metric_name: str,
    device_type: str | None,
    device_id: str | None,
    statistic: str,
    unit: str,
) -> None:
    value_suffix = "value_milli_percent" if unit == "percent" else "value_bytes"
    aggregate = _stage_resource_aggregate(
        calculated,
        stage=stage,
        metric_name=metric_name,
        device_type=device_type,
        device_id=device_id,
        statistic=statistic,
    )
    if aggregate is None:
        _add(values, f"{base}.{value_suffix}", Availability.NOT_AVAILABLE.value)
        _add(values, f"{base}.sample_count", 0)
        _add(values, f"{base}.aggregation", "not_available")
        _add(values, f"{base}.reason", "no canonical stage-window resource aggregate")
        return
    multiplier = 1_000 if unit == "percent" else 1
    _emit_metric(
        values,
        base=base,
        metric=aggregate,
        expected_name=f"{metric_name}.{statistic}",
        expected_unit=unit,
        value_suffix=value_suffix,
        multiplier=multiplier,
    )


def build_performance_trace_attributes(
    loaded: object,
    calculated: Mapping[str, object],
) -> tuple[TraceAttributeSpec, ...]:
    """Build the complete sorted allowlist from canonical Overview KPI records."""

    if not isinstance(calculated, Mapping):
        raise TypeError("calculated must be a canonical KPI mapping")
    values: dict[str, int | str] = {}
    _add(values, "schema_version", TRACE_ATTRIBUTE_SCHEMA_VERSION)

    count = _find_kpi(calculated, "throughput_and_tokens", "request.count")
    count_contract = _metric_contract(
        count, expected_name="request.count", expected_unit="requests"
    )
    request_count = count_contract[3]
    measurement_scope = (
        "single_request_smoke"
        if count_contract[0] == Availability.AVAILABLE.value and request_count == 1
        else (
            "measured_smoke_window"
            if count_contract[0] == Availability.AVAILABLE.value
            else "not_available"
        )
    )
    _add(values, "measurement_scope", measurement_scope)

    models = _model_names(loaded)
    for role in ("prefill", "decode"):
        if role not in models:
            raise TraceAttributeExportError(f"manifest has no {role} model")
        _add(values, f"run.model.{role}", models[role])

    aliases = _device_aliases(loaded)
    for device_type in ("gpu", "npu"):
        stable = sorted(aliases[device_type].values())
        if not stable:
            raise TraceAttributeExportError(f"manifest has no {device_type} device")
        _add(values, f"run.hardware.{device_type}", ",".join(stable))

    _add(values, "run.measurement_scope", measurement_scope)
    for name, suffix in (
        ("request.count", "request_count"),
        ("request.input_tokens", "input_tokens"),
        ("request.output_tokens", "output_tokens"),
    ):
        metric = _find_kpi(calculated, "throughput_and_tokens", name)
        availability, _, _, raw_value, _ = _metric_contract(
            metric,
            expected_name=name,
            expected_unit="requests" if name == "request.count" else "tokens",
        )
        if availability == Availability.AVAILABLE.value:
            _add(values, f"run.{suffix}", _int64(raw_value, field=name))

    for section, name, base in _LATENCY_EXPORTS:
        _emit_metric(
            values,
            base=base,
            metric=_find_kpi(calculated, section, name),
            expected_name=name,
            expected_unit="ns",
            value_suffix="value_ns",
            multiplier=1,
        )
    for name, base, unit, suffix in _THROUGHPUT_EXPORTS:
        _emit_metric(
            values,
            base=base,
            metric=_find_kpi(calculated, "throughput_and_tokens", name),
            expected_name=name,
            expected_unit=unit,
            value_suffix=suffix,
            multiplier=1_000,
        )
    for section, name, base, unit, suffix, multiplier in _TRANSFER_EXPORTS:
        _emit_metric(
            values,
            base=base,
            metric=_find_kpi(calculated, section, name),
            expected_name=name,
            expected_unit=unit,
            value_suffix=suffix,
            multiplier=multiplier,
        )

    _emit_resource(
        values, calculated, base="resource.prefill.cpu.utilization_mean",
        stage="prefill", metric_name="resource.cpu.utilization", device_type=None,
        device_id=None, statistic="mean", unit="percent",
    )
    _emit_resource(
        values, calculated, base="resource.prefill.cpu.utilization_peak",
        stage="prefill", metric_name="resource.cpu.utilization", device_type=None,
        device_id=None, statistic="max", unit="percent",
    )
    _emit_resource(
        values, calculated, base="resource.prefill.system_memory_peak",
        stage="prefill", metric_name="resource.system.memory_used", device_type=None,
        device_id=None, statistic="max", unit="bytes",
    )
    for physical, alias in sorted(aliases["gpu"].items(), key=lambda item: item[1]):
        for suffix, statistic, metric_name, unit in (
            ("utilization_mean", "mean", "resource.gpu.utilization", "percent"),
            ("utilization_peak", "max", "resource.gpu.utilization", "percent"),
            ("memory_peak", "max", "resource.gpu.memory_used", "bytes"),
        ):
            _emit_resource(
                values, calculated, base=f"resource.prefill.{alias}.{suffix}",
                stage="prefill", metric_name=metric_name, device_type="gpu",
                device_id=physical, statistic=statistic, unit=unit,
            )

    _emit_resource(
        values, calculated, base="resource.decode.cpu.utilization_mean",
        stage="decode", metric_name="resource.cpu.utilization", device_type=None,
        device_id=None, statistic="mean", unit="percent",
    )
    _emit_resource(
        values, calculated, base="resource.decode.cpu.utilization_peak",
        stage="decode", metric_name="resource.cpu.utilization", device_type=None,
        device_id=None, statistic="max", unit="percent",
    )
    _emit_resource(
        values, calculated, base="resource.decode.system_memory_peak",
        stage="decode", metric_name="resource.system.memory_used", device_type=None,
        device_id=None, statistic="max", unit="bytes",
    )
    for physical, alias in sorted(aliases["npu"].items(), key=lambda item: item[1]):
        for suffix, statistic, metric_name, unit in (
            ("utilization_mean", "mean", "resource.npu.utilization", "percent"),
            ("utilization_peak", "max", "resource.npu.utilization", "percent"),
            ("memory_peak", "max", "resource.npu.memory_used", "bytes"),
        ):
            _emit_resource(
                values, calculated, base=f"resource.decode.{alias}.{suffix}",
                stage="decode", metric_name=metric_name, device_type="npu",
                device_id=physical, statistic=statistic, unit=unit,
            )

    return tuple(
        TraceAttributeSpec(key=key, value=values[key]) for key in sorted(values)
    )


def trace_attribute_validation_report(
    attributes: Sequence[TraceAttributeSpec],
    trace_validation: Mapping[str, object],
) -> dict[str, object]:
    """Return a path-free detached summary of the exact metadata SQL check."""

    queries = trace_validation.get("queries")
    if not isinstance(queries, list):
        raise TraceAttributeExportError("trace validation query list is invalid")
    matches = [
        item
        for item in queries
        if isinstance(item, dict) and item.get("name") == "trace_attributes"
    ]
    if not attributes and not matches:
        return {
            "schema_version": TRACE_ATTRIBUTE_SCHEMA_VERSION,
            "record_type": TRACE_ATTRIBUTE_RECORD_TYPE,
            "valid": True,
            "namespace": TRACE_ATTRIBUTE_NAMESPACE,
            "attribute_count": 0,
            "integer_count": 0,
            "string_count": 0,
            "duplicate_key_count": 0,
            "keys_sorted": True,
            "trace_processor_query_matched": "not_applicable_legacy_trace",
            "metadata_row_count": 0,
            "metadata_rows_sha256": hashlib.sha256(b"[]").hexdigest(),
            "mismatches": [],
        }
    if len(matches) != 1:
        raise TraceAttributeExportError("trace attribute SQL validation is missing")
    query = matches[0]
    keys = [item.key for item in attributes]
    mismatches: list[str] = []
    if keys != sorted(keys):
        mismatches.append("attribute keys are not sorted")
    if len(keys) != len(set(keys)):
        mismatches.append("attribute keys are not unique")
    schema_key = f"{TRACE_ATTRIBUTE_NAMESPACE}schema_version"
    schema_values = [item.value for item in attributes if item.key == schema_key]
    attribute_schema_version = (
        schema_values[0] if len(schema_values) == 1 else None
    )
    if attribute_schema_version not in {
        _LEGACY_TRACE_ATTRIBUTE_SCHEMA_VERSION,
        TRACE_ATTRIBUTE_SCHEMA_VERSION,
    }:
        mismatches.append("trace attribute schema version is unsupported")
    if (
        attribute_schema_version == TRACE_ATTRIBUTE_SCHEMA_VERSION
        and any(key.endswith(".availability") for key in keys)
    ):
        mismatches.append("schema 1.1.0 must not contain availability keys")
    for item in attributes:
        if isinstance(item.value, bool) or not isinstance(item.value, (int, str)):
            mismatches.append(f"unsupported value type for {item.key}")
    if query.get("matched") is not True:
        mismatches.append("Trace Processor metadata rows differ")
    integer_count = sum(type(item.value) is int for item in attributes)
    string_count = sum(type(item.value) is str for item in attributes)
    return {
        "schema_version": (
            attribute_schema_version
            if isinstance(attribute_schema_version, str)
            else TRACE_ATTRIBUTE_SCHEMA_VERSION
        ),
        "record_type": TRACE_ATTRIBUTE_RECORD_TYPE,
        "valid": not mismatches,
        "namespace": TRACE_ATTRIBUTE_NAMESPACE,
        "attribute_count": len(attributes),
        "integer_count": integer_count,
        "string_count": string_count,
        "duplicate_key_count": len(keys) - len(set(keys)),
        "keys_sorted": keys == sorted(keys),
        "trace_processor_query_matched": query.get("matched") is True,
        "metadata_row_count": query.get("row_count"),
        "metadata_rows_sha256": query.get("rows_sha256"),
        "mismatches": mismatches,
    }


__all__ = [
    "TRACE_ATTRIBUTE_NAMESPACE",
    "TRACE_ATTRIBUTE_RECORD_TYPE",
    "TRACE_ATTRIBUTE_SCHEMA_VERSION",
    "TraceAttributeExportError",
    "build_performance_trace_attributes",
    "fixed_point_half_even",
    "trace_attribute_validation_report",
]
