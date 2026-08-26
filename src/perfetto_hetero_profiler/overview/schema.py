"""Strict semantic validation and canonical JSON for Overview model v1.

The checked-in JSON Schema documents are the portable contract.  This module
implements the same contract without depending on a third-party JSON Schema
runtime, and adds cross-field rules that JSON Schema cannot express clearly
(catalog units, availability/null coupling, resource count reconciliation,
and comparison run coverage).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
from importlib import resources
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from ..schema import Availability, METRIC_CATALOG
from ..schema.catalog import (
    INTERVAL_RESOURCE_METRICS,
    KPI_SECTION_METRICS,
    RESOURCE_AGGREGATIONS,
)
from ..schema.constants import JSON_SCHEMA_DRAFT, SCHEMA_VERSION, SHA256_RE
from .model import (
    OVERVIEW_COMPARISON_RECORD_TYPE,
    OVERVIEW_MODEL_VERSION,
    OVERVIEW_REPORT_RECORD_TYPE,
    Comparability,
    ComparisonDelta,
    ComparisonKpi,
    ComparisonMetadata,
    ComparisonRun,
    ComparisonValue,
    DeltaValue,
    DisplayRule,
    KpiCalculation,
    KpiClock,
    KpiDirection,
    KpiSections,
    KpiScope,
    KpiSource,
    KpiValue,
    OverviewComparison,
    OverviewDocument,
    OverviewReport,
    ResourceSummary,
)


OVERVIEW_REPORT_SCHEMA_NAME = "overview_report.schema.json"
OVERVIEW_COMPARISON_SCHEMA_NAME = "overview_comparison.schema.json"

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_EMBEDDED_POSIX_PATH_RE = re.compile(
    r"""(?:^|[\s="'(])/(?!/)[A-Za-z0-9._-]"""
)
_OBSERVATION_LAYERS = {
    "request_facing_client",
    "hybrid_pipeline",
    "normalized_resource_metric",
    "gpu_only",
    "npu_only",
    "run",
}
_SCOPE_TYPES = {
    "run",
    "request",
    "phase",
    "host",
    "process",
    "device",
    "transfer",
}
_ALIGNMENT_STATUSES = {
    "canonical",
    "aligned",
    "partial",
    "unaligned",
    "not_applicable",
    "not_available",
    "unknown",
}
_RUN_MODES = {"gpu_only", "npu_only", "hybrid"}
_RUN_STATUSES = {
    "planned",
    "running",
    "succeeded",
    "failed",
    "partial",
    "cancelled",
}
_PROFILE_MODES = {"monitor", "detailed_profile"}
_RESOURCE_AGGREGATIONS = dict(RESOURCE_AGGREGATIONS)
_INTERVAL_RESOURCE_METRICS = INTERVAL_RESOURCE_METRICS
_RUN_FIELDS = {
    "run_id",
    "mode",
    "profile_mode",
    "status",
    "profiler_kind",
    "canonical_clock_domain_id",
}
_REQUEST_FACING_KPIS = KPI_SECTION_METRICS["request_facing_latency"]
_PIPELINE_KPIS = KPI_SECTION_METRICS["pipeline_latency"]
_THROUGHPUT_TOKEN_KPIS = KPI_SECTION_METRICS["throughput_and_tokens"]
_TRANSFER_KPIS = KPI_SECTION_METRICS["transfer"]
_COMPARISON_SECTION_CONTRACT = {
    "request_facing_latency": (
        _REQUEST_FACING_KPIS,
        {"request_facing_client", "gpu_only", "npu_only"},
    ),
    "pipeline_latency": (_PIPELINE_KPIS, {"hybrid_pipeline"}),
    "throughput_and_tokens": (
        _THROUGHPUT_TOKEN_KPIS,
        {"request_facing_client", "gpu_only", "npu_only", "run"},
    ),
    "transfer": (_TRANSFER_KPIS, {"hybrid_pipeline"}),
}
_DATA_QUALITY_FIELDS = {
    "run_status",
    "canonical_marker_count",
    "marker_validation",
    "request_join",
    "alignment",
    "resource_samples",
    "profiler",
    "source_artifact_validation",
    "perfetto_sql_validation",
    "trace_sha256",
    "per_sample_stream_preserved",
    "cleanup_complete",
    "rbln_pb_policy",
    "sample_limitations",
}
_WORKLOAD_FIELDS = {
    "request_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "concurrency",
    "request_rate_per_s",
    "warmup_requests",
    "max_output_tokens",
    "temperature",
    "retry_count",
    "prompt_sha256",
    "request_body_sha256",
    "offline",
    "max_model_len",
    "block_size",
}
_MODEL_FIELDS = {"role", "model_id", "revision", "dtype"}
_HARDWARE_FIELDS = {
    "device_type",
    "device_id",
    "vendor",
    "model",
    "memory_total_bytes",
}
_NATIVE_PROFILE_FIELDS = {
    "profiler_type",
    "source_role",
    "timestamp_ns",
    "duration_ns",
    "alignment_status",
    "alignment_method",
    "uncertainty_ns",
    "native_clock_domain",
    "native_timestamp_unit",
    "artifact_count",
    "opaque_rbln_pb",
    "native_event_alignment",
    "structure_analysis",
}
_INTERPRETATION_FIELDS = {
    "comparison_scope",
    "benchmark_claim_allowed",
    "limitations",
    "policies",
}
_INTERPRETATION_POLICY_FIELDS = {
    "request_observation_layers_separate",
    "timestamp_proximity_join",
    "unavailable_zero_fill",
    "native_clock_inference",
    "rbln_pb_parsing",
    "resource_device_aggregation",
}
_PERFETTO_FIELDS = {
    "valid",
    "trace",
    "counts",
    "query_count",
    "queries",
    "mismatches",
    "flow_endpoint_reconciliation",
    "artifact_validation",
    "toolchain",
}
_PERFETTO_COUNT_FIELDS = {
    "annotations",
    "counters",
    "dangling_flows",
    "flows",
    "import_errors",
    "native_policy",
    "process",
    "slices",
    "step_annotations",
    "tracks",
}
_DISPLAY_SCALES: dict[str, dict[str, tuple[int, int]]] = {
    "ns": {
        "ns": (1, 1),
        "us": (1, 1_000),
        "ms": (1, 1_000_000),
        "s": (1, 1_000_000_000),
    },
    "bytes": {
        "bytes": (1, 1),
        "KiB": (1, 1_024),
        "MiB": (1, 1_048_576),
        "GiB": (1, 1_073_741_824),
    },
    "percent": {"percent": (1, 1)},
    "W": {"W": (1, 1)},
    "ratio": {"ratio": (1, 1), "percent": (100, 1)},
    "requests": {"requests": (1, 1)},
    "tokens": {"tokens": (1, 1)},
    "requests/s": {"requests/s": (1, 1)},
    "tokens/s": {"tokens/s": (1, 1)},
    "bytes/s": {
        "bytes/s": (1, 1),
        "KiB/s": (1, 1_024),
        "MiB/s": (1, 1_048_576),
        "GiB/s": (1, 1_073_741_824),
    },
}

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


def _sorted_models(
    values: object,
    path: str,
    *,
    key,
    allow_empty: bool = True,
) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        _fail(path, "must be an immutable tuple")
    if not allow_empty and not values:
        _fail(path, "must not be empty")
    keys = [key(value) for value in values]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted with unique keys")
    return values


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


def validate_kpi_source(source: KpiSource, path: str = "kpi.sources[0]") -> None:
    _require_type(source, KpiSource, path)
    _nonempty(source.source_kind, f"{path}.source_kind")
    record_ids = _sorted_unique_strings(source.record_ids, f"{path}.record_ids")
    metric_names = _sorted_unique_strings(
        source.metric_names,
        f"{path}.metric_names",
    )
    if source.root_id is not None:
        root_id = _nonempty(source.root_id, f"{path}.root_id")
        if _ROOT_ID_RE.fullmatch(root_id) is None:
            _fail(f"{path}.root_id", "is not a safe logical root id")
    if source.relative_path is not None:
        _safe_relative_path(source.relative_path, f"{path}.relative_path")
    _json_object(source.details, f"{path}.details")
    if not record_ids and not metric_names and source.relative_path is None and not source.details:
        _fail(path, "must include record, metric, artifact, or detail provenance")


def validate_kpi_scope(scope: KpiScope, path: str = "kpi.scope") -> None:
    _require_type(scope, KpiScope, path)
    _nonempty(scope.run_id, f"{path}.run_id")
    if scope.scope_type not in _SCOPE_TYPES:
        _fail(f"{path}.scope_type", f"must be one of {sorted(_SCOPE_TYPES)}")
    if scope.observation_layer not in _OBSERVATION_LAYERS:
        _fail(
            f"{path}.observation_layer",
            f"must be one of {sorted(_OBSERVATION_LAYERS)}",
        )
    for name in (
        "request_id",
        "host_id",
        "device_type",
        "device_id",
        "phase",
        "window",
    ):
        value = getattr(scope, name)
        if value is not None:
            _nonempty(value, f"{path}.{name}")
    if scope.scope_type in {"request", "phase", "transfer"} and scope.request_id is None:
        _fail(f"{path}.request_id", f"is required for {scope.scope_type} scope")
    if scope.scope_type in {"host", "process", "device"} and scope.host_id is None:
        _fail(f"{path}.host_id", f"is required for {scope.scope_type} scope")
    if scope.scope_type == "device":
        if scope.device_type is None or scope.device_id is None:
            _fail(path, "device scope requires device_type and device_id")
    elif scope.device_type is not None or scope.device_id is not None:
        _fail(path, "device identity is only valid for device scope")
    if scope.scope_type == "phase" and scope.phase is None:
        _fail(f"{path}.phase", "is required for phase scope")


def validate_kpi_clock(clock: KpiClock, path: str = "kpi.clock") -> None:
    _require_type(clock, KpiClock, path)
    domains = _sorted_unique_strings(clock.domain_ids, f"{path}.domain_ids")
    if clock.alignment_status not in _ALIGNMENT_STATUSES:
        _fail(
            f"{path}.alignment_status",
            f"must be one of {sorted(_ALIGNMENT_STATUSES)}",
        )
    if clock.alignment_method is not None:
        _nonempty(clock.alignment_method, f"{path}.alignment_method")
    _integer(clock.offset_ns, f"{path}.offset_ns", nullable=True)
    _integer(
        clock.uncertainty_ns,
        f"{path}.uncertainty_ns",
        minimum=0,
        nullable=True,
    )
    if clock.alignment_status in {"canonical", "aligned", "partial"} and not domains:
        _fail(f"{path}.domain_ids", "must identify the observed clock")
    if clock.alignment_status == "aligned" and clock.alignment_method is None:
        _fail(f"{path}.alignment_method", "is required for aligned data")
    if clock.alignment_status == "aligned" and clock.uncertainty_ns is None:
        _fail(f"{path}.uncertainty_ns", "is required for aligned data")
    if clock.alignment_status == "not_applicable" and (
        clock.offset_ns is not None or clock.uncertainty_ns is not None
    ):
        _fail(path, "not_applicable clock must not invent offset/uncertainty")


def validate_display_rule(
    display: DisplayRule,
    canonical_unit: str,
    path: str = "kpi.display",
) -> None:
    _require_type(display, DisplayRule, path)
    _nonempty(display.unit, f"{path}.unit")
    numerator = _integer(
        display.scale_numerator,
        f"{path}.scale_numerator",
        minimum=1,
    )
    denominator = _integer(
        display.scale_denominator,
        f"{path}.scale_denominator",
        minimum=1,
    )
    assert numerator is not None and denominator is not None
    if math.gcd(numerator, denominator) != 1:
        _fail(path, "display scale must be a reduced rational")
    _integer(
        display.decimal_places,
        f"{path}.decimal_places",
        minimum=0,
    )
    if display.decimal_places > 12:
        _fail(f"{path}.decimal_places", "must be <= 12")
    if display.rounding != "half_even":
        _fail(f"{path}.rounding", "must be half_even")
    supported = _DISPLAY_SCALES.get(canonical_unit)
    expected = None if supported is None else supported.get(display.unit)
    if expected != (numerator, denominator):
        _fail(
            path,
            f"unsupported {canonical_unit!r} to {display.unit!r} display conversion",
        )


def _validate_available_scalar(
    availability: Availability,
    value: object,
    reason: object,
    path: str,
) -> None:
    if not isinstance(availability, Availability):
        _fail(f"{path}.availability", "must be an Availability enum")
    if availability is Availability.AVAILABLE:
        _number(value, f"{path}.value")
        if reason is not None:
            _fail(f"{path}.unavailable_reason", "must be null when available")
    else:
        if value is not None:
            _fail(f"{path}.value", "must be null when unavailable")
        _nonempty(reason, f"{path}.unavailable_reason")


def _catalog_definition(name: str, path: str):
    definition = METRIC_CATALOG.get(name)
    if definition is not None:
        return definition
    for suffix in _RESOURCE_AGGREGATIONS:
        marker = f".{suffix}"
        if name.endswith(marker):
            base = name[: -len(marker)]
            candidate = METRIC_CATALOG.get(base)
            if candidate is not None and base.startswith("resource."):
                return candidate
    _fail(path, "must be an official METRIC_CATALOG KPI or resource aggregate")
    raise AssertionError


def validate_kpi(kpi: KpiValue, path: str = "kpi") -> None:
    _require_type(kpi, KpiValue, path)
    name = _nonempty(kpi.name, f"{path}.name")
    definition = _catalog_definition(name, f"{path}.name")
    if kpi.canonical_unit != definition.unit:
        _fail(
            f"{path}.canonical_unit",
            f"must be catalog unit {definition.unit!r}",
        )
    _validate_available_scalar(
        kpi.availability,
        kpi.value,
        kpi.unavailable_reason,
        path,
    )
    if kpi.availability is Availability.AVAILABLE:
        assert kpi.value is not None
        if definition.value_type == "integer" and (
            not isinstance(kpi.value, int) or isinstance(kpi.value, bool)
        ):
            _fail(f"{path}.value", "must be an integer for this catalog KPI")
        if definition.minimum is not None and kpi.value < definition.minimum:
            _fail(f"{path}.value", f"must be >= {definition.minimum}")
        if definition.maximum is not None and kpi.value > definition.maximum:
            _fail(f"{path}.value", f"must be <= {definition.maximum}")
    _nonempty(kpi.aggregation_method, f"{path}.aggregation_method")
    sample_count = _integer(
        kpi.sample_count,
        f"{path}.sample_count",
        minimum=0,
    )
    if kpi.availability is Availability.AVAILABLE and sample_count == 0:
        _fail(f"{path}.sample_count", "available KPI requires at least one sample")
    if not isinstance(kpi.sources, tuple):
        _fail(f"{path}.sources", "must be an immutable tuple")
    if kpi.availability is Availability.AVAILABLE and not kpi.sources:
        _fail(f"{path}.sources", "available KPI requires provenance")
    for index, source in enumerate(kpi.sources):
        validate_kpi_source(source, f"{path}.sources[{index}]")
    validate_kpi_scope(kpi.scope, f"{path}.scope")
    scope_allowed = kpi.scope.scope_type in {
        scope.value for scope in definition.allowed_scopes
    }
    aggregate_run_scope = (
        kpi.scope.scope_type == "run"
        and kpi.aggregation_method
        in {
            "arithmetic_mean_across_measured_requests_v1",
            "not_available_across_measured_requests_v1",
            "ratio_of_measured_request_means_v1",
        }
    )
    if not scope_allowed and not aggregate_run_scope:
        _fail(
            f"{path}.scope.scope_type",
            "is not allowed by METRIC_CATALOG for this KPI",
        )
    _require_type(kpi.calculation, KpiCalculation, f"{path}.calculation")
    _nonempty(kpi.calculation.method_id, f"{path}.calculation.method_id")
    _nonempty(kpi.calculation.formula, f"{path}.calculation.formula")
    validate_kpi_clock(kpi.clock, f"{path}.clock")
    _sorted_unique_strings(
        kpi.quality_warnings,
        f"{path}.quality_warnings",
    )
    validate_display_rule(
        kpi.display,
        kpi.canonical_unit,
        f"{path}.display",
    )


def _kpi_key(kpi: KpiValue) -> tuple[str, ...]:
    scope = kpi.scope
    return (
        kpi.name,
        scope.observation_layer,
        scope.scope_type,
        scope.request_id or "",
        scope.host_id or "",
        scope.device_type or "",
        scope.device_id or "",
        scope.phase or "",
        scope.window or "",
        kpi.aggregation_method,
    )


def validate_resource_summary(
    summary: ResourceSummary,
    path: str = "resource",
) -> None:
    _require_type(summary, ResourceSummary, path)
    definition = METRIC_CATALOG.get(summary.metric_name)
    if definition is None or not summary.metric_name.startswith("resource."):
        _fail(f"{path}.metric_name", "must be an official resource KPI")
    if summary.canonical_unit != definition.unit:
        _fail(
            f"{path}.canonical_unit",
            f"must be catalog unit {definition.unit!r}",
        )
    validate_kpi_scope(summary.scope, f"{path}.scope")
    if summary.scope.scope_type not in {"host", "process", "device"}:
        _fail(f"{path}.scope.scope_type", "must be host, process, or device")
    validate_kpi_clock(summary.clock, f"{path}.clock")
    total = _integer(
        summary.total_sample_count,
        f"{path}.total_sample_count",
        minimum=0,
    )
    available = _integer(
        summary.available_sample_count,
        f"{path}.available_sample_count",
        minimum=0,
    )
    unavailable = _integer(
        summary.unavailable_sample_count,
        f"{path}.unavailable_sample_count",
        minimum=0,
    )
    assert total is not None and available is not None and unavailable is not None
    if total != available + unavailable:
        _fail(path, "total_sample_count must equal available + unavailable")
    ratio = _number(
        summary.availability_ratio,
        f"{path}.availability_ratio",
        minimum=0,
        maximum=1,
    )
    expected_ratio = 0.0 if total == 0 else available / total
    if ratio != expected_ratio:
        _fail(
            f"{path}.availability_ratio",
            "must exactly equal available_sample_count / total_sample_count",
        )
    first = _integer(
        summary.first_timestamp_ns,
        f"{path}.first_timestamp_ns",
        minimum=0,
        nullable=True,
    )
    last = _integer(
        summary.last_timestamp_ns,
        f"{path}.last_timestamp_ns",
        minimum=0,
        nullable=True,
    )
    coverage = _integer(
        summary.coverage_ns,
        f"{path}.coverage_ns",
        minimum=0,
        nullable=True,
    )
    if total == 0:
        if any(value is not None for value in (first, last, coverage)):
            _fail(path, "empty resource stream must have null timestamps/coverage")
    else:
        if first is None or last is None or coverage is None:
            _fail(path, "non-empty resource stream requires timestamps/coverage")
        if last < first or coverage != last - first:
            _fail(
                f"{path}.coverage_ns",
                "must exactly equal last_timestamp_ns - first_timestamp_ns",
            )
    if not isinstance(summary.aggregates, tuple):
        _fail(f"{path}.aggregates", "must be an immutable tuple")
    expected_aggregations = dict(_RESOURCE_AGGREGATIONS)
    if summary.scope.window is not None and summary.metric_name in _INTERVAL_RESOURCE_METRICS:
        expected_aggregations["mean"] = "trailing_interval_overlap_weighted_mean_v1"
    aggregate_contract: dict[str, str] = {}
    for index, aggregate in enumerate(summary.aggregates):
        validate_kpi(aggregate, f"{path}.aggregates[{index}]")
        prefix = f"{summary.metric_name}."
        if not aggregate.name.startswith(prefix):
            _fail(
                f"{path}.aggregates[{index}].name",
                "must use the resource metric name plus a statistic suffix",
            )
        suffix = aggregate.name[len(prefix) :]
        expected_method = expected_aggregations.get(suffix)
        if expected_method is None or aggregate.aggregation_method != expected_method:
            _fail(
                f"{path}.aggregates[{index}].aggregation_method",
                "does not match the resource statistic contract",
            )
        if suffix in aggregate_contract:
            _fail(f"{path}.aggregates", "contains a duplicate statistic")
        aggregate_contract[suffix] = aggregate.aggregation_method
        if (
            aggregate.canonical_unit != summary.canonical_unit
            or aggregate.scope != summary.scope
            or aggregate.clock != summary.clock
        ):
            _fail(
                f"{path}.aggregates[{index}]",
                "must retain the resource metric, unit, scope, and clock",
            )
    if aggregate_contract != expected_aggregations:
        _fail(
            f"{path}.aggregates",
            f"must contain exactly {sorted(_RESOURCE_AGGREGATIONS)}",
        )
    _sorted_unique_strings(
        summary.quality_warnings,
        f"{path}.quality_warnings",
    )


def _resource_key(summary: ResourceSummary) -> tuple[str, ...]:
    scope = summary.scope
    return (
        summary.metric_name,
        scope.observation_layer,
        scope.scope_type,
        scope.host_id or "",
        scope.device_type or "",
        scope.device_id or "",
        scope.window or "",
    )


def _validate_identity_objects(
    values: object,
    path: str,
    *,
    allow_empty: bool,
) -> None:
    if not isinstance(values, tuple):
        _fail(path, "must be an immutable tuple")
    if not allow_empty and not values:
        _fail(path, "must not be empty")
    for index, value in enumerate(values):
        _json_object(value, f"{path}[{index}]", nonempty=True)
    keys = [_canonical_sort_key(value) for value in values]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted without duplicates")


def _validate_kpi_section(
    values: tuple[KpiValue, ...],
    *,
    path: str,
    run_id: str,
    allowed_names: set[str],
    observation_layers: set[str],
) -> None:
    for index, kpi in enumerate(values):
        validate_kpi(kpi, f"{path}[{index}]")
        if kpi.name not in allowed_names:
            _fail(
                f"{path}[{index}].name",
                "does not belong to this Overview section",
            )
        if kpi.scope.run_id != run_id:
            _fail(
                f"{path}[{index}].scope.run_id",
                "must match run.run_id",
            )
        if kpi.scope.observation_layer not in observation_layers:
            _fail(
                f"{path}[{index}].scope.observation_layer",
                "does not match this Overview section",
            )
    keys = [_kpi_key(value) for value in values]
    if len(keys) != len(set(keys)):
        _fail(path, "must not contain duplicate KPI identities")


def validate_kpi_sections(
    sections: KpiSections,
    *,
    run_id: str,
    path: str = "overview.kpis",
) -> None:
    _require_type(sections, KpiSections, path)
    _validate_kpi_section(
        sections.request_facing_latency,
        path=f"{path}.request_facing_latency",
        run_id=run_id,
        allowed_names=_REQUEST_FACING_KPIS,
        observation_layers={"request_facing_client", "gpu_only", "npu_only"},
    )
    _validate_kpi_section(
        sections.pipeline_latency,
        path=f"{path}.pipeline_latency",
        run_id=run_id,
        allowed_names=_PIPELINE_KPIS,
        observation_layers={"hybrid_pipeline"},
    )
    _validate_kpi_section(
        sections.throughput_and_tokens,
        path=f"{path}.throughput_and_tokens",
        run_id=run_id,
        allowed_names=_THROUGHPUT_TOKEN_KPIS,
        observation_layers={"request_facing_client", "gpu_only", "npu_only", "run"},
    )
    _validate_kpi_section(
        sections.transfer,
        path=f"{path}.transfer",
        run_id=run_id,
        allowed_names=_TRANSFER_KPIS,
        observation_layers={"hybrid_pipeline"},
    )


def _exact_mapping(
    value: object,
    *,
    fields: set[str],
    path: str,
) -> dict[str, Any]:
    _json_object(value, path, nonempty=True)
    copied = dict(value)
    if set(copied) != fields:
        _fail(path, f"must contain exactly {sorted(fields)}")
    return copied


def _validate_data_quality(value: object, path: str) -> None:
    data = _exact_mapping(
        value,
        fields=_DATA_QUALITY_FIELDS,
        path=path,
    )
    if data["run_status"] not in _RUN_STATUSES:
        _fail(f"{path}.run_status", "is not a valid run status")
    _integer(
        data["canonical_marker_count"],
        f"{path}.canonical_marker_count",
        minimum=0,
    )

    marker = _exact_mapping(
        data["marker_validation"],
        fields={
            "status",
            "missing_count",
            "duplicate_count",
            "pairing_violation_count",
            "order_violation_count",
        },
        path=f"{path}.marker_validation",
    )
    _nonempty(marker["status"], f"{path}.marker_validation.status")
    for name in (
        "missing_count",
        "duplicate_count",
        "pairing_violation_count",
        "order_violation_count",
    ):
        _integer(
            marker[name],
            f"{path}.marker_validation.{name}",
            minimum=0,
        )

    joined = _exact_mapping(
        data["request_join"],
        fields={"joined_count", "unjoined_count", "method"},
        path=f"{path}.request_join",
    )
    _integer(joined["joined_count"], f"{path}.request_join.joined_count", minimum=0)
    _integer(
        joined["unjoined_count"],
        f"{path}.request_join.unjoined_count",
        minimum=0,
    )
    _nonempty(joined["method"], f"{path}.request_join.method")

    alignment = _exact_mapping(
        data["alignment"],
        fields={"status", "method", "offset_ns", "uncertainty_ns"},
        path=f"{path}.alignment",
    )
    if alignment["status"] not in _ALIGNMENT_STATUSES:
        _fail(f"{path}.alignment.status", "is invalid")
    if alignment["method"] is not None:
        _nonempty(alignment["method"], f"{path}.alignment.method")
    _integer(
        alignment["offset_ns"],
        f"{path}.alignment.offset_ns",
        nullable=True,
    )
    _integer(
        alignment["uncertainty_ns"],
        f"{path}.alignment.uncertainty_ns",
        minimum=0,
        nullable=True,
    )

    samples = _exact_mapping(
        data["resource_samples"],
        fields={"total", "available", "unavailable"},
        path=f"{path}.resource_samples",
    )
    total = _integer(samples["total"], f"{path}.resource_samples.total", minimum=0)
    available = _integer(
        samples["available"],
        f"{path}.resource_samples.available",
        minimum=0,
    )
    unavailable = _integer(
        samples["unavailable"],
        f"{path}.resource_samples.unavailable",
        minimum=0,
    )
    if total != available + unavailable:
        _fail(
            f"{path}.resource_samples",
            "total must equal available + unavailable",
        )

    profiler = _exact_mapping(
        data["profiler"],
        fields={"kind", "native_alignment_status"},
        path=f"{path}.profiler",
    )
    _nonempty(profiler["kind"], f"{path}.profiler.kind")
    if profiler["native_alignment_status"] not in _ALIGNMENT_STATUSES:
        _fail(f"{path}.profiler.native_alignment_status", "is invalid")

    source = _exact_mapping(
        data["source_artifact_validation"],
        fields={
            "valid",
            "closeout_artifact_count",
            "closeout_manifest_sha256",
            "roots",
        },
        path=f"{path}.source_artifact_validation",
    )
    if not isinstance(source["valid"], bool):
        _fail(f"{path}.source_artifact_validation.valid", "must be boolean")
    _integer(
        source["closeout_artifact_count"],
        f"{path}.source_artifact_validation.closeout_artifact_count",
        minimum=0,
    )
    if (
        not isinstance(source["closeout_manifest_sha256"], str)
        or SHA256_RE.fullmatch(source["closeout_manifest_sha256"]) is None
    ):
        _fail(
            f"{path}.source_artifact_validation.closeout_manifest_sha256",
            "must be lowercase SHA-256",
        )
    roots = source["roots"]
    if not isinstance(roots, list):
        _fail(f"{path}.source_artifact_validation.roots", "must be an array")
    root_keys: list[str] = []
    for index, root in enumerate(roots):
        root_data = _exact_mapping(
            root,
            fields={"root_id", "file_count", "fingerprint_sha256"},
            path=f"{path}.source_artifact_validation.roots[{index}]",
        )
        root_id = _nonempty(
            root_data["root_id"],
            f"{path}.source_artifact_validation.roots[{index}].root_id",
        )
        if _ROOT_ID_RE.fullmatch(root_id) is None:
            _fail(
                f"{path}.source_artifact_validation.roots[{index}].root_id",
                "is not a safe root id",
            )
        _integer(
            root_data["file_count"],
            f"{path}.source_artifact_validation.roots[{index}].file_count",
            minimum=0,
        )
        if (
            not isinstance(root_data["fingerprint_sha256"], str)
            or SHA256_RE.fullmatch(root_data["fingerprint_sha256"]) is None
        ):
            _fail(
                f"{path}.source_artifact_validation.roots[{index}].fingerprint_sha256",
                "must be lowercase SHA-256",
            )
        root_keys.append(root_id)
    if root_keys != sorted(set(root_keys)):
        _fail(
            f"{path}.source_artifact_validation.roots",
            "must be sorted by unique root_id",
        )

    perfetto = _exact_mapping(
        data["perfetto_sql_validation"],
        fields={"valid", "query_count", "mismatches"},
        path=f"{path}.perfetto_sql_validation",
    )
    if not isinstance(perfetto["valid"], bool):
        _fail(f"{path}.perfetto_sql_validation.valid", "must be boolean")
    _integer(
        perfetto["query_count"],
        f"{path}.perfetto_sql_validation.query_count",
        minimum=0,
    )
    mismatches = perfetto["mismatches"]
    if not isinstance(mismatches, list):
        _fail(f"{path}.perfetto_sql_validation.mismatches", "must be an array")
    mismatch_values = tuple(
        _nonempty(item, f"{path}.perfetto_sql_validation.mismatches[{index}]")
        for index, item in enumerate(mismatches)
    )
    if mismatch_values != tuple(sorted(set(mismatch_values))):
        _fail(
            f"{path}.perfetto_sql_validation.mismatches",
            "must be sorted without duplicates",
        )

    if (
        not isinstance(data["trace_sha256"], str)
        or SHA256_RE.fullmatch(data["trace_sha256"]) is None
    ):
        _fail(f"{path}.trace_sha256", "must be lowercase SHA-256")
    for name in ("per_sample_stream_preserved", "cleanup_complete"):
        if not isinstance(data[name], bool):
            _fail(f"{path}.{name}", "must be boolean")
    rbln_policy = _exact_mapping(
        data["rbln_pb_policy"],
        fields={"classification", "structure_analysis", "raw_bytes_embedded"},
        path=f"{path}.rbln_pb_policy",
    )
    _nonempty(
        rbln_policy["classification"],
        f"{path}.rbln_pb_policy.classification",
    )
    _nonempty(
        rbln_policy["structure_analysis"],
        f"{path}.rbln_pb_policy.structure_analysis",
    )
    if not isinstance(rbln_policy["raw_bytes_embedded"], bool):
        _fail(f"{path}.rbln_pb_policy.raw_bytes_embedded", "must be boolean")
    if rbln_policy["raw_bytes_embedded"]:
        _fail(
            f"{path}.rbln_pb_policy.raw_bytes_embedded",
            "must be false; RBLN PB is published as a separate native-relative "
            "Perfetto trace rather than embedded in the Overview report",
        )
    limitations = data["sample_limitations"]
    if not isinstance(limitations, list):
        _fail(f"{path}.sample_limitations", "must be an array")
    limitation_values = tuple(
        _nonempty(item, f"{path}.sample_limitations[{index}]")
        for index, item in enumerate(limitations)
    )
    if limitation_values != tuple(sorted(set(limitation_values))):
        _fail(f"{path}.sample_limitations", "must be sorted without duplicates")


def _validate_workload(value: object, path: str) -> None:
    workload = _exact_mapping(value, fields=_WORKLOAD_FIELDS, path=path)
    integer_fields = (
        "request_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "concurrency",
        "warmup_requests",
        "max_output_tokens",
        "retry_count",
        "max_model_len",
        "block_size",
    )
    for name in integer_fields:
        _integer(
            workload[name],
            f"{path}.{name}",
            minimum=0,
            nullable=True,
        )
    _number(
        workload["request_rate_per_s"],
        f"{path}.request_rate_per_s",
        minimum=0,
        nullable=True,
    )
    _number(
        workload["temperature"],
        f"{path}.temperature",
        minimum=0,
        nullable=True,
    )
    for name in ("prompt_sha256", "request_body_sha256"):
        digest = workload[name]
        if digest is not None and (
            not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None
        ):
            _fail(f"{path}.{name}", "must be null or lowercase SHA-256")
    if workload["offline"] is not None and not isinstance(workload["offline"], bool):
        _fail(f"{path}.offline", "must be boolean or null")
    input_tokens = workload["input_tokens"]
    output_tokens = workload["output_tokens"]
    total_tokens = workload["total_tokens"]
    if (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        _fail(
            f"{path}.total_tokens",
            "must equal input_tokens + output_tokens when all are available",
        )


def _validate_models(values: object, path: str) -> None:
    if not isinstance(values, tuple) or not values:
        _fail(path, "must be a non-empty immutable tuple")
    keys: list[bytes] = []
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        model = _exact_mapping(value, fields=_MODEL_FIELDS, path=item_path)
        _nonempty(model["role"], f"{item_path}.role")
        _nonempty(model["model_id"], f"{item_path}.model_id")
        for name in ("revision", "dtype"):
            if model[name] is not None:
                _nonempty(model[name], f"{item_path}.{name}")
        keys.append(_canonical_sort_key(model))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted without duplicates")


def _validate_hardware(values: object, path: str) -> None:
    if not isinstance(values, tuple):
        _fail(path, "must be an immutable tuple")
    keys: list[bytes] = []
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        device = _exact_mapping(
            value,
            fields=_HARDWARE_FIELDS,
            path=item_path,
        )
        for name in ("device_type", "device_id", "vendor", "model"):
            _nonempty(device[name], f"{item_path}.{name}")
        _integer(
            device["memory_total_bytes"],
            f"{item_path}.memory_total_bytes",
            minimum=0,
            nullable=True,
        )
        keys.append(_canonical_sort_key(device))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted without duplicates")


def _validate_native_profiles(values: object, path: str) -> None:
    if not isinstance(values, tuple):
        _fail(path, "must be an immutable tuple")
    keys: list[bytes] = []
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        profile = _exact_mapping(
            value,
            fields=_NATIVE_PROFILE_FIELDS,
            path=item_path,
        )
        for name in (
            "profiler_type",
            "source_role",
            "alignment_method",
            "native_clock_domain",
            "native_timestamp_unit",
            "native_event_alignment",
            "structure_analysis",
        ):
            _nonempty(profile[name], f"{item_path}.{name}")
        for name in ("timestamp_ns", "duration_ns", "uncertainty_ns", "artifact_count"):
            _integer(profile[name], f"{item_path}.{name}", minimum=0)
        if profile["alignment_status"] not in _ALIGNMENT_STATUSES:
            _fail(f"{item_path}.alignment_status", "is invalid")
        if not isinstance(profile["opaque_rbln_pb"], bool):
            _fail(f"{item_path}.opaque_rbln_pb", "must be boolean")
        keys.append(_canonical_sort_key(profile))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail(path, "must be deterministically sorted without duplicates")


def _validate_interpretation(value: object, path: str) -> None:
    interpretation = _exact_mapping(
        value,
        fields=_INTERPRETATION_FIELDS,
        path=path,
    )
    _nonempty(interpretation["comparison_scope"], f"{path}.comparison_scope")
    if not isinstance(interpretation["benchmark_claim_allowed"], bool):
        _fail(f"{path}.benchmark_claim_allowed", "must be boolean")
    limitations = interpretation["limitations"]
    if not isinstance(limitations, list) or not limitations:
        _fail(f"{path}.limitations", "must be a non-empty array")
    limitation_values = tuple(
        _nonempty(item, f"{path}.limitations[{index}]")
        for index, item in enumerate(limitations)
    )
    if limitation_values != tuple(sorted(set(limitation_values))):
        _fail(f"{path}.limitations", "must be sorted without duplicates")
    policies = _exact_mapping(
        interpretation["policies"],
        fields=_INTERPRETATION_POLICY_FIELDS,
        path=f"{path}.policies",
    )
    for name in _INTERPRETATION_POLICY_FIELDS:
        if not isinstance(policies[name], bool):
            _fail(f"{path}.policies.{name}", "must be boolean")
    required_policy_values = {
        "request_observation_layers_separate": True,
        "timestamp_proximity_join": False,
        "unavailable_zero_fill": False,
        "native_clock_inference": False,
        "rbln_pb_parsing": False,
        "resource_device_aggregation": False,
    }
    for name, expected in required_policy_values.items():
        if policies[name] is not expected:
            _fail(
                f"{path}.policies.{name}",
                f"must be {expected} for Overview v1",
            )


def _sorted_json_array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    keys = [_canonical_sort_key(item) for item in value]
    if keys != sorted(keys):
        _fail(path, "must be deterministically sorted")
    for index, item in enumerate(value):
        _json_value(item, f"{path}[{index}]")
    return value


def _validate_perfetto(value: object, path: str) -> None:
    perfetto = _exact_mapping(value, fields=_PERFETTO_FIELDS, path=path)
    if not isinstance(perfetto["valid"], bool):
        _fail(f"{path}.valid", "must be boolean")
    trace = _exact_mapping(
        perfetto["trace"],
        fields={"size_bytes", "sha256"},
        path=f"{path}.trace",
    )
    _integer(trace["size_bytes"], f"{path}.trace.size_bytes", minimum=0)
    if (
        not isinstance(trace["sha256"], str)
        or SHA256_RE.fullmatch(trace["sha256"]) is None
    ):
        _fail(f"{path}.trace.sha256", "must be lowercase SHA-256")
    counts = _exact_mapping(
        perfetto["counts"],
        fields=_PERFETTO_COUNT_FIELDS,
        path=f"{path}.counts",
    )
    for name in _PERFETTO_COUNT_FIELDS:
        _integer(counts[name], f"{path}.counts.{name}", minimum=0)
    query_count = _integer(
        perfetto["query_count"],
        f"{path}.query_count",
        minimum=0,
    )
    queries = perfetto["queries"]
    if not isinstance(queries, list):
        _fail(f"{path}.queries", "must be an array")
    query_names: list[str] = []
    for index, value in enumerate(queries):
        item_path = f"{path}.queries[{index}]"
        query = _exact_mapping(
            value,
            fields={
                "name",
                "row_count",
                "rows_sha256",
                "expected_row_count",
                "expected_rows_sha256",
                "matched",
            },
            path=item_path,
        )
        query_names.append(_nonempty(query["name"], f"{item_path}.name"))
        for name in ("row_count", "expected_row_count"):
            _integer(query[name], f"{item_path}.{name}", minimum=0)
        for name in ("rows_sha256", "expected_rows_sha256"):
            if (
                not isinstance(query[name], str)
                or SHA256_RE.fullmatch(query[name]) is None
            ):
                _fail(f"{item_path}.{name}", "must be lowercase SHA-256")
        if not isinstance(query["matched"], bool):
            _fail(f"{item_path}.matched", "must be boolean")
    if query_count != len(queries):
        _fail(f"{path}.query_count", "must equal len(queries)")
    if len(query_names) != len(set(query_names)):
        _fail(f"{path}.queries", "must contain unique query names")

    mismatches = perfetto["mismatches"]
    if not isinstance(mismatches, list):
        _fail(f"{path}.mismatches", "must be an array")
    mismatch_values = tuple(
        _nonempty(item, f"{path}.mismatches[{index}]")
        for index, item in enumerate(mismatches)
    )
    if mismatch_values != tuple(sorted(set(mismatch_values))):
        _fail(f"{path}.mismatches", "must be sorted without duplicates")

    flow = _exact_mapping(
        perfetto["flow_endpoint_reconciliation"],
        fields={
            "declared_flow_ids",
            "source_endpoint_ids",
            "destination_endpoint_ids",
            "matched",
        },
        path=f"{path}.flow_endpoint_reconciliation",
    )
    for name in (
        "declared_flow_ids",
        "source_endpoint_ids",
        "destination_endpoint_ids",
    ):
        values = flow[name]
        if not isinstance(values, list):
            _fail(f"{path}.flow_endpoint_reconciliation.{name}", "must be an array")
        normalized = tuple(
            _integer(
                item,
                f"{path}.flow_endpoint_reconciliation.{name}[{index}]",
                minimum=1,
            )
            for index, item in enumerate(values)
        )
        if normalized != tuple(sorted(set(normalized))):
            _fail(
                f"{path}.flow_endpoint_reconciliation.{name}",
                "must be sorted without duplicates",
            )
    if not isinstance(flow["matched"], bool):
        _fail(f"{path}.flow_endpoint_reconciliation.matched", "must be boolean")

    artifact = _exact_mapping(
        perfetto["artifact_validation"],
        fields={"valid", "checked", "mismatches", "manifest_sha256"},
        path=f"{path}.artifact_validation",
    )
    if not isinstance(artifact["valid"], bool):
        _fail(f"{path}.artifact_validation.valid", "must be boolean")
    _integer(
        artifact["checked"],
        f"{path}.artifact_validation.checked",
        minimum=0,
    )
    _sorted_json_array(
        artifact["mismatches"],
        f"{path}.artifact_validation.mismatches",
    )
    if (
        not isinstance(artifact["manifest_sha256"], str)
        or SHA256_RE.fullmatch(artifact["manifest_sha256"]) is None
    ):
        _fail(
            f"{path}.artifact_validation.manifest_sha256",
            "must be lowercase SHA-256",
        )

    toolchain = _exact_mapping(
        perfetto["toolchain"],
        fields={
            "filename",
            "version",
            "sha256",
            "perfetto_package_version",
            "protobuf_package_version",
            "trace_processor_rpc_api_version",
        },
        path=f"{path}.toolchain",
    )
    for name in (
        "filename",
        "version",
        "perfetto_package_version",
        "protobuf_package_version",
    ):
        _nonempty(toolchain[name], f"{path}.toolchain.{name}")
    if (
        not isinstance(toolchain["sha256"], str)
        or SHA256_RE.fullmatch(toolchain["sha256"]) is None
    ):
        _fail(f"{path}.toolchain.sha256", "must be lowercase SHA-256")
    _integer(
        toolchain["trace_processor_rpc_api_version"],
        f"{path}.toolchain.trace_processor_rpc_api_version",
        minimum=0,
    )
    if perfetto["valid"]:
        if mismatch_values or not all(query["matched"] for query in queries):
            _fail(path, "valid Perfetto reconciliation cannot contain mismatches")
        if not flow["matched"] or not artifact["valid"]:
            _fail(path, "valid Perfetto reconciliation requires flow/artifact validity")
        if counts["dangling_flows"] != 0 or counts["import_errors"] != 0:
            _fail(path, "valid Perfetto reconciliation requires zero dangling/import errors")


def validate_overview_report(report: OverviewReport) -> None:
    _require_type(report, OverviewReport, "overview")
    if report.schema_version != SCHEMA_VERSION:
        _fail("overview.schema_version", f"must be {SCHEMA_VERSION}")
    if report.record_type != OVERVIEW_REPORT_RECORD_TYPE:
        _fail(
            "overview.record_type",
            f"must be {OVERVIEW_REPORT_RECORD_TYPE}",
        )
    _json_object(report.run, "overview.run", nonempty=True)
    if set(report.run) != _RUN_FIELDS:
        _fail(
            "overview.run",
            f"must contain exactly {sorted(_RUN_FIELDS)}",
        )
    run_id = _nonempty(report.run["run_id"], "overview.run.run_id")
    if report.run["mode"] not in _RUN_MODES:
        _fail("overview.run.mode", f"must be one of {sorted(_RUN_MODES)}")
    if report.run["status"] not in _RUN_STATUSES:
        _fail("overview.run.status", f"must be one of {sorted(_RUN_STATUSES)}")
    if report.run["profile_mode"] not in _PROFILE_MODES:
        _fail(
            "overview.run.profile_mode",
            f"must be one of {sorted(_PROFILE_MODES)}",
        )
    _nonempty(report.run["profiler_kind"], "overview.run.profiler_kind")
    _nonempty(
        report.run["canonical_clock_domain_id"],
        "overview.run.canonical_clock_domain_id",
    )
    _validate_workload(report.workload, "overview.workload")
    _validate_models(report.models, "overview.models")
    _validate_hardware(report.hardware, "overview.hardware")
    validate_kpi_sections(report.kpis, run_id=run_id)
    for index, summary in enumerate(report.resources):
        validate_resource_summary(
            summary,
            f"overview.resources[{index}]",
        )
        if summary.scope.run_id != run_id:
            _fail(
                f"overview.resources[{index}].scope.run_id",
                "must match run.run_id",
            )
    resource_keys = tuple(_resource_key(summary) for summary in report.resources)
    if len(resource_keys) != len(set(resource_keys)):
        _fail("overview.resources", "must not contain duplicate resource streams")
    _validate_data_quality(report.data_quality, "overview.data_quality")
    _validate_perfetto(report.perfetto, "overview.perfetto")
    _validate_native_profiles(report.native_profiles, "overview.native_profiles")
    _validate_interpretation(report.interpretation, "overview.interpretation")
    if report.data_quality["run_status"] != report.run["status"]:
        _fail("overview.data_quality.run_status", "must match run.status")
    if report.data_quality["trace_sha256"] != report.perfetto["trace"]["sha256"]:
        _fail(
            "overview.data_quality.trace_sha256",
            "must match perfetto.trace.sha256",
        )
    sql = report.data_quality["perfetto_sql_validation"]
    if (
        sql["valid"] != report.perfetto["valid"]
        or sql["query_count"] != report.perfetto["query_count"]
        or sql["mismatches"] != report.perfetto["mismatches"]
    ):
        _fail(
            "overview.data_quality.perfetto_sql_validation",
            "must match the Perfetto summary",
        )


def validate_comparison_run(
    run: ComparisonRun,
    path: str = "comparison.runs[0]",
) -> None:
    _require_type(run, ComparisonRun, path)
    _nonempty(run.run_id, f"{path}.run_id")
    if run.run_mode not in _RUN_MODES:
        _fail(f"{path}.run_mode", "is not a supported run mode")
    if run.profile_mode not in _PROFILE_MODES:
        _fail(f"{path}.profile_mode", "is not a supported profile mode")
    _nonempty(run.profile_kind, f"{path}.profile_kind")
    for name in (
        "overview_sha256",
        "model_identity_sha256",
        "hardware_identity_sha256",
        "workload_identity_sha256",
    ):
        value = getattr(run, name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            _fail(f"{path}.{name}", "must be lowercase SHA-256")
    _integer(
        run.request_sample_count,
        f"{path}.request_sample_count",
        minimum=0,
    )
    _nonempty(
        run.canonical_clock_domain_id,
        f"{path}.canonical_clock_domain_id",
    )
    if run.clock_alignment_status not in _ALIGNMENT_STATUSES:
        _fail(f"{path}.clock_alignment_status", "is invalid")
    if not isinstance(run.source_integrity_valid, bool):
        _fail(f"{path}.source_integrity_valid", "must be boolean")
    _sorted_unique_strings(
        run.quality_warnings,
        f"{path}.quality_warnings",
    )


def validate_comparison_value(
    value: ComparisonValue,
    path: str,
) -> None:
    _require_type(value, ComparisonValue, path)
    _nonempty(value.run_id, f"{path}.run_id")
    _validate_available_scalar(
        value.availability,
        value.value,
        value.unavailable_reason,
        path,
    )
    sample_count = _integer(
        value.sample_count,
        f"{path}.sample_count",
        minimum=0,
    )
    if value.availability is Availability.AVAILABLE and sample_count == 0:
        _fail(f"{path}.sample_count", "available value requires samples")


def validate_delta_value(value: DeltaValue, path: str) -> None:
    _require_type(value, DeltaValue, path)
    _validate_available_scalar(
        value.availability,
        value.value,
        value.unavailable_reason,
        path,
    )


def validate_comparison_delta(
    delta: ComparisonDelta,
    path: str,
) -> None:
    _require_type(delta, ComparisonDelta, path)
    _nonempty(delta.run_id, f"{path}.run_id")
    _nonempty(delta.baseline_run_id, f"{path}.baseline_run_id")
    validate_delta_value(delta.absolute, f"{path}.absolute")
    validate_delta_value(delta.percentage, f"{path}.percentage")


def validate_comparison_kpi(
    kpi: ComparisonKpi,
    run_ids: tuple[str, ...],
    baseline_run_id: str | None,
    path: str,
    *,
    comparability: Comparability | None = None,
) -> None:
    _require_type(kpi, ComparisonKpi, path)
    _nonempty(kpi.section, f"{path}.section")
    section_contract = _COMPARISON_SECTION_CONTRACT.get(kpi.section)
    if section_contract is None:
        _fail(f"{path}.section", "is not a supported KPI section")
    if kpi.observation_layer not in _OBSERVATION_LAYERS:
        _fail(f"{path}.observation_layer", "is not a supported observation layer")
    definition = METRIC_CATALOG.get(kpi.name)
    if definition is None:
        _fail(f"{path}.name", "must be an official METRIC_CATALOG name")
    allowed_names, allowed_layers = section_contract
    if kpi.name not in allowed_names:
        _fail(f"{path}.name", "does not belong to this comparison section")
    if kpi.observation_layer not in allowed_layers:
        _fail(
            f"{path}.observation_layer",
            "does not match this comparison section",
        )
    if kpi.canonical_unit != definition.unit:
        _fail(f"{path}.canonical_unit", "must match METRIC_CATALOG")
    if not isinstance(kpi.direction, KpiDirection):
        _fail(f"{path}.direction", "must be a KpiDirection")
    _sorted_models(
        kpi.values,
        f"{path}.values",
        key=lambda value: value.run_id,
        allow_empty=False,
    )
    for index, value in enumerate(kpi.values):
        validate_comparison_value(value, f"{path}.values[{index}]")
    if tuple(value.run_id for value in kpi.values) != run_ids:
        _fail(f"{path}.values", "must contain every comparison run exactly once")
    _sorted_models(
        kpi.deltas,
        f"{path}.deltas",
        key=lambda value: value.run_id,
    )
    for index, delta in enumerate(kpi.deltas):
        validate_comparison_delta(delta, f"{path}.deltas[{index}]")
        if delta.baseline_run_id != baseline_run_id:
            _fail(
                f"{path}.deltas[{index}].baseline_run_id",
                "must match comparison baseline",
            )
    expected_delta_runs = (
        ()
        if comparability is Comparability.NOT_COMPARABLE
        else (
            tuple(run_id for run_id in run_ids if run_id != baseline_run_id)
            if baseline_run_id is not None
            else ()
        )
    )
    if tuple(delta.run_id for delta in kpi.deltas) != expected_delta_runs:
        _fail(
            f"{path}.deltas",
            "must cover every non-baseline run, or be empty without a baseline",
        )
    _sorted_unique_strings(
        kpi.quality_warnings,
        f"{path}.quality_warnings",
    )


def validate_comparison_metadata(
    metadata: ComparisonMetadata,
    run_ids: tuple[str, ...],
    path: str = "comparison.comparison",
) -> None:
    _require_type(metadata, ComparisonMetadata, path)
    if not isinstance(metadata.comparability, Comparability):
        _fail(f"{path}.comparability", "must be a Comparability enum")
    _sorted_unique_strings(
        metadata.comparability_reasons,
        f"{path}.comparability_reasons",
        allow_empty=metadata.comparability is Comparability.COMPARABLE,
    )
    if metadata.baseline_run_id is not None:
        _nonempty(metadata.baseline_run_id, f"{path}.baseline_run_id")
        if metadata.baseline_run_id not in run_ids:
            _fail(f"{path}.baseline_run_id", "must identify a compared run")


def validate_overview_comparison(comparison: OverviewComparison) -> None:
    _require_type(comparison, OverviewComparison, "comparison")
    if comparison.schema_version != SCHEMA_VERSION:
        _fail("comparison.schema_version", f"must be {SCHEMA_VERSION}")
    if comparison.record_type != OVERVIEW_COMPARISON_RECORD_TYPE:
        _fail(
            "comparison.record_type",
            f"must be {OVERVIEW_COMPARISON_RECORD_TYPE}",
        )
    _sorted_models(
        comparison.runs,
        "comparison.runs",
        key=lambda run: run.run_id,
        allow_empty=False,
    )
    if len(comparison.runs) < 2:
        _fail("comparison.runs", "must contain at least two runs")
    for index, run in enumerate(comparison.runs):
        validate_comparison_run(run, f"comparison.runs[{index}]")
    run_ids = tuple(run.run_id for run in comparison.runs)
    validate_comparison_metadata(comparison.comparison, run_ids)
    if comparison.comparison.comparability is Comparability.NOT_COMPARABLE:
        for index, kpi in enumerate(comparison.metrics):
            if kpi.deltas:
                _fail(
                    f"comparison.metrics[{index}].deltas",
                    "not_comparable report must not calculate deltas",
                )
    _sorted_models(
        comparison.metrics,
        "comparison.metrics",
        key=lambda kpi: (kpi.section, kpi.observation_layer, kpi.name),
    )
    for index, kpi in enumerate(comparison.metrics):
        validate_comparison_kpi(
            kpi,
            run_ids,
            comparison.comparison.baseline_run_id,
            f"comparison.metrics[{index}]",
            comparability=comparison.comparison.comparability,
        )
    _sorted_unique_strings(
        comparison.limitations,
        "comparison.limitations",
        allow_empty=False,
    )


def validate_overview_document(document: OverviewDocument) -> None:
    """Validate one report or comparison with all semantic invariants."""

    if isinstance(document, OverviewReport):
        validate_overview_report(document)
    elif isinstance(document, OverviewComparison):
        validate_overview_comparison(document)
    else:
        _fail("overview", "must be OverviewReport or OverviewComparison")


def overview_to_dict(document: OverviewDocument) -> dict[str, Any]:
    """Return a validated JSON-compatible object with no host paths."""

    validate_overview_document(document)
    value = _raw_primitive(document)
    assert isinstance(value, dict)
    _json_value(value, "overview")
    return value


def canonical_json_bytes(document: OverviewDocument) -> bytes:
    """Serialize a validated document to stable, path-free canonical bytes."""

    value = overview_to_dict(document)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover - guarded above
        raise OverviewSchemaError(
            "overview",
            f"cannot be serialized as finite canonical JSON: {error}",
        ) from error


def canonical_sha256(document: OverviewDocument) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _strict_object(
    value: object,
    cls: type[Any],
    path: str,
) -> dict[str, Any]:
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


def _tuple_of(
    value: object,
    parser,
    path: str,
) -> tuple[Any, ...]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return tuple(parser(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return tuple(value)


def _availability(value: object, path: str) -> Availability:
    try:
        return Availability(value)
    except (TypeError, ValueError) as error:
        _fail(path, "is not a valid availability")
        raise AssertionError from error


def _kpi_source_from_dict(value: object, path: str) -> KpiSource:
    data = _strict_object(value, KpiSource, path)
    data["record_ids"] = _string_tuple(data["record_ids"], f"{path}.record_ids")
    data["metric_names"] = _string_tuple(
        data["metric_names"],
        f"{path}.metric_names",
    )
    return KpiSource(**data)


def _scope_from_dict(value: object, path: str) -> KpiScope:
    return KpiScope(**_strict_object(value, KpiScope, path))


def _calculation_from_dict(value: object, path: str) -> KpiCalculation:
    return KpiCalculation(**_strict_object(value, KpiCalculation, path))


def _clock_from_dict(value: object, path: str) -> KpiClock:
    data = _strict_object(value, KpiClock, path)
    data["domain_ids"] = _string_tuple(data["domain_ids"], f"{path}.domain_ids")
    return KpiClock(**data)


def _display_from_dict(value: object, path: str) -> DisplayRule:
    return DisplayRule(**_strict_object(value, DisplayRule, path))


def _kpi_from_dict(value: object, path: str) -> KpiValue:
    data = _strict_object(value, KpiValue, path)
    data["availability"] = _availability(
        data["availability"],
        f"{path}.availability",
    )
    data["sources"] = _tuple_of(
        data["sources"],
        _kpi_source_from_dict,
        f"{path}.sources",
    )
    data["scope"] = _scope_from_dict(data["scope"], f"{path}.scope")
    data["calculation"] = _calculation_from_dict(
        data["calculation"],
        f"{path}.calculation",
    )
    data["clock"] = _clock_from_dict(data["clock"], f"{path}.clock")
    data["quality_warnings"] = _string_tuple(
        data["quality_warnings"],
        f"{path}.quality_warnings",
    )
    data["display"] = _display_from_dict(data["display"], f"{path}.display")
    return KpiValue(**data)


def _resource_from_dict(value: object, path: str) -> ResourceSummary:
    data = _strict_object(value, ResourceSummary, path)
    data["scope"] = _scope_from_dict(data["scope"], f"{path}.scope")
    data["clock"] = _clock_from_dict(data["clock"], f"{path}.clock")
    data["aggregates"] = _tuple_of(
        data["aggregates"],
        _kpi_from_dict,
        f"{path}.aggregates",
    )
    data["quality_warnings"] = _string_tuple(
        data["quality_warnings"],
        f"{path}.quality_warnings",
    )
    return ResourceSummary(**data)


def _kpi_sections_from_dict(value: object, path: str) -> KpiSections:
    data = _strict_object(value, KpiSections, path)
    for name in (
        "request_facing_latency",
        "pipeline_latency",
        "throughput_and_tokens",
        "transfer",
    ):
        data[name] = _tuple_of(
            data[name],
            _kpi_from_dict,
            f"{path}.{name}",
        )
    return KpiSections(**data)


def overview_report_from_dict(value: object) -> OverviewReport:
    """Parse and semantically validate a strict Overview report object."""

    data = _strict_object(value, OverviewReport, "overview")
    data["models"] = _tuple_of(
        data["models"],
        lambda item, path: dict(item) if isinstance(item, dict) else _fail(
            path, "must be an object"
        ),
        "overview.models",
    )
    data["hardware"] = _tuple_of(
        data["hardware"],
        lambda item, path: dict(item) if isinstance(item, dict) else _fail(
            path, "must be an object"
        ),
        "overview.hardware",
    )
    data["kpis"] = _kpi_sections_from_dict(data["kpis"], "overview.kpis")
    data["resources"] = _tuple_of(
        data["resources"],
        _resource_from_dict,
        "overview.resources",
    )
    data["native_profiles"] = _tuple_of(
        data["native_profiles"],
        lambda item, path: dict(item) if isinstance(item, dict) else _fail(
            path, "must be an object"
        ),
        "overview.native_profiles",
    )
    report = OverviewReport(**data)
    validate_overview_report(report)
    return report


def _comparison_run_from_dict(value: object, path: str) -> ComparisonRun:
    data = _strict_object(value, ComparisonRun, path)
    data["quality_warnings"] = _string_tuple(
        data["quality_warnings"],
        f"{path}.quality_warnings",
    )
    return ComparisonRun(**data)


def _comparison_value_from_dict(value: object, path: str) -> ComparisonValue:
    data = _strict_object(value, ComparisonValue, path)
    data["availability"] = _availability(
        data["availability"],
        f"{path}.availability",
    )
    return ComparisonValue(**data)


def _delta_value_from_dict(value: object, path: str) -> DeltaValue:
    data = _strict_object(value, DeltaValue, path)
    data["availability"] = _availability(
        data["availability"],
        f"{path}.availability",
    )
    return DeltaValue(**data)


def _comparison_delta_from_dict(value: object, path: str) -> ComparisonDelta:
    data = _strict_object(value, ComparisonDelta, path)
    data["absolute"] = _delta_value_from_dict(
        data["absolute"],
        f"{path}.absolute",
    )
    data["percentage"] = _delta_value_from_dict(
        data["percentage"],
        f"{path}.percentage",
    )
    return ComparisonDelta(**data)


def _comparison_kpi_from_dict(value: object, path: str) -> ComparisonKpi:
    data = _strict_object(value, ComparisonKpi, path)
    try:
        data["direction"] = KpiDirection(data["direction"])
    except (TypeError, ValueError):
        _fail(f"{path}.direction", "is not a valid KPI direction")
    data["values"] = _tuple_of(
        data["values"],
        _comparison_value_from_dict,
        f"{path}.values",
    )
    data["deltas"] = _tuple_of(
        data["deltas"],
        _comparison_delta_from_dict,
        f"{path}.deltas",
    )
    data["quality_warnings"] = _string_tuple(
        data["quality_warnings"],
        f"{path}.quality_warnings",
    )
    return ComparisonKpi(**data)


def _comparison_metadata_from_dict(
    value: object,
    path: str,
) -> ComparisonMetadata:
    data = _strict_object(value, ComparisonMetadata, path)
    try:
        data["comparability"] = Comparability(data["comparability"])
    except (TypeError, ValueError):
        _fail(f"{path}.comparability", "is not a valid comparability")
    data["comparability_reasons"] = _string_tuple(
        data["comparability_reasons"],
        f"{path}.comparability_reasons",
    )
    return ComparisonMetadata(**data)


def overview_comparison_from_dict(value: object) -> OverviewComparison:
    """Parse and semantically validate a strict Overview comparison object."""

    data = _strict_object(value, OverviewComparison, "comparison")
    data["comparison"] = _comparison_metadata_from_dict(
        data["comparison"],
        "comparison.comparison",
    )
    data["runs"] = _tuple_of(
        data["runs"],
        _comparison_run_from_dict,
        "comparison.runs",
    )
    data["metrics"] = _tuple_of(
        data["metrics"],
        _comparison_kpi_from_dict,
        "comparison.metrics",
    )
    data["limitations"] = _string_tuple(
        data["limitations"],
        "comparison.limitations",
    )
    comparison = OverviewComparison(**data)
    validate_overview_comparison(comparison)
    return comparison


def overview_document_from_dict(value: object) -> OverviewDocument:
    """Parse either model by its required record_type."""

    if not isinstance(value, dict):
        _fail("overview", "must be an object")
    record_type = value.get("record_type")
    if record_type == OVERVIEW_REPORT_RECORD_TYPE:
        return overview_report_from_dict(value)
    if record_type == OVERVIEW_COMPARISON_RECORD_TYPE:
        return overview_comparison_from_dict(value)
    _fail("overview.record_type", "is not a supported Overview record type")
    raise AssertionError


def overview_document_from_json(payload: str | bytes) -> OverviewDocument:
    """Decode strict finite JSON and return a validated Overview document."""

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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


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
    resource = resources.files(__package__) / "json" / "v1" / name
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
