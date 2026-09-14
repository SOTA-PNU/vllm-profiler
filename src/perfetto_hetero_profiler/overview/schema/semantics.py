"""Semantic rules an Overview report must satisfy beyond its JSON structure.

JSON Schema cannot say that a KPI's unit matches the official catalog, that an
available value carries provenance and a sample, or that ``data_quality``
agrees with the Perfetto summary.  Those cross-field contracts live here.
"""

from __future__ import annotations

import math
import re
from typing import Mapping

from ...schema import METRIC_CATALOG, Availability
from ...schema.catalog import (
    INTERVAL_RESOURCE_METRICS,
    KPI_SECTION_METRICS,
    RESOURCE_AGGREGATIONS,
)
from ..model import (
    DisplayRule,
    KpiCalculation,
    KpiClock,
    KpiScope,
    KpiSections,
    KpiSource,
    KpiValue,
    OverviewReport,
    ResourceSummary,
)
from .primitives import (
    _deterministic_object_tuple,
    _fail,
    _integer,
    _json_object,
    _json_value,
    _nonempty,
    _number,
    _raw_primitive,
    _require_type,
    _safe_relative_path,
    _sorted_json_array,
    _sorted_unique_strings,
    _validate_report_structure,
)

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
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
_RESOURCE_AGGREGATIONS = dict(RESOURCE_AGGREGATIONS)
_INTERVAL_RESOURCE_METRICS = INTERVAL_RESOURCE_METRICS
_REQUEST_FACING_KPIS = KPI_SECTION_METRICS["request_facing_latency"]
_PIPELINE_KPIS = KPI_SECTION_METRICS["pipeline_latency"]
_THROUGHPUT_TOKEN_KPIS = KPI_SECTION_METRICS["throughput_and_tokens"]
_TRANSFER_KPIS = KPI_SECTION_METRICS["transfer"]
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
_AGGREGATE_RUN_METHODS = {
    "arithmetic_mean_across_measured_requests_v1",
    "not_available_across_measured_requests_v1",
    "ratio_of_measured_request_means_v1",
}
_KPI_SECTION_CONTRACT = (
    (
        "request_facing_latency",
        _REQUEST_FACING_KPIS,
        {"request_facing_client", "gpu_only", "npu_only"},
    ),
    ("pipeline_latency", _PIPELINE_KPIS, {"hybrid_pipeline"}),
    (
        "throughput_and_tokens",
        _THROUGHPUT_TOKEN_KPIS,
        {"request_facing_client", "gpu_only", "npu_only", "run"},
    ),
    ("transfer", _TRANSFER_KPIS, {"hybrid_pipeline"}),
)


def _require_sorted_unique(values: object, path: str) -> None:
    """Require a JSON array that is already sorted and free of duplicates."""

    normalized = tuple(values)
    if normalized != tuple(sorted(set(normalized))):
        _fail(path, "must be sorted without duplicates")


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
        and kpi.aggregation_method in _AGGREGATE_RUN_METHODS
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


def validate_resource_summary(summary: ResourceSummary, path: str = "resource") -> None:
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
    for name, allowed_names, observation_layers in _KPI_SECTION_CONTRACT:
        _validate_kpi_section(
            getattr(sections, name),
            path=f"{path}.{name}",
            run_id=run_id,
            allowed_names=allowed_names,
            observation_layers=observation_layers,
        )


def _validate_data_quality(value: object, path: str) -> None:
    assert isinstance(value, Mapping)
    data = value
    samples = data["resource_samples"]
    assert isinstance(samples, Mapping)
    total = samples["total"]
    available = samples["available"]
    unavailable = samples["unavailable"]
    if total != available + unavailable:
        _fail(
            f"{path}.resource_samples",
            "total must equal available + unavailable",
        )

    source = data["source_artifact_validation"]
    assert isinstance(source, Mapping)
    roots = source["roots"]
    assert isinstance(roots, list)
    root_keys: list[str] = []
    for index, root in enumerate(roots):
        assert isinstance(root, Mapping)
        root_id = str(root["root_id"])
        if _ROOT_ID_RE.fullmatch(root_id) is None:
            _fail(
                f"{path}.source_artifact_validation.roots[{index}].root_id",
                "is not a safe root id",
            )
        root_keys.append(root_id)
    if root_keys != sorted(set(root_keys)):
        _fail(
            f"{path}.source_artifact_validation.roots",
            "must be sorted by unique root_id",
        )

    perfetto = data["perfetto_sql_validation"]
    assert isinstance(perfetto, Mapping)
    _require_sorted_unique(
        perfetto["mismatches"], f"{path}.perfetto_sql_validation.mismatches"
    )
    _require_sorted_unique(data["sample_limitations"], f"{path}.sample_limitations")


def _validate_workload(value: object, path: str) -> None:
    assert isinstance(value, Mapping)
    workload = value
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


def _validate_interpretation(value: object, path: str) -> None:
    assert isinstance(value, Mapping)
    interpretation = value
    _require_sorted_unique(interpretation["limitations"], f"{path}.limitations")


def _validate_perfetto(value: object, path: str) -> None:
    assert isinstance(value, Mapping)
    perfetto = value
    counts = perfetto["counts"]
    assert isinstance(counts, Mapping)
    query_count = perfetto["query_count"]
    queries = perfetto["queries"]
    assert isinstance(queries, list)
    query_names: list[str] = []
    for query in queries:
        assert isinstance(query, Mapping)
        query_names.append(str(query["name"]))
    if query_count != len(queries):
        _fail(f"{path}.query_count", "must equal len(queries)")
    if len(query_names) != len(set(query_names)):
        _fail(f"{path}.queries", "must contain unique query names")

    mismatch_values = tuple(perfetto["mismatches"])
    _require_sorted_unique(mismatch_values, f"{path}.mismatches")

    flow = perfetto["flow_endpoint_reconciliation"]
    assert isinstance(flow, Mapping)
    for name in (
        "declared_flow_ids",
        "source_endpoint_ids",
        "destination_endpoint_ids",
    ):
        _require_sorted_unique(
            flow[name], f"{path}.flow_endpoint_reconciliation.{name}"
        )
    artifact = perfetto["artifact_validation"]
    assert isinstance(artifact, Mapping)
    _sorted_json_array(
        artifact["mismatches"],
        f"{path}.artifact_validation.mismatches",
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
    primitive = _raw_primitive(report)
    _json_value(primitive, "overview")
    _validate_report_structure(primitive)
    run_id = str(report.run["run_id"])
    _validate_workload(report.workload, "overview.workload")
    _deterministic_object_tuple(report.models, "overview.models")
    _deterministic_object_tuple(report.hardware, "overview.hardware")
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
    _deterministic_object_tuple(
        report.native_profiles, "overview.native_profiles"
    )
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



__all__ = [
    "validate_display_rule",
    "validate_kpi",
    "validate_kpi_clock",
    "validate_kpi_scope",
    "validate_kpi_sections",
    "validate_kpi_source",
    "validate_overview_report",
    "validate_resource_summary",
]
