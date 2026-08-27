"""Versioned, path-free data model for deterministic Overview reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from ..schema import Availability
from ..schema.constants import SCHEMA_VERSION


OVERVIEW_MODEL_VERSION = "1.0.0"
OVERVIEW_REPORT_RECORD_TYPE = "overview_report"

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiSource:
    """Logical, path-free provenance for one KPI."""

    source_kind: str
    record_ids: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()
    root_id: str | None = None
    relative_path: str | None = None
    details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiScope:
    """Run/request/host/device scope and observation layer."""

    run_id: str
    scope_type: str
    observation_layer: str
    request_id: str | None = None
    host_id: str | None = None
    device_type: str | None = None
    device_id: str | None = None
    phase: str | None = None
    window: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiCalculation:
    """Stable formula identifier plus human-readable formula."""

    method_id: str
    formula: str


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiClock:
    """Clock and alignment evidence attached to a KPI."""

    domain_ids: tuple[str, ...]
    alignment_status: str
    alignment_method: str | None
    offset_ns: int | None
    uncertainty_ns: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DisplayRule:
    """Exact rational conversion from canonical units to display units."""

    unit: str
    scale_numerator: int
    scale_denominator: int
    decimal_places: int
    rounding: str = "half_even"


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiValue:
    """One measured or unavailable KPI with complete provenance."""

    name: str
    canonical_unit: str
    availability: Availability
    value: int | float | None
    unavailable_reason: str | None
    aggregation_method: str
    sample_count: int
    sources: tuple[KpiSource, ...]
    scope: KpiScope
    calculation: KpiCalculation
    clock: KpiClock
    quality_warnings: tuple[str, ...]
    display: DisplayRule


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSummary:
    """Per-host or per-device resource stream aggregation."""

    metric_name: str
    canonical_unit: str
    scope: KpiScope
    clock: KpiClock
    total_sample_count: int
    available_sample_count: int
    unavailable_sample_count: int
    availability_ratio: float
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    coverage_ns: int | None
    aggregates: tuple[KpiValue, ...]
    quality_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class KpiSections:
    """Stable display sections; observation layers never share one list."""

    request_facing_latency: tuple[KpiValue, ...]
    pipeline_latency: tuple[KpiValue, ...]
    throughput_and_tokens: tuple[KpiValue, ...]
    transfer: tuple[KpiValue, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class OverviewReport:
    """Deterministic single-run Overview JSON document."""

    run: JsonObject
    workload: JsonObject
    models: tuple[JsonObject, ...]
    hardware: tuple[JsonObject, ...]
    kpis: KpiSections
    resources: tuple[ResourceSummary, ...]
    data_quality: JsonObject
    perfetto: JsonObject
    native_profiles: tuple[JsonObject, ...]
    interpretation: JsonObject
    schema_version: str = SCHEMA_VERSION
    record_type: str = OVERVIEW_REPORT_RECORD_TYPE


__all__ = [
    "DisplayRule",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "KpiCalculation",
    "KpiClock",
    "KpiSections",
    "KpiScope",
    "KpiSource",
    "KpiValue",
    "OVERVIEW_MODEL_VERSION",
    "OVERVIEW_REPORT_RECORD_TYPE",
    "OverviewReport",
    "ResourceSummary",
]
