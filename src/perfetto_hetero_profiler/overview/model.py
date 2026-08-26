"""Versioned, path-free data model for deterministic Overview reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias

from ..schema import Availability
from ..schema.constants import SCHEMA_VERSION


OVERVIEW_MODEL_VERSION = "1.0.0"
OVERVIEW_REPORT_RECORD_TYPE = "overview_report"
OVERVIEW_COMPARISON_RECORD_TYPE = "overview_comparison"

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


class Comparability(str, Enum):
    """Comparison eligibility determined from immutable run dimensions."""

    COMPARABLE = "comparable"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    NOT_COMPARABLE = "not_comparable"


class KpiDirection(str, Enum):
    """How a consumer may interpret a numeric delta without ranking runs."""

    LOWER_IS_PREFERRED = "lower_is_preferred"
    HIGHER_IS_PREFERRED = "higher_is_preferred"
    NEUTRAL = "neutral"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonRun:
    """Path-free identity and sample context for one compared Overview."""

    run_id: str
    run_mode: str
    profile_mode: str
    profile_kind: str
    overview_sha256: str
    request_sample_count: int
    model_identity_sha256: str
    hardware_identity_sha256: str
    workload_identity_sha256: str
    canonical_clock_domain_id: str
    clock_alignment_status: str
    source_integrity_valid: bool
    quality_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonValue:
    """One run's value in a comparison row."""

    run_id: str
    availability: Availability
    value: int | float | None
    unavailable_reason: str | None
    sample_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DeltaValue:
    """Available or unavailable scalar delta."""

    availability: Availability
    value: int | float | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonDelta:
    """Absolute and percentage delta from the selected baseline."""

    run_id: str
    baseline_run_id: str
    absolute: DeltaValue
    percentage: DeltaValue


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonKpi:
    """One KPI row across all compared runs."""

    section: str
    observation_layer: str
    name: str
    canonical_unit: str
    direction: KpiDirection
    values: tuple[ComparisonValue, ...]
    deltas: tuple[ComparisonDelta, ...]
    quality_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonMetadata:
    """Eligibility classification and optional baseline selection."""

    comparability: Comparability
    comparability_reasons: tuple[str, ...]
    baseline_run_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class OverviewComparison:
    """Deterministic comparison of independently validated Overview outputs."""

    comparison: ComparisonMetadata
    runs: tuple[ComparisonRun, ...]
    metrics: tuple[ComparisonKpi, ...]
    limitations: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    record_type: str = OVERVIEW_COMPARISON_RECORD_TYPE


OverviewDocument: TypeAlias = OverviewReport | OverviewComparison


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
