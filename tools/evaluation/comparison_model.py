"""Repository-only data model for deterministic Overview comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perfetto_hetero_profiler.schema import Availability
from perfetto_hetero_profiler.schema.constants import SCHEMA_VERSION


OVERVIEW_COMPARISON_RECORD_TYPE = "overview_comparison"


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


__all__ = [
    "Comparability",
    "ComparisonDelta",
    "ComparisonKpi",
    "ComparisonMetadata",
    "ComparisonRun",
    "ComparisonValue",
    "DeltaValue",
    "KpiDirection",
    "OVERVIEW_COMPARISON_RECORD_TYPE",
    "OverviewComparison",
]
