"""Strict repository-only contract for Overview comparison documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from perfetto_hetero_profiler.overview.schema import (
    OverviewSchemaError,
    _ALIGNMENT_STATUSES,
    _OBSERVATION_LAYERS,
    _PROFILE_MODES,
    _RUN_MODES,
    _availability,
    _fail,
    _integer,
    _json_value,
    _nonempty,
    _raw_primitive,
    _reject_duplicate_pairs,
    _require_type,
    _sorted_models,
    _sorted_unique_strings,
    _strict_object,
    _string_tuple,
    _tuple_of,
    _validate_available_scalar,
)
from perfetto_hetero_profiler.schema import Availability, METRIC_CATALOG
from perfetto_hetero_profiler.schema.catalog import KPI_SECTION_METRICS
from perfetto_hetero_profiler.schema.constants import (
    JSON_SCHEMA_DRAFT,
    SCHEMA_VERSION,
    SHA256_RE,
)

from .comparison_model import (
    OVERVIEW_COMPARISON_RECORD_TYPE,
    Comparability,
    ComparisonDelta,
    ComparisonKpi,
    ComparisonMetadata,
    ComparisonRun,
    ComparisonValue,
    DeltaValue,
    KpiDirection,
    OverviewComparison,
)


OVERVIEW_COMPARISON_SCHEMA_NAME = "overview_comparison.schema.json"
_COMPARISON_SECTION_CONTRACT = {
    "request_facing_latency": (
        KPI_SECTION_METRICS["request_facing_latency"],
        {"request_facing_client", "gpu_only", "npu_only"},
    ),
    "pipeline_latency": (
        KPI_SECTION_METRICS["pipeline_latency"],
        {"hybrid_pipeline"},
    ),
    "throughput_and_tokens": (
        KPI_SECTION_METRICS["throughput_and_tokens"],
        {"request_facing_client", "gpu_only", "npu_only", "run"},
    ),
    "transfer": (
        KPI_SECTION_METRICS["transfer"],
        {"hybrid_pipeline"},
    ),
}


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
        _fail(
            f"{path}.observation_layer",
            "is not a supported observation layer",
        )
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
        _fail(
            f"{path}.values",
            "must contain every comparison run exactly once",
        )
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


def comparison_to_dict(comparison: OverviewComparison) -> dict[str, Any]:
    """Return a validated, path-free JSON-compatible comparison object."""

    validate_overview_comparison(comparison)
    value = _raw_primitive(comparison)
    assert isinstance(value, dict)
    _json_value(value, "comparison")
    return value


def canonical_comparison_json_bytes(comparison: OverviewComparison) -> bytes:
    """Serialize a comparison to stable canonical JSON bytes."""

    value = comparison_to_dict(comparison)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover - prevalidated
        raise OverviewSchemaError(
            "comparison",
            f"cannot be serialized as finite canonical JSON: {error}",
        ) from error


def canonical_comparison_sha256(comparison: OverviewComparison) -> str:
    return hashlib.sha256(canonical_comparison_json_bytes(comparison)).hexdigest()


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
    """Parse and semantically validate a strict comparison object."""

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


def overview_document_from_json(payload: str | bytes) -> OverviewComparison:
    """Decode strict finite JSON into a repository-only comparison model."""

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise OverviewSchemaError("comparison", f"invalid JSON: {error}") from error
    return overview_comparison_from_dict(value)


def load_comparison_schema() -> dict[str, Any]:
    """Load the repository-only checked-in comparison schema."""

    path = Path(__file__).with_name("schema") / OVERVIEW_COMPARISON_SCHEMA_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OverviewSchemaError(
            "json_schema",
            f"cannot load {OVERVIEW_COMPARISON_SCHEMA_NAME}: {error}",
        ) from error
    if not isinstance(value, dict):
        _fail("json_schema", "must be an object")
    return value


def validate_comparison_schema_contract() -> None:
    """Verify schema identity and top-level model field parity."""

    schema = load_comparison_schema()
    if schema.get("$schema") != JSON_SCHEMA_DRAFT:
        _fail(
            OVERVIEW_COMPARISON_SCHEMA_NAME,
            f"$schema must be {JSON_SCHEMA_DRAFT}",
        )
    if schema.get("type") != "object":
        _fail(OVERVIEW_COMPARISON_SCHEMA_NAME, "top-level type must be object")
    if schema.get("additionalProperties") is not False:
        _fail(
            OVERVIEW_COMPARISON_SCHEMA_NAME,
            "must reject additional properties",
        )
    properties = schema.get("properties")
    required = schema.get("required")
    expected = {item.name for item in fields(OverviewComparison)}
    if not isinstance(properties, dict) or set(properties) != expected:
        _fail(
            OVERVIEW_COMPARISON_SCHEMA_NAME,
            "top-level properties differ from dataclass fields",
        )
    if not isinstance(required, list) or set(required) != expected:
        _fail(
            OVERVIEW_COMPARISON_SCHEMA_NAME,
            "all top-level properties must be required",
        )
    record_schema = properties.get("record_type")
    if (
        not isinstance(record_schema, dict)
        or record_schema.get("const") != OVERVIEW_COMPARISON_RECORD_TYPE
    ):
        _fail(
            OVERVIEW_COMPARISON_SCHEMA_NAME,
            "record_type const differs from model",
        )


__all__ = [
    "OVERVIEW_COMPARISON_SCHEMA_NAME",
    "canonical_comparison_json_bytes",
    "canonical_comparison_sha256",
    "comparison_to_dict",
    "load_comparison_schema",
    "overview_comparison_from_dict",
    "overview_document_from_json",
    "validate_comparison_delta",
    "validate_comparison_kpi",
    "validate_comparison_metadata",
    "validate_comparison_run",
    "validate_comparison_schema_contract",
    "validate_comparison_value",
    "validate_delta_value",
    "validate_overview_comparison",
]
