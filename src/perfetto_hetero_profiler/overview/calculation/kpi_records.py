"""Catalog-checked construction of one Overview KPI record with provenance.

Domain modules decide *which* KPI to emit; this module decides what a valid KPI
record is — catalog contract, availability policy, provenance and coercion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ...artifact_compatibility import LEGACY_MEASURED_WINDOW
from ...schema import Availability
from ...schema.catalog import display_rule
from ...schema.metric_catalog import METRIC_CATALOG
from ...schema.records import EventRecord, MetricSample
from ..resources import _enum_value


class OverviewCalculationError(ValueError):
    """Raised when KPI provenance is contradictory or unsafe."""


def _finite_number(value: object, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OverviewCalculationError(f"{field} must be a non-bool number")
    if not math.isfinite(value):
        raise OverviewCalculationError(f"{field} must be finite")
    return value


def _non_bool_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OverviewCalculationError(f"{field} must be a non-bool integer")
    return value


def _scope(
    loaded: object,
    *,
    scope_type: str,
    observation_layer: str,
    request_id: str | None = None,
    host_id: str | None = None,
    device_type: str | None = None,
    device_id: str | None = None,
    phase: str | None = None,
    window: str | None = None,
) -> dict[str, object]:
    manifest = getattr(loaded, "manifest", None)
    run_id = getattr(manifest, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise OverviewCalculationError("loaded manifest has no run_id")
    return {
        "run_id": run_id,
        "scope_type": scope_type,
        "observation_layer": observation_layer,
        "request_id": request_id,
        "host_id": host_id,
        "device_type": device_type,
        "device_id": device_id,
        "phase": phase,
        "window": window,
    }


def _source_metric(
    metrics: Sequence[MetricSample],
    *,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    event_ids = sorted(
        {
            event_id
            for metric in metrics
            for event_id in (metric.source_event_ids or ())
        }
    )
    return {
        "source_kind": "normalized_metric_stream",
        "record_ids": event_ids,
        "metric_names": sorted({metric.metric_name for metric in metrics}),
        "root_id": None,
        "relative_path": "metrics/metrics.jsonl",
        "details": details or {},
    }


def _source_events(
    events: Sequence[EventRecord],
    *,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "source_kind": "canonical_runtime_markers",
        "record_ids": sorted(event.event_id for event in events),
        "metric_names": [],
        "root_id": None,
        "relative_path": "events/events.jsonl",
        "details": details or {},
    }


def _unknown_clock() -> dict[str, object]:
    """Return a fresh clock record for KPIs with no observed clock evidence."""

    return {
        "domain_ids": [],
        "alignment_status": "unknown",
        "alignment_method": None,
        "offset_ns": None,
        "uncertainty_ns": None,
    }


def _manifest_alignment_offset(loaded: object) -> int | None:
    manifest = getattr(loaded, "manifest", None)
    attributes = getattr(manifest, "attributes", {})
    value = (
        attributes.get("hybrid.alignment_offset_ns")
        if isinstance(attributes, dict)
        else None
    )
    if value is None:
        return None
    return _non_bool_int(value, field="hybrid.alignment_offset_ns")


def _clock(
    loaded: object,
    records: Sequence[EventRecord | MetricSample],
) -> dict[str, object]:
    domains = tuple(sorted({record.clock_domain_id for record in records}))
    canonical = getattr(loaded, "canonical_clock_domain_id", None)
    if canonical is None:
        canonical = getattr(
            getattr(loaded, "canonical_clock", None), "clock_domain_id", None
        )
    methods = {
        record.attributes.get("hybrid.alignment_method")
        for record in records
        if isinstance(record.attributes.get("hybrid.alignment_method"), str)
    }
    uncertainty_values = [
        record.attributes.get("hybrid.alignment_uncertainty_ns")
        for record in records
        if record.attributes.get("hybrid.alignment_uncertainty_ns") is not None
    ]
    uncertainty: int | None = None
    if uncertainty_values:
        parsed = [
            _non_bool_int(value, field="hybrid.alignment_uncertainty_ns")
            for value in uncertainty_values
        ]
        if any(value < 0 for value in parsed):
            raise OverviewCalculationError(
                "hybrid.alignment_uncertainty_ns must be non-negative"
            )
        uncertainty = max(parsed)
    method = next(iter(methods)) if len(methods) == 1 else None
    aligned = (
        len(domains) == 1
        and domains[0] == canonical
        and method is not None
        and uncertainty is not None
    )
    return {
        "domain_ids": list(domains),
        "alignment_status": "aligned" if aligned else "unknown",
        "alignment_method": method,
        "offset_ns": _manifest_alignment_offset(loaded) if aligned else None,
        "uncertainty_ns": uncertainty if aligned else None,
    }


def _kpi(
    *,
    name: str,
    canonical_unit: str,
    value: int | float | None,
    unavailable_reason: str | None,
    aggregation_method: str,
    sample_count: int,
    sources: Sequence[dict[str, object]],
    scope: dict[str, object],
    calculation_method: str,
    formula: str,
    clock: dict[str, object],
    warnings: Iterable[str] = (),
) -> dict[str, object]:
    definition = METRIC_CATALOG.get(name)
    if definition is None:
        raise OverviewCalculationError(f"{name!r} is not an official KPI")
    if canonical_unit != definition.unit:
        raise OverviewCalculationError(
            f"{name} canonical unit does not match METRIC_CATALOG"
        )
    if value is not None:
        value = _finite_number(value, field=f"{name} value")
        if definition.value_type == "integer" and not (
            isinstance(value, int) and not isinstance(value, bool)
        ):
            raise OverviewCalculationError(f"{name} must be an integer")
        if definition.minimum is not None and value < definition.minimum:
            raise OverviewCalculationError(
                f"{name} is below its catalog minimum"
            )
        if definition.maximum is not None and value > definition.maximum:
            raise OverviewCalculationError(
                f"{name} is above its catalog maximum"
            )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise OverviewCalculationError(f"{name} sample_count must be an integer")
    if sample_count < 0:
        raise OverviewCalculationError(f"{name} sample_count must be non-negative")
    available = value is not None
    if available and unavailable_reason is not None:
        raise OverviewCalculationError(
            f"{name} cannot have both a value and unavailable_reason"
        )
    if not available and not unavailable_reason:
        raise OverviewCalculationError(
            f"{name} unavailable KPI requires an unavailable_reason"
        )
    return {
        "name": name,
        "canonical_unit": canonical_unit,
        "availability": (
            Availability.AVAILABLE.value
            if available
            else Availability.NOT_AVAILABLE.value
        ),
        "value": value,
        "unavailable_reason": unavailable_reason,
        "aggregation_method": aggregation_method,
        "sample_count": sample_count,
        "sources": list(sources),
        "scope": scope,
        "calculation": {
            "method_id": calculation_method,
            "formula": formula,
        },
        "clock": clock,
        "quality_warnings": sorted(set(warnings)),
        "display": display_rule(canonical_unit),
    }


def _metric_contract(metric: MetricSample, expected_name: str) -> int | float | None:
    definition = METRIC_CATALOG[expected_name]
    if metric.metric_name != expected_name:
        raise OverviewCalculationError("internal metric selection error")
    if metric.unit != definition.unit:
        raise OverviewCalculationError(
            f"{expected_name} unit mismatch: "
            f"{metric.unit!r} != {definition.unit!r}"
        )
    if metric.metric_kind != definition.kind:
        raise OverviewCalculationError(
            f"{expected_name} metric_kind does not match the catalog"
        )
    if metric.scope not in definition.allowed_scopes:
        raise OverviewCalculationError(
            f"{expected_name} scope does not match the catalog"
        )
    availability = _enum_value(metric.availability)
    if availability not in {item.value for item in Availability}:
        raise OverviewCalculationError(
            f"{expected_name} has an invalid availability"
        )
    if availability == Availability.AVAILABLE.value:
        value = _finite_number(metric.value, field=f"{expected_name} value")
        if definition.value_type == "integer" and not (
            isinstance(value, int) and not isinstance(value, bool)
        ):
            raise OverviewCalculationError(f"{expected_name} must be an integer")
        if definition.minimum is not None and value < definition.minimum:
            raise OverviewCalculationError(
                f"{expected_name} is below its catalog minimum"
            )
        if definition.maximum is not None and value > definition.maximum:
            raise OverviewCalculationError(
                f"{expected_name} is above its catalog maximum"
            )
        return value
    if metric.value is not None:
        raise OverviewCalculationError(
            f"{expected_name} unavailable metric must have value=null"
        )
    return None


def _is_pipeline_metric(metric: MetricSample) -> bool:
    method = metric.dimensions.get("hybrid.join_method")
    return method in {"correlation_id", "transfer_id"}


def _select_metric(
    metrics: Sequence[MetricSample],
    name: str,
    *,
    request_id: str | None | object = ...,
    window: str | None | object = ...,
    pipeline: bool | None = None,
) -> MetricSample | None:
    candidates = [metric for metric in metrics if metric.metric_name == name]
    if request_id is not ...:
        candidates = [
            metric for metric in candidates if metric.request_id == request_id
        ]
    if window is not ...:
        candidates = [
            metric
            for metric in candidates
            if metric.dimensions.get("window") == window
        ]
    if pipeline is True:
        candidates = [metric for metric in candidates if _is_pipeline_metric(metric)]
    elif pipeline is False:
        candidates = [metric for metric in candidates if not _is_pipeline_metric(metric)]
    if len(candidates) > 1:
        raise OverviewCalculationError(
            f"ambiguous normalized metric provenance for {name}"
        )
    return candidates[0] if candidates else None


def _normalized_metric_kpi(
    loaded: object,
    metric: MetricSample | None,
    *,
    name: str,
    observation_layer: str,
    formula: str,
    warning: str | None = None,
) -> dict[str, object]:
    definition = METRIC_CATALOG[name]
    if metric is None:
        scope = _scope(
            loaded,
            scope_type="run",
            observation_layer=observation_layer,
        )
        return _kpi(
            name=name,
            canonical_unit=definition.unit,
            value=None,
            unavailable_reason=f"normalized {name} metric is not present",
            aggregation_method="single_normalized_metric_v1",
            sample_count=0,
            sources=[],
            scope=scope,
            calculation_method="normalized_metric_validation_v1",
            formula=formula,
            clock=_unknown_clock(),
        )
    value = _metric_contract(metric, name)
    return _kpi(
        name=name,
        canonical_unit=definition.unit,
        value=value,
        unavailable_reason=(
            None
            if value is not None
            else metric.reason or f"normalized {name} metric is unavailable"
        ),
        aggregation_method="single_normalized_metric_v1",
        sample_count=1 if value is not None else 0,
        sources=[_source_metric([metric])],
        scope=_scope(
            loaded,
            scope_type=str(_enum_value(metric.scope)),
            observation_layer=observation_layer,
            request_id=metric.request_id,
            host_id=metric.host_id,
            phase=_enum_value(metric.phase),
        ),
        calculation_method="normalized_metric_validation_v1",
        formula=formula,
        clock=_clock(loaded, [metric]),
        warnings=(() if warning is None else (warning,)),
    )


def _aggregate_request_kpis(
    loaded: object,
    rows: Sequence[Sequence[dict[str, object]]],
    request_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate like-named request KPIs without hiding unavailable values."""

    if not rows or len(rows) != len(request_ids):
        raise OverviewCalculationError("request KPI aggregation inputs are incomplete")
    by_name = [{str(item["name"]): item for item in group} for group in rows]
    names = tuple(item["name"] for item in rows[0])
    if any(set(group) != set(names) for group in by_name):
        raise OverviewCalculationError("request KPI sets do not match")
    result: list[dict[str, object]] = []
    for name in names:
        values = [group[name] for group in by_name]
        available = all(
            item["availability"] == Availability.AVAILABLE.value
            for item in values
        )
        numeric = [
            _finite_number(item["value"], field=f"{name} aggregate input")
            for item in values
            if item["availability"] == Availability.AVAILABLE.value
        ]
        value = math.fsum(numeric) / len(numeric) if available else None
        first = values[0]
        scope = first["scope"]
        if not isinstance(scope, dict):
            raise OverviewCalculationError("request KPI scope is invalid")
        calculation = first["calculation"]
        if not isinstance(calculation, dict):
            raise OverviewCalculationError("request KPI calculation is invalid")
        result.append(
            _kpi(
                name=name,
                canonical_unit=str(first["canonical_unit"]),
                value=value,
                unavailable_reason=(
                    None
                    if available
                    else "one or more measured request values are unavailable"
                ),
                aggregation_method="arithmetic_mean_across_measured_requests_v1",
                sample_count=len(numeric),
                sources=[
                    source
                    for item in values
                    for source in item.get("sources", [])
                ],
                scope=_scope(
                    loaded,
                    scope_type="run",
                    observation_layer=str(scope["observation_layer"]),
                    phase=scope.get("phase"),
                    window=LEGACY_MEASURED_WINDOW,
                ),
                calculation_method="request_arithmetic_mean_v1",
                formula=f"mean({calculation.get('formula', name)})",
                clock=first["clock"],
                warnings=(
                    f"arithmetic mean across {len(request_ids)} explicitly "
                    "identified measured requests",
                ),
            )
        )
    return result


__all__ = [
    "OverviewCalculationError",
    "_aggregate_request_kpis",
    "_clock",
    "_enum_value",
    "_finite_number",
    "_is_pipeline_metric",
    "_kpi",
    "_metric_contract",
    "_non_bool_int",
    "_normalized_metric_kpi",
    "_scope",
    "_select_metric",
    "_source_events",
    "_source_metric",
    "_unknown_clock",
]
