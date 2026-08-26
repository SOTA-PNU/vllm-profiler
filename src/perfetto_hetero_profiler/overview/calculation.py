"""Deterministic Overview KPI calculation from a validated normalized run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..schema import Availability
from ..schema.catalog import METRIC_CATALOG
from ..schema.catalog import DERIVED_LATENCY_METRICS, STAGE_BY_METRIC
from ..schema.records import EventRecord, MetricSample
from .resources import ResourceCalculationError, StageWindow, summarize_resources


class OverviewCalculationError(ValueError):
    """Raised when KPI provenance is contradictory or unsafe."""


_PAIRINGS = tuple(
    (
        metric_name,
        STAGE_BY_METRIC[metric_name].start_event,
        STAGE_BY_METRIC[metric_name].end_event,
        STAGE_BY_METRIC[metric_name].window,
    )
    for metric_name in DERIVED_LATENCY_METRICS
)


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


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


def _display_rule(unit: str) -> dict[str, object]:
    rules: dict[str, tuple[str, int, int, int]] = {
        "ns": ("ms", 1, 1_000_000, 3),
        "bytes": ("MiB", 1, 1_048_576, 3),
        "bytes/s": ("MiB/s", 1, 1_048_576, 3),
        "requests": ("requests", 1, 1, 0),
        "requests/s": ("requests/s", 1, 1, 3),
        "tokens": ("tokens", 1, 1, 0),
        "tokens/s": ("tokens/s", 1, 1, 3),
        "ratio": ("percent", 100, 1, 2),
    }
    display_unit, numerator, denominator, places = rules.get(
        unit, (unit, 1, 1, 6)
    )
    return {
        "unit": display_unit,
        "scale_numerator": numerator,
        "scale_denominator": denominator,
        "decimal_places": places,
        "rounding": "half_even",
    }


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
        "display": _display_rule(canonical_unit),
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
            clock={
                "domain_ids": [],
                "alignment_status": "unknown",
                "alignment_method": None,
                "offset_ns": None,
                "uncertainty_ns": None,
            },
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


def _reject_duplicate_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _raw_request_provenance(
    loaded: object, request_id: str
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Load the validated measured-request row without exposing an absolute path."""

    for source in getattr(loaded, "sources", ()):
        for artifact in getattr(source, "artifacts", ()):
            if artifact.relative_path != "raw/client/measured_requests.jsonl":
                continue
            root = Path(source.root)
            path = root / artifact.relative_path
            if path.is_symlink() or not path.is_file():
                raise OverviewCalculationError(
                    "measured request artifact is missing or is a symlink"
                )
            payload = path.read_bytes()
            if artifact.size_bytes is not None and len(payload) != artifact.size_bytes:
                raise OverviewCalculationError(
                    "measured request artifact size changed after validation"
                )
            if artifact.sha256 is not None:
                digest = hashlib.sha256(payload).hexdigest()
                if digest != artifact.sha256:
                    raise OverviewCalculationError(
                        "measured request artifact hash changed after validation"
                    )
            matches: list[dict[str, object]] = []
            try:
                for line in payload.decode("utf-8").splitlines():
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_object,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON number {token}")
                        ),
                    )
                    if not isinstance(row, dict):
                        raise OverviewCalculationError(
                            "measured request JSONL row must be an object"
                        )
                    if row.get("client_request_id") == request_id:
                        matches.append(row)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                raise OverviewCalculationError(
                    "measured request artifact is not valid UTF-8 JSONL"
                ) from exc
            if len(matches) > 1:
                raise OverviewCalculationError(
                    "measured request artifact has duplicate request rows"
                )
            root_id = None
            for fingerprint in getattr(loaded, "root_fingerprints", ()):
                if Path(fingerprint.root) == root:
                    root_id = fingerprint.root_id
                    break
            provenance = {
                "source_kind": "raw_measured_request",
                "record_ids": [request_id],
                "metric_names": [],
                "root_id": root_id,
                "relative_path": artifact.relative_path,
                "details": {
                    "artifact_id": artifact.artifact_id,
                    "artifact_sha256": artifact.sha256,
                },
            }
            return (matches[0] if matches else None), provenance
    return None, None


def _request_facing_latency(
    loaded: object, metrics: Sequence[MetricSample]
) -> list[dict[str, object]]:
    request_ids = sorted(
        {
            metric.request_id
            for metric in metrics
            if metric.metric_name == "latency.e2e"
            and not _is_pipeline_metric(metric)
            and isinstance(metric.request_id, str)
        }
    )
    if len(request_ids) > 1:
        rows = [
            _request_facing_latency(
                loaded,
                [metric for metric in metrics if metric.request_id == request_id],
            )
            for request_id in request_ids
        ]
        return _aggregate_request_kpis(loaded, rows, request_ids)
    e2e = _select_metric(metrics, "latency.e2e", pipeline=False)
    if e2e is None or not isinstance(e2e.request_id, str):
        candidates = [
            metric
            for metric in metrics
            if metric.metric_name == "latency.e2e"
            and not _is_pipeline_metric(metric)
        ]
        if not candidates:
            return [
                _normalized_metric_kpi(
                    loaded,
                    None,
                    name=name,
                    observation_layer="request_facing_client",
                    formula=formula,
                )
                for name, formula in (
                    ("latency.e2e", "response_done_ns - request_received_ns"),
                    ("latency.ttft", "first_token_ns - request_received_ns"),
                    (
                        "latency.tpot",
                        "(last_token_ns - first_token_ns) / (output_tokens - 1)",
                    ),
                )
            ]
        raise OverviewCalculationError(
            "request-facing latency.e2e requires an explicit request_id"
        )
    request_id = e2e.request_id
    ttft = _select_metric(
        metrics, "latency.ttft", request_id=request_id, pipeline=False
    )
    tpot = _select_metric(
        metrics, "latency.tpot", request_id=request_id, pipeline=False
    )
    output_metric = _select_metric(
        metrics, "request.output_tokens", request_id=request_id
    )
    output_tokens: int | None = None
    if output_metric is not None:
        output_value = _metric_contract(output_metric, "request.output_tokens")
        if output_value is not None:
            output_tokens = _non_bool_int(
                output_value, field="request.output_tokens"
            )

    raw_row, raw_source = _raw_request_provenance(loaded, request_id)
    raw_warning = (
        "raw per-token timestamps are unavailable; normalized latency metric "
        "provenance was retained"
    )
    if raw_row is not None:
        raw_warning += " and aggregate values were reconciled"
    kpis = [
        _normalized_metric_kpi(
            loaded,
            e2e,
            name="latency.e2e",
            observation_layer="request_facing_client",
            formula="response_done_ns - request_received_ns",
            warning=raw_warning,
        ),
        _normalized_metric_kpi(
            loaded,
            ttft,
            name="latency.ttft",
            observation_layer="request_facing_client",
            formula="first_token_ns - request_received_ns",
            warning=raw_warning,
        ),
    ]
    tpot_timestamp_provenance = bool(
        tpot is not None
        and tpot.source_event_ids is not None
        and len(tpot.source_event_ids) >= 2
    )
    if (
        output_tokens is None
        or output_tokens <= 1
        or not tpot_timestamp_provenance
    ):
        clock = _clock(loaded, [tpot] if tpot is not None else [e2e])
        tpot_kpi = _kpi(
            name="latency.tpot",
            canonical_unit="ns",
            value=None,
            unavailable_reason=(
                "TPOT requires an explicit output token count of at least two "
                "and first/last token timestamp provenance"
            ),
            aggregation_method="client_token_interval_mean_v1",
            sample_count=0,
            sources=[
                _source_metric(
                    [metric for metric in (tpot, output_metric) if metric is not None]
                )
            ],
            scope=_scope(
                loaded,
                scope_type="request",
                observation_layer="request_facing_client",
                request_id=request_id,
                host_id=e2e.host_id,
            ),
            calculation_method="tpot_v1",
            formula=(
                "(last_token_ns - first_token_ns) / (output_tokens - 1)"
            ),
            clock=clock,
        )
    else:
        tpot_kpi = _normalized_metric_kpi(
            loaded,
            tpot,
            name="latency.tpot",
            observation_layer="request_facing_client",
            formula="(last_token_ns - first_token_ns) / (output_tokens - 1)",
            warning=raw_warning,
        )
    kpis.append(tpot_kpi)

    if raw_row is not None and raw_source is not None:
        raw_hash = raw_row.get("client_request_hash")
        marker_hashes = {
            event.attributes.get("proxy.client_request_id_hash")
            for event in getattr(loaded, "events", ())
            if event.event_name == "request_received"
            and isinstance(
                event.attributes.get("proxy.client_request_id_hash"), str
            )
        }
        if marker_hashes and (
            not isinstance(raw_hash, str) or raw_hash not in marker_hashes
        ):
            raise OverviewCalculationError(
                "raw client request does not match an explicit pipeline request hash"
            )
        if isinstance(raw_hash, str) and raw_hash in marker_hashes:
            raw_source["details"]["pipeline_link"] = {
                "method": "client_request_hash",
                "value": raw_hash,
            }
        raw_fields = {
            "latency.e2e": "e2e_ns",
            "latency.ttft": "ttft_ns",
            "latency.tpot": "tpot_ns",
        }
        start = raw_row.get("start_monotonic_ns")
        end = raw_row.get("end_monotonic_ns")
        if start is not None and end is not None:
            start_ns = _non_bool_int(start, field="raw start_monotonic_ns")
            end_ns = _non_bool_int(end, field="raw end_monotonic_ns")
            if end_ns - start_ns != _finite_number(
                raw_row.get("e2e_ns"), field="raw e2e_ns"
            ):
                raise OverviewCalculationError(
                    "raw request E2E does not match raw start/end timestamps"
                )
        for kpi in kpis:
            raw_field = raw_fields[kpi["name"]]
            raw_value = raw_row.get(raw_field)
            if kpi["availability"] == Availability.AVAILABLE.value:
                if _finite_number(raw_value, field=f"raw {raw_field}") != kpi["value"]:
                    raise OverviewCalculationError(
                        f"raw {raw_field} does not match normalized metric"
                    )
            kpi["sources"].append(raw_source)
    return kpis


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
                    window="measured_smoke",
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


def _correlation_id(event: EventRecord) -> str:
    value = event.attributes.get("hybrid.correlation_id")
    if not isinstance(value, str) or not value:
        raise OverviewCalculationError(
            f"{event.event_name} lacks an explicit hybrid.correlation_id"
        )
    return value


def _correlated_events(events: Sequence[EventRecord]) -> tuple[str, list[EventRecord]]:
    relevant_names = {
        name for _, start, end, _ in _PAIRINGS for name in (start, end)
    } | {"sampling_start", "sampling_end"}
    relevant = [event for event in events if event.event_name in relevant_names]
    if not relevant:
        raise OverviewCalculationError("no canonical hybrid runtime markers found")
    by_correlation: dict[str, list[EventRecord]] = {}
    for event in relevant:
        correlation = _correlation_id(event)
        by_correlation.setdefault(correlation, []).append(event)
    if len(by_correlation) != 1:
        raise OverviewCalculationError(
            "Overview requires exactly one explicit correlated pipeline request"
        )
    correlation = next(iter(by_correlation))
    return correlation, by_correlation[correlation]


def _pair(
    events: Sequence[EventRecord],
    start_name: str,
    end_name: str,
    *,
    step_index: int | None = None,
) -> tuple[EventRecord | None, EventRecord | None, str | None]:
    def matches(event: EventRecord, name: str) -> bool:
        if event.event_name != name:
            return False
        if step_index is None:
            return True
        return event.attributes.get("decode.step_index") == step_index

    starts = [event for event in events if matches(event, start_name)]
    ends = [event for event in events if matches(event, end_name)]
    if len(starts) != 1 or len(ends) != 1:
        return (
            starts[0] if len(starts) == 1 else None,
            ends[0] if len(ends) == 1 else None,
            (
                f"marker pair {start_name}/{end_name} is "
                f"missing or ambiguous ({len(starts)} starts, {len(ends)} ends)"
            ),
        )
    start, end = starts[0], ends[0]
    if end.timestamp_ns < start.timestamp_ns:
        return start, end, f"marker pair {start_name}/{end_name} is reversed"
    return start, end, None


def union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Return interval union duration, rejecting bool and reversed endpoints."""

    normalized: list[tuple[int, int]] = []
    for start, end in intervals:
        start_ns = _non_bool_int(start, field="interval start")
        end_ns = _non_bool_int(end, field="interval end")
        if end_ns < start_ns:
            raise OverviewCalculationError("wait interval is reversed")
        normalized.append((start_ns, end_ns))
    if not normalized:
        return 0
    normalized.sort()
    total = 0
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _pipeline_kpi(
    loaded: object,
    *,
    name: str,
    phase: str,
    start: EventRecord | None,
    end: EventRecord | None,
    reason: str | None,
    correlation: str,
    normalized_metric: MetricSample | None,
) -> dict[str, object]:
    records = [event for event in (start, end) if event is not None]
    clock = _clock(loaded, records) if records else {
        "domain_ids": [],
        "alignment_status": "unknown",
        "alignment_method": None,
        "offset_ns": None,
        "uncertainty_ns": None,
    }
    value: int | None = None
    if reason is None and start is not None and end is not None:
        if start.clock_domain_id != end.clock_domain_id:
            reason = "markers use different clock domains without a direct transform"
        elif clock["alignment_status"] != "aligned":
            reason = "canonical clock alignment evidence is incomplete"
        else:
            value = end.timestamp_ns - start.timestamp_ns
    if normalized_metric is not None:
        normalized_value = _metric_contract(normalized_metric, name)
        if value is not None and normalized_value != value:
            raise OverviewCalculationError(
                f"{name} canonical marker duration disagrees with normalized metric"
            )
    sources = [_source_events(records, details={"correlation_id": correlation})]
    if normalized_metric is not None:
        sources.append(_source_metric([normalized_metric]))
    host = start.host_id if start is not None else (end.host_id if end else None)
    return _kpi(
        name=name,
        canonical_unit="ns",
        value=value,
        unavailable_reason=reason,
        aggregation_method="canonical_marker_pair_v1",
        sample_count=1 if value is not None else 0,
        sources=sources,
        scope=_scope(
            loaded,
            scope_type="request",
            observation_layer="hybrid_pipeline",
            request_id=correlation,
            host_id=host,
            phase=phase,
        ),
        calculation_method="marker_duration_v1",
        formula=f"{end.event_name if end else 'end'} - "
        f"{start.event_name if start else 'start'}",
        clock=clock,
    )


def _pipeline_latency(
    loaded: object, metrics: Sequence[MetricSample], events: Sequence[EventRecord]
) -> tuple[list[dict[str, object]], dict[str, tuple[EventRecord, EventRecord]]]:
    relevant_names = {
        name for _, start, end, _ in _PAIRINGS for name in (start, end)
    } | {"sampling_start", "sampling_end"}
    correlations = sorted(
        {
            _correlation_id(event)
            for event in events
            if event.event_name in relevant_names
        }
    )
    if len(correlations) > 1:
        rows = []
        for correlation in correlations:
            selected_events = [
                event
                for event in events
                if event.event_name in relevant_names
                and _correlation_id(event) == correlation
            ]
            selected_metrics = [
                metric
                for metric in metrics
                if metric.request_id == correlation
            ]
            row, _ = _pipeline_latency(
                loaded, selected_metrics, selected_events
            )
            rows.append(row)
        return _aggregate_request_kpis(loaded, rows, correlations), {}
    if not any(event.event_name in relevant_names for event in events):
        clock = {
            "domain_ids": [],
            "alignment_status": "unknown",
            "alignment_method": None,
            "offset_ns": None,
            "uncertainty_ns": None,
        }
        unavailable = []
        for name, start_name, end_name, phase in _PAIRINGS:
            unavailable.append(
                _kpi(
                    name=name,
                    canonical_unit="ns",
                    value=None,
                    unavailable_reason="canonical hybrid runtime markers are absent",
                    aggregation_method="canonical_marker_pair_v1",
                    sample_count=0,
                    sources=[],
                    scope=_scope(
                        loaded,
                        scope_type="request",
                        observation_layer="hybrid_pipeline",
                        phase=phase,
                    ),
                    calculation_method="marker_duration_v1",
                    formula=f"{end_name} - {start_name}",
                    clock=clock,
                )
            )
        unavailable.extend(
            [
                _kpi(
                    name="latency.sampling",
                    canonical_unit="ns",
                    value=None,
                    unavailable_reason="canonical hybrid runtime markers are absent",
                    aggregation_method="sum_explicit_sampling_pairs_v1",
                    sample_count=0,
                    sources=[],
                    scope=_scope(
                        loaded,
                        scope_type="request",
                        observation_layer="hybrid_pipeline",
                        phase="sampling",
                    ),
                    calculation_method="sampling_pair_sum_v1",
                    formula=(
                        "sum(sampling_end[step] - sampling_start[step])"
                    ),
                    clock=clock,
                ),
                _kpi(
                    name="latency.wait",
                    canonical_unit="ns",
                    value=None,
                    unavailable_reason="explicit classified wait intervals are absent",
                    aggregation_method="interval_union_v1",
                    sample_count=0,
                    sources=[],
                    scope=_scope(
                        loaded,
                        scope_type="request",
                        observation_layer="hybrid_pipeline",
                        phase="request",
                    ),
                    calculation_method="wait_interval_union_v1",
                    formula="union duration of explicit wait intervals",
                    clock=clock,
                ),
            ]
        )
        return unavailable, {}
    correlation, selected = _correlated_events(events)
    result: list[dict[str, object]] = []
    pairs: dict[str, tuple[EventRecord, EventRecord]] = {}
    for name, start_name, end_name, phase in _PAIRINGS:
        start, end, reason = _pair(selected, start_name, end_name)
        normalized = _select_metric(
            metrics, name, request_id=correlation, pipeline=True
        )
        result.append(
            _pipeline_kpi(
                loaded,
                name=name,
                phase=phase,
                start=start,
                end=end,
                reason=reason,
                correlation=correlation,
                normalized_metric=normalized,
            )
        )
        if reason is None and start is not None and end is not None:
            pairs[name] = (start, end)

    sampling_events = [
        event
        for event in selected
        if event.event_name in {"sampling_start", "sampling_end"}
    ]
    indices: set[int] = set()
    invalid_step = False
    for event in sampling_events:
        value = event.attributes.get("decode.step_index")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid_step = True
            break
        indices.add(value)
    sampling_reason: str | None = None
    sampling_pairs: list[tuple[EventRecord, EventRecord]] = []
    if invalid_step or not indices:
        sampling_reason = (
            "sampling markers require explicit non-negative decode.step_index"
        )
    elif sorted(indices) != list(range(max(indices) + 1)):
        sampling_reason = "sampling decode.step_index values are not contiguous"
    else:
        for index in sorted(indices):
            start, end, reason = _pair(
                selected,
                "sampling_start",
                "sampling_end",
                step_index=index,
            )
            if reason is not None or start is None or end is None:
                sampling_reason = reason
                break
            sampling_pairs.append((start, end))
        if sampling_reason is None:
            for previous, current in zip(sampling_pairs, sampling_pairs[1:]):
                if current[0].timestamp_ns < previous[1].timestamp_ns:
                    sampling_reason = "sampling intervals overlap"
                    break
    sampling_records = [
        event for pair in sampling_pairs for event in pair
    ] or sampling_events
    sampling_clock = _clock(loaded, sampling_records) if sampling_records else {
        "domain_ids": [],
        "alignment_status": "unknown",
        "alignment_method": None,
        "offset_ns": None,
        "uncertainty_ns": None,
    }
    sampling_value: int | None = None
    if sampling_reason is None:
        if sampling_clock["alignment_status"] != "aligned":
            sampling_reason = "canonical clock alignment evidence is incomplete"
        else:
            sampling_value = sum(
                end.timestamp_ns - start.timestamp_ns
                for start, end in sampling_pairs
            )
    normalized_sampling = _select_metric(
        metrics, "latency.sampling", request_id=correlation, pipeline=True
    )
    sampling_sources = [
        _source_events(
            sampling_records,
            details={
                "correlation_id": correlation,
                "decode_step_indices": sorted(indices),
            },
        )
    ]
    warnings: list[str] = []
    if normalized_sampling is not None:
        _metric_contract(normalized_sampling, "latency.sampling")
        sampling_sources.append(_source_metric([normalized_sampling]))
        if (
            sampling_value is not None
            and normalized_sampling.value != sampling_value
        ):
            warnings.append(
                "normalized latency.sampling records only the first repeated "
                "marker pair; Overview sums every explicit decode.step_index pair"
            )
    result.append(
        _kpi(
            name="latency.sampling",
            canonical_unit="ns",
            value=sampling_value,
            unavailable_reason=sampling_reason,
            aggregation_method="sum_explicit_sampling_pairs_v1",
            sample_count=len(sampling_pairs) if sampling_value is not None else 0,
            sources=sampling_sources,
            scope=_scope(
                loaded,
                scope_type="request",
                observation_layer="hybrid_pipeline",
                request_id=correlation,
                phase="sampling",
            ),
            calculation_method="sampling_pair_sum_v1",
            formula=(
                "sum(sampling_end[step] - sampling_start[step]) over "
                "contiguous decode.step_index values"
            ),
            clock=sampling_clock,
            warnings=warnings,
        )
    )
    result.append(
        _kpi(
            name="latency.wait",
            canonical_unit="ns",
            value=None,
            unavailable_reason=(
                "no explicit classified wait intervals are present; gaps between "
                "phases are not assumed to be wait time"
            ),
            aggregation_method="interval_union_v1",
            sample_count=0,
            sources=[],
            scope=_scope(
                loaded,
                scope_type="request",
                observation_layer="hybrid_pipeline",
                request_id=correlation,
                phase="request",
            ),
            calculation_method="wait_interval_union_v1",
            formula="union duration of explicit wait intervals",
            clock=(
                _clock(loaded, selected)
                if selected
                else {
                    "domain_ids": [],
                    "alignment_status": "unknown",
                    "alignment_method": None,
                    "offset_ns": None,
                    "uncertainty_ns": None,
                }
            ),
        )
    )
    return result, pairs


def _canonical_stage_windows(
    loaded: object,
    pipeline: Sequence[dict[str, object]],
    pairs: dict[str, tuple[EventRecord, EventRecord]],
) -> tuple[StageWindow, ...]:
    """Build the three required windows from the already-validated marker join."""

    by_name = {str(item.get("name")): item for item in pipeline}
    specifications = (
        ("prefill", "latency.prefill", "latency.prefill"),
        ("transfer", "latency.kv_export", "latency.kv_transform"),
        ("decode", "latency.decode", "latency.decode"),
    )
    windows: list[StageWindow] = []
    for stage, start_pair_name, end_pair_name in specifications:
        start_pair = pairs.get(start_pair_name)
        end_pair = pairs.get(end_pair_name)
        start_kpi = by_name.get(start_pair_name)
        end_kpi = by_name.get(end_pair_name)
        reason: str | None = None
        if start_pair is None or end_pair is None:
            reason = "no valid canonical stage window"
        elif (
            start_kpi is None
            or end_kpi is None
            or start_kpi.get("availability") != Availability.AVAILABLE.value
            or end_kpi.get("availability") != Availability.AVAILABLE.value
        ):
            reason = "no valid canonical stage window"

        start = start_pair[0] if start_pair is not None else None
        end = end_pair[1] if end_pair is not None else None
        correlation: str | None = None
        clock_domain: str | None = None
        host_ids: tuple[str, ...] = ()
        marker_ids: tuple[str, ...] = ()
        if reason is None and start is not None and end is not None:
            start_correlation = _correlation_id(start)
            end_correlation = _correlation_id(end)
            if start_correlation != end_correlation:
                reason = "no valid canonical stage window"
            elif start.clock_domain_id != end.clock_domain_id:
                reason = "no valid canonical stage window"
            elif end.timestamp_ns <= start.timestamp_ns:
                reason = "no valid canonical stage window"
            elif _clock(loaded, (start, end))["alignment_status"] != "aligned":
                reason = "no valid canonical stage window"
            else:
                correlation = start_correlation
                clock_domain = start.clock_domain_id
                host_ids = tuple(sorted({start.host_id, end.host_id}))
                marker_ids = (start.event_id, end.event_id)
        windows.append(
            StageWindow(
                phase=stage,
                window=stage,
                request_id=correlation,
                start_ns=start.timestamp_ns if start is not None else None,
                end_ns=end.timestamp_ns if end is not None else None,
                clock_domain_id=clock_domain,
                host_ids=host_ids,
                marker_event_ids=marker_ids,
                unavailable_reason=reason,
            )
        )
    return tuple(windows)


def _throughput_and_tokens(
    loaded: object, metrics: Sequence[MetricSample]
) -> list[dict[str, object]]:
    count_names = (
        "request.count",
        "request.input_tokens",
        "request.output_tokens",
        "request.total_tokens",
    )
    rate_names = (
        "throughput.requests",
        "throughput.input_tokens",
        "throughput.output_tokens",
        "throughput.total_tokens",
    )
    candidates_by_name = {
        name: [
            metric
            for metric in metrics
            if metric.metric_name == name
            and (
                metric.dimensions.get("window") == "measured_smoke"
                or metric.attributes.get("vllm.measurement_window")
                == "measured_smoke"
            )
        ]
        for name in (*count_names, *rate_names)
    }
    if any(len(items) > 1 for items in candidates_by_name.values()):
        return _multi_request_throughput_and_tokens(
            loaded,
            candidates_by_name,
            count_names=count_names,
            rate_names=rate_names,
        )
    selected: dict[str, MetricSample | None] = {}
    for name in (*count_names, *rate_names):
        candidates = candidates_by_name[name]
        if len(candidates) > 1:
            raise OverviewCalculationError(
                f"{name} has ambiguous measured_smoke metric provenance"
            )
        selected[name] = candidates[0] if candidates else None

    missing = [name for name, metric in selected.items() if metric is None]
    if len(missing) == len(selected):
        formulas = {
            **{name: "measured count" for name in count_names},
            **{
                name: "count * 1_000_000_000 / window_duration_ns"
                for name in rate_names
            },
        }
        return [
            _normalized_metric_kpi(
                loaded,
                selected[name],
                name=name,
                observation_layer="request_facing_client",
                formula=formulas[name],
            )
            for name in (*count_names, *rate_names)
        ]
    if missing:
        raise OverviewCalculationError(
            f"{missing[0]} requires exactly one measured_smoke metric"
        )

    values: dict[str, int | float] = {}
    for name, metric in selected.items():
        if metric is None:  # pragma: no cover - branch above
            raise OverviewCalculationError("measured_smoke selection changed")
        value = _metric_contract(metric, name)
        if value is None:
            raise OverviewCalculationError(
                f"{name} measured_smoke metric is unavailable"
            )
        values[name] = value
    if values["request.input_tokens"] + values["request.output_tokens"] != values[
        "request.total_tokens"
    ]:
        raise OverviewCalculationError("request.total_tokens reconciliation failed")

    intervals = {
        selected[name].interval_ns
        for name in ("request.count", *rate_names)
        if selected[name] is not None
    }
    if len(intervals) != 1:
        raise OverviewCalculationError(
            "measured_smoke throughput metrics disagree on window duration"
        )
    interval = intervals.pop()
    interval_ns = _non_bool_int(interval, field="measured_smoke interval_ns")
    if interval_ns <= 0:
        raise OverviewCalculationError(
            "measured_smoke throughput window must be positive"
        )
    interval_seconds = interval_ns / 1_000_000_000
    expected = {
        "throughput.requests": values["request.count"] / interval_seconds,
        "throughput.input_tokens": values["request.input_tokens"] / interval_seconds,
        "throughput.output_tokens": values["request.output_tokens"]
        / interval_seconds,
        "throughput.total_tokens": values["request.total_tokens"]
        / interval_seconds,
    }
    for name, recomputed in expected.items():
        if values[name] != recomputed:
            raise OverviewCalculationError(
                f"{name} does not equal count / measured_smoke duration"
            )

    warning = (
        "measured_smoke contains one request; this observation is not a "
        "generalizable throughput benchmark"
    )
    result = []
    for name in (*count_names, *rate_names):
        metric = selected[name]
        if metric is None:  # pragma: no cover - branch above
            raise OverviewCalculationError("measured_smoke selection changed")
        result.append(
            _kpi(
                name=name,
                canonical_unit=METRIC_CATALOG[name].unit,
                value=values[name],
                unavailable_reason=None,
                aggregation_method="measured_smoke_window_v1",
                sample_count=1,
                sources=[
                    _source_metric(
                        [metric],
                        details={
                            "window": "measured_smoke",
                            "window_duration_ns": interval_ns,
                        },
                    )
                ],
                scope=_scope(
                    loaded,
                    scope_type=str(_enum_value(metric.scope)),
                    observation_layer="request_facing_client",
                    request_id=metric.request_id,
                    host_id=metric.host_id,
                    window="measured_smoke",
                ),
                calculation_method=(
                    "measured_smoke_count_v1"
                    if name in count_names
                    else "count_per_window_second_v1"
                ),
                formula=(
                    "measured count"
                    if name in count_names
                    else "count * 1_000_000_000 / window_duration_ns"
                ),
                clock=_clock(loaded, [metric]),
                warnings=(warning,),
            )
        )
    return result


def _multi_request_throughput_and_tokens(
    loaded: object,
    candidates: dict[str, list[MetricSample]],
    *,
    count_names: Sequence[str],
    rate_names: Sequence[str],
) -> list[dict[str, object]]:
    """Reconcile per-request token counts with one measured run window."""

    singleton_names = ("request.count", *rate_names)
    for name in singleton_names:
        if len(candidates[name]) != 1:
            raise OverviewCalculationError(
                f"{name} requires exactly one measured_smoke run metric"
            )
    token_names = (
        "request.input_tokens",
        "request.output_tokens",
        "request.total_tokens",
    )
    request_ids: set[str] | None = None
    count_values: dict[str, int | float] = {}
    for name in token_names:
        rows = candidates[name]
        if not rows:
            raise OverviewCalculationError(
                f"{name} requires measured request metrics"
            )
        ids = {metric.request_id for metric in rows}
        if None in ids or len(ids) != len(rows):
            raise OverviewCalculationError(
                f"{name} has missing or duplicate request provenance"
            )
        typed_ids = {str(item) for item in ids}
        if request_ids is None:
            request_ids = typed_ids
        elif typed_ids != request_ids:
            raise OverviewCalculationError(
                "measured request token metrics disagree on request IDs"
            )
        values = [_metric_contract(metric, name) for metric in rows]
        if any(value is None for value in values):
            raise OverviewCalculationError(
                f"{name} measured request metric is unavailable"
            )
        count_values[name] = sum(value for value in values if value is not None)
    if request_ids is None:  # pragma: no cover - token rows required above
        raise OverviewCalculationError("measured request IDs are unavailable")
    request_count_metric = candidates["request.count"][0]
    request_count = _metric_contract(request_count_metric, "request.count")
    if request_count != len(request_ids):
        raise OverviewCalculationError(
            "request.count does not match measured request provenance"
        )
    count_values["request.count"] = request_count
    if (
        count_values["request.input_tokens"]
        + count_values["request.output_tokens"]
        != count_values["request.total_tokens"]
    ):
        raise OverviewCalculationError("request.total_tokens reconciliation failed")

    rate_metrics = {name: candidates[name][0] for name in rate_names}
    intervals = {
        metric.interval_ns
        for metric in (request_count_metric, *rate_metrics.values())
    }
    if len(intervals) != 1:
        raise OverviewCalculationError(
            "measured_smoke throughput metrics disagree on window duration"
        )
    interval_ns = _non_bool_int(
        intervals.pop(), field="measured_smoke interval_ns"
    )
    if interval_ns <= 0:
        raise OverviewCalculationError(
            "measured_smoke throughput window must be positive"
        )
    interval_seconds = interval_ns / 1_000_000_000
    expected_rates = {
        "throughput.requests": count_values["request.count"] / interval_seconds,
        "throughput.input_tokens": (
            count_values["request.input_tokens"] / interval_seconds
        ),
        "throughput.output_tokens": (
            count_values["request.output_tokens"] / interval_seconds
        ),
        "throughput.total_tokens": (
            count_values["request.total_tokens"] / interval_seconds
        ),
    }
    for name, expected in expected_rates.items():
        actual = _metric_contract(rate_metrics[name], name)
        if actual != expected:
            raise OverviewCalculationError(
                f"{name} does not equal count / measured_smoke duration"
            )

    warning = (
        f"measured_smoke contains {len(request_ids)} requests; this validation "
        "workload is not a generalizable throughput benchmark"
    )
    result: list[dict[str, object]] = []
    for name in (*count_names, *rate_names):
        if name in token_names:
            source_metrics = candidates[name]
            value = count_values[name]
            sample_count = len(source_metrics)
            method = "sum_measured_request_counts_v1"
            formula = "sum(per-request measured count)"
        elif name == "request.count":
            source_metrics = [request_count_metric]
            value = count_values[name]
            sample_count = 1
            method = "measured_smoke_count_v1"
            formula = "measured count"
        else:
            source_metrics = [rate_metrics[name]]
            value = expected_rates[name]
            sample_count = 1
            method = "count_per_window_second_v1"
            formula = "count * 1_000_000_000 / window_duration_ns"
        result.append(
            _kpi(
                name=name,
                canonical_unit=METRIC_CATALOG[name].unit,
                value=value,
                unavailable_reason=None,
                aggregation_method="measured_smoke_window_v1",
                sample_count=sample_count,
                sources=[
                    _source_metric(
                        source_metrics,
                        details={
                            "window": "measured_smoke",
                            "window_duration_ns": interval_ns,
                            "measured_request_count": len(request_ids),
                        },
                    )
                ],
                scope=_scope(
                    loaded,
                    scope_type="run",
                    observation_layer="request_facing_client",
                    window="measured_smoke",
                ),
                calculation_method=method,
                formula=formula,
                clock=_clock(loaded, source_metrics),
                warnings=(warning,),
            )
        )
    return result


def _transfer_kpis(
    loaded: object,
    pipeline: Sequence[dict[str, object]],
    pairs: dict[str, tuple[EventRecord, EventRecord]],
) -> list[dict[str, object]]:
    transfer_pair = pairs.get("latency.kv_transfer")
    pipeline_by_name = {item["name"]: item for item in pipeline}
    pipeline_e2e = pipeline_by_name["latency.e2e"]
    transfer_latency = pipeline_by_name["latency.kv_transfer"]
    transform_latency = pipeline_by_name["latency.kv_transform"]
    records = list(transfer_pair or ())
    correlation = (
        _correlation_id(records[0]) if records else pipeline_e2e["scope"]["request_id"]
    )
    aggregate_run = correlation is None
    scope = _scope(
        loaded,
        scope_type="run" if aggregate_run else "transfer",
        observation_layer="hybrid_pipeline",
        request_id=correlation,
        phase="kv_transfer",
        window="measured_smoke" if aggregate_run else None,
    )
    clock = (
        _clock(loaded, records)
        if records
        else transfer_latency["clock"]
    )
    sources = (
        [_source_events(records, details={"correlation_id": correlation})]
        if records
        else []
    )
    transfer_bytes: int | None = None
    bytes_reason: str | None = None
    if transfer_pair is None:
        bytes_reason = "no unambiguous KV transfer marker pair is available"
    else:
        start, end = transfer_pair
        start_id = start.attributes.get("hybrid.transfer_id")
        end_id = end.attributes.get("hybrid.transfer_id")
        if not isinstance(start_id, str) or not start_id or start_id != end_id:
            raise OverviewCalculationError(
                "KV transfer markers require one equal explicit hybrid.transfer_id"
            )
        raw_start = start.attributes.get("kv.transfer_bytes")
        raw_end = end.attributes.get("kv.transfer_bytes")
        if raw_start is None or raw_end is None:
            bytes_reason = "KV transfer markers do not contain a byte count"
        else:
            start_bytes = _non_bool_int(raw_start, field="kv.transfer_bytes")
            end_bytes = _non_bool_int(raw_end, field="kv.transfer_bytes")
            if start_bytes < 0 or end_bytes < 0:
                raise OverviewCalculationError(
                    "kv.transfer_bytes must be non-negative"
                )
            if start_bytes != end_bytes:
                raise OverviewCalculationError(
                    "KV transfer start/end byte counts disagree"
                )
            transfer_bytes = start_bytes

    bytes_kpi = _kpi(
        name="transfer.bytes",
        canonical_unit="bytes",
        value=transfer_bytes,
        unavailable_reason=bytes_reason,
        aggregation_method=(
            "not_available_across_measured_requests_v1"
            if aggregate_run
            else "equal_transfer_marker_attributes_v1"
        ),
        sample_count=1 if transfer_bytes is not None else 0,
        sources=sources,
        scope=scope,
        calculation_method="transfer_bytes_v1",
        formula="equal kv.transfer_bytes on transfer start and end markers",
        clock=clock,
    )
    duration_value = transfer_latency["value"]
    duration_kpi = _kpi(
        name="transfer.duration",
        canonical_unit="ns",
        value=duration_value,
        unavailable_reason=transfer_latency["unavailable_reason"],
        aggregation_method=(
            "arithmetic_mean_across_measured_requests_v1"
            if aggregate_run
            else "canonical_marker_pair_v1"
        ),
        sample_count=transfer_latency["sample_count"],
        sources=transfer_latency["sources"],
        scope=scope,
        calculation_method="marker_duration_v1",
        formula="kv_transfer_end - kv_transfer_start",
        clock=transfer_latency["clock"],
    )
    if transfer_bytes is None:
        bandwidth = None
        bandwidth_reason = "transfer byte count is unavailable"
    elif duration_value is None:
        bandwidth = None
        bandwidth_reason = "transfer duration is unavailable"
    elif duration_value == 0:
        bandwidth = None
        bandwidth_reason = "transfer duration is zero"
    else:
        bandwidth = transfer_bytes * 1_000_000_000 / duration_value
        bandwidth_reason = None
    bandwidth_kpi = _kpi(
        name="transfer.effective_bandwidth",
        canonical_unit="bytes/s",
        value=bandwidth,
        unavailable_reason=bandwidth_reason,
        aggregation_method=(
            "not_available_across_measured_requests_v1"
            if aggregate_run
            else "bytes_per_transfer_duration_v1"
        ),
        sample_count=1 if bandwidth is not None else 0,
        sources=sources,
        scope=scope,
        calculation_method="effective_bandwidth_v1",
        formula="transfer_bytes * 1_000_000_000 / transfer_duration_ns",
        clock=clock,
    )
    transform_kpi = _kpi(
        name="transfer.transform_duration",
        canonical_unit="ns",
        value=transform_latency["value"],
        unavailable_reason=transform_latency["unavailable_reason"],
        aggregation_method=(
            "arithmetic_mean_across_measured_requests_v1"
            if aggregate_run
            else "canonical_marker_pair_v1"
        ),
        sample_count=transform_latency["sample_count"],
        sources=transform_latency["sources"],
        scope=scope,
        calculation_method="marker_duration_v1",
        formula="kv_transform_end - kv_transform_start",
        clock=transform_latency["clock"],
    )
    e2e_value = pipeline_e2e["value"]
    if duration_value is None:
        share = None
        share_reason = "transfer duration is unavailable"
    elif e2e_value is None:
        share = None
        share_reason = "pipeline E2E duration is unavailable"
    elif e2e_value == 0:
        share = None
        share_reason = "pipeline E2E duration is zero"
    else:
        share = duration_value / e2e_value
        share_reason = None
    share_kpi = _kpi(
        name="transfer.e2e_share",
        canonical_unit="ratio",
        value=share,
        unavailable_reason=share_reason,
        aggregation_method=(
            "ratio_of_measured_request_means_v1"
            if aggregate_run
            else "transfer_to_pipeline_e2e_ratio_v1"
        ),
        sample_count=1 if share is not None else 0,
        sources=transfer_latency["sources"] + pipeline_e2e["sources"],
        scope=scope,
        calculation_method="transfer_e2e_share_v1",
        formula="transfer_duration_ns / pipeline_e2e_ns",
        clock=clock,
    )
    normalized_metrics = tuple(getattr(loaded, "metrics", ()))

    def observability_kpi(
        name: str,
        formula: str,
        *,
        warning: str,
    ) -> dict[str, object]:
        candidates = [
            metric
            for metric in normalized_metrics
            if metric.metric_name == name
            and (
                correlation is None
                or metric.request_id == correlation
            )
        ]
        available_values: list[int | float] = []
        unavailable_reasons: list[str] = []
        for metric in candidates:
            value = _metric_contract(metric, name)
            if value is None:
                unavailable_reasons.append(
                    metric.reason or "normalized interval is unavailable"
                )
            else:
                available_values.append(value)
        fully_available = bool(candidates) and not unavailable_reasons
        value = (
            math.fsum(available_values) / len(available_values)
            if fully_available and available_values
            else None
        )
        reason = None
        if not candidates:
            reason = (
                "runtime marker capability transfer_wait_observability_v1 "
                "or its normalized metric is absent"
            )
        elif unavailable_reasons:
            reason = "; ".join(sorted(set(unavailable_reasons)))
        kpi_scope = scope
        if name == "decode.schedule_wait_duration" and not aggregate_run:
            kpi_scope = _scope(
                loaded,
                scope_type="request",
                observation_layer="hybrid_pipeline",
                request_id=correlation,
                phase="decode",
            )
        return _kpi(
            name=name,
            canonical_unit="ns",
            value=value,
            unavailable_reason=reason,
            aggregation_method=(
                "arithmetic_mean_across_measured_requests_v1"
                if aggregate_run
                else "arithmetic_mean_across_explicit_intervals_v1"
                if len(candidates) > 1
                else "canonical_marker_pair_v1"
            ),
            sample_count=len(available_values),
            sources=(
                [
                    _source_metric(
                        candidates,
                        details={
                            "source_markers": list(
                                METRIC_CATALOG[name].source_events
                            )
                        },
                    )
                ]
                if candidates
                else []
            ),
            scope=kpi_scope,
            calculation_method="explicit_runtime_boundary_duration_v1",
            formula=formula,
            clock=_clock(loaded, candidates) if candidates else clock,
            warnings=(warning,),
        )

    handoff_kpi = observability_kpi(
        "transfer.handoff_duration",
        "kv_handoff_end - kv_handoff_start",
        warning="Handoff covers exported metadata delivery to NIXL setup entry.",
    )
    setup_kpi = observability_kpi(
        "transfer.setup_duration",
        "kv_transfer_setup_end - kv_transfer_setup_start",
        warning=(
            "Setup and transfer/wait intervals can overlap by definition and "
            "must not be summed as total transfer delay."
        ),
    )
    wait_kpi = observability_kpi(
        "transfer.wait_duration",
        "kv_transfer_wait_end - kv_transfer_wait_start",
        warning=(
            "Wait is bounded by host status observations; polling cadence is "
            "not exact device completion time."
        ),
    )
    decode_wait_kpi = observability_kpi(
        "decode.schedule_wait_duration",
        "decode_schedule_wait_end - decode_schedule_wait_start",
        warning="Decode scheduling wait ends at the first actual model step.",
    )
    return [
        bytes_kpi,
        duration_kpi,
        bandwidth_kpi,
        transform_kpi,
        handoff_kpi,
        setup_kpi,
        wait_kpi,
        decode_wait_kpi,
        share_kpi,
    ]


def calculate_overview_kpis(loaded: object) -> dict[str, object]:
    """Calculate every deterministic Overview KPI from one validated run."""

    metrics = tuple(getattr(loaded, "metrics", ()))
    events = tuple(getattr(loaded, "events", ()))
    request_facing = _request_facing_latency(loaded, metrics)
    pipeline, pairs = _pipeline_latency(loaded, metrics, events)
    stage_windows = _canonical_stage_windows(loaded, pipeline, pairs)
    throughput = _throughput_and_tokens(loaded, metrics)
    transfer = _transfer_kpis(loaded, pipeline, pairs)
    try:
        resources = summarize_resources(loaded, stage_windows=stage_windows)
    except ResourceCalculationError as exc:
        raise OverviewCalculationError(str(exc)) from exc
    return {
        "request_facing_latency": request_facing,
        "pipeline_latency": pipeline,
        "throughput_and_tokens": throughput,
        "transfer": transfer,
        "resource_summaries": resources,
    }


__all__ = [
    "OverviewCalculationError",
    "calculate_overview_kpis",
    "union_duration_ns",
]
