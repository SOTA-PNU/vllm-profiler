"""Request-facing and canonical pipeline latency, kept strictly separate.

Request-facing KPIs report what the client observed; pipeline KPIs come only
from unambiguous marker pairs sharing one correlation id and one aligned clock
domain.  Nothing falls back to timestamp proximity, gaps between phases are
never assumed to be wait time, and the marker join also yields the canonical
stage windows.  Reading the raw measured-request row is the one filesystem
access here; it re-verifies size and SHA-256 before parsing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from ...schema import Availability
from ...schema.catalog import DERIVED_LATENCY_METRICS, STAGE_BY_METRIC
from ...schema.records import EventRecord, MetricSample
from ..resources import StageWindow
from .kpi_records import (
    OverviewCalculationError,
    _aggregate_request_kpis,
    _clock,
    _finite_number,
    _is_pipeline_metric,
    _kpi,
    _metric_contract,
    _non_bool_int,
    _normalized_metric_kpi,
    _scope,
    _select_metric,
    _source_events,
    _source_metric,
    _unknown_clock,
)

_MEASURED_REQUESTS_PATH = "raw/client/measured_requests.jsonl"


def _reject_duplicate_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
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
            if artifact.relative_path != _MEASURED_REQUESTS_PATH:
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


_REQUEST_FACING_FORMULAS = (
    ("latency.e2e", "response_done_ns - request_received_ns"),
    ("latency.ttft", "first_token_ns - request_received_ns"),
    ("latency.tpot", "(last_token_ns - first_token_ns) / (output_tokens - 1)"),
)
_RAW_LATENCY_FIELDS = {
    "latency.e2e": "e2e_ns",
    "latency.ttft": "ttft_ns",
    "latency.tpot": "tpot_ns",
}


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
                for name, formula in _REQUEST_FACING_FORMULAS
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
            formula="(last_token_ns - first_token_ns) / (output_tokens - 1)",
            clock=_clock(loaded, [tpot] if tpot is not None else [e2e]),
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
            raw_field = _RAW_LATENCY_FIELDS[kpi["name"]]
            raw_value = raw_row.get(raw_field)
            if kpi["availability"] == Availability.AVAILABLE.value:
                if _finite_number(raw_value, field=f"raw {raw_field}") != kpi["value"]:
                    raise OverviewCalculationError(
                        f"raw {raw_field} does not match normalized metric"
                    )
            kpi["sources"].append(raw_source)
    return kpis


_PAIRINGS = tuple(
    (
        metric_name,
        STAGE_BY_METRIC[metric_name].start_event,
        STAGE_BY_METRIC[metric_name].end_event,
        STAGE_BY_METRIC[metric_name].window,
    )
    for metric_name in DERIVED_LATENCY_METRICS
)
_SAMPLING_EVENTS = {"sampling_start", "sampling_end"}
_MARKER_EVENT_NAMES = {
    name for _, start, end, _ in _PAIRINGS for name in (start, end)
} | _SAMPLING_EVENTS
_STAGE_WINDOW_SPECS = (
    ("prefill", "latency.prefill", "latency.prefill"),
    ("transfer", "latency.kv_export", "latency.kv_transform"),
    ("decode", "latency.decode", "latency.decode"),
)
_NO_STAGE_WINDOW = "no valid canonical stage window"


def _correlation_id(event: EventRecord) -> str:
    value = event.attributes.get("hybrid.correlation_id")
    if not isinstance(value, str) or not value:
        raise OverviewCalculationError(
            f"{event.event_name} lacks an explicit hybrid.correlation_id"
        )
    return value


def _correlated_events(events: Sequence[EventRecord]) -> tuple[str, list[EventRecord]]:
    relevant = [event for event in events if event.event_name in _MARKER_EVENT_NAMES]
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
    clock = _clock(loaded, records) if records else _unknown_clock()
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
    correlations = sorted(
        {
            _correlation_id(event)
            for event in events
            if event.event_name in _MARKER_EVENT_NAMES
        }
    )
    if len(correlations) > 1:
        rows = []
        for correlation in correlations:
            selected_events = [
                event
                for event in events
                if event.event_name in _MARKER_EVENT_NAMES
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
    if not any(event.event_name in _MARKER_EVENT_NAMES for event in events):
        unavailable = [
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
                clock=_unknown_clock(),
            )
            for name, start_name, end_name, phase in _PAIRINGS
        ]
        unavailable.append(
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
                formula="sum(sampling_end[step] - sampling_start[step])",
                clock=_unknown_clock(),
            )
        )
        unavailable.append(
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
                clock=_unknown_clock(),
            )
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

    def _sampling_pairs(
        selected: Sequence[EventRecord],
    ) -> tuple[list[tuple[EventRecord, EventRecord]], set[int], str | None]:
        sampling_events = [
            event for event in selected if event.event_name in _SAMPLING_EVENTS
        ]
        indices: set[int] = set()
        invalid_step = False
        for event in sampling_events:
            value = event.attributes.get("decode.step_index")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                invalid_step = True
                break
            indices.add(value)
        pairs: list[tuple[EventRecord, EventRecord]] = []
        if invalid_step or not indices:
            return pairs, indices, (
                "sampling markers require explicit non-negative decode.step_index"
            )
        if sorted(indices) != list(range(max(indices) + 1)):
            return pairs, indices, "sampling decode.step_index values are not contiguous"
        for index in sorted(indices):
            start, end, reason = _pair(
                selected,
                "sampling_start",
                "sampling_end",
                step_index=index,
            )
            if reason is not None or start is None or end is None:
                return pairs, indices, reason
            pairs.append((start, end))
        for previous, current in zip(pairs, pairs[1:]):
            if current[0].timestamp_ns < previous[1].timestamp_ns:
                return pairs, indices, "sampling intervals overlap"
        return pairs, indices, None

    sampling_events = [
        event for event in selected if event.event_name in _SAMPLING_EVENTS
    ]
    sampling_pairs, indices, sampling_reason = _sampling_pairs(selected)
    sampling_records = [
        event for pair in sampling_pairs for event in pair
    ] or sampling_events
    sampling_clock = (
        _clock(loaded, sampling_records) if sampling_records else _unknown_clock()
    )
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
    sampling = _kpi(
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
    result.append(sampling)
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
            clock=_clock(loaded, selected) if selected else _unknown_clock(),
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
    windows: list[StageWindow] = []
    for stage, start_pair_name, end_pair_name in _STAGE_WINDOW_SPECS:
        start_pair = pairs.get(start_pair_name)
        end_pair = pairs.get(end_pair_name)
        start_kpi = by_name.get(start_pair_name)
        end_kpi = by_name.get(end_pair_name)
        reason: str | None = None
        if start_pair is None or end_pair is None:
            reason = _NO_STAGE_WINDOW
        elif (
            start_kpi is None
            or end_kpi is None
            or start_kpi.get("availability") != Availability.AVAILABLE.value
            or end_kpi.get("availability") != Availability.AVAILABLE.value
        ):
            reason = _NO_STAGE_WINDOW

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
                reason = _NO_STAGE_WINDOW
            elif start.clock_domain_id != end.clock_domain_id:
                reason = _NO_STAGE_WINDOW
            elif end.timestamp_ns <= start.timestamp_ns:
                reason = _NO_STAGE_WINDOW
            elif _clock(loaded, (start, end))["alignment_status"] != "aligned":
                reason = _NO_STAGE_WINDOW
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


__all__ = [
    "_canonical_stage_windows",
    "_correlation_id",
    "_pipeline_latency",
    "_request_facing_latency",
    "union_duration_ns",
]
