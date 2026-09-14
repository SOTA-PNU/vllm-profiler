"""Measured-window throughput, token counts, and KV transfer KPIs.

Counts and rates are recomputed and must reconcile exactly: input + output ==
total, and every rate equals its count over the one shared window duration.
Transfer size comes only from equal explicit byte counts on both markers, and
the overlapping observability intervals each carry a warning against summing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...artifact_compatibility import (
    LEGACY_MEASURED_COUNT_AGGREGATION,
    LEGACY_MEASURED_WINDOW,
    LEGACY_MEASURED_WINDOW_AGGREGATION,
)
from ...schema.metric_catalog import METRIC_CATALOG
from ...schema.records import EventRecord, MetricSample
from .kpi_records import (
    OverviewCalculationError,
    _clock,
    _enum_value,
    _kpi,
    _metric_contract,
    _non_bool_int,
    _normalized_metric_kpi,
    _scope,
    _source_events,
    _source_metric,
)
from .latency import _correlation_id

_COUNT_NAMES = (
    "request.count",
    "request.input_tokens",
    "request.output_tokens",
    "request.total_tokens",
)
_RATE_NAMES = (
    "throughput.requests",
    "throughput.input_tokens",
    "throughput.output_tokens",
    "throughput.total_tokens",
)
_TOKEN_NAMES = (
    "request.input_tokens",
    "request.output_tokens",
    "request.total_tokens",
)
_RATE_BY_COUNT = {
    "throughput.requests": "request.count",
    "throughput.input_tokens": "request.input_tokens",
    "throughput.output_tokens": "request.output_tokens",
    "throughput.total_tokens": "request.total_tokens",
}
_COUNT_FORMULA = "measured count"
_RATE_FORMULA = "count * 1_000_000_000 / window_duration_ns"


def _measured_window_interval(metrics: Sequence[MetricSample]) -> int:
    """Return the single positive measured-window duration shared by rates."""

    intervals = {metric.interval_ns for metric in metrics}
    if len(intervals) != 1:
        raise OverviewCalculationError(
            f"{LEGACY_MEASURED_WINDOW} throughput metrics disagree on "
            "window duration"
        )
    interval_ns = _non_bool_int(
        intervals.pop(),
        field=f"{LEGACY_MEASURED_WINDOW} interval_ns",
    )
    if interval_ns <= 0:
        raise OverviewCalculationError(
            f"{LEGACY_MEASURED_WINDOW} throughput window must be positive"
        )
    return interval_ns


def _expected_rates(
    counts: dict[str, int | float], interval_ns: int
) -> dict[str, float]:
    interval_seconds = interval_ns / 1_000_000_000
    return {
        rate_name: counts[count_name] / interval_seconds
        for rate_name, count_name in _RATE_BY_COUNT.items()
    }


def _window_kpi(
    loaded: object,
    *,
    name: str,
    value: int | float,
    source_metrics: Sequence[MetricSample],
    sample_count: int,
    method: str,
    formula: str,
    scope: dict[str, object],
    details: dict[str, object],
    warning: str,
) -> dict[str, object]:
    """Build one measured-window count or rate KPI."""

    return _kpi(
        name=name,
        canonical_unit=METRIC_CATALOG[name].unit,
        value=value,
        unavailable_reason=None,
        aggregation_method=LEGACY_MEASURED_WINDOW_AGGREGATION,
        sample_count=sample_count,
        sources=[_source_metric(source_metrics, details=details)],
        scope=scope,
        calculation_method=method,
        formula=formula,
        clock=_clock(loaded, source_metrics),
        warnings=(warning,),
    )


def _throughput_and_tokens(
    loaded: object, metrics: Sequence[MetricSample]
) -> list[dict[str, object]]:
    candidates_by_name = {
        name: [
            metric
            for metric in metrics
            if metric.metric_name == name
            and (
                metric.dimensions.get("window") == LEGACY_MEASURED_WINDOW
                or metric.attributes.get("vllm.measurement_window")
                == LEGACY_MEASURED_WINDOW
            )
        ]
        for name in (*_COUNT_NAMES, *_RATE_NAMES)
    }
    if any(len(items) > 1 for items in candidates_by_name.values()):
        return _multi_request_throughput_and_tokens(loaded, candidates_by_name)
    selected: dict[str, MetricSample | None] = {}
    for name in (*_COUNT_NAMES, *_RATE_NAMES):
        candidates = candidates_by_name[name]
        if len(candidates) > 1:
            raise OverviewCalculationError(
                f"{name} has ambiguous {LEGACY_MEASURED_WINDOW} metric provenance"
            )
        selected[name] = candidates[0] if candidates else None

    missing = [name for name, metric in selected.items() if metric is None]
    if len(missing) == len(selected):
        return [
            _normalized_metric_kpi(
                loaded,
                selected[name],
                name=name,
                observation_layer="request_facing_client",
                formula=_COUNT_FORMULA if name in _COUNT_NAMES else _RATE_FORMULA,
            )
            for name in (*_COUNT_NAMES, *_RATE_NAMES)
        ]
    if missing:
        raise OverviewCalculationError(
            f"{missing[0]} requires exactly one {LEGACY_MEASURED_WINDOW} metric"
        )

    values: dict[str, int | float] = {}
    for name, metric in selected.items():
        if metric is None:  # pragma: no cover - branch above
            raise OverviewCalculationError(
                f"{LEGACY_MEASURED_WINDOW} selection changed"
            )
        value = _metric_contract(metric, name)
        if value is None:
            raise OverviewCalculationError(
                f"{name} {LEGACY_MEASURED_WINDOW} metric is unavailable"
            )
        values[name] = value
    if values["request.input_tokens"] + values["request.output_tokens"] != values[
        "request.total_tokens"
    ]:
        raise OverviewCalculationError("request.total_tokens reconciliation failed")

    interval_ns = _measured_window_interval(
        [
            selected[name]
            for name in ("request.count", *_RATE_NAMES)
            if selected[name] is not None
        ]
    )
    for name, recomputed in _expected_rates(values, interval_ns).items():
        if values[name] != recomputed:
            raise OverviewCalculationError(
                f"{name} does not equal count / {LEGACY_MEASURED_WINDOW} duration"
            )

    warning = (
        f"{LEGACY_MEASURED_WINDOW} contains one request; this observation is not a "
        "generalizable throughput benchmark"
    )
    details = {
        "window": LEGACY_MEASURED_WINDOW,
        "window_duration_ns": interval_ns,
    }
    result = []
    for name in (*_COUNT_NAMES, *_RATE_NAMES):
        metric = selected[name]
        if metric is None:  # pragma: no cover - branch above
            raise OverviewCalculationError(
                f"{LEGACY_MEASURED_WINDOW} selection changed"
            )
        counted = name in _COUNT_NAMES
        result.append(
            _window_kpi(
                loaded,
                name=name,
                value=values[name],
                source_metrics=[metric],
                sample_count=1,
                method=(
                    LEGACY_MEASURED_COUNT_AGGREGATION
                    if counted
                    else "count_per_window_second_v1"
                ),
                formula=_COUNT_FORMULA if counted else _RATE_FORMULA,
                scope=_scope(
                    loaded,
                    scope_type=str(_enum_value(metric.scope)),
                    observation_layer="request_facing_client",
                    request_id=metric.request_id,
                    host_id=metric.host_id,
                    window=LEGACY_MEASURED_WINDOW,
                ),
                details=dict(details),
                warning=warning,
            )
        )
    return result


def _multi_request_throughput_and_tokens(
    loaded: object,
    candidates: dict[str, list[MetricSample]],
) -> list[dict[str, object]]:
    """Reconcile per-request token counts with one measured run window."""

    for name in ("request.count", *_RATE_NAMES):
        if len(candidates[name]) != 1:
            raise OverviewCalculationError(
                f"{name} requires exactly one {LEGACY_MEASURED_WINDOW} run metric"
            )
    # Sum per-request token metrics, rejecting missing or duplicate requests.
    request_ids: set[str] | None = None
    count_values: dict[str, int | float] = {}
    for name in _TOKEN_NAMES:
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

    rate_metrics = {name: candidates[name][0] for name in _RATE_NAMES}
    interval_ns = _measured_window_interval(
        [request_count_metric, *rate_metrics.values()]
    )
    expected_rates = _expected_rates(count_values, interval_ns)
    for name, expected in expected_rates.items():
        actual = _metric_contract(rate_metrics[name], name)
        if actual != expected:
            raise OverviewCalculationError(
                f"{name} does not equal count / {LEGACY_MEASURED_WINDOW} duration"
            )

    warning = (
        f"{LEGACY_MEASURED_WINDOW} contains {len(request_ids)} requests; "
        "this validation "
        "workload is not a generalizable throughput benchmark"
    )
    run_scope = _scope(
        loaded,
        scope_type="run",
        observation_layer="request_facing_client",
        window=LEGACY_MEASURED_WINDOW,
    )
    result: list[dict[str, object]] = []
    for name in (*_COUNT_NAMES, *_RATE_NAMES):
        if name in _TOKEN_NAMES:
            source_metrics = candidates[name]
            value = count_values[name]
            sample_count = len(source_metrics)
            method = "sum_measured_request_counts_v1"
            formula = "sum(per-request measured count)"
        elif name == "request.count":
            source_metrics = [request_count_metric]
            value = count_values[name]
            sample_count = 1
            method = LEGACY_MEASURED_COUNT_AGGREGATION
            formula = _COUNT_FORMULA
        else:
            source_metrics = [rate_metrics[name]]
            value = expected_rates[name]
            sample_count = 1
            method = "count_per_window_second_v1"
            formula = _RATE_FORMULA
        result.append(
            _window_kpi(
                loaded,
                name=name,
                value=value,
                source_metrics=source_metrics,
                sample_count=sample_count,
                method=method,
                formula=formula,
                scope=run_scope,
                details={
                    "window": LEGACY_MEASURED_WINDOW,
                    "window_duration_ns": interval_ns,
                    "measured_request_count": len(request_ids),
                },
                warning=warning,
            )
        )
    return result


_OBSERVABILITY_KPIS = (
    (
        "transfer.handoff_duration",
        "kv_handoff_end - kv_handoff_start",
        "Handoff covers exported metadata delivery to NIXL setup entry.",
    ),
    (
        "transfer.setup_duration",
        "kv_transfer_setup_end - kv_transfer_setup_start",
        "Setup and transfer/wait intervals can overlap by definition and "
        "must not be summed as total transfer delay.",
    ),
    (
        "transfer.wait_duration",
        "kv_transfer_wait_end - kv_transfer_wait_start",
        "Wait is bounded by host status observations; polling cadence is "
        "not exact device completion time.",
    ),
    (
        "transfer.device_sync_duration",
        "kv_device_sync_end - kv_device_sync_start",
        "Device sync covers host-buffer to NPU KV-cache synchronization.",
    ),
    (
        "decode.schedule_wait_duration",
        "decode_schedule_wait_end - decode_schedule_wait_start",
        "Decode scheduling wait ends at the first actual model step.",
    ),
)


def _derived_ratio(
    numerator: int | float | None,
    denominator: int | float | None,
    *,
    scale: int,
    numerator_reason: str,
    denominator_reason: str,
    zero_reason: str,
) -> tuple[float | None, str | None]:
    """Divide two KPI values, preserving each distinct unavailable reason."""

    if numerator is None:
        return None, numerator_reason
    if denominator is None:
        return None, denominator_reason
    if denominator == 0:
        return None, zero_reason
    return numerator * scale / denominator, None


def _observability_kpi(
    loaded: object,
    *,
    name: str,
    formula: str,
    warning: str,
    correlation: str | None,
    aggregate_run: bool,
    scope: dict[str, object],
    clock: dict[str, object],
) -> dict[str, object]:
    candidates = [
        metric
        for metric in tuple(getattr(loaded, "metrics", ()))
        if metric.metric_name == name
        and (correlation is None or metric.request_id == correlation)
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
                        "source_markers": list(METRIC_CATALOG[name].source_events)
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
        window=LEGACY_MEASURED_WINDOW if aggregate_run else None,
    )
    clock = _clock(loaded, records) if records else transfer_latency["clock"]
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
    bandwidth, bandwidth_reason = _derived_ratio(
        transfer_bytes,
        duration_value,
        scale=1_000_000_000,
        numerator_reason="transfer byte count is unavailable",
        denominator_reason="transfer duration is unavailable",
        zero_reason="transfer duration is zero",
    )
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
    share, share_reason = _derived_ratio(
        duration_value,
        pipeline_e2e["value"],
        scale=1,
        numerator_reason="transfer duration is unavailable",
        denominator_reason="pipeline E2E duration is unavailable",
        zero_reason="pipeline E2E duration is zero",
    )
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
    observability = [
        _observability_kpi(
            loaded,
            name=name,
            formula=formula,
            warning=warning,
            correlation=correlation,
            aggregate_run=aggregate_run,
            scope=scope,
            clock=clock,
        )
        for name, formula, warning in _OBSERVABILITY_KPIS
    ]
    return [
        bytes_kpi,
        duration_kpi,
        bandwidth_kpi,
        transform_kpi,
        *observability,
        share_kpi,
    ]


__all__ = ["_throughput_and_tokens", "_transfer_kpis"]
