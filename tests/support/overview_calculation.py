"""Shared normalized records for Overview calculation tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from perfetto_hetero_profiler.hybrid.runtime_markers import (
    CANONICAL_MARKER_PHASES,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceType,
    EventRecord,
    EventType,
    MetricSample,
    RunMode,
    RunStatus,
    ValueOrigin,
)
from perfetto_hetero_profiler.schema.metric_catalog import METRIC_CATALOG

RUN_ID = "overview-fixture"
CLIENT_REQUEST_ID = "client-m01"
CORRELATION_ID = "correlation-1"
CLOCK_ID = "canonical"
ALIGNMENT = {
    "hybrid.alignment_method": "same_clock_domain",
    "hybrid.alignment_uncertainty_ns": 0,
}


def _event(
    name: str,
    timestamp_ns: int,
    *,
    attributes: dict[str, object] | None = None,
    event_id: str | None = None,
) -> EventRecord:
    merged = {
        **ALIGNMENT,
        "hybrid.correlation_id": CORRELATION_ID,
        **(attributes or {}),
    }
    return EventRecord(
        run_id=RUN_ID,
        event_id=event_id or f"event-{name}-{timestamp_ns}",
        event_name=name,
        event_type=EventType.INSTANT,
        phase=CANONICAL_MARKER_PHASES[name],
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=timestamp_ns,
        request_id=CORRELATION_ID,
        attributes=merged,
    )


def _metric(
    name: str,
    value: int | float | None,
    *,
    request_id: str | None = None,
    timestamp_ns: int = 110,
    interval_ns: int | None = None,
    dimensions: dict[str, object] | None = None,
    attributes: dict[str, object] | None = None,
    availability: Availability = Availability.AVAILABLE,
    source_event_ids: list[str] | None = None,
) -> MetricSample:
    definition = METRIC_CATALOG[name]
    scope = definition.allowed_scopes[0]
    return MetricSample(
        run_id=RUN_ID,
        metric_name=name,
        metric_kind=definition.kind,
        scope=scope,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=timestamp_ns,
        availability=availability,
        origin=ValueOrigin.DERIVED,
        unit=definition.unit,
        value=value,
        dimensions=dimensions or {},
        attributes={**ALIGNMENT, **(attributes or {})},
        request_id=request_id,
        interval_ns=interval_ns,
        source_event_ids=source_event_ids,
    )


def _fixture(*, output_tokens: int = 2) -> SimpleNamespace:
    events = [
        _event(
            "request_received",
            0,
            attributes={"proxy.client_request_id_hash": "explicit-hash"},
        ),
        _event("prefill_start", 10),
        _event("prefill_end", 20),
        _event("kv_export_start", 21),
        _event("kv_export_end", 22),
        _event(
            "kv_transfer_start",
            30,
            attributes={
                "hybrid.transfer_id": "transfer-1",
                "kv.transfer_bytes": 100,
            },
        ),
        _event(
            "kv_transfer_end",
            40,
            attributes={
                "hybrid.transfer_id": "transfer-1",
                "kv.transfer_bytes": 100,
            },
        ),
        _event("kv_transform_start", 41),
        _event("kv_transform_end", 45),
        _event("decode_loop_start", 46),
    ]
    for step in range(8):
        start = 50 + step * 3
        events.extend(
            [
                _event(
                    "sampling_start",
                    start,
                    attributes={"decode.step_index": step},
                    event_id=f"sampling-start-{step}",
                ),
                _event(
                    "sampling_end",
                    start + 1,
                    attributes={"decode.step_index": step},
                    event_id=f"sampling-end-{step}",
                ),
            ]
        )
    events.extend([_event("decode_loop_end", 80), _event("response_done", 100)])

    input_tokens = 3
    total_tokens = input_tokens + output_tokens
    interval_ns = 110
    interval_seconds = interval_ns / 1_000_000_000
    measured = {
        "vllm.measurement_window": "measured_smoke",
    }
    metrics = [
        _metric(
            "latency.e2e",
            110,
            request_id=CLIENT_REQUEST_ID,
            attributes=measured,
        ),
        _metric(
            "latency.ttft",
            20,
            request_id=CLIENT_REQUEST_ID,
            attributes=measured,
        ),
        _metric(
            "latency.tpot",
            5.0,
            request_id=CLIENT_REQUEST_ID,
            source_event_ids=["client-token-first", "client-token-last"],
        ),
        _metric(
            "request.count",
            1,
            interval_ns=interval_ns,
            dimensions={"window": "measured_smoke"},
            attributes=measured,
        ),
        _metric(
            "request.input_tokens",
            input_tokens,
            request_id=CLIENT_REQUEST_ID,
            attributes=measured,
        ),
        _metric(
            "request.output_tokens",
            output_tokens,
            request_id=CLIENT_REQUEST_ID,
            attributes=measured,
        ),
        _metric(
            "request.total_tokens",
            total_tokens,
            request_id=CLIENT_REQUEST_ID,
            attributes=measured,
        ),
        _metric(
            "throughput.requests",
            1 / interval_seconds,
            interval_ns=interval_ns,
            dimensions={"window": "measured_smoke"},
            attributes=measured,
        ),
        _metric(
            "throughput.input_tokens",
            input_tokens / interval_seconds,
            interval_ns=interval_ns,
            dimensions={"window": "measured_smoke"},
            attributes=measured,
        ),
        _metric(
            "throughput.output_tokens",
            output_tokens / interval_seconds,
            interval_ns=interval_ns,
            dimensions={"window": "measured_smoke"},
            attributes=measured,
        ),
        _metric(
            "throughput.total_tokens",
            total_tokens / interval_seconds,
            interval_ns=interval_ns,
            dimensions={"window": "measured_smoke"},
            attributes=measured,
        ),
    ]
    pipeline_values = {
        "latency.e2e": 100,
        "latency.prefill": 10,
        "latency.kv_export": 1,
        "latency.kv_transfer": 10,
        "latency.kv_transform": 4,
        "latency.decode": 34,
        # The existing bundle metric intentionally captures only step zero.
        "latency.sampling": 1,
    }
    for name, value in pipeline_values.items():
        metrics.append(
            _metric(
                name,
                value,
                request_id=CORRELATION_ID,
                interval_ns=value,
                dimensions={"hybrid.join_method": "correlation_id"},
            )
        )
    manifest = SimpleNamespace(
        run_id=RUN_ID,
        mode=RunMode.HYBRID,
        status=RunStatus.SUCCEEDED,
        attributes={"hybrid.alignment_offset_ns": 0},
    )
    return SimpleNamespace(
        manifest=manifest,
        canonical_clock_domain_id=CLOCK_ID,
        events=tuple(events),
        metrics=tuple(metrics),
        sources=(),
        root_fingerprints=(),
    )


def _section_by_name(result: dict[str, object], section: str) -> dict[str, dict]:
    return {item["name"]: item for item in result[section]}


def _two_request_fixture() -> SimpleNamespace:
    loaded = _fixture()
    duplicate_events = []
    for event in loaded.events:
        attributes = dict(event.attributes)
        attributes["hybrid.correlation_id"] = "correlation-2"
        if "hybrid.transfer_id" in attributes:
            attributes["hybrid.transfer_id"] = "transfer-2"
        if "proxy.client_request_id_hash" in attributes:
            attributes["proxy.client_request_id_hash"] = "explicit-hash-2"
        duplicate_events.append(
            replace(
                event,
                event_id=f"{event.event_id}-request-2",
                timestamp_ns=event.timestamp_ns + 200,
                request_id="correlation-2",
                attributes=attributes,
            )
        )

    run_names = {
        "request.count",
        "throughput.requests",
        "throughput.input_tokens",
        "throughput.output_tokens",
        "throughput.total_tokens",
    }
    interval_ns = 220
    interval_seconds = interval_ns / 1_000_000_000
    run_values = {
        "request.count": 2,
        "throughput.requests": 2 / interval_seconds,
        "throughput.input_tokens": 6 / interval_seconds,
        "throughput.output_tokens": 4 / interval_seconds,
        "throughput.total_tokens": 10 / interval_seconds,
    }
    metrics = []
    for metric in loaded.metrics:
        if metric.metric_name in run_names and metric.request_id is None:
            metrics.append(
                replace(
                    metric,
                    value=run_values[metric.metric_name],
                    interval_ns=interval_ns,
                )
            )
            continue
        metrics.append(metric)
        if metric.request_id is not None:
            metrics.append(
                replace(
                    metric,
                    request_id=(
                        "client-m02"
                        if metric.request_id == CLIENT_REQUEST_ID
                        else "correlation-2"
                    ),
                    timestamp_ns=metric.timestamp_ns + 200,
                )
            )
    return SimpleNamespace(
        **{
            **loaded.__dict__,
            "events": tuple((*loaded.events, *duplicate_events)),
            "metrics": tuple(metrics),
        }
    )
