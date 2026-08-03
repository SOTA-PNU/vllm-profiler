"""Translate externally observed completions into schema v1 records."""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import (
    Availability,
    EventRecord,
    EventType,
    MetricKind,
    MetricSample,
    MetricScope,
    Phase,
    ValueOrigin,
)
from .openai_client import CompletionObservation


HOST_ID = "localhost"
CLOCK_DOMAIN_ID = "host-monotonic"


def observation_events(
    run_id: str,
    observation: CompletionObservation,
) -> list[EventRecord]:
    request_id = observation.request_id
    records = [
        _event(
            run_id,
            f"{request_id}-received",
            "request_received",
            Phase.REQUEST,
            observation.received_ns,
            request_id,
            {"vllm.observation": "client_request_send"},
        )
    ]
    for index, timestamp_ns in enumerate(observation.token_timestamps_ns):
        name = "first_token_emitted" if index == 0 else "token_emitted"
        records.append(
            _event(
                run_id,
                f"{request_id}-token-{index}",
                name,
                Phase.RESPONSE,
                timestamp_ns,
                request_id,
                {
                    "vllm.observation": "client_stream_arrival",
                    "vllm.token_sequence": index,
                },
            )
        )
    records.append(
        _event(
            run_id,
            f"{request_id}-done",
            "response_done",
            Phase.RESPONSE,
            observation.done_ns,
            request_id,
            {
                "vllm.observation": "client_done_arrival",
                "vllm.http_status": observation.http_status,
            },
        )
    )
    return records


def observation_metrics(
    run_id: str,
    observation: CompletionObservation,
) -> list[MetricSample]:
    request_id = observation.request_id
    metrics = [
        _metric(
            run_id,
            "latency.e2e",
            MetricKind.DURATION,
            MetricScope.REQUEST,
            "ns",
            observation.e2e_ns,
            observation.done_ns,
            request_id=request_id,
            source_event_ids=[
                f"{request_id}-received",
                f"{request_id}-done",
            ],
        ),
        _metric(
            run_id,
            "request.input_tokens",
            MetricKind.COUNT,
            MetricScope.REQUEST,
            "tokens",
            observation.input_tokens,
            observation.done_ns,
            request_id=request_id,
        ),
        _metric(
            run_id,
            "request.output_tokens",
            MetricKind.COUNT,
            MetricScope.REQUEST,
            "tokens",
            observation.output_tokens,
            observation.done_ns,
            request_id=request_id,
        ),
        _metric(
            run_id,
            "request.total_tokens",
            MetricKind.COUNT,
            MetricScope.REQUEST,
            "tokens",
            observation.total_tokens,
            observation.done_ns,
            request_id=request_id,
        ),
    ]
    if observation.ttft_ns is not None:
        metrics.append(
            _metric(
                run_id,
                "latency.ttft",
                MetricKind.DURATION,
                MetricScope.REQUEST,
                "ns",
                observation.ttft_ns,
                observation.token_timestamps_ns[0],
                request_id=request_id,
                source_event_ids=[
                    f"{request_id}-received",
                    f"{request_id}-token-0",
                ],
            )
        )
    if observation.tpot_ns is not None:
        metrics.append(
            _metric(
                run_id,
                "latency.tpot",
                MetricKind.DURATION,
                MetricScope.REQUEST,
                "ns",
                observation.tpot_ns,
                observation.done_ns,
                request_id=request_id,
                source_event_ids=[
                    f"{request_id}-token-0",
                    f"{request_id}-token-{observation.output_tokens - 1}",
                ],
                attributes={
                    "vllm.calculation": (
                        "(last_token_arrival_ns-first_token_arrival_ns)"
                        "/(output_tokens-1)"
                    ),
                    "vllm.timestamp_source": "client_stream_arrival",
                },
            )
        )
    return metrics


def measured_window_metrics(
    run_id: str,
    observations: Iterable[CompletionObservation],
) -> list[MetricSample]:
    items = tuple(observations)
    if not items:
        return []
    start_ns = min(item.received_ns for item in items)
    end_ns = max(item.done_ns for item in items)
    interval_ns = max(1, end_ns - start_ns)
    interval_sec = interval_ns / 1_000_000_000
    input_tokens = sum(item.input_tokens for item in items)
    output_tokens = sum(item.output_tokens for item in items)
    total_tokens = sum(item.total_tokens for item in items)
    dimensions = {"window": "measured_smoke"}
    return [
        _metric(
            run_id,
            "request.count",
            MetricKind.COUNT,
            MetricScope.RUN,
            "requests",
            len(items),
            end_ns,
            interval_ns=interval_ns,
            dimensions=dimensions,
        ),
        _metric(
            run_id,
            "throughput.requests",
            MetricKind.RATE,
            MetricScope.RUN,
            "requests/s",
            len(items) / interval_sec,
            end_ns,
            interval_ns=interval_ns,
            dimensions=dimensions,
        ),
        _metric(
            run_id,
            "throughput.input_tokens",
            MetricKind.RATE,
            MetricScope.RUN,
            "tokens/s",
            input_tokens / interval_sec,
            end_ns,
            interval_ns=interval_ns,
            dimensions=dimensions,
        ),
        _metric(
            run_id,
            "throughput.output_tokens",
            MetricKind.RATE,
            MetricScope.RUN,
            "tokens/s",
            output_tokens / interval_sec,
            end_ns,
            interval_ns=interval_ns,
            dimensions=dimensions,
        ),
        _metric(
            run_id,
            "throughput.total_tokens",
            MetricKind.RATE,
            MetricScope.RUN,
            "tokens/s",
            total_tokens / interval_sec,
            end_ns,
            interval_ns=interval_ns,
            dimensions=dimensions,
        ),
    ]


def _event(
    run_id: str,
    event_id: str,
    event_name: str,
    phase: Phase,
    timestamp_ns: int,
    request_id: str,
    attributes: dict[str, object],
) -> EventRecord:
    return EventRecord(
        run_id=run_id,
        event_id=event_id,
        event_name=event_name,
        event_type=EventType.INSTANT,
        phase=phase,
        host_id=HOST_ID,
        clock_domain_id=CLOCK_DOMAIN_ID,
        timestamp_ns=timestamp_ns,
        request_id=request_id,
        attributes=attributes,
    )


def _metric(
    run_id: str,
    name: str,
    kind: MetricKind,
    scope: MetricScope,
    unit: str,
    value: int | float,
    timestamp_ns: int,
    *,
    request_id: str | None = None,
    interval_ns: int | None = None,
    dimensions: dict[str, object] | None = None,
    source_event_ids: list[str] | None = None,
    attributes: dict[str, object] | None = None,
) -> MetricSample:
    return MetricSample(
        run_id=run_id,
        metric_name=name,
        metric_kind=kind,
        scope=scope,
        host_id=HOST_ID,
        clock_domain_id=CLOCK_DOMAIN_ID,
        timestamp_ns=timestamp_ns,
        availability=Availability.AVAILABLE,
        origin=ValueOrigin.DERIVED,
        unit=unit,
        value=value,
        request_id=request_id,
        interval_ns=interval_ns,
        dimensions=dimensions or {},
        attributes=attributes
        or {"vllm.measurement_window": "measured_smoke"},
        source_event_ids=source_event_ids,
    )
