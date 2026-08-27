"""Normalize direct, non-token RBLN inference observations."""

from __future__ import annotations

from dataclasses import dataclass

from ..artifact_compatibility import LEGACY_MEASURED_WINDOW
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


HOST_ID = "host-0"
CLOCK_DOMAIN_ID = "host-monotonic"
NON_TOKEN_REASON = (
    "selected NPU runtime workload does not expose streamed token timing"
)


@dataclass(frozen=True)
class InferenceObservation:
    request_id: str
    started_ns: int
    ended_ns: int

    @property
    def latency_ns(self) -> int:
        return self.ended_ns - self.started_ns


def parse_observations(summary: dict[str, object]) -> tuple[InferenceObservation, ...]:
    """Parse the child summary without accepting malformed durations."""
    raw = summary.get("measured")
    if not isinstance(raw, list):
        raise ValueError("workload summary measured must be a list")
    observations: list[InferenceObservation] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"workload summary measured[{index}] must be an object")
        request_id = item.get("request_id")
        started_ns = item.get("started_ns")
        ended_ns = item.get("ended_ns")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"workload summary measured[{index}].request_id is invalid")
        if not isinstance(started_ns, int) or not isinstance(ended_ns, int):
            raise ValueError(f"workload summary measured[{index}] timestamps are invalid")
        if ended_ns < started_ns:
            raise ValueError(f"workload summary measured[{index}] has negative duration")
        observations.append(InferenceObservation(request_id, started_ns, ended_ns))
    return tuple(observations)


def observation_events(
    run_id: str, observation: InferenceObservation
) -> list[EventRecord]:
    attributes = {"rbln.observation": "direct_runtime_boundary"}
    return [
        EventRecord(
            run_id=run_id,
            event_id=f"{observation.request_id}-received",
            event_name="request_received",
            event_type=EventType.INSTANT,
            phase=Phase.REQUEST,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            timestamp_ns=observation.started_ns,
            request_id=observation.request_id,
            attributes=attributes,
        ),
        EventRecord(
            run_id=run_id,
            event_id=f"{observation.request_id}-done",
            event_name="response_done",
            event_type=EventType.INSTANT,
            phase=Phase.RESPONSE,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            timestamp_ns=observation.ended_ns,
            request_id=observation.request_id,
            attributes=attributes,
        ),
    ]


def observation_metrics(
    run_id: str, observation: InferenceObservation
) -> list[MetricSample]:
    common = {
        "run_id": run_id,
        "scope": MetricScope.REQUEST,
        "host_id": HOST_ID,
        "clock_domain_id": CLOCK_DOMAIN_ID,
        "timestamp_ns": observation.ended_ns,
        "request_id": observation.request_id,
        "dimensions": {"workload.unit": "inference"},
        "attributes": {"rbln.timestamp_source": "time.monotonic_ns"},
    }
    return [
        MetricSample(
            **common,
            metric_name="latency.e2e",
            metric_kind=MetricKind.DURATION,
            availability=Availability.AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit="ns",
            value=observation.latency_ns,
            source_event_ids=[
                f"{observation.request_id}-received",
                f"{observation.request_id}-done",
            ],
        ),
        MetricSample(
            **common,
            metric_name="latency.ttft",
            metric_kind=MetricKind.DURATION,
            availability=Availability.NOT_AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit="ns",
            value=None,
            reason=NON_TOKEN_REASON,
        ),
        MetricSample(
            **common,
            metric_name="latency.tpot",
            metric_kind=MetricKind.DURATION,
            availability=Availability.NOT_AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit="ns",
            value=None,
            reason=NON_TOKEN_REASON,
        ),
    ]


def measured_window_metrics(
    run_id: str, observations: tuple[InferenceObservation, ...]
) -> list[MetricSample]:
    if not observations:
        return []
    start_ns = observations[0].started_ns
    end_ns = observations[-1].ended_ns
    interval_ns = max(1, end_ns - start_ns)
    common = {
        "run_id": run_id,
        "scope": MetricScope.RUN,
        "host_id": HOST_ID,
        "clock_domain_id": CLOCK_DOMAIN_ID,
        "timestamp_ns": end_ns,
        "availability": Availability.AVAILABLE,
        "origin": ValueOrigin.DERIVED,
        "interval_ns": interval_ns,
        "dimensions": {
            "window": LEGACY_MEASURED_WINDOW,
            "workload.unit": "inference",
        },
        "attributes": {"rbln.warmup_excluded": True},
    }
    return [
        MetricSample(
            **common,
            metric_name="request.count",
            metric_kind=MetricKind.COUNT,
            unit="requests",
            value=len(observations),
        ),
        MetricSample(
            **common,
            metric_name="throughput.requests",
            metric_kind=MetricKind.RATE,
            unit="requests/s",
            value=len(observations) / (interval_ns / 1_000_000_000),
        ),
    ]
