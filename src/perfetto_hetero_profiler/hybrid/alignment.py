"""Canonical nanosecond timestamp conversion with provenance preservation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from ..schema import EventRecord, MetricSample


class AlignmentError(RuntimeError):
    """A source timestamp cannot be represented on the canonical timeline."""


@dataclass(frozen=True)
class TimestampTransform:
    source_clock_domain_id: str
    target_clock_domain_id: str
    offset_ns: int
    uncertainty_ns: int
    method: str
    available: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.source_clock_domain_id or not self.target_clock_domain_id:
            raise ValueError("clock domain ids must be non-empty")
        if self.uncertainty_ns < 0:
            raise ValueError("uncertainty_ns must be non-negative")
        if self.available and self.reason is not None:
            raise ValueError("available transform must not have a failure reason")
        if not self.available and not self.reason:
            raise ValueError("unavailable transform requires a reason")


@dataclass(frozen=True)
class AlignedTimestamp:
    original_timestamp_ns: int
    original_clock_domain_id: str
    timestamp_ns: int
    target_clock_domain_id: str
    uncertainty_ns: int
    method: str


def align_timestamp(
    timestamp_ns: int,
    clock_domain_id: str,
    transform: TimestampTransform,
) -> AlignedTimestamp:
    if timestamp_ns < 0:
        raise AlignmentError("source timestamp must be non-negative")
    if clock_domain_id != transform.source_clock_domain_id:
        raise AlignmentError(
            f"transform source {transform.source_clock_domain_id!r} does not match "
            f"record domain {clock_domain_id!r}"
        )
    if not transform.available:
        raise AlignmentError(transform.reason or "clock alignment unavailable")
    aligned = timestamp_ns + transform.offset_ns
    if aligned < 0:
        raise AlignmentError("clock transform would create a negative timestamp")
    return AlignedTimestamp(
        original_timestamp_ns=timestamp_ns,
        original_clock_domain_id=clock_domain_id,
        timestamp_ns=aligned,
        target_clock_domain_id=transform.target_clock_domain_id,
        uncertainty_ns=transform.uncertainty_ns,
        method=transform.method,
    )


def _provenance(
    attributes: dict[str, object],
    aligned: AlignedTimestamp,
    source_role: str,
) -> dict[str, object]:
    return {
        **attributes,
        "hybrid.original_timestamp_ns": aligned.original_timestamp_ns,
        "hybrid.original_clock_domain_id": aligned.original_clock_domain_id,
        "hybrid.aligned_clock_domain_id": aligned.target_clock_domain_id,
        "hybrid.alignment_uncertainty_ns": aligned.uncertainty_ns,
        "hybrid.alignment_method": aligned.method,
        "hybrid.source_role": source_role,
    }


def align_event(
    event: EventRecord,
    *,
    hybrid_run_id: str,
    source_role: str,
    transform: TimestampTransform,
) -> EventRecord:
    aligned = align_timestamp(
        event.timestamp_ns, event.clock_domain_id, transform
    )
    duration_ns = event.duration_ns
    return replace(
        event,
        run_id=hybrid_run_id,
        event_id=f"{source_role}:{event.event_id}",
        parent_event_id=(
            f"{source_role}:{event.parent_event_id}"
            if event.parent_event_id is not None
            else None
        ),
        timestamp_ns=aligned.timestamp_ns,
        clock_domain_id=aligned.target_clock_domain_id,
        duration_ns=duration_ns,
        attributes=_provenance(event.attributes, aligned, source_role),
    )


def align_metric(
    metric: MetricSample,
    *,
    hybrid_run_id: str,
    source_role: str,
    transform: TimestampTransform,
) -> MetricSample:
    aligned = align_timestamp(
        metric.timestamp_ns, metric.clock_domain_id, transform
    )
    return replace(
        metric,
        run_id=hybrid_run_id,
        timestamp_ns=aligned.timestamp_ns,
        clock_domain_id=aligned.target_clock_domain_id,
        source_event_ids=(
            [f"{source_role}:{event_id}" for event_id in metric.source_event_ids]
            if metric.source_event_ids is not None
            else None
        ),
        attributes=_provenance(metric.attributes, aligned, source_role),
    )


def align_event_stream(
    events: Iterable[EventRecord],
    *,
    hybrid_run_id: str,
    source_role: str,
    transforms: dict[str, TimestampTransform],
) -> list[EventRecord]:
    aligned: list[EventRecord] = []
    last_by_stream: dict[tuple[str, str | None], tuple[int, int]] = {}
    for event in events:
        transform = transforms.get(event.clock_domain_id)
        if transform is None:
            raise AlignmentError(
                f"no transform for source clock {event.clock_domain_id!r}"
            )
        converted = align_event(
            event,
            hybrid_run_id=hybrid_run_id,
            source_role=source_role,
            transform=transform,
        )
        stream_key = (event.clock_domain_id, event.request_id)
        previous = last_by_stream.get(stream_key)
        if previous is not None:
            previous_source, previous_target = previous
            if event.timestamp_ns < previous_source:
                raise AlignmentError("source event stream timestamp decreased")
            if converted.timestamp_ns < previous_target:
                raise AlignmentError("clock transform reversed source ordering")
        last_by_stream[stream_key] = (
            event.timestamp_ns,
            converted.timestamp_ns,
        )
        aligned.append(converted)
    return aligned


def align_metric_stream(
    metrics: Iterable[MetricSample],
    *,
    hybrid_run_id: str,
    source_role: str,
    transforms: dict[str, TimestampTransform],
) -> list[MetricSample]:
    aligned: list[MetricSample] = []
    last_by_stream: dict[
        tuple[str, str | None, str, str | None], tuple[int, int]
    ] = {}
    for metric in metrics:
        transform = transforms.get(metric.clock_domain_id)
        if transform is None:
            raise AlignmentError(
                f"no transform for source clock {metric.clock_domain_id!r}"
            )
        converted = align_metric(
            metric,
            hybrid_run_id=hybrid_run_id,
            source_role=source_role,
            transform=transform,
        )
        stream_key = (
            metric.clock_domain_id,
            metric.request_id,
            metric.metric_name,
            metric.device_id,
        )
        previous = last_by_stream.get(stream_key)
        if previous is not None:
            previous_source, previous_target = previous
            if metric.timestamp_ns < previous_source:
                raise AlignmentError("source metric stream timestamp decreased")
            if converted.timestamp_ns < previous_target:
                raise AlignmentError("clock transform reversed source ordering")
        last_by_stream[stream_key] = (
            metric.timestamp_ns,
            converted.timestamp_ns,
        )
        aligned.append(converted)
    return aligned
