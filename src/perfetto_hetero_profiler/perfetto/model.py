"""Deterministic intermediate model for Perfetto trace generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


AnnotationValue: TypeAlias = bool | int | float | str
TraceAttributeValue: TypeAlias = int | str


@dataclass(frozen=True)
class TraceAttributeSpec:
    """One official path-free ``TraceAttributes`` entry."""

    key: str
    value: TraceAttributeValue


@dataclass(frozen=True)
class TrackSpec:
    """One deterministic Perfetto track descriptor."""

    key: str
    uuid: int
    name: str
    kind: str
    description: str
    unit: str | None = None
    parent_key: str | None = None
    child_ordering: str = "unknown"
    sibling_order_rank: int | None = None


@dataclass(frozen=True)
class SliceSpec:
    """One complete slice and its explicit flow endpoints."""

    track_key: str
    name: str
    timestamp_ns: int
    duration_ns: int
    annotations: tuple[tuple[str, AnnotationValue], ...] = ()
    begin_flow_ids: tuple[int, ...] = ()
    end_flow_ids: tuple[int, ...] = ()
    begin_terminating_flow_ids: tuple[int, ...] = ()
    end_terminating_flow_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class InstantSpec:
    """One zero-duration TrackEvent."""

    track_key: str
    name: str
    timestamp_ns: int
    annotations: tuple[tuple[str, AnnotationValue], ...] = ()


@dataclass(frozen=True)
class CounterSpec:
    """One available normalized resource counter sample."""

    track_key: str
    timestamp_ns: int
    value: int | float
    annotations: tuple[tuple[str, AnnotationValue], ...] = ()


@dataclass(frozen=True)
class FlowSpec:
    """Expected SQL-visible flow derived from explicit identifiers."""

    flow_id: int
    source_slice_name: str
    destination_slice_name: str
    correlation_id: str
    source_event_id: str | None = None
    destination_event_id: str | None = None
    evidence_kind: str | None = None
    evidence_id: str | None = None


@dataclass(frozen=True)
class UnclassifiedGapSpec:
    """One observed marker gap without an evidence-backed classification."""

    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_ns: int
    preceding_marker: str
    following_marker: str
    reason: str


@dataclass(frozen=True)
class TracePlan:
    """All deterministic packets and validation expectations for one trace."""

    run_id: str
    canonical_clock_domain_id: str
    process_uuid: int
    process_id: int
    packet_sequence_id: int
    tracks: tuple[TrackSpec, ...]
    slices: tuple[SliceSpec, ...]
    instants: tuple[InstantSpec, ...]
    counters: tuple[CounterSpec, ...]
    flows: tuple[FlowSpec, ...]
    trace_attributes: tuple[TraceAttributeSpec, ...] = ()
    mapping_version: str = "legacy-unversioned-phase5-v1"
    source_identity_sha256: str | None = None
    presentation_mode: bool = False
    unclassified_gaps: tuple[UnclassifiedGapSpec, ...] = ()

    @property
    def track_by_key(self) -> dict[str, TrackSpec]:
        return {track.key: track for track in self.tracks}
