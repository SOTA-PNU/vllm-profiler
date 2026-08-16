"""Deterministic Perfetto protobuf serialization for :mod:`.model` plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
    BUILTIN_CLOCK_MONOTONIC,
    CounterDescriptor,
    Trace,
    TrackDescriptor,
    TrackEvent,
)
from perfetto.trace_builder.proto_builder import TraceProtoBuilder

from .model import AnnotationValue, TracePlan, TrackSpec
from .trace_attributes import TRACE_ATTRIBUTE_NAMESPACE


_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1
_UINT32_MAX: Final = 2**32 - 1
_UINT64_MAX: Final = 2**64 - 1

_COUNTER_UNITS: Final = {
    "ns": CounterDescriptor.UNIT_TIME_NS,
    "nanosecond": CounterDescriptor.UNIT_TIME_NS,
    "nanoseconds": CounterDescriptor.UNIT_TIME_NS,
    "time_ns": CounterDescriptor.UNIT_TIME_NS,
    "count": CounterDescriptor.UNIT_COUNT,
    "counts": CounterDescriptor.UNIT_COUNT,
    "byte": CounterDescriptor.UNIT_SIZE_BYTES,
    "bytes": CounterDescriptor.UNIT_SIZE_BYTES,
    "size_bytes": CounterDescriptor.UNIT_SIZE_BYTES,
}
_CHILD_ORDERINGS: Final = {
    "unknown": TrackDescriptor.UNKNOWN,
    "lexicographic": TrackDescriptor.LEXICOGRAPHIC,
    "chronological": TrackDescriptor.CHRONOLOGICAL,
    "explicit": TrackDescriptor.EXPLICIT,
}


@dataclass(frozen=True)
class _PendingEvent:
    """One TrackEvent packet before canonical ordering and serialization."""

    timestamp_ns: int
    order: int
    track_uuid: int
    name: str
    event_type: int
    nesting_sort_ns: int = 0
    annotations: tuple[tuple[str, AnnotationValue], ...] = ()
    counter_value: int | float | None = None
    flow_ids: tuple[int, ...] = ()
    terminating_flow_ids: tuple[int, ...] = ()

    @property
    def sort_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.timestamp_ns,
            self.order,
            self.track_uuid,
            self.nesting_sort_ns,
            self.name,
        )


class PerfettoTraceWriter:
    """Build and serialize a canonical Perfetto ``Trace`` from a ``TracePlan``."""

    def build(self, plan: TracePlan) -> Trace:
        """Return the official Perfetto protobuf for ``plan``."""

        _validate_plan(plan)
        tracks = plan.track_by_key
        builder = TraceProtoBuilder()

        clock_packet = builder.add_packet()
        clock_snapshot = clock_packet.clock_snapshot
        clock_snapshot.primary_trace_clock = BUILTIN_CLOCK_MONOTONIC
        canonical_clock = clock_snapshot.clocks.add()
        canonical_clock.clock_id = BUILTIN_CLOCK_MONOTONIC
        canonical_clock.timestamp = 0
        canonical_clock.unit_multiplier_ns = 1

        if plan.trace_attributes:
            attributes_packet = builder.add_packet()
            trace_attributes = attributes_packet.trace_attributes
            for spec in sorted(
                plan.trace_attributes,
                key=lambda item: item.key,
            ):
                attribute = trace_attributes.attribute.add()
                attribute.key = spec.key
                if isinstance(spec.value, int):
                    attribute.long_value = spec.value
                else:
                    attribute.string_value = spec.value

        process_packet = builder.add_packet()
        process_descriptor = process_packet.track_descriptor
        process_descriptor.uuid = plan.process_uuid
        process_descriptor.name = plan.run_id
        process_descriptor.description = (
            f"synthetic process for canonical clock domain "
            f"{plan.canonical_clock_domain_id}"
        )
        process_descriptor.process.pid = plan.process_id
        process_descriptor.process.process_name = (
            f"perfetto-hetero-profiler:{plan.run_id}"
        )

        for track in _descriptor_tracks(plan, tracks):
            packet = builder.add_packet()
            descriptor = packet.track_descriptor
            descriptor.uuid = track.uuid
            descriptor.parent_uuid = (
                plan.process_uuid
                if track.parent_key is None
                else tracks[track.parent_key].uuid
            )
            descriptor.name = track.name
            descriptor.description = track.description
            if track.child_ordering != "unknown":
                descriptor.child_ordering = _child_ordering(track)
            if track.sibling_order_rank is not None:
                descriptor.sibling_order_rank = track.sibling_order_rank
            if _is_counter_track(track):
                _set_counter_descriptor(descriptor.counter, track.unit)

        for pending in _pending_events(plan, tracks):
            packet = builder.add_packet()
            packet.timestamp = pending.timestamp_ns
            packet.timestamp_clock_id = BUILTIN_CLOCK_MONOTONIC
            packet.trusted_packet_sequence_id = plan.packet_sequence_id

            event = packet.track_event
            event.type = pending.event_type
            event.track_uuid = pending.track_uuid
            if (
                pending.name
                and pending.event_type != TrackEvent.TYPE_SLICE_END
            ):
                event.name = pending.name
            if pending.counter_value is not None:
                _set_counter_value(event, pending.counter_value)
            if pending.flow_ids:
                event.flow_ids.extend(pending.flow_ids)
            if pending.terminating_flow_ids:
                event.terminating_flow_ids.extend(pending.terminating_flow_ids)
            _add_debug_annotations(event, pending.annotations)

        return builder.trace

    def serialize(self, plan: TracePlan) -> bytes:
        """Serialize ``plan`` with protobuf deterministic mode enabled."""

        return self.build(plan).SerializeToString(deterministic=True)

    def write(self, plan: TracePlan, path: str | PathLike[str]) -> Path:
        """Write a deterministic ``.pftrace`` and return its path."""

        output_path = Path(path)
        output_path.write_bytes(self.serialize(plan))
        return output_path


def build_trace(plan: TracePlan) -> Trace:
    """Build an official Perfetto ``Trace`` protobuf."""

    return PerfettoTraceWriter().build(plan)


def serialize_trace(plan: TracePlan) -> bytes:
    """Serialize ``plan`` deterministically."""

    return PerfettoTraceWriter().serialize(plan)


def write_trace(plan: TracePlan, path: str | PathLike[str]) -> Path:
    """Serialize ``plan`` deterministically to ``path``."""

    return PerfettoTraceWriter().write(plan, path)


def _pending_events(
    plan: TracePlan,
    tracks: dict[str, TrackSpec],
) -> tuple[_PendingEvent, ...]:
    pending: list[_PendingEvent] = []

    for spec in plan.slices:
        track = _event_track(tracks, spec.track_key)
        _timestamp(spec.timestamp_ns, f"slice {spec.name!r} timestamp")
        _duration(spec.duration_ns, spec.name)
        end_timestamp_ns = spec.timestamp_ns + spec.duration_ns
        _timestamp(end_timestamp_ns, f"slice {spec.name!r} end timestamp")
        begin_flow_ids = tuple(
            _flow_id(value, f"slice {spec.name!r} begin_flow_ids")
            for value in spec.begin_flow_ids
        )
        end_flow_ids = tuple(
            _flow_id(value, f"slice {spec.name!r} end_flow_ids")
            for value in spec.end_flow_ids
        )
        begin_terminating_flow_ids = tuple(
            _flow_id(
                value,
                f"slice {spec.name!r} begin_terminating_flow_ids",
            )
            for value in spec.begin_terminating_flow_ids
        )
        end_terminating_flow_ids = tuple(
            _flow_id(
                value,
                f"slice {spec.name!r} end_terminating_flow_ids",
            )
            for value in spec.end_terminating_flow_ids
        )
        _validate_annotations(spec.annotations, f"slice {spec.name!r}")
        pending.append(
            _PendingEvent(
                timestamp_ns=spec.timestamp_ns,
                order=1,
                track_uuid=track.uuid,
                name=spec.name,
                event_type=TrackEvent.TYPE_SLICE_BEGIN,
                # Same-start nested slices must open outermost first.
                nesting_sort_ns=-end_timestamp_ns,
                annotations=spec.annotations,
                flow_ids=begin_flow_ids,
                terminating_flow_ids=begin_terminating_flow_ids,
            )
        )
        pending.append(
            _PendingEvent(
                timestamp_ns=end_timestamp_ns,
                order=0,
                track_uuid=track.uuid,
                name=spec.name,
                event_type=TrackEvent.TYPE_SLICE_END,
                # Same-end nested slices must close innermost first.
                nesting_sort_ns=-spec.timestamp_ns,
                flow_ids=end_flow_ids,
                terminating_flow_ids=end_terminating_flow_ids,
            )
        )

    for spec in plan.instants:
        track = _event_track(tracks, spec.track_key)
        _timestamp(spec.timestamp_ns, f"instant {spec.name!r} timestamp")
        _validate_annotations(spec.annotations, f"instant {spec.name!r}")
        pending.append(
            _PendingEvent(
                timestamp_ns=spec.timestamp_ns,
                order=2,
                track_uuid=track.uuid,
                name=spec.name,
                event_type=TrackEvent.TYPE_INSTANT,
                annotations=spec.annotations,
            )
        )

    for spec in plan.counters:
        track = _counter_track(tracks, spec.track_key)
        _timestamp(spec.timestamp_ns, f"counter {track.name!r} timestamp")
        _validate_counter_value(spec.value, track.name)
        _validate_annotations(spec.annotations, f"counter {track.name!r}")
        pending.append(
            _PendingEvent(
                timestamp_ns=spec.timestamp_ns,
                order=3,
                track_uuid=track.uuid,
                name=track.name,
                event_type=TrackEvent.TYPE_COUNTER,
                annotations=spec.annotations,
                counter_value=spec.value,
            )
        )

    return tuple(sorted(pending, key=lambda item: item.sort_key))


def _validate_plan(plan: TracePlan) -> None:
    if not plan.run_id:
        raise ValueError("run_id must be non-empty")
    if not plan.canonical_clock_domain_id:
        raise ValueError("canonical_clock_domain_id must be non-empty")
    _uuid(plan.process_uuid, "process_uuid")
    if not _INT32_MIN <= plan.process_id <= _INT32_MAX:
        raise ValueError("synthetic process_id must fit the int32 protobuf field")
    if not 0 < plan.packet_sequence_id <= _UINT32_MAX:
        raise ValueError("packet_sequence_id must be in the uint32 range and non-zero")

    keys: set[str] = set()
    uuids = {plan.process_uuid}
    for track in plan.tracks:
        if not track.key:
            raise ValueError("track key must be non-empty")
        if track.key in keys:
            raise ValueError(f"duplicate track key: {track.key!r}")
        keys.add(track.key)
        _uuid(track.uuid, f"track {track.key!r} uuid")
        if track.uuid in uuids:
            raise ValueError(f"duplicate process/track uuid: {track.uuid}")
        uuids.add(track.uuid)
        if not track.name:
            raise ValueError(f"track {track.key!r} name must be non-empty")

    _validate_track_hierarchy(plan)
    _validate_slice_nesting(plan)

    for flow in plan.flows:
        _flow_id(flow.flow_id, f"flow {flow.correlation_id!r}")
    _validate_trace_attributes(plan)


def _validate_trace_attributes(plan: TracePlan) -> None:
    keys: set[str] = set()
    for spec in plan.trace_attributes:
        if (
            not isinstance(spec.key, str)
            or not spec.key.startswith(TRACE_ATTRIBUTE_NAMESPACE)
            or spec.key in keys
        ):
            raise ValueError(
                "trace attribute keys must be unique names in the public namespace"
            )
        keys.add(spec.key)
        value = spec.value
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError(
                f"trace attribute {spec.key!r} must be an integer or string"
            )
        if isinstance(value, str):
            if not value:
                raise ValueError(
                    f"trace attribute {spec.key!r} must not be empty"
                )
            if (
                PurePosixPath(value).is_absolute()
                or PureWindowsPath(value).is_absolute()
                or value.startswith("file://")
            ):
                raise ValueError(
                    f"trace attribute {spec.key!r} must not expose a path"
                )
        else:
            if not _INT64_MIN <= value <= _INT64_MAX:
                raise ValueError(
                    f"trace attribute {spec.key!r} must fit signed int64"
                )


def _validate_slice_nesting(plan: TracePlan) -> None:
    """Reject intervals that cannot be represented by one TrackEvent stack."""

    by_track: dict[str, list[tuple[int, int, str]]] = {}
    for spec in plan.slices:
        _timestamp(spec.timestamp_ns, f"slice {spec.name!r} timestamp")
        _duration(spec.duration_ns, spec.name)
        end_ns = spec.timestamp_ns + spec.duration_ns
        _timestamp(end_ns, f"slice {spec.name!r} end timestamp")
        by_track.setdefault(spec.track_key, []).append(
            (spec.timestamp_ns, end_ns, spec.name)
        )

    for track_key, intervals in sorted(by_track.items()):
        stack: list[tuple[int, int, str]] = []
        for interval in sorted(
            intervals,
            key=lambda item: (item[0], -item[1], item[2]),
        ):
            start_ns, end_ns, name = interval
            while stack and start_ns >= stack[-1][1]:
                stack.pop()
            if stack:
                parent_start, parent_end, parent_name = stack[-1]
                if start_ns == parent_start and end_ns == parent_end:
                    raise ValueError(
                        f"track {track_key!r} has indistinguishable duplicate "
                        f"slice intervals for {parent_name!r} and {name!r}"
                    )
                if end_ns > parent_end:
                    raise ValueError(
                        f"track {track_key!r} has crossing slice intervals for "
                        f"{parent_name!r} and {name!r}"
                    )
            stack.append(interval)


def _validate_track_hierarchy(plan: TracePlan) -> None:
    tracks = plan.track_by_key
    children: dict[str, list[TrackSpec]] = {}

    for track in plan.tracks:
        _child_ordering(track)
        _sibling_order_rank(track)

        if track.parent_key is None:
            if track.sibling_order_rank is not None:
                raise ValueError(
                    f"track {track.key!r} cannot set sibling_order_rank "
                    "without a parent track"
                )
            continue
        if not isinstance(track.parent_key, str) or not track.parent_key:
            raise ValueError(
                f"track {track.key!r} parent_key must be a non-empty string "
                "when provided"
            )
        if track.parent_key == track.key:
            raise ValueError(f"track {track.key!r} cannot be its own parent")
        if track.parent_key not in tracks:
            raise ValueError(
                f"track {track.key!r} references unknown parent "
                f"{track.parent_key!r}"
            )
        children.setdefault(track.parent_key, []).append(track)

    states: dict[str, int] = {}

    def visit(key: str) -> None:
        state = states.get(key, 0)
        if state == 1:
            raise ValueError(f"track hierarchy contains a cycle at {key!r}")
        if state == 2:
            return
        states[key] = 1
        parent_key = tracks[key].parent_key
        if parent_key is not None:
            visit(parent_key)
        states[key] = 2

    for key in sorted(tracks):
        visit(key)

    for parent_key, siblings in children.items():
        parent = tracks[parent_key]
        if parent.child_ordering == "explicit":
            ranks: dict[int, str] = {}
            for sibling in siblings:
                rank = sibling.sibling_order_rank
                if rank is None:
                    raise ValueError(
                        f"track {sibling.key!r} must set sibling_order_rank "
                        f"because parent {parent_key!r} uses explicit ordering"
                    )
                previous = ranks.get(rank)
                if previous is not None:
                    raise ValueError(
                        f"tracks {previous!r} and {sibling.key!r} have "
                        f"duplicate sibling_order_rank {rank} under explicit "
                        f"parent {parent_key!r}"
                    )
                ranks[rank] = sibling.key
        else:
            ranked = [
                sibling.key
                for sibling in siblings
                if sibling.sibling_order_rank is not None
            ]
            if ranked:
                raise ValueError(
                    f"track {ranked[0]!r} sets sibling_order_rank but parent "
                    f"{parent_key!r} does not use explicit ordering"
                )


def _descriptor_tracks(
    plan: TracePlan,
    tracks: dict[str, TrackSpec],
) -> tuple[TrackSpec, ...]:
    if all(
        track.parent_key is None
        and track.child_ordering == "unknown"
        and track.sibling_order_rank is None
        for track in plan.tracks
    ):
        return tuple(sorted(plan.tracks, key=lambda item: (item.uuid, item.key)))

    depths: dict[str, int] = {}

    def depth(track: TrackSpec) -> int:
        known = depths.get(track.key)
        if known is not None:
            return known
        value = (
            0
            if track.parent_key is None
            else depth(tracks[track.parent_key]) + 1
        )
        depths[track.key] = value
        return value

    return tuple(
        sorted(
            plan.tracks,
            key=lambda item: (depth(item), item.uuid, item.key),
        )
    )


def _child_ordering(track: TrackSpec) -> int:
    if not isinstance(track.child_ordering, str):
        raise TypeError(
            f"track {track.key!r} child_ordering must be a string"
        )
    try:
        return _CHILD_ORDERINGS[track.child_ordering]
    except KeyError as error:
        choices = ", ".join(sorted(_CHILD_ORDERINGS))
        raise ValueError(
            f"track {track.key!r} child_ordering must be one of: {choices}"
        ) from error


def _sibling_order_rank(track: TrackSpec) -> None:
    rank = track.sibling_order_rank
    if rank is None:
        return
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(
            f"track {track.key!r} sibling_order_rank must be an integer"
        )
    if not _INT32_MIN <= rank <= _INT32_MAX:
        raise ValueError(
            f"track {track.key!r} sibling_order_rank must fit signed int32"
        )


def _event_track(tracks: dict[str, TrackSpec], key: str) -> TrackSpec:
    track = _track(tracks, key)
    if _is_counter_track(track):
        raise ValueError(f"event references counter track {key!r}")
    return track


def _counter_track(tracks: dict[str, TrackSpec], key: str) -> TrackSpec:
    track = _track(tracks, key)
    if not _is_counter_track(track):
        raise ValueError(f"counter references non-counter track {key!r}")
    return track


def _track(tracks: dict[str, TrackSpec], key: str) -> TrackSpec:
    try:
        return tracks[key]
    except KeyError as error:
        raise ValueError(f"unknown track key: {key!r}") from error


def _is_counter_track(track: TrackSpec) -> bool:
    return track.kind.strip().casefold() == "counter"


def _set_counter_descriptor(
    descriptor: CounterDescriptor,
    unit: str | None,
) -> None:
    # Accessing a protobuf submessage alone does not set message presence.
    descriptor.SetInParent()
    if unit is None:
        return
    normalized = unit.strip().casefold()
    if not normalized:
        raise ValueError("counter unit must be non-empty when provided")
    known_unit = _COUNTER_UNITS.get(normalized)
    if known_unit is None:
        descriptor.unit = CounterDescriptor.UNIT_UNSPECIFIED
        descriptor.unit_name = unit
    else:
        descriptor.unit = known_unit


def _add_debug_annotations(
    event: TrackEvent,
    annotations: tuple[tuple[str, AnnotationValue], ...],
) -> None:
    for name, value in annotations:
        annotation = event.debug_annotations.add()
        annotation.name = name
        if isinstance(value, bool):
            annotation.bool_value = value
        elif isinstance(value, int):
            annotation.int_value = value
        elif isinstance(value, float):
            annotation.double_value = value
        elif isinstance(value, str):
            annotation.string_value = value
        else:
            raise TypeError(
                f"unsupported debug annotation value for {name!r}: "
                f"{type(value).__name__}"
            )


def _validate_annotations(
    annotations: tuple[tuple[str, AnnotationValue], ...],
    owner: str,
) -> None:
    for name, value in annotations:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{owner} annotation name must be a non-empty string")
        if isinstance(value, bool) or isinstance(value, str):
            continue
        if isinstance(value, int):
            if not _INT64_MIN <= value <= _INT64_MAX:
                raise ValueError(
                    f"{owner} annotation {name!r} must fit signed int64"
                )
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"{owner} annotation {name!r} must be finite"
                )
            continue
        raise TypeError(
            f"{owner} annotation {name!r} has unsupported value type "
            f"{type(value).__name__}"
        )


def _set_counter_value(event: TrackEvent, value: int | float) -> None:
    if isinstance(value, bool):
        raise TypeError("counter value must not be bool")
    if isinstance(value, int):
        event.counter_value = value
    elif isinstance(value, float):
        event.double_counter_value = value
    else:
        raise TypeError(f"unsupported counter value type: {type(value).__name__}")


def _validate_counter_value(value: int | float, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"counter {name!r} value must not be bool")
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"counter {name!r} value must fit signed int64")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"counter {name!r} value must be finite")
        return
    raise TypeError(
        f"counter {name!r} has unsupported value type {type(value).__name__}"
    )


def _uuid(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 < value <= _UINT64_MAX:
        raise ValueError(f"{name} must be in the uint64 range and non-zero")
    return value


def _flow_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must contain integers")
    if not 0 < value <= _UINT64_MAX:
        raise ValueError(f"{name} must contain non-zero fixed64 values")
    return value


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer number of nanoseconds")
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must fit the uint64 timestamp field")
    return value


def _duration(value: int, slice_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"slice {slice_name!r} duration must be integer nanoseconds")
    if value <= 0:
        raise ValueError(
            f"slice {slice_name!r} duration must be positive; use InstantSpec "
            "for zero-duration events"
        )
    if value > _UINT64_MAX:
        raise ValueError(f"slice {slice_name!r} duration is out of range")
    return value
