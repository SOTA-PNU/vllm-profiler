"""Read-only conversion of supported native profiler artifacts.

The conversion in this module deliberately separates two questions:

* whether a profiler's native timestamp can be interpreted without guessing;
* whether that timestamp can be placed on the run's canonical monotonic axis.

Kineto Chrome traces and Nsight SQLite exports have documented Unix-time
reconstructions. Earlier captures also contain paired Unix/monotonic
samples.  Those samples support a deterministic point transform, but they are
not atomic clock snapshots.  Emitted events are consequently labelled
``partial_derived`` and retain a conservative uncertainty; they are never
reported as exactly aligned.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
from typing import Any, Final

from google.protobuf.message import DecodeError
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

from .loader import LoadedHybridRun, SourceRunMetadata
from .model import CounterSpec, FlowSpec, InstantSpec, SliceSpec, TrackSpec, TracePlan


class NativeDetailError(RuntimeError):
    """A native artifact cannot be converted without weakening its evidence."""


@dataclass(frozen=True, slots=True)
class NativeDetailSummary:
    """Path-free conversion and limitation metadata for one native profiler."""

    profiler_type: str
    source_role: str
    support_status: str
    alignment_status: str
    alignment_method: str
    native_clock_domain: str
    native_timestamp_unit: str
    emitted_event_count: int
    emitted_slice_count: int
    emitted_instant_count: int
    emitted_flow_count: int
    metadata_only_event_count: int
    skipped_event_count: int
    timestamp_fallback_count: int
    fabricated_event_count: int
    alignment_uncertainty_ns: int | None
    clock_offset_ns: int | None
    observed_offset_half_range_ns: int | None
    native_epoch_base_ns: int | None
    clock_sample_offsets_ns: tuple[int, ...]
    canonical_transform_offset_ns: int | None
    clock_formula: str | None
    alignment_valid_interval_ns: tuple[int, int] | None
    mapped_event_interval_ns: tuple[int, int] | None
    event_counts: tuple[tuple[str, int], ...]
    artifact_count: int
    artifact_sha256: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def metadata(self) -> dict[str, object]:
        """Return deterministic JSON-ready metadata without filesystem paths."""

        return {
            "profiler_type": self.profiler_type,
            "source_role": self.source_role,
            "support_status": self.support_status,
            "alignment_status": self.alignment_status,
            "alignment_method": self.alignment_method,
            "native_clock_domain": self.native_clock_domain,
            "native_timestamp_unit": self.native_timestamp_unit,
            "native_details_emitted": self.emitted_event_count > 0,
            "emitted_event_count": self.emitted_event_count,
            "emitted_slice_count": self.emitted_slice_count,
            "emitted_instant_count": self.emitted_instant_count,
            "emitted_flow_count": self.emitted_flow_count,
            "metadata_only_event_count": self.metadata_only_event_count,
            "skipped_event_count": self.skipped_event_count,
            "timestamp_fallback_count": self.timestamp_fallback_count,
            "fabricated_event_count": self.fabricated_event_count,
            "alignment_uncertainty_ns": self.alignment_uncertainty_ns,
            "alignment_uncertainty_kind": (
                "empirical_display_window_not_proven_clock_error_bound"
                if self.alignment_uncertainty_ns is not None
                else "not_available"
            ),
            "clock_error_bound_proven": False,
            "clock_offset_ns": self.clock_offset_ns,
            "observed_offset_half_range_ns": (
                self.observed_offset_half_range_ns
            ),
            "native_epoch_base_ns": self.native_epoch_base_ns,
            "clock_sample_offsets_ns": list(self.clock_sample_offsets_ns),
            "canonical_transform_offset_ns": (
                self.canonical_transform_offset_ns
            ),
            "clock_formula": self.clock_formula,
            "alignment_valid_interval_ns": (
                list(self.alignment_valid_interval_ns)
                if self.alignment_valid_interval_ns is not None
                else None
            ),
            "mapped_event_interval_ns": (
                list(self.mapped_event_interval_ns)
                if self.mapped_event_interval_ns is not None
                else None
            ),
            "event_counts": dict(self.event_counts),
            "artifact_count": self.artifact_count,
            "artifact_sha256": list(self.artifact_sha256),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class NativeDetailResult:
    """Native tracks/events plus their explicit evidence summary."""

    tracks: tuple[TrackSpec, ...] = ()
    slices: tuple[SliceSpec, ...] = ()
    instants: tuple[InstantSpec, ...] = ()
    flows: tuple[FlowSpec, ...] = ()
    summaries: tuple[NativeDetailSummary, ...] = ()
    separate_traces: tuple["NativeTraceView", ...] = ()

    @property
    def emitted_event_count(self) -> int:
        return len(self.slices) + len(self.instants)


@dataclass(frozen=True, slots=True)
class NativeTraceView:
    """Byte-identical native-clock Perfetto trace published separately."""

    profiler_type: str
    source_role: str
    source_relative_path: str
    output_name: str
    validation_name: str
    payload: bytes
    size_bytes: int
    sha256: str
    expected_slice_count: int
    expected_track_count: int
    expected_flow_count: int
    alignment_status: str = "partial_unaligned"
    timestamp_rebased: bool = False

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "profiler_type": self.profiler_type,
            "source_role": self.source_role,
            "source_relative_path": self.source_relative_path,
            "relative_path": self.output_name,
            "validation_relative_path": self.validation_name,
            "format": "perfetto_protobuf",
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "expected_slice_count": self.expected_slice_count,
            "expected_track_count": self.expected_track_count,
            "expected_flow_count": self.expected_flow_count,
            "alignment_status": self.alignment_status,
            "timestamp_rebased": self.timestamp_rebased,
            "canonical_merge": False,
        }


@dataclass(frozen=True, slots=True)
class _ClockBridge:
    source_role: str
    native_clock_domain: str
    native_timestamp_unit: str
    offset_ns: int
    observed_half_range_ns: int
    uncertainty_ns: int
    canonical_offset_ns: int
    sample_offsets_ns: tuple[int, ...]

    def unix_to_canonical(self, unix_ns: int) -> int:
        value = unix_ns - self.offset_ns + self.canonical_offset_ns
        if value < 0:
            raise NativeDetailError("native timestamp maps before canonical zero")
        return value


@dataclass(frozen=True, slots=True)
class _NativeSlice:
    spec: SliceSpec
    category: str
    correlation_id: int | None
    endpoint_kind: str
    correlation_scope: str | None = None


@dataclass(frozen=True, slots=True)
class _ChromeEvent:
    artifact_index: int
    phase: str
    category: str
    name: str
    pid: str
    tid: str
    timestamp: Decimal | None
    duration: Decimal | None
    event_id: str | None
    args: Mapping[str, Any]


_SAFE_ANNOTATION_RE: Final = re.compile(r"[^A-Za-z0-9_]+")
_NSYS_REQUIRED_TABLES: Final = {
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "ENUM_CUDA_MEMCPY_OPER",
    "ENUM_CUDA_MEM_KIND",
    "NVTX_EVENTS",
    "PROCESSES",
    "StringIds",
    "TARGET_INFO_SESSION_START_TIME",
}
# Nsight's installed official reports join CUDA API rows to device rows with
# ``globalPid == (globalTid & 0xFFFFFFFFFF000000)``. Preserve that process
# scope so equal numeric correlation IDs from different processes never join.
_NSYS_GLOBAL_PID_MASK: Final = 0xFFFFFFFFFF000000
_CHROME_ARG_KEYS: Final = (
    "External id",
    "Record function id",
    "Ev Idx",
    "correlation",
    "cbid",
    "device",
    "context",
    "stream",
    "bytes",
    "bandwidth",
    "grid",
    "block",
    "registers per thread",
    "shared memory",
    "memory bandwidth (GB/s)",
    "blocks per SM",
    "warps per SM",
    "est. achieved occupancy %",
    "Op count",
    "queued",
)


def build_native_detail_plan(
    loaded: LoadedHybridRun,
    base_plan: TracePlan,
) -> NativeDetailResult:
    """Convert the detailed profiler in ``loaded`` without changing inputs."""

    if not isinstance(loaded, LoadedHybridRun):
        raise TypeError("loaded must be LoadedHybridRun")
    if not isinstance(base_plan, TracePlan):
        raise TypeError("base_plan must be TracePlan")
    if not loaded.native_envelopes:
        return NativeDetailResult()

    results: list[NativeDetailResult] = []
    source_by_role = loaded.source_by_role
    for envelope in sorted(
        loaded.native_envelopes,
        key=lambda item: (item.source_role, item.profiler_type),
    ):
        source = source_by_role[envelope.source_role]
        if envelope.profiler_type in {"gpu_torch", "npu_vllm"}:
            results.append(
                _chrome_detail_result(
                    loaded,
                    source,
                    profiler_type=envelope.profiler_type,
                    native_clock_domain=envelope.native_clock_domain,
                    native_timestamp_unit=envelope.native_timestamp_unit,
                    host_boundary_uncertainty_ns=envelope.uncertainty_ns,
                )
            )
        elif envelope.profiler_type == "gpu_nsys":
            results.append(
                _nsys_detail_result(
                    loaded,
                    source,
                    native_clock_domain=envelope.native_clock_domain,
                    native_timestamp_unit=envelope.native_timestamp_unit,
                    host_boundary_uncertainty_ns=envelope.uncertainty_ns,
                )
            )
        elif envelope.profiler_type == "npu_rbln":
            results.append(
                _rbln_native_only_result(
                    source,
                    native_clock_domain=envelope.native_clock_domain,
                    native_timestamp_unit=envelope.native_timestamp_unit,
                )
            )
        else:
            raise NativeDetailError(
                f"unsupported native profiler type: {envelope.profiler_type}"
            )

    tracks = tuple(track for result in results for track in result.tracks)
    slices = tuple(spec for result in results for spec in result.slices)
    instants = tuple(spec for result in results for spec in result.instants)
    flows = tuple(flow for result in results for flow in result.flows)
    summaries = tuple(
        summary for result in results for summary in result.summaries
    )
    separate_traces = tuple(
        trace for result in results for trace in result.separate_traces
    )
    _validate_combined_native_plan(base_plan, tracks, slices, instants, flows)
    return NativeDetailResult(
        tracks=tuple(sorted(tracks, key=lambda item: (item.uuid, item.key))),
        slices=tuple(
            sorted(
                slices,
                key=lambda item: (
                    item.timestamp_ns,
                    item.track_key,
                    item.duration_ns,
                    item.name,
                    item.annotations,
                ),
            )
        ),
        instants=tuple(
            sorted(
                instants,
                key=lambda item: (
                    item.timestamp_ns,
                    item.track_key,
                    item.name,
                    item.annotations,
                ),
            )
        ),
        flows=tuple(sorted(flows, key=lambda item: item.flow_id)),
        summaries=summaries,
        separate_traces=separate_traces,
    )


def augment_trace_plan(
    base_plan: TracePlan,
    native: NativeDetailResult,
) -> TracePlan:
    """Return ``base_plan`` with evidence-backed native details appended."""

    if not native.summaries:
        return base_plan
    summaries = list(native.summaries)
    native_types = {
        item.profiler_type for item in summaries if item.emitted_event_count
    }
    summary_by_type = {item.profiler_type: item for item in summaries}
    slices: list[SliceSpec] = []
    for spec in base_plan.slices:
        if spec.track_key != "profiler":
            slices.append(spec)
            continue
        annotations = dict(spec.annotations)
        profiler_type = annotations.get("hetero.profiler_type")
        summary = summary_by_type.get(str(profiler_type))
        if summary is None:
            slices.append(spec)
            continue
        emitted = profiler_type in native_types
        annotations.update(
            {
                "hetero.native_details_emitted": emitted,
                "hetero.unaligned_profiler_events": not emitted,
                "hetero.native_event_alignment": summary.alignment_status,
                "hetero.native_alignment_method": summary.alignment_method,
                "hetero.timestamp_fallback_count": 0,
                "hetero.fabricated_event_count": 0,
            }
        )
        if summary.alignment_uncertainty_ns is not None:
            annotations["hetero.native_alignment_uncertainty_ns"] = (
                summary.alignment_uncertainty_ns
            )
        if profiler_type == "npu_rbln":
            annotations.update(
                {
                    "hetero.rbln_pb_classification": (
                        "perfetto_compatible_rbln_trace"
                    ),
                    "hetero.rbln_pb_structure_analysis": (
                        "official_perfetto_protobuf_schema"
                    ),
                }
            )
        slices.append(
            replace(spec, annotations=tuple(sorted(annotations.items())))
        )

    instants: list[InstantSpec] = []
    for spec in base_plan.instants:
        if spec.track_key == "clock_metadata":
            annotations = dict(spec.annotations)
            annotations["hetero.native_details_emitted"] = bool(native_types)
            instants.append(
                replace(spec, annotations=tuple(sorted(annotations.items())))
            )
            continue
        if spec.track_key != "summary.data_quality":
            instants.append(spec)
            continue
        annotations = dict(spec.annotations)
        annotations["hetero.native_details_emitted"] = bool(native_types)
        annotations["hetero.native_profiler_alignment"] = (
            "partial_derived"
            if native_types
            else "partial_or_unaligned"
        )
        annotations["hetero.timestamp_fallback_count"] = 0
        annotations["hetero.fabricated_event_count"] = 0
        if "npu_rbln" in summary_by_type:
            annotations.update(
                {
                    "hetero.rbln_pb_state": (
                        "perfetto_trace_proto_native_unaligned"
                    ),
                    "hetero.rbln_pb_structure_analysis": (
                        "official_perfetto_protobuf_schema"
                    ),
                }
            )
        instants.append(
            replace(spec, annotations=tuple(sorted(annotations.items())))
        )

    base_track_keys = {track.key for track in base_plan.tracks}
    native_parent = (
        "summary.native_details"
        if "summary.native_details" in base_track_keys
        else None
    )
    native_tracks = tuple(
        replace(
            track,
            parent_key=native_parent,
            # Rank 0 is reserved for the diagnostic capture envelope.
            sibling_order_rank=(1 if native_parent is not None else None),
        )
        if track.parent_key == "summary.root"
        else track
        for track in native.tracks
    )
    return replace(
        base_plan,
        tracks=tuple((*base_plan.tracks, *native_tracks)),
        slices=tuple(
            sorted(
                (*slices, *native.slices),
                key=lambda item: (
                    item.timestamp_ns,
                    item.track_key,
                    item.duration_ns,
                    item.name,
                    item.annotations,
                ),
            )
        ),
        instants=tuple(
            sorted(
                (*instants, *native.instants),
                key=lambda item: (
                    item.timestamp_ns,
                    item.track_key,
                    item.name,
                    item.annotations,
                ),
            )
        ),
        flows=tuple(
            sorted((*base_plan.flows, *native.flows), key=lambda item: item.flow_id)
        ),
    )


def _client_request_window(plan: TracePlan) -> tuple[int, int, str]:
    window = plan.request_window
    if (
        window is None
        or not window.request_id
        or window.start_ns < 0
        or window.end_ns < window.start_ns
        or window.target_clock_domain_id != plan.canonical_clock_domain_id
    ):
        raise NativeDetailError(
            "request-focused trace requires one canonical client request window"
        )
    return window.start_ns, window.end_ns, window.request_id


def _request_window_counters(
    plan: TracePlan,
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[CounterSpec, ...]:
    by_stream: dict[str, list[CounterSpec]] = defaultdict(list)
    for spec in plan.counters:
        if spec.interval_ns is not None and (
            isinstance(spec.interval_ns, bool) or spec.interval_ns < 0
        ):
            raise NativeDetailError("resource counter interval is invalid")
        by_stream[spec.track_key].append(spec)

    selected: dict[tuple[str, int], CounterSpec] = {}

    def add(spec: CounterSpec) -> None:
        identity = (spec.track_key, spec.timestamp_ns)
        existing = selected.get(identity)
        if existing is not None and existing != spec:
            raise NativeDetailError(
                "resource stream has conflicting samples at one timestamp"
            )
        selected[identity] = spec

    for rows in by_stream.values():
        ordered = sorted(
            rows,
            key=lambda item: (
                item.timestamp_ns,
                repr(item.value),
                repr(item.annotations),
            ),
        )
        baseline = [
            item
            for item in ordered
            if item.sample_role == "baseline" and item.timestamp_ns <= start_ns
        ]
        if baseline:
            add(max(baseline, key=lambda item: item.timestamp_ns))
        for item in ordered:
            if item.sample_role != "background" or item.interval_ns is None:
                continue
            coverage_start = item.timestamp_ns - item.interval_ns
            if item.timestamp_ns >= start_ns and coverage_start < end_ns:
                add(item)
        final = [
            item
            for item in ordered
            if item.sample_role == "final" and item.timestamp_ns >= end_ns
        ]
        if final:
            add(min(final, key=lambda item: item.timestamp_ns))
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                item.timestamp_ns,
                item.track_key,
                repr(item.value),
            ),
        )
    )


def _request_resource_tracks(
    plan: TracePlan,
    counters: tuple[CounterSpec, ...],
) -> tuple[TrackSpec, ...]:
    if not counters:
        return ()
    by_key = plan.track_by_key
    root_key = "summary.request_resources"
    group_keys = {
        by_key[counter.track_key].parent_key for counter in counters
    }
    if None in group_keys or any(
        not str(key).startswith("telemetry.resources.") for key in group_keys
    ):
        raise NativeDetailError("request resource counter hierarchy is invalid")
    group_mapping = {
        str(key): root_key + str(key).removeprefix("telemetry.resources")
        for key in group_keys
    }
    tracks = [
        TrackSpec(
            key=root_key,
            uuid=_stable_uint64(plan.run_id, "request-resource-track", root_key),
            name="Request-window Resource Telemetry",
            kind="group",
            description=(
                "Source-backed baseline, overlapping background, and final "
                "resource samples for the canonical client request window."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=3,
        )
    ]
    tracks.extend(
        replace(
            by_key[group_key],
            key=focused_key,
            uuid=_stable_uint64(
                plan.run_id, "request-resource-track", focused_key
            ),
            parent_key=root_key,
        )
        for group_key, focused_key in sorted(group_mapping.items())
    )
    counter_keys = {counter.track_key for counter in counters}
    tracks.extend(
        replace(
            by_key[key],
            parent_key=group_mapping[str(by_key[key].parent_key)],
        )
        for key in sorted(counter_keys)
    )
    return tuple(tracks)


def request_focused_plan(plan: TracePlan) -> TracePlan:
    """Keep request processing and its source-backed resource subset."""

    start, end, _ = _client_request_window(plan)
    by_key = plan.track_by_key

    def is_under(track_key: str, root_key: str) -> bool:
        current = by_key[track_key]
        while True:
            if current.key == root_key:
                return True
            if current.parent_key is None:
                return False
            current = by_key[current.parent_key]

    def overlaps(spec: SliceSpec) -> bool:
        spec_end = spec.timestamp_ns + spec.duration_ns
        return spec.timestamp_ns < end and spec_end > start

    candidate_slices = tuple(
        spec
        for spec in plan.slices
        if not is_under(spec.track_key, "telemetry.resources")
        and spec.track_key not in {"request", "profiler"}
        and spec.name not in {"Hybrid Request", "Request Summary"}
        and (
            not spec.track_key.startswith("native.")
            or overlaps(spec)
        )
    )
    instants = tuple(
        spec
        for spec in plan.instants
        if not is_under(spec.track_key, "telemetry.resources")
        and (
            spec.track_key == "summary.boundaries.events"
            or (
                spec.track_key.startswith("native.")
                and start <= spec.timestamp_ns < end
            )
        )
    )
    counters = _request_window_counters(plan, start_ns=start, end_ns=end)
    endpoint_counts = Counter(
        flow_id
        for spec in candidate_slices
        for flow_id in (
            *spec.begin_flow_ids,
            *spec.end_flow_ids,
            *spec.begin_terminating_flow_ids,
            *spec.end_terminating_flow_ids,
        )
    )
    flow_ids = {
        flow.flow_id
        for flow in plan.flows
        if endpoint_counts[flow.flow_id] == 2
    }
    slices = tuple(
        replace(
            spec,
            begin_flow_ids=tuple(
                value for value in spec.begin_flow_ids if value in flow_ids
            ),
            end_flow_ids=tuple(
                value for value in spec.end_flow_ids if value in flow_ids
            ),
            begin_terminating_flow_ids=tuple(
                value
                for value in spec.begin_terminating_flow_ids
                if value in flow_ids
            ),
            end_terminating_flow_ids=tuple(
                value
                for value in spec.end_terminating_flow_ids
                if value in flow_ids
            ),
        )
        for spec in candidate_slices
    )
    flows = tuple(flow for flow in plan.flows if flow.flow_id in flow_ids)
    used_keys = {
        *(spec.track_key for spec in slices),
        *(spec.track_key for spec in instants),
    }
    for key in tuple(used_keys):
        current = by_key[key]
        while current.parent_key is not None:
            used_keys.add(current.parent_key)
            current = by_key[current.parent_key]
    tracks = tuple(
        replace(
            track,
            sibling_order_rank=(
                4
                if counters and track.key == "summary.native_details"
                else track.sibling_order_rank
            ),
        )
        for track in plan.tracks
        if track.key in used_keys
    ) + _request_resource_tracks(plan, counters)
    return replace(
        plan,
        tracks=tracks,
        slices=slices,
        instants=instants,
        counters=counters,
        flows=flows,
        presentation_mode=True,
    )


def native_validation_metadata(
    plan: TracePlan,
    native: NativeDetailResult,
    *,
    filtered_subset: bool = False,
) -> dict[str, object]:
    """Build validation facts additionally reconciled by generic TP validation."""

    native_tracks = {
        track.key: track
        for track in plan.tracks
        if track.key.startswith("native.")
    }
    native_slices = [
        spec for spec in plan.slices if spec.track_key in native_tracks
    ]
    native_instants = [
        spec for spec in plan.instants if spec.track_key in native_tracks
    ]
    native_events = [*native_slices, *native_instants]
    invalid_duration_count = sum(spec.duration_ns <= 0 for spec in native_slices)
    timestamp_fallback_count = sum(
        dict(spec.annotations).get("hetero.timestamp_fallback") is not False
        for spec in native_events
    )
    fabricated_event_count = sum(
        dict(spec.annotations).get("hetero.fabricated_event") is not False
        for spec in native_events
    )
    native_flow_ids = {flow.flow_id for flow in native.flows}
    plan_flow_ids = {flow.flow_id for flow in plan.flows}
    emitted_native_flow_ids = native_flow_ids & plan_flow_ids
    (
        expected_slices,
        expected_instants,
        expected_track_keys,
        expected_flow_ids,
    ) = _expected_native_subset(
        plan,
        native,
        filtered_subset=filtered_subset,
    )
    expected = {
        "event": len(expected_slices) + len(expected_instants),
        "slice": len(expected_slices),
        "instant": len(expected_instants),
        "track": len(expected_track_keys),
        "flow": len(expected_flow_ids),
    }
    actual = {
        "event": len(native_slices) + len(native_instants),
        "slice": len(native_slices),
        "instant": len(native_instants),
        "track": len(native_tracks),
        "flow": len(emitted_native_flow_ids),
    }
    counts_reconciled = actual == expected
    identity_reconciled = (
        Counter(_native_event_identity(spec) for spec in native_slices)
        == Counter(_native_event_identity(spec) for spec in expected_slices)
        and Counter(_native_event_identity(spec) for spec in native_instants)
        == Counter(_native_event_identity(spec) for spec in expected_instants)
        and set(native_tracks) == expected_track_keys
        and emitted_native_flow_ids == expected_flow_ids
    )
    clock_alignment_evidence = [
        {
            "profiler_type": item.profiler_type,
            "status": item.alignment_status,
            "method": item.alignment_method,
            "native_clock_domain": item.native_clock_domain,
            "native_timestamp_unit": item.native_timestamp_unit,
            "formula": item.clock_formula,
            "native_epoch_base_ns": item.native_epoch_base_ns,
            "clock_sample_offsets_ns": list(item.clock_sample_offsets_ns),
            "selected_clock_offset_ns": item.clock_offset_ns,
            "canonical_transform_offset_ns": (
                item.canonical_transform_offset_ns
            ),
            "alignment_valid_interval_ns": (
                list(item.alignment_valid_interval_ns)
                if item.alignment_valid_interval_ns is not None
                else None
            ),
            "mapped_event_interval_ns": (
                list(item.mapped_event_interval_ns)
                if item.mapped_event_interval_ns is not None
                else None
            ),
            "uncertainty_ns": item.alignment_uncertainty_ns,
            "uncertainty_kind": (
                "empirical_display_window_not_proven_clock_error_bound"
                if item.alignment_uncertainty_ns is not None
                else "not_available"
            ),
            "clock_error_bound_proven": False,
        }
        for item in native.summaries
    ]
    return {
        "valid": (
            invalid_duration_count == 0
            and counts_reconciled
            and identity_reconciled
            and timestamp_fallback_count == 0
            and fabricated_event_count == 0
            and all(item.timestamp_fallback_count == 0 for item in native.summaries)
            and all(item.fabricated_event_count == 0 for item in native.summaries)
        ),
        "native_event_count": len(native_slices) + len(native_instants),
        "native_slice_count": len(native_slices),
        "native_instant_count": len(native_instants),
        "native_track_count": len(native_tracks),
        "native_flow_count": len(emitted_native_flow_ids),
        "track_event_counts": dict(
            sorted(
                Counter(
                    spec.track_key
                    for spec in native_events
                ).items()
            )
        ),
        "invalid_duration_count": invalid_duration_count,
        "expected_native_counts": expected,
        "native_counts_reconciled": counts_reconciled,
        "native_identity_reconciled": identity_reconciled,
        "filtered_subset_allowed": filtered_subset,
        "parent_child_range_violation_count": None,
        "parent_child_range_status": (
            "not_available_no_explicit_native_parent_id"
        ),
        "cuda_correlation_policy": "explicit_unique_id_only",
        "timestamp_fallback_count": timestamp_fallback_count,
        "fabricated_event_count": fabricated_event_count,
        "hybrid_alignment_status": clock_alignment_evidence,
        "clock_alignment_evidence": clock_alignment_evidence,
    }


def _expected_native_subset(
    plan: TracePlan,
    native: NativeDetailResult,
    *,
    filtered_subset: bool,
) -> tuple[
    tuple[SliceSpec, ...],
    tuple[InstantSpec, ...],
    set[str],
    set[int],
]:
    if filtered_subset:
        start, end, _ = _client_request_window(plan)
        slices = tuple(
            spec
            for spec in native.slices
            if spec.timestamp_ns < end
            and spec.timestamp_ns + spec.duration_ns > start
        )
        instants = tuple(
            spec
            for spec in native.instants
            if start <= spec.timestamp_ns < end
        )
    else:
        slices = native.slices
        instants = native.instants

    endpoint_counts = Counter(
        flow_id
        for spec in slices
        for flow_id in (
            *spec.begin_flow_ids,
            *spec.end_flow_ids,
            *spec.begin_terminating_flow_ids,
            *spec.end_terminating_flow_ids,
        )
    )
    flow_ids = {
        flow.flow_id
        for flow in native.flows
        if endpoint_counts[flow.flow_id] == 2
    }
    track_by_key = {track.key: track for track in native.tracks}
    track_keys = {
        *(spec.track_key for spec in slices),
        *(spec.track_key for spec in instants),
    }
    for key in tuple(track_keys):
        current = track_by_key[key]
        while (
            current.parent_key is not None
            and current.parent_key in track_by_key
        ):
            track_keys.add(current.parent_key)
            current = track_by_key[current.parent_key]
    return slices, instants, track_keys, flow_ids


def _native_event_identity(
    spec: SliceSpec | InstantSpec,
) -> tuple[object, ...]:
    if isinstance(spec, SliceSpec):
        return (
            "slice",
            spec.track_key,
            spec.name,
            spec.timestamp_ns,
            spec.duration_ns,
            spec.annotations,
        )
    return (
        "instant",
        spec.track_key,
        spec.name,
        spec.timestamp_ns,
        spec.annotations,
    )


def _chrome_detail_result(
    loaded: LoadedHybridRun,
    source: SourceRunMetadata,
    *,
    profiler_type: str,
    native_clock_domain: str,
    native_timestamp_unit: str,
    host_boundary_uncertainty_ns: int,
) -> NativeDetailResult:
    artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.format == "chrome_trace_json_gzip"
        ),
        key=lambda item: item.relative_path,
    )
    if not artifacts:
        raise NativeDetailError(f"{profiler_type} has no Chrome trace artifacts")
    alignment = _read_alignment(source)
    bridge = _clock_bridge(
        loaded,
        source,
        alignment,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        host_boundary_uncertainty_ns=host_boundary_uncertainty_ns,
    )

    events: list[_ChromeEvent] = []
    base_times: set[int] = set()
    process_names: dict[str, str] = {}
    thread_names: dict[tuple[str, str], str] = {}
    metadata_count = 0
    skipped = 0
    for artifact_index, artifact in enumerate(artifacts):
        path = _artifact_path(source, artifact)
        document = _stable_gzip_json(path, artifact)
        base = document.get("baseTimeNanoseconds")
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise NativeDetailError("Chrome trace lacks integer baseTimeNanoseconds")
        base_times.add(base)
        raw_events = document.get("traceEvents")
        if not isinstance(raw_events, list):
            raise NativeDetailError("Chrome trace traceEvents must be an array")
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise NativeDetailError("Chrome trace event must be an object")
            phase = str(raw.get("ph", ""))
            category = str(raw.get("cat", ""))
            name = str(raw.get("name", ""))
            pid = _identity(raw.get("pid"))
            tid = _identity(raw.get("tid"))
            args = raw.get("args")
            if not isinstance(args, dict):
                args = {}
            if phase == "M":
                metadata_count += 1
                label = args.get("name")
                if isinstance(label, str) and label:
                    if name == "process_name":
                        process_names[pid] = label
                    elif name == "thread_name":
                        thread_names[(pid, tid)] = label
                continue
            timestamp = _decimal_or_none(raw.get("ts"))
            duration = _decimal_or_none(raw.get("dur"))
            event_id = (
                str(raw["id"]) if raw.get("id") is not None else None
            )
            if phase not in {"X", "i", "I", "s", "f"}:
                skipped += 1
                continue
            events.append(
                _ChromeEvent(
                    artifact_index=artifact_index,
                    phase=phase,
                    category=category,
                    name=name,
                    pid=pid,
                    tid=tid,
                    timestamp=timestamp,
                    duration=duration,
                    event_id=event_id,
                    args=args,
                )
            )
    if len(base_times) != 1:
        raise NativeDetailError("Chrome traces have inconsistent time bases")
    base_time_ns = next(iter(base_times))

    root_key = f"native.{profiler_type}"
    root_name = (
        "GPU Torch native details (partial alignment)"
        if profiler_type == "gpu_torch"
        else "NPU vLLM native details (partial alignment)"
    )
    tracks: dict[str, TrackSpec] = {
        root_key: TrackSpec(
            key=root_key,
            uuid=_stable_uint64(loaded.manifest.run_id, "track", root_key),
            name=root_name,
            kind="group",
            description=(
                "Native profiler events derived from documented Kineto Unix "
                "timestamps and recorded Unix/monotonic samples; alignment is "
                "partial, never exact."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=5,
        )
    }
    category_keys: dict[str, str] = {}
    native_slices: list[_NativeSlice] = []
    instants: list[InstantSpec] = []
    counts: Counter[str] = Counter()
    flow_host_pids: dict[int, set[str]] = defaultdict(set)
    for event in events:
        _, endpoint_kind = _chrome_category(
            profiler_type,
            event.category,
            event.name,
            event.phase,
        )
        if (
            endpoint_kind == "host_api"
            and _non_bool_int_or_none(
                event.args.get("correlation")
            )
            is not None
        ):
            flow_host_pids[event.artifact_index].add(event.pid)

    category_order = _chrome_category_order(profiler_type)
    chrome_flow_marker_count = 0
    for event in events:
        if event.phase in {"s", "f"}:
            chrome_flow_marker_count += 1
            continue
        if event.timestamp is None:
            raise NativeDetailError("Chrome activity event lacks timestamp")
        unix_ns = base_time_ns + _microseconds_to_ns(event.timestamp)
        timestamp_ns = bridge.unix_to_canonical(unix_ns)
        category_name, endpoint_kind = _chrome_category(
            profiler_type,
            event.category,
            event.name,
            event.phase,
        )
        category_key = category_keys.get(category_name)
        if category_key is None:
            category_key = (
                f"{root_key}.category."
                f"{_stable_token(category_name)}"
            )
            category_keys[category_name] = category_key
            tracks[category_key] = TrackSpec(
                key=category_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", category_key
                ),
                name=category_name,
                kind="group",
                description=f"{profiler_type} {category_name} events.",
                parent_key=root_key,
                child_ordering="lexicographic",
                sibling_order_rank=category_order[category_name],
            )
        leaf_identity = _chrome_leaf_identity(event, endpoint_kind)
        leaf_key = (
            f"{category_key}.lane.{_stable_token(leaf_identity)}"
        )
        if leaf_key not in tracks:
            tracks[leaf_key] = TrackSpec(
                key=leaf_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", leaf_key
                ),
                name=_chrome_leaf_name(
                    event,
                    endpoint_kind,
                    process_names=process_names,
                    thread_names=thread_names,
                ),
                kind="slice",
                description=(
                    "Original native process/thread/stream identity; timestamp "
                    "point is conditionally mapped and annotated with uncertainty."
                ),
                parent_key=category_key,
            )
        annotations = _chrome_annotations(
            event,
            profiler_type=profiler_type,
            bridge=bridge,
            original_timestamp=event.timestamp,
            original_duration=event.duration,
            process_name=process_names.get(event.pid),
            thread_name=thread_names.get((event.pid, event.tid)),
        )
        correlation = _non_bool_int_or_none(event.args.get("correlation"))
        if event.phase == "X":
            if event.duration is None:
                raise NativeDetailError("Chrome X event lacks duration")
            duration_ns = _microseconds_to_ns(event.duration)
            if duration_ns == 0:
                instant_annotations = dict(annotations)
                instant_annotations[
                    "hetero.native_zero_duration_complete_event"
                ] = True
                instants.append(
                    InstantSpec(
                        track_key=leaf_key,
                        name=event.name,
                        timestamp_ns=timestamp_ns,
                        annotations=tuple(
                            sorted(instant_annotations.items())
                        ),
                    )
                )
                counts[
                    f"{category_name} (zero-duration complete instant)"
                ] += 1
                continue
            host_pids = flow_host_pids[event.artifact_index]
            if len(host_pids) == 1:
                correlation_scope = (
                    f"artifact:{event.artifact_index}:host-pid:"
                    f"{next(iter(host_pids))}"
                )
            elif endpoint_kind == "host_api":
                correlation_scope = (
                    f"artifact:{event.artifact_index}:host-pid:{event.pid}"
                )
            else:
                correlation_scope = (
                    f"artifact:{event.artifact_index}:"
                    "device-without-unique-host-process"
                )
            native_slices.append(
                _NativeSlice(
                    spec=SliceSpec(
                        track_key=leaf_key,
                        name=event.name,
                        timestamp_ns=timestamp_ns,
                        duration_ns=duration_ns,
                        annotations=annotations,
                    ),
                    category=category_name,
                    correlation_id=correlation,
                    endpoint_kind=endpoint_kind,
                    correlation_scope=correlation_scope,
                )
            )
            counts[category_name] += 1
        else:
            instants.append(
                InstantSpec(
                    track_key=leaf_key,
                    name=event.name,
                    timestamp_ns=timestamp_ns,
                    annotations=annotations,
                )
            )
            counts[f"{category_name} (instant)"] += 1

    skipped += chrome_flow_marker_count
    if chrome_flow_marker_count:
        counts["Chrome flow markers (not emitted)"] = (
            chrome_flow_marker_count
        )
    native_slices, flows = _attach_explicit_flows(
        loaded.manifest.run_id,
        profiler_type,
        native_slices,
    )
    valid_interval, mapped_interval = _validate_mapped_interval(
        alignment,
        bridge,
        tuple(item.spec for item in native_slices),
        tuple(instants),
    )
    summary = NativeDetailSummary(
        profiler_type=profiler_type,
        source_role=source.source_role,
        support_status="converted",
        alignment_status="partial_derived",
        alignment_method=(
            "documented_kineto_unix_time_plus_recorded_host_clock_samples"
        ),
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=len(native_slices) + len(instants),
        emitted_slice_count=len(native_slices),
        emitted_instant_count=len(instants),
        emitted_flow_count=len(flows),
        metadata_only_event_count=metadata_count,
        skipped_event_count=skipped,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=bridge.uncertainty_ns,
        clock_offset_ns=bridge.offset_ns,
        observed_offset_half_range_ns=bridge.observed_half_range_ns,
        native_epoch_base_ns=base_time_ns,
        clock_sample_offsets_ns=bridge.sample_offsets_ns,
        canonical_transform_offset_ns=bridge.canonical_offset_ns,
        clock_formula=(
            "canonical_ns = baseTimeNanoseconds + Decimal(ts_us)*1000 "
            "- clock_offset_ns + canonical_transform_offset_ns"
        ),
        alignment_valid_interval_ns=valid_interval,
        mapped_event_interval_ns=mapped_interval,
        event_counts=tuple(sorted(counts.items())),
        artifact_count=len(artifacts),
        artifact_sha256=tuple(item.sha256 for item in artifacts),
        notes=(
            "baseTimeNanoseconds + exact Decimal(ts_us)*1000 reconstructs Unix ns",
            "Unix/monotonic samples are non-atomic; alignment remains partial",
            "reported uncertainty is not a proven clock-error bound",
            (
                "Chrome s/f markers are counted but not emitted as "
                "API-to-device flows"
            ),
            (
                "no event is classified as NPU device execution; "
                "unrecognized source events remain execution-domain unverified"
                if profiler_type == "npu_vllm"
                else "CUDA flows require one unique explicit correlation ID"
            ),
        ),
    )
    return NativeDetailResult(
        tracks=tuple(tracks.values()),
        slices=tuple(item.spec for item in native_slices),
        instants=tuple(instants),
        flows=flows,
        summaries=(summary,),
    )


def _nsys_detail_result(
    loaded: LoadedHybridRun,
    source: SourceRunMetadata,
    *,
    native_clock_domain: str,
    native_timestamp_unit: str,
    host_boundary_uncertainty_ns: int,
) -> NativeDetailResult:
    sqlite_artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.format == "sqlite"
            and artifact.relative_path.endswith(".sqlite")
        ),
        key=lambda item: item.relative_path,
    )
    report_artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.format == "nsys-rep"
        ),
        key=lambda item: item.relative_path,
    )
    if len(sqlite_artifacts) != 1 or len(report_artifacts) != 1:
        raise NativeDetailError(
            "gpu_nsys requires exactly one existing SQLite export and report"
        )
    sqlite_artifact = sqlite_artifacts[0]
    sqlite_path = _artifact_path(source, sqlite_artifact)
    _stable_file_identity(sqlite_path, sqlite_artifact)
    _stable_file_identity(
        _artifact_path(source, report_artifacts[0]), report_artifacts[0]
    )
    alignment = _read_alignment(source)
    bridge = _clock_bridge(
        loaded,
        source,
        alignment,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        host_boundary_uncertainty_ns=host_boundary_uncertainty_ns,
    )

    uri = f"file:{sqlite_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise NativeDetailError("Nsight SQLite quick_check failed")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = sorted(_NSYS_REQUIRED_TABLES - tables)
        if missing:
            raise NativeDetailError(
                f"Nsight SQLite lacks required tables: {missing}"
            )
        start_rows = connection.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchall()
        if (
            len(start_rows) != 1
            or isinstance(start_rows[0][0], bool)
            or not isinstance(start_rows[0][0], int)
        ):
            raise NativeDetailError("Nsight session start time is invalid")
        session_unix_ns = start_rows[0][0]
        strings = {
            int(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT id, value FROM StringIds ORDER BY id"
            )
        }
        process_names = {
            int(row[0]): (int(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT globalPid, pid, name FROM PROCESSES ORDER BY globalPid"
            )
        }
        slices, tracks, counts, metadata_count = _read_nsys_rows(
            loaded,
            connection,
            strings=strings,
            process_names=process_names,
            session_unix_ns=session_unix_ns,
            bridge=bridge,
        )
    finally:
        connection.close()
    _stable_file_identity(sqlite_path, sqlite_artifact)

    slices, flows = _attach_explicit_flows(
        loaded.manifest.run_id,
        "gpu_nsys",
        slices,
    )
    valid_interval, mapped_interval = _validate_mapped_interval(
        alignment,
        bridge,
        tuple(item.spec for item in slices),
        (),
    )
    summary = NativeDetailSummary(
        profiler_type="gpu_nsys",
        source_role=source.source_role,
        support_status="converted_from_existing_official_sqlite_export",
        alignment_status="partial_derived",
        alignment_method=(
            "nsight_utcEpochNs_plus_native_ns_plus_recorded_host_clock_samples"
        ),
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=len(slices),
        emitted_slice_count=len(slices),
        emitted_instant_count=0,
        emitted_flow_count=len(flows),
        metadata_only_event_count=metadata_count,
        skipped_event_count=0,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=bridge.uncertainty_ns,
        clock_offset_ns=bridge.offset_ns,
        observed_offset_half_range_ns=bridge.observed_half_range_ns,
        native_epoch_base_ns=session_unix_ns,
        clock_sample_offsets_ns=bridge.sample_offsets_ns,
        canonical_transform_offset_ns=bridge.canonical_offset_ns,
        clock_formula=(
            "canonical_ns = utcEpochNs + activity.start "
            "- clock_offset_ns + canonical_transform_offset_ns"
        ),
        alignment_valid_interval_ns=valid_interval,
        mapped_event_interval_ns=mapped_interval,
        event_counts=tuple(sorted(counts.items())),
        artifact_count=2,
        artifact_sha256=(
            report_artifacts[0].sha256,
            sqlite_artifact.sha256,
        ),
        notes=(
            "existing SQLite export is opened read-only and immutable",
            "API-to-device flows require one unique official correlationId",
            "Unix/monotonic samples are non-atomic; alignment remains partial",
            "reported uncertainty is not a proven clock-error bound",
        ),
    )
    return NativeDetailResult(
        tracks=tuple(tracks.values()),
        slices=tuple(item.spec for item in slices),
        flows=flows,
        summaries=(summary,),
    )


def _read_nsys_rows(
    loaded: LoadedHybridRun,
    connection: sqlite3.Connection,
    *,
    strings: Mapping[int, str],
    process_names: Mapping[int, tuple[int, str]],
    session_unix_ns: int,
    bridge: _ClockBridge,
) -> tuple[
    list[_NativeSlice],
    dict[str, TrackSpec],
    Counter[str],
    int,
]:
    root_key = "native.gpu_nsys"
    tracks: dict[str, TrackSpec] = {
        root_key: TrackSpec(
            key=root_key,
            uuid=_stable_uint64(loaded.manifest.run_id, "track", root_key),
            name="GPU Nsight native details (partial alignment)",
            kind="group",
            description=(
                "Nsight activities from the immutable official SQLite export; "
                "timestamps use session UTC evidence and partial clock samples."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=5,
        )
    }
    category_order = {
        "NVTX ranges": 0,
        "CUDA Runtime API": 1,
        "CUDA Driver API": 2,
        "CUDA kernels": 3,
        "CUDA memcpy": 4,
        "CUDA memset": 5,
    }
    category_keys: dict[str, str] = {}
    result: list[_NativeSlice] = []
    counts: Counter[str] = Counter()

    def ensure_track(
        category: str,
        identity: str,
        lane_name: str,
    ) -> str:
        category_key = category_keys.get(category)
        if category_key is None:
            category_key = (
                f"{root_key}.category.{_stable_token(category)}"
            )
            category_keys[category] = category_key
            tracks[category_key] = TrackSpec(
                key=category_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", category_key
                ),
                name=category,
                kind="group",
                description=f"Nsight {category}.",
                parent_key=root_key,
                child_ordering="lexicographic",
                sibling_order_rank=category_order[category],
            )
        leaf_key = (
            f"{category_key}.lane.{_stable_token(identity)}"
        )
        if leaf_key not in tracks:
            tracks[leaf_key] = TrackSpec(
                key=leaf_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", leaf_key
                ),
                name=lane_name,
                kind="slice",
                description="Original Nsight thread or CUDA stream identity.",
                parent_key=category_key,
            )
        return leaf_key

    runtime_rows = connection.execute(
        """
        SELECT start, end, eventClass, globalTid, correlationId, nameId,
               returnValue
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start, end, eventClass, globalTid, correlationId, nameId
        """
    )
    for start, end, event_class, global_tid, correlation, name_id, return_value in runtime_rows:
        category = _nsys_api_category(event_class)
        lane = f"Nsight globalTid {global_tid}"
        leaf = ensure_track(category, f"tid:{global_tid}", lane)
        name = strings.get(int(name_id), f"StringId {name_id}")
        annotations = _nsys_annotations(
            bridge,
            native_start_ns=int(start),
            values={
                "category": category,
                "global_tid": global_tid,
                "correlation_id": correlation,
                "return_value": return_value,
                "event_class": event_class,
            },
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=name,
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight API"),
                    annotations=annotations,
                ),
                category=category,
                correlation_id=_non_bool_int_or_none(correlation),
                endpoint_kind="host_api",
                correlation_scope=(
                    f"nsight-process:{int(global_tid) & _NSYS_GLOBAL_PID_MASK}"
                ),
            )
        )
        counts[category] += 1

    kernel_rows = connection.execute(
        """
        SELECT start, end, deviceId, contextId, streamId, correlationId,
               globalPid, demangledName, shortName, gridX, gridY, gridZ,
               blockX, blockY, blockZ, registersPerThread,
               staticSharedMemory, dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start, end, deviceId, contextId, streamId, correlationId
        """
    )
    for row in kernel_rows:
        (
            start,
            end,
            device,
            context,
            stream,
            correlation,
            global_pid,
            demangled_name,
            short_name,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            registers,
            static_shared,
            dynamic_shared,
        ) = row
        category = "CUDA kernels"
        lane = (
            f"GPU {device} / context {context} / stream {stream}"
        )
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            lane,
        )
        process = process_names.get(int(global_pid)) if global_pid is not None else None
        values = {
            "category": category,
            "device": device,
            "context": context,
            "stream": stream,
            "correlation_id": correlation,
            "global_pid": global_pid,
            "pid": process[0] if process else None,
            "process_name": process[1] if process else None,
            "grid": f"{grid_x},{grid_y},{grid_z}",
            "block": f"{block_x},{block_y},{block_z}",
            "registers_per_thread": registers,
            "static_shared_memory": static_shared,
            "dynamic_shared_memory": dynamic_shared,
        }
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=strings.get(
                        int(demangled_name),
                        strings.get(int(short_name), f"StringId {short_name}"),
                    ),
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight kernel"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values=values,
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    memcpy_rows = connection.execute(
        """
        SELECT m.start, m.end, m.deviceId, m.contextId, m.streamId,
               m.correlationId, m.globalPid, m.bytes, m.copyKind,
               copy.label, m.srcKind, src.label, m.dstKind, dst.label
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS copy ON copy.id = m.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS src ON src.id = m.srcKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS dst ON dst.id = m.dstKind
        ORDER BY m.start, m.end, m.deviceId, m.contextId, m.streamId,
                 m.correlationId
        """
    )
    for row in memcpy_rows:
        (
            start,
            end,
            device,
            context,
            stream,
            correlation,
            global_pid,
            byte_count,
            copy_kind,
            copy_label,
            src_kind,
            src_label,
            dst_kind,
            dst_label,
        ) = row
        category = "CUDA memcpy"
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            f"GPU {device} / context {context} / stream {stream}",
        )
        process = process_names.get(int(global_pid)) if global_pid is not None else None
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=f"Memcpy {copy_label or copy_kind}",
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight memcpy"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "device": device,
                            "context": context,
                            "stream": stream,
                            "correlation_id": correlation,
                            "global_pid": global_pid,
                            "pid": process[0] if process else None,
                            "process_name": process[1] if process else None,
                            "bytes": byte_count,
                            "copy_kind": copy_kind,
                            "copy_label": copy_label,
                            "source_kind": src_kind,
                            "source_label": src_label,
                            "destination_kind": dst_kind,
                            "destination_label": dst_label,
                        },
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    memset_rows = connection.execute(
        """
        SELECT start, end, deviceId, contextId, streamId, correlationId,
               globalPid, value, bytes, memKind
        FROM CUPTI_ACTIVITY_KIND_MEMSET
        ORDER BY start, end, deviceId, contextId, streamId, correlationId
        """
    )
    for start, end, device, context, stream, correlation, global_pid, value, byte_count, mem_kind in memset_rows:
        category = "CUDA memset"
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            f"GPU {device} / context {context} / stream {stream}",
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name="Memset",
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight memset"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "device": device,
                            "context": context,
                            "stream": stream,
                            "correlation_id": correlation,
                            "global_pid": global_pid,
                            "value": value,
                            "bytes": byte_count,
                            "memory_kind": mem_kind,
                        },
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    metadata_count = 0
    nvtx_rows = connection.execute(
        """
        SELECT start, end, eventType, rangeId, text, globalTid, textId,
               domainId
        FROM NVTX_EVENTS
        ORDER BY start, end, eventType, globalTid, rangeId
        """
    )
    for start, end, event_type, range_id, text, global_tid, text_id, domain_id in nvtx_rows:
        if end is None or int(end) <= int(start):
            metadata_count += 1
            continue
        category = "NVTX ranges"
        leaf = ensure_track(
            category,
            f"tid:{global_tid}:domain:{domain_id}",
            f"Nsight globalTid {global_tid} / domain {domain_id}",
        )
        name = (
            str(text)
            if text
            else strings.get(int(text_id), f"NVTX range {range_id}")
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=name,
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "NVTX range"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "event_type": event_type,
                            "range_id": range_id,
                            "global_tid": global_tid,
                            "domain_id": domain_id,
                        },
                    ),
                ),
                category=category,
                correlation_id=None,
                endpoint_kind="annotation",
            )
        )
        counts[category] += 1
    return result, tracks, counts, metadata_count


def _rbln_flow_edge_count(
    endpoints: Mapping[int, Sequence[tuple[int, int, bool]]],
) -> int:
    """Count timestamp-directed Perfetto flow edges without inventing links."""

    edge_count = 0
    for flow_id, rows in endpoints.items():
        active = False
        for _timestamp_ns, _packet_index, terminating in sorted(rows):
            if terminating:
                if not active:
                    raise NativeDetailError(
                        "RBLN Perfetto flow terminates before it starts "
                        f"(flow_id={flow_id})"
                    )
                edge_count += 1
                active = False
            elif active:
                edge_count += 1
            else:
                active = True
    return edge_count


def _rbln_native_only_result(
    source: SourceRunMetadata,
    *,
    native_clock_domain: str,
    native_timestamp_unit: str,
) -> NativeDetailResult:
    artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.relative_path.endswith(".pb")
        ),
        key=lambda item: item.relative_path,
    )
    if not artifacts:
        raise NativeDetailError("npu_rbln has no PB artifacts")
    parsed: list[tuple[Any, int, int, int, int, int]] = []
    payloads: dict[str, bytes] = {}
    for artifact in artifacts:
        path = _artifact_path(source, artifact)
        _stable_file_identity(path, artifact)
        before = path.lstat()
        payload = path.read_bytes()
        payloads[artifact.relative_path] = payload
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise NativeDetailError("RBLN PB changed while parsing")
        trace = Trace()
        try:
            trace.ParseFromString(payload)
        except DecodeError as error:
            raise NativeDetailError(
                "RBLN PB is not a standard Perfetto Trace protobuf"
            ) from error
        descriptor_uuids = {
            packet.track_descriptor.uuid
            for packet in trace.packet
            if packet.HasField("track_descriptor")
            and packet.track_descriptor.uuid
        }
        depths: Counter[int] = Counter()
        flow_endpoints: defaultdict[int, list[tuple[int, int, bool]]] = (
            defaultdict(list)
        )
        begin_count = 0
        end_count = 0
        instant_count = 0
        used_track_uuids: set[int] = set()
        for packet_index, packet in enumerate(trace.packet):
            if not packet.HasField("track_event"):
                continue
            event = packet.track_event
            if event.type not in {
                TrackEvent.TYPE_SLICE_BEGIN,
                TrackEvent.TYPE_SLICE_END,
                TrackEvent.TYPE_INSTANT,
            }:
                if event.flow_ids or event.terminating_flow_ids:
                    raise NativeDetailError(
                        "RBLN Perfetto flow uses an unsupported TrackEvent type"
                    )
                continue
            if event.track_uuid == 0:
                raise NativeDetailError(
                    "RBLN Perfetto slice lacks an explicit track UUID"
                )
            used_track_uuids.add(event.track_uuid)
            continuing = tuple(int(value) for value in event.flow_ids)
            terminating = tuple(
                int(value) for value in event.terminating_flow_ids
            )
            if (
                any(value <= 0 for value in (*continuing, *terminating))
                or len(set(continuing)) != len(continuing)
                or len(set(terminating)) != len(terminating)
                or set(continuing).intersection(terminating)
            ):
                raise NativeDetailError(
                    "RBLN Perfetto flow identifiers are invalid"
                )
            if continuing or terminating:
                if not packet.HasField("timestamp"):
                    raise NativeDetailError(
                        "RBLN Perfetto flow endpoint lacks an absolute timestamp"
                    )
                for flow_id in continuing:
                    flow_endpoints[flow_id].append(
                        (int(packet.timestamp), packet_index, False)
                    )
                for flow_id in terminating:
                    flow_endpoints[flow_id].append(
                        (int(packet.timestamp), packet_index, True)
                    )
            if event.type == TrackEvent.TYPE_INSTANT:
                instant_count += 1
                continue
            if event.type == TrackEvent.TYPE_SLICE_BEGIN:
                begin_count += 1
                depths[event.track_uuid] += 1
            else:
                end_count += 1
                depths[event.track_uuid] -= 1
                if depths[event.track_uuid] < 0:
                    raise NativeDetailError(
                        "RBLN Perfetto slice stream closes before it opens"
                    )
        expected_flow_count = _rbln_flow_edge_count(flow_endpoints)
        descriptor_count = sum(
            packet.HasField("track_descriptor") for packet in trace.packet
        )
        clock_snapshot_count = sum(
            packet.HasField("clock_snapshot") for packet in trace.packet
        )
        if clock_snapshot_count:
            raise NativeDetailError(
                "RBLN clock snapshots require an explicit clock-mapping policy"
            )
        if (
            begin_count <= 0
            or begin_count != end_count
            or descriptor_count <= 0
            or any(depths.values())
            or not used_track_uuids.issubset(descriptor_uuids)
        ):
            raise NativeDetailError(
                "RBLN PB lacks a balanced standard Perfetto slice stream"
            )
        parsed.append(
            (
                artifact,
                begin_count + instant_count,
                descriptor_count,
                len(used_track_uuids),
                expected_flow_count,
                clock_snapshot_count,
            )
        )
    aggregate_rows = []
    for candidate in parsed:
        candidate_path = PurePosixPath(candidate[0].relative_path)
        shard_pattern = re.compile(
            rf"{re.escape(candidate_path.stem)}_\d+\.pb$"
        )
        direct_shards = [
            item
            for item in parsed
            if PurePosixPath(item[0].relative_path).parent
            == candidate_path.parent
            and shard_pattern.fullmatch(
                PurePosixPath(item[0].relative_path).name
            )
        ]
        if len(parsed) == 1 or len(direct_shards) == len(parsed) - 1:
            aggregate_rows.append(candidate)
    if len(aggregate_rows) != 1:
        raise NativeDetailError(
            "RBLN capture must have exactly one unnumbered aggregate PB"
        )
    aggregate = aggregate_rows[0]
    aggregate_slices = aggregate[1]
    aggregate_flows = aggregate[4]
    shard_slices = sum(
        item[1]
        for item in parsed
        if item[0].relative_path != aggregate[0].relative_path
    )
    if len(parsed) > 1 and aggregate_slices != shard_slices:
        raise NativeDetailError(
            "RBLN aggregate/shard Perfetto slice counts do not reconcile"
        )
    shard_flows = sum(
        item[4]
        for item in parsed
        if item[0].relative_path != aggregate[0].relative_path
    )
    if len(parsed) > 1 and aggregate_flows != shard_flows:
        raise NativeDetailError(
            "RBLN aggregate/shard Perfetto flow counts do not reconcile"
        )
    clock_snapshots = sum(item[5] for item in parsed)
    summary = NativeDetailSummary(
        profiler_type="npu_rbln",
        source_role=source.source_role,
        support_status="separate_native_perfetto_trace_unaligned",
        alignment_status="partial_unaligned",
        alignment_method="none_no_clock_snapshot_or_shared_anchor",
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=0,
        emitted_slice_count=0,
        emitted_instant_count=0,
        emitted_flow_count=0,
        metadata_only_event_count=0,
        skipped_event_count=0,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=None,
        clock_offset_ns=None,
        observed_offset_half_range_ns=None,
        native_epoch_base_ns=None,
        clock_sample_offsets_ns=(),
        canonical_transform_offset_ns=None,
        clock_formula=None,
        alignment_valid_interval_ns=None,
        mapped_event_interval_ns=None,
        event_counts=(
            ("aggregate_perfetto_flow_count", aggregate_flows),
            ("aggregate_perfetto_slice_count", aggregate_slices),
            ("aggregate_track_descriptor_packet_count", aggregate[2]),
            ("aggregate_used_track_count", aggregate[3]),
            ("clock_snapshot_count", clock_snapshots),
            ("shard_perfetto_flow_count", shard_flows),
            ("shard_perfetto_slice_count", shard_slices),
        ),
        artifact_count=len(artifacts),
        artifact_sha256=tuple(item.sha256 for item in artifacts),
        notes=(
            "official Perfetto protobuf schema parses every PB artifact",
            "unnumbered PB is the aggregate; numbered PB files are shards",
            "no clock_snapshot or canonical anchor; canonical merge is forbidden",
            "aggregate PB is published byte-identically as a separate native timeline",
        ),
    )
    aggregate_artifact = aggregate[0]
    view = NativeTraceView(
        profiler_type="npu_rbln",
        source_role=source.source_role,
        source_relative_path=aggregate_artifact.relative_path,
        output_name="trace.rbln-native.pftrace",
        validation_name="trace.rbln-native.validation.json",
        payload=payloads[aggregate_artifact.relative_path],
        size_bytes=aggregate_artifact.size_bytes,
        sha256=aggregate_artifact.sha256,
        expected_slice_count=aggregate_slices,
        expected_track_count=aggregate[3],
        expected_flow_count=aggregate_flows,
    )
    return NativeDetailResult(
        summaries=(summary,),
        separate_traces=(view,),
    )


def _validate_mapped_interval(
    alignment: Mapping[str, Any],
    bridge: _ClockBridge,
    slices: Sequence[SliceSpec],
    instants: Sequence[InstantSpec],
) -> tuple[tuple[int, int], tuple[int, int]]:
    raw_interval = alignment.get("valid_interval_monotonic_ns")
    if (
        not isinstance(raw_interval, list)
        or len(raw_interval) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_interval
        )
        or raw_interval[0] < 0
        or raw_interval[1] < raw_interval[0]
    ):
        raise NativeDetailError("native alignment valid interval is invalid")
    valid_interval = (
        raw_interval[0] + bridge.canonical_offset_ns,
        raw_interval[1] + bridge.canonical_offset_ns,
    )
    starts = [
        *(spec.timestamp_ns for spec in slices),
        *(spec.timestamp_ns for spec in instants),
    ]
    ends = [
        *(spec.timestamp_ns + spec.duration_ns for spec in slices),
        *(spec.timestamp_ns for spec in instants),
    ]
    if not starts or not ends:
        raise NativeDetailError("native detail capture has no emitted interval")
    mapped_interval = (min(starts), max(ends))
    if (
        mapped_interval[0] < valid_interval[0]
        or mapped_interval[1] > valid_interval[1]
    ):
        raise NativeDetailError(
            "mapped native interval falls outside recorded capture bracket"
        )
    return valid_interval, mapped_interval


def _clock_bridge(
    loaded: LoadedHybridRun,
    source: SourceRunMetadata,
    alignment: Mapping[str, Any],
    *,
    native_clock_domain: str,
    native_timestamp_unit: str,
    host_boundary_uncertainty_ns: int,
) -> _ClockBridge:
    anchors = alignment.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 2:
        raise NativeDetailError("native alignment needs two API anchors")
    offsets: list[int] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise NativeDetailError("native alignment anchor must be an object")
        for prefix in ("before", "after"):
            unix = anchor.get(f"{prefix}_unix_ns")
            monotonic = anchor.get(f"{prefix}_monotonic_ns")
            if (
                isinstance(unix, bool)
                or not isinstance(unix, int)
                or isinstance(monotonic, bool)
                or not isinstance(monotonic, int)
            ):
                raise NativeDetailError(
                    "native alignment lacks paired Unix/monotonic samples"
                )
            offsets.append(unix - monotonic)
    lower = min(offsets)
    upper = max(offsets)
    midpoint = (lower + upper) // 2
    observed_half_range = max(midpoint - lower, upper - midpoint)

    transforms = [
        transform
        for transform in loaded.transforms
        if transform.target_clock_domain_id
        == loaded.canonical_clock_domain_id
        and transform.attributes.get("hybrid.source_role")
        == source.source_role
    ]
    if len(transforms) != 1 or transforms[0].scale != 1.0:
        raise NativeDetailError(
            "native clock bridge lacks an exact unit-scale host transform"
        )
    transform = transforms[0]
    if (
        isinstance(host_boundary_uncertainty_ns, bool)
        or not isinstance(host_boundary_uncertainty_ns, int)
        or host_boundary_uncertainty_ns < 0
    ):
        raise NativeDetailError("host boundary uncertainty is invalid")
    # This display window is deliberately no smaller than either the recorded
    # API bracket or the observed clock-pair dispersion. It is not a proven
    # bound on clock adjustment/drift and is never described as exact.
    uncertainty = (
        max(host_boundary_uncertainty_ns, observed_half_range)
        + transform.uncertainty_ns
    )
    return _ClockBridge(
        source_role=source.source_role,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        offset_ns=midpoint,
        observed_half_range_ns=observed_half_range,
        uncertainty_ns=uncertainty,
        canonical_offset_ns=transform.offset_ns,
        sample_offsets_ns=tuple(offsets),
    )


def _read_alignment(source: SourceRunMetadata) -> Mapping[str, Any]:
    artifacts = [
        artifact
        for artifact in source.artifacts
        if artifact.relative_path == "clocks/profiler_alignment.json"
    ]
    if len(artifacts) != 1:
        raise NativeDetailError("source lacks one profiler alignment artifact")
    path = _artifact_path(source, artifacts[0])
    return _stable_json(path, artifacts[0])


def _artifact_path(source: SourceRunMetadata, artifact: Any) -> Path:
    relative = PurePosixPath(artifact.relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise NativeDetailError("native artifact path is unsafe")
    current = source.root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            file_stat = current.lstat()
        except OSError as error:
            raise NativeDetailError("native artifact cannot be inspected") from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise NativeDetailError("native artifact path must not use symlinks")
        if index < len(relative.parts) - 1:
            if not stat.S_ISDIR(file_stat.st_mode):
                raise NativeDetailError("native artifact parent is not a directory")
        elif not stat.S_ISREG(file_stat.st_mode):
            raise NativeDetailError("native artifact is not a regular file")
    return current


def _stable_file_identity(path: Path, artifact: Any) -> tuple[int, str]:
    before = path.lstat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise NativeDetailError("native artifact changed while reading")
    identity = (after.st_size, digest.hexdigest())
    if identity != (artifact.size_bytes, artifact.sha256):
        raise NativeDetailError("native artifact identity differs from manifest")
    return identity


def _stable_json(path: Path, artifact: Any) -> Mapping[str, Any]:
    _stable_file_identity(path, artifact)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeDetailError("native JSON artifact is invalid") from error
    _stable_file_identity(path, artifact)
    if not isinstance(value, dict):
        raise NativeDetailError("native JSON artifact must be an object")
    return value


def _stable_gzip_json(path: Path, artifact: Any) -> Mapping[str, Any]:
    _stable_file_identity(path, artifact)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream, parse_float=Decimal)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeDetailError("native gzip JSON artifact is invalid") from error
    _stable_file_identity(path, artifact)
    if not isinstance(value, dict):
        raise NativeDetailError("native gzip JSON must be an object")
    return value


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise NativeDetailError("native timestamp is not decimal") from error
    if not converted.is_finite():
        raise NativeDetailError("native timestamp is not finite")
    return converted


def _microseconds_to_ns(value: Decimal) -> int:
    converted = value * 1000
    integral = converted.to_integral_value()
    if converted != integral:
        raise NativeDetailError(
            "Chrome timestamp has precision finer than integer nanoseconds"
        )
    result = int(integral)
    if result < 0:
        raise NativeDetailError("native Chrome timestamp is negative")
    return result


def _identity(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    raise NativeDetailError("native process/thread identity is invalid")


def _chrome_category(
    profiler_type: str,
    category: str,
    name: str,
    phase: str,
) -> tuple[str, str]:
    if profiler_type == "gpu_torch":
        mapping = {
            "cpu_op": ("PyTorch / ATen operators", "host"),
            "cuda_runtime": ("CUDA Runtime API", "host_api"),
            "cuda_driver": ("CUDA Driver API", "host_api"),
            "kernel": ("CUDA kernels", "device"),
            "gpu_memcpy": ("CUDA memcpy", "device"),
            "gpu_memset": ("CUDA memset", "device"),
            "user_annotation": ("vLLM annotations", "annotation"),
            "gpu_user_annotation": ("GPU annotations", "device"),
            "overhead": ("Profiler overhead", "host"),
            "Trace": ("Profiler lifecycle", "host"),
        }
        return mapping.get(category, ("Other native events", "host"))
    lowered_category = category.casefold()
    if (
        category
        in {
            "cuda_runtime",
            "cuda_driver",
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "gpu_user_annotation",
        }
        or any(
            token in lowered_category
            for token in ("cuda", "gpu", "kernel", "memcpy", "memset")
        )
    ):
        raise NativeDetailError(
            "NPU vLLM trace contains unsupported GPU/device activity"
        )
    if category == "cpu_op":
        if name.startswith("aten::"):
            return "ATen operators", "host"
        return "PyTorch host events", "host"
    if (
        "Torch-Compiled Region" in name
        or "TorchDynamo" in name
        or "Pregraph bytecode" in name
    ):
        return "Torch compiled regions", "host"
    if "vllm" in name.casefold() or name.startswith(("_C::", "_C_cache_ops::")):
        return "vLLM host events", "host"
    if category == "Trace":
        return "Profiler lifecycle", "host"
    if phase in {"i", "I"} and not category:
        return "PyTorch host events", "host"
    return "Other NPU vLLM events (device identity unverified)", "unknown"


def _chrome_category_order(profiler_type: str) -> dict[str, int]:
    names = (
        (
            "PyTorch / ATen operators",
            "vLLM annotations",
            "CUDA Runtime API",
            "CUDA Driver API",
            "CUDA kernels",
            "CUDA memcpy",
            "CUDA memset",
            "GPU annotations",
            "Profiler lifecycle",
            "Profiler overhead",
            "Other native events",
        )
        if profiler_type == "gpu_torch"
        else (
            "ATen operators",
            "Torch compiled regions",
            "vLLM host events",
            "PyTorch host events",
            "Profiler lifecycle",
            "Other NPU vLLM events (device identity unverified)",
        )
    )
    return {name: index for index, name in enumerate(names)}


def _chrome_leaf_identity(event: _ChromeEvent, endpoint_kind: str) -> str:
    if endpoint_kind == "device":
        return (
            f"device:{event.args.get('device', 'unknown')}:"
            f"context:{event.args.get('context', 'unknown')}:"
            f"stream:{event.args.get('stream', event.tid)}"
        )
    return f"pid:{event.pid}:tid:{event.tid}"


def _chrome_leaf_name(
    event: _ChromeEvent,
    endpoint_kind: str,
    *,
    process_names: Mapping[str, str],
    thread_names: Mapping[tuple[str, str], str],
) -> str:
    if endpoint_kind == "device":
        return (
            f"GPU {event.args.get('device', 'unknown')} / "
            f"context {event.args.get('context', 'unknown')} / "
            f"stream {event.args.get('stream', event.tid)}"
        )
    process = process_names.get(event.pid)
    thread = thread_names.get((event.pid, event.tid))
    suffix = " / ".join(item for item in (process, thread) if item)
    label = f"PID {event.pid} / TID {event.tid}"
    return f"{label} — {suffix}" if suffix else label


def _chrome_annotations(
    event: _ChromeEvent,
    *,
    profiler_type: str,
    bridge: _ClockBridge,
    original_timestamp: Decimal,
    original_duration: Decimal | None,
    process_name: str | None,
    thread_name: str | None,
) -> tuple[tuple[str, bool | int | float | str], ...]:
    values: dict[str, bool | int | float | str] = {
        "hetero.native_profiler": profiler_type,
        "hetero.native_category": event.category or "not_provided",
        "hetero.native_phase": event.phase,
        "hetero.native_pid": event.pid,
        "hetero.native_tid": event.tid,
        "hetero.native_timestamp_original": str(original_timestamp),
        "hetero.native_timestamp_unit": bridge.native_timestamp_unit,
        "hetero.native_clock_domain": bridge.native_clock_domain,
        "hetero.native_alignment_status": "partial_derived",
        "hetero.native_alignment_uncertainty_ns": bridge.uncertainty_ns,
        "hetero.native_alignment_uncertainty_kind": (
            "empirical_display_window_not_proven_clock_error_bound"
        ),
        "hetero.timestamp_fallback": False,
        "hetero.fabricated_event": False,
    }
    if original_duration is not None:
        values["hetero.native_duration_original"] = str(
            original_duration
        )
    if process_name:
        values["hetero.native_process_name"] = process_name
    if thread_name:
        values["hetero.native_thread_name"] = thread_name
    if event.artifact_index >= 0:
        values["hetero.native_artifact_index"] = event.artifact_index
    for key in _CHROME_ARG_KEYS:
        value = event.args.get(key)
        converted = _annotation_value(value)
        if converted is not None:
            values[f"native.{_safe_annotation_key(key)}"] = converted
    return tuple(sorted(values.items()))


def _nsys_annotations(
    bridge: _ClockBridge,
    *,
    native_start_ns: int,
    values: Mapping[str, object],
) -> tuple[tuple[str, bool | int | float | str], ...]:
    result: dict[str, bool | int | float | str] = {
        "hetero.native_profiler": "gpu_nsys",
        "hetero.native_clock_domain": bridge.native_clock_domain,
        "hetero.native_timestamp_original_ns": native_start_ns,
        "hetero.native_timestamp_unit": bridge.native_timestamp_unit,
        "hetero.native_alignment_status": "partial_derived",
        "hetero.native_alignment_uncertainty_ns": bridge.uncertainty_ns,
        "hetero.native_alignment_uncertainty_kind": (
            "empirical_display_window_not_proven_clock_error_bound"
        ),
        "hetero.timestamp_fallback": False,
        "hetero.fabricated_event": False,
    }
    for key, value in values.items():
        converted = _annotation_value(value)
        if converted is not None:
            result[f"native.{_safe_annotation_key(key)}"] = converted
    return tuple(sorted(result.items()))


def _annotation_value(value: object) -> bool | int | float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if -(2**63) <= value <= 2**63 - 1:
            return value
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    return None


def _safe_annotation_key(value: str) -> str:
    result = _SAFE_ANNOTATION_RE.sub("_", value.strip()).strip("_").casefold()
    return result or "value"


def _attach_explicit_flows(
    run_id: str,
    profiler_type: str,
    slices: Sequence[_NativeSlice],
) -> tuple[list[_NativeSlice], tuple[FlowSpec, ...]]:
    sources: dict[tuple[str, int], list[int]] = defaultdict(list)
    destinations: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, item in enumerate(slices):
        correlation = item.correlation_id
        if correlation is None or correlation <= 0:
            continue
        correlation_key = (
            item.correlation_scope or "single-artifact",
            correlation,
        )
        if item.endpoint_kind == "host_api":
            sources[correlation_key].append(index)
        elif item.endpoint_kind == "device":
            destinations[correlation_key].append(index)
    updated = list(slices)
    flows: list[FlowSpec] = []
    for correlation_key in sorted(set(sources) & set(destinations)):
        scope, correlation = correlation_key
        source_rows = sources[correlation_key]
        destination_rows = destinations[correlation_key]
        if len(source_rows) != 1 or len(destination_rows) != 1:
            continue
        source_index = source_rows[0]
        destination_index = destination_rows[0]
        source = updated[source_index]
        destination = updated[destination_index]
        flow_id = _stable_uint64(
            run_id,
            "native-flow",
            (
                f"{profiler_type}\0{scope}\0{correlation}\0"
                f"{source.spec.timestamp_ns}\0{destination.spec.timestamp_ns}"
            ),
        )
        correlation_text = f"{profiler_type}:{scope}:{correlation}"
        source_annotations = dict(source.spec.annotations)
        source_annotations["hetero.correlation_id"] = correlation_text
        destination_annotations = dict(destination.spec.annotations)
        destination_annotations["hetero.correlation_id"] = correlation_text
        source = replace(
            source,
            spec=replace(
                source.spec,
                annotations=tuple(sorted(source_annotations.items())),
                begin_flow_ids=tuple(
                    sorted((*source.spec.begin_flow_ids, flow_id))
                ),
            ),
        )
        destination = replace(
            destination,
            spec=replace(
                destination.spec,
                annotations=tuple(sorted(destination_annotations.items())),
                begin_terminating_flow_ids=tuple(
                    sorted(
                        (*destination.spec.begin_terminating_flow_ids, flow_id)
                    )
                ),
            ),
        )
        updated[source_index] = source
        updated[destination_index] = destination
        flows.append(
            FlowSpec(
                flow_id=flow_id,
                source_slice_name=source.spec.name,
                destination_slice_name=destination.spec.name,
                correlation_id=correlation_text,
            )
        )
    return updated, tuple(sorted(flows, key=lambda item: item.flow_id))


def _positive_duration(start: object, end: object, label: str) -> int:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise NativeDetailError(f"{label} timestamp is not integer ns")
    duration = end - start
    if duration <= 0:
        raise NativeDetailError(f"{label} duration must be positive")
    return duration


def _nsys_api_category(event_class: object) -> str:
    if (
        isinstance(event_class, bool)
        or not isinstance(event_class, int)
        or event_class not in {0, 1}
    ):
        raise NativeDetailError(
            f"unsupported Nsight CUDA API eventClass: {event_class!r}"
        )
    return "CUDA Runtime API" if event_class == 0 else "CUDA Driver API"


def _non_bool_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        return None
    return value


def _stable_token(value: str) -> str:
    readable = _SAFE_ANNOTATION_RE.sub("-", value.strip()).strip("-").casefold()
    readable = readable[:40] or "item"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _stable_uint64(run_id: str, namespace: str, value: str) -> int:
    digest = hashlib.sha256(
        f"{run_id}\0{namespace}\0{value}".encode("utf-8")
    ).digest()
    result = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return result or 1


def _validate_combined_native_plan(
    base: TracePlan,
    tracks: Sequence[TrackSpec],
    slices: Sequence[SliceSpec],
    instants: Sequence[InstantSpec],
    flows: Sequence[FlowSpec],
) -> None:
    base_keys = {track.key for track in base.tracks}
    base_uuids = {base.process_uuid, *(track.uuid for track in base.tracks)}
    native_keys = [track.key for track in tracks]
    native_uuids = [track.uuid for track in tracks]
    if len(native_keys) != len(set(native_keys)) or base_keys & set(native_keys):
        raise NativeDetailError("native track key collision")
    if len(native_uuids) != len(set(native_uuids)) or base_uuids & set(native_uuids):
        raise NativeDetailError("native track UUID collision")
    known = base_keys | set(native_keys)
    if any(spec.track_key not in known for spec in (*slices, *instants)):
        raise NativeDetailError("native event references an unknown track")
    base_flow_ids = {flow.flow_id for flow in base.flows}
    native_flow_ids = [flow.flow_id for flow in flows]
    if (
        len(native_flow_ids) != len(set(native_flow_ids))
        or base_flow_ids & set(native_flow_ids)
    ):
        raise NativeDetailError("native flow ID collision")


__all__ = [
    "NativeDetailError",
    "NativeDetailResult",
    "NativeDetailSummary",
    "NativeTraceView",
    "augment_trace_plan",
    "build_native_detail_plan",
    "native_validation_metadata",
    "request_focused_plan",
]
