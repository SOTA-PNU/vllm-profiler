"""Map validated normalized records onto deterministic Perfetto tracks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Iterable, Mapping

from ..schema import (
    Availability,
    EventRecord,
    EventType,
    MetricKind,
    MetricSample,
    Phase,
    RunManifest,
)
from .model import (
    AnnotationValue,
    CounterSpec,
    FlowSpec,
    InstantSpec,
    SliceSpec,
    TrackSpec,
    TracePlan,
)
from .timeline_summary import (
    LEGACY_MAPPING_VERSION,
    TIMELINE_SUMMARY_MAPPING_VERSION,
    TIMELINE_SUMMARY_ROOT_NAME,
    TimelineSummaryContext,
    TimelineSummaryKpi,
)


class PerfettoPlanningError(ValueError):
    """Normalized records cannot be represented without inventing evidence."""


@dataclass(frozen=True)
class NativeProfileEnvelope:
    """Evidence-backed host boundary for one otherwise unaligned native trace."""

    profiler_type: str
    source_role: str
    timestamp_ns: int
    duration_ns: int
    alignment_status: str
    alignment_method: str
    uncertainty_ns: int
    native_clock_domain: str
    native_timestamp_unit: str
    artifact_count: int
    opaque_rbln_pb: bool = False


@dataclass(frozen=True)
class PlanBuildMetadata:
    """Counts required to reconcile normalized inputs with emitted packets."""

    input_event_count: int
    input_metric_count: int
    resource_metric_count: int
    available_resource_metric_count: int
    unavailable_resource_metric_count: int
    skipped_non_resource_metric_count: int
    emitted_track_count: int
    emitted_slice_count: int
    emitted_instant_count: int
    emitted_counter_count: int
    emitted_flow_count: int
    native_envelope_count: int
    timeline_summary_track_count: int = 0
    timeline_summary_slice_count: int = 0
    timeline_summary_kpi_counter_count: int = 0
    timeline_summary_unavailable_kpi_count: int = 0
    timeline_summary_data_quality_instant_count: int = 0
    resource_telemetry_track_count: int = 0


@dataclass(frozen=True)
class PlanBuildResult:
    plan: TracePlan
    metadata: PlanBuildMetadata


@dataclass(frozen=True)
class _PairDefinition:
    start_name: str
    end_name: str
    track_key: str
    track_name: str
    slice_name: str
    start_phase: Phase
    end_phase: Phase | None = None
    discriminator: str | None = None


@dataclass(frozen=True)
class _PairedSlice:
    definition: _PairDefinition
    key: tuple[object, ...]
    correlation_id: str
    start: EventRecord
    end: EventRecord
    spec: SliceSpec


_PAIR_DEFINITIONS = (
    _PairDefinition(
        "request_received",
        "response_done",
        "request",
        "Request lifecycle",
        "Request",
        Phase.REQUEST,
        Phase.RESPONSE,
    ),
    _PairDefinition(
        "prefill_start",
        "prefill_end",
        "gpu_prefill",
        "GPU Prefill",
        "GPU Prefill",
        Phase.PREFILL,
    ),
    _PairDefinition(
        "kv_export_start",
        "kv_export_end",
        "kv_export",
        "KV Export",
        "KV Export",
        Phase.KV_EXPORT,
    ),
    _PairDefinition(
        "kv_transfer_start",
        "kv_transfer_end",
        "kv_transfer",
        "KV Transfer",
        "KV Transfer",
        Phase.KV_TRANSFER,
        discriminator="transfer",
    ),
    _PairDefinition(
        "kv_transform_start",
        "kv_transform_end",
        "kv_transform",
        "KV Transform",
        "KV Transform",
        Phase.KV_TRANSFORM,
    ),
    _PairDefinition(
        "decode_loop_start",
        "decode_loop_end",
        "npu_decode",
        "NPU Decode",
        "NPU Decode",
        Phase.DECODE,
    ),
    _PairDefinition(
        "decode_step_start",
        "decode_step_end",
        "npu_decode_step",
        "NPU Decode Step",
        "NPU Decode Step",
        Phase.DECODE,
        discriminator="step",
    ),
    _PairDefinition(
        "sampling_start",
        "sampling_end",
        "sampling",
        "Sampling",
        "Sampling",
        Phase.SAMPLING,
        discriminator="step",
    ),
)

_PAIR_EVENT_NAMES = {
    name
    for definition in _PAIR_DEFINITIONS
    for name in (definition.start_name, definition.end_name)
}

_TRACK_DESCRIPTIONS = {
    "request": "End-to-end request lifecycle on the canonical clock.",
    "gpu_prefill": "GPU prefill markers paired without timestamp inference.",
    "kv_export": "GPU KV export markers paired by explicit request identity.",
    "kv_transfer": "GPU-to-NPU KV transfer paired by explicit transfer identity.",
    "kv_transform": "NPU KV transform markers on the canonical clock.",
    "npu_decode": "NPU decode loop markers on the canonical clock.",
    "npu_decode_step": "Ordered NPU decode steps with preserved step index.",
    "sampling": "Ordered sampling steps with preserved step index.",
    "response": "Canonical response completion point.",
    "profiler": "Host API bracket for native profiler capture; partially aligned.",
    "clock_metadata": "Canonical and native clock alignment policy.",
}

_COUNTER_NAMES = {
    "resource.cpu.utilization": "CPU utilization",
    "resource.process.cpu_memory": "Process CPU memory",
    "resource.process.memory_used": "Process CPU memory",
    "resource.system.memory_used": "System memory",
    "resource.gpu.utilization": "GPU utilization",
    "resource.gpu.memory_used": "GPU memory",
    "resource.gpu.power": "GPU power",
    "resource.npu.utilization": "NPU utilization",
    "resource.npu.memory_used": "NPU memory",
    "resource.npu.power": "NPU power",
}

_TIMELINE_SUMMARY_STAGE_TRACKS = (
    ("gpu_prefill", "summary.pipeline.gpu_prefill", "GPU Prefill", 0),
    ("kv_export", "summary.pipeline.kv_export", "KV Export", 1),
    ("kv_transfer", "summary.pipeline.kv_transfer", "KV Transfer", 2),
    ("kv_transform", "summary.pipeline.kv_transform", "KV Transform", 3),
    ("npu_decode", "summary.pipeline.npu_decode", "NPU Decode", 4),
)

_KPI_ANCHOR_TRACKS = {
    ("pipeline_latency", "latency.e2e"): "request",
    ("pipeline_latency", "latency.prefill"): "gpu_prefill",
    ("pipeline_latency", "latency.kv_export"): "kv_export",
    ("pipeline_latency", "latency.kv_transfer"): "kv_transfer",
    ("pipeline_latency", "latency.kv_transform"): "kv_transform",
    ("pipeline_latency", "latency.decode"): "npu_decode",
    ("transfer", "transfer.bytes"): "kv_transfer",
    ("transfer", "transfer.duration"): "kv_transfer",
    ("transfer", "transfer.effective_bandwidth"): "kv_transfer",
    ("transfer", "transfer.transform_duration"): "kv_transform",
    ("transfer", "transfer.e2e_share"): "kv_transfer",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _stable_uint64(run_id: str, namespace: str, value: str) -> int:
    payload = f"{run_id}\0{namespace}\0{value}".encode("utf-8")
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result or 1


def _stable_uint31(run_id: str, namespace: str) -> int:
    payload = f"{run_id}\0{namespace}".encode("utf-8")
    return (int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x3FFFFFFF) + 1


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PerfettoPlanningError(
            "metric dimensions must be finite JSON values"
        ) from error


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PerfettoPlanningError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _step_index(event: EventRecord) -> int:
    value = event.attributes.get("decode.step_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PerfettoPlanningError(
            f"{event.event_name} decode.step_index must be a non-negative integer"
        )
    return value


def _correlation_id(event: EventRecord) -> str:
    value = event.attributes.get("hybrid.correlation_id")
    if value is None:
        value = event.request_id
    return _required_string(
        value,
        f"event {event.event_id!r} correlation/request identity",
    )


def _pair_key(
    definition: _PairDefinition,
    event: EventRecord,
) -> tuple[object, ...]:
    correlation_id = _correlation_id(event)
    if definition.discriminator == "step":
        return (correlation_id, _step_index(event))
    if definition.discriminator == "transfer":
        transfer_id = _required_string(
            event.attributes.get("hybrid.transfer_id"),
            f"event {event.event_id!r} hybrid.transfer_id",
        )
        return (correlation_id, transfer_id)
    return (correlation_id,)


def _event_annotations(
    start: EventRecord,
    end: EventRecord,
    *,
    correlation_id: str,
    discriminator: str | None,
) -> tuple[tuple[str, AnnotationValue], ...]:
    annotations: dict[str, AnnotationValue] = {
        "hetero.correlation_id": correlation_id,
        "hetero.start_event_id": start.event_id,
        "hetero.end_event_id": end.event_id,
        "hetero.clock_domain_id": start.clock_domain_id,
    }
    if start.request_id is not None:
        annotations["hetero.request_id"] = start.request_id
    source_role = _optional_string(
        start.attributes.get("hybrid.source_role"),
        f"event {start.event_id!r} hybrid.source_role",
    )
    if source_role is not None:
        annotations["hetero.source_role"] = source_role
    if discriminator == "step":
        annotations["hetero.step_index"] = _step_index(start)
    if discriminator == "transfer":
        annotations["hetero.transfer_id"] = _required_string(
            start.attributes.get("hybrid.transfer_id"),
            f"event {start.event_id!r} hybrid.transfer_id",
        )
    return tuple(sorted(annotations.items()))


def _pair_slices(
    events: tuple[EventRecord, ...],
    canonical_clock_domain_id: str,
) -> list[_PairedSlice]:
    by_name: dict[str, list[EventRecord]] = {}
    for event in events:
        by_name.setdefault(event.event_name, []).append(event)
        if (
            event.event_name not in _PAIR_EVENT_NAMES
            and event.event_type is EventType.INSTANT
            and (
                event.event_name.endswith("_start")
                or event.event_name.endswith("_end")
            )
        ):
            raise PerfettoPlanningError(
                f"unsupported start/end marker cannot be guessed: {event.event_name}"
            )

    paired: list[_PairedSlice] = []
    for definition in _PAIR_DEFINITIONS:
        starts = by_name.get(definition.start_name, [])
        ends = by_name.get(definition.end_name, [])
        if not starts and not ends:
            continue
        start_map: dict[tuple[object, ...], EventRecord] = {}
        end_map: dict[tuple[object, ...], EventRecord] = {}
        for event, target in (
            *((event, start_map) for event in starts),
            *((event, end_map) for event in ends),
        ):
            if event.event_type is not EventType.INSTANT:
                raise PerfettoPlanningError(
                    f"paired marker {event.event_name} must be an instant"
                )
            expected_phase = (
                definition.start_phase
                if target is start_map
                else definition.end_phase or definition.start_phase
            )
            if event.phase is not expected_phase:
                raise PerfettoPlanningError(
                    f"paired marker {event.event_name} has an unexpected phase"
                )
            if event.clock_domain_id != canonical_clock_domain_id:
                raise PerfettoPlanningError(
                    f"paired marker {event.event_name} is not on the canonical clock"
                )
            key = _pair_key(definition, event)
            if key in target:
                raise PerfettoPlanningError(
                    f"duplicate {event.event_name} marker for key {key!r}"
                )
            target[key] = event
        if set(start_map) != set(end_map):
            missing_end = sorted(set(start_map) - set(end_map), key=repr)
            missing_start = sorted(set(end_map) - set(start_map), key=repr)
            raise PerfettoPlanningError(
                f"incomplete {definition.slice_name} pairing: "
                f"missing_end={missing_end!r}, missing_start={missing_start!r}"
            )
        for key in sorted(start_map, key=repr):
            start = start_map[key]
            end = end_map[key]
            if end.timestamp_ns < start.timestamp_ns:
                raise PerfettoPlanningError(
                    f"negative duration for {definition.slice_name}: "
                    f"{start.event_id} -> {end.event_id}"
                )
            correlation_id = _correlation_id(start)
            if _correlation_id(end) != correlation_id:
                raise PerfettoPlanningError(
                    f"{definition.slice_name} correlation identity changed"
                )
            spec = SliceSpec(
                track_key=definition.track_key,
                name=definition.slice_name,
                timestamp_ns=start.timestamp_ns,
                duration_ns=end.timestamp_ns - start.timestamp_ns,
                annotations=_event_annotations(
                    start,
                    end,
                    correlation_id=correlation_id,
                    discriminator=definition.discriminator,
                ),
            )
            paired.append(
                _PairedSlice(
                    definition=definition,
                    key=key,
                    correlation_id=correlation_id,
                    start=start,
                    end=end,
                    spec=spec,
                )
            )
    return paired


def _annotation_dict(spec: SliceSpec) -> dict[str, AnnotationValue]:
    return dict(spec.annotations)


def _matching_pair(
    paired: Iterable[_PairedSlice],
    track_key: str,
    correlation_id: str,
) -> _PairedSlice | None:
    matches = [
        item
        for item in paired
        if item.definition.track_key == track_key
        and item.correlation_id == correlation_id
    ]
    if len(matches) > 1:
        raise PerfettoPlanningError(
            f"flow endpoint {track_key!r} is ambiguous for {correlation_id!r}"
        )
    return matches[0] if matches else None


def _add_flow_endpoint(
    item: _PairedSlice,
    flow_id: int,
    *,
    endpoint: str,
    terminating: bool,
) -> _PairedSlice:
    field = f"{endpoint}_{'terminating_flow_ids' if terminating else 'flow_ids'}"
    values = tuple(sorted((*getattr(item.spec, field), flow_id)))
    return replace(item, spec=replace(item.spec, **{field: values}))


def _build_flows(
    run_id: str,
    paired: list[_PairedSlice],
) -> tuple[list[_PairedSlice], tuple[FlowSpec, ...]]:
    by_correlation = sorted({item.correlation_id for item in paired})
    flows: list[FlowSpec] = []
    updated = list(paired)

    transitions = (
        ("request", "begin", "gpu_prefill", "begin", "request_to_prefill"),
        ("gpu_prefill", "end", "kv_export", "begin", "prefill_to_kv_export"),
        ("kv_export", "end", "kv_transfer", "begin", "kv_export_to_transfer"),
        ("kv_transfer", "end", "kv_transform", "begin", "transfer_to_transform"),
        ("kv_transform", "end", "npu_decode", "begin", "transform_to_decode"),
    )
    for correlation_id in by_correlation:
        for source_key, source_endpoint, destination_key, destination_endpoint, kind in transitions:
            source = _matching_pair(updated, source_key, correlation_id)
            destination = _matching_pair(updated, destination_key, correlation_id)
            if source is None or destination is None:
                continue
            source_event = (
                source.start if source_endpoint == "begin" else source.end
            )
            destination_event = (
                destination.start
                if destination_endpoint == "begin"
                else destination.end
            )
            explicit_source = source_event.attributes.get(
                "hybrid.correlation_id"
            )
            explicit_destination = destination_event.attributes.get(
                "hybrid.correlation_id"
            )
            if explicit_source is None or explicit_destination is None:
                # Request identity or timestamp adjacency is not sufficient
                # evidence for a Perfetto flow.
                continue
            explicit_source = _required_string(
                explicit_source,
                f"event {source_event.event_id!r} hybrid.correlation_id",
            )
            explicit_destination = _required_string(
                explicit_destination,
                f"event {destination_event.event_id!r} hybrid.correlation_id",
            )
            if (
                explicit_source != correlation_id
                or explicit_destination != correlation_id
            ):
                raise PerfettoPlanningError(
                    f"explicit correlation identity changed across {kind}"
                )
            if kind == "kv_export_to_transfer":
                source_suffix = _required_string(
                    source.end.attributes.get("hybrid.remote_request_id_suffix"),
                    f"event {source.end.event_id!r} hybrid.remote_request_id_suffix",
                )
                destination_suffix = _required_string(
                    destination.start.attributes.get(
                        "hybrid.remote_request_id_suffix"
                    ),
                    f"event {destination.start.event_id!r} "
                    "hybrid.remote_request_id_suffix",
                )
                if source_suffix != destination_suffix:
                    raise PerfettoPlanningError(
                        "cross-device flow remote request suffix mismatch"
                    )
            flow_id = _stable_uint64(
                run_id,
                "flow",
                f"{kind}\0{correlation_id}",
            )
            source_index = updated.index(source)
            source = _add_flow_endpoint(
                source,
                flow_id,
                endpoint=source_endpoint,
                terminating=False,
            )
            updated[source_index] = source
            destination = _matching_pair(updated, destination_key, correlation_id)
            if destination is None:  # pragma: no cover - defensive
                raise PerfettoPlanningError("flow destination disappeared")
            destination_index = updated.index(destination)
            destination = _add_flow_endpoint(
                destination,
                flow_id,
                endpoint=destination_endpoint,
                terminating=True,
            )
            updated[destination_index] = destination
            flows.append(
                FlowSpec(
                    flow_id=flow_id,
                    source_slice_name=source.spec.name,
                    destination_slice_name=destination.spec.name,
                    correlation_id=correlation_id,
                )
            )
    if len({flow.flow_id for flow in flows}) != len(flows):
        raise PerfettoPlanningError("deterministic flow id collision")
    return updated, tuple(sorted(flows, key=lambda item: item.flow_id))


def _counter_track_key(metric: MetricSample) -> str:
    components = (
        metric.scope.value,
        metric.host_id,
        metric.device_type.value if metric.device_type is not None else "",
        metric.device_id or "",
        metric.request_id or "",
        metric.phase.value if metric.phase is not None else "",
        metric.metric_name,
        metric.unit,
        _canonical_json(metric.dimensions),
    )
    return "counter:" + "\0".join(components)


def _counter_track_name(metric: MetricSample) -> str:
    base = _COUNTER_NAMES.get(
        metric.metric_name,
        metric.metric_name.removeprefix("resource.").replace(".", " "),
    )
    entity = metric.device_id or metric.host_id
    return f"{base} [{entity}]"


def _resource_counters(
    metrics: tuple[MetricSample, ...],
    canonical_clock_domain_id: str,
) -> tuple[list[TrackSpec], list[CounterSpec], int, int, int]:
    track_metrics: dict[str, MetricSample] = {}
    counters: list[CounterSpec] = []
    resource_count = 0
    available_count = 0
    unavailable_count = 0
    for metric in metrics:
        if not metric.metric_name.startswith("resource."):
            continue
        resource_count += 1
        if metric.clock_domain_id != canonical_clock_domain_id:
            raise PerfettoPlanningError(
                f"resource metric {metric.metric_name!r} is not on the canonical clock"
            )
        if metric.metric_kind is not MetricKind.GAUGE:
            raise PerfettoPlanningError(
                f"resource metric {metric.metric_name!r} must be a gauge"
            )
        if metric.availability is not Availability.AVAILABLE:
            unavailable_count += 1
            continue
        value = metric.value
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise PerfettoPlanningError(
                f"available resource metric {metric.metric_name!r} "
                "must have a finite non-boolean number"
            )
        available_count += 1
        key = _counter_track_key(metric)
        existing = track_metrics.get(key)
        if existing is not None and existing.unit != metric.unit:
            raise PerfettoPlanningError("counter track mixes different units")
        track_metrics[key] = metric
        counters.append(
            CounterSpec(
                track_key=key,
                timestamp_ns=metric.timestamp_ns,
                value=value,
            )
        )
    tracks = [
        TrackSpec(
            key=key,
            uuid=0,
            name=_counter_track_name(metric),
            kind="counter",
            description=(
                f"Normalized available samples for {metric.metric_name}; "
                f"scope={metric.scope.value}, unit={metric.unit}."
            ),
            unit=metric.unit,
        )
        for key, metric in sorted(track_metrics.items())
    ]
    counters.sort(key=lambda item: (item.timestamp_ns, item.track_key, float(item.value)))
    return tracks, counters, resource_count, available_count, unavailable_count


def _native_slices(
    envelopes: tuple[NativeProfileEnvelope, ...],
) -> list[SliceSpec]:
    slices: list[SliceSpec] = []
    for envelope in sorted(
        envelopes,
        key=lambda item: (item.timestamp_ns, item.profiler_type),
    ):
        if (
            envelope.timestamp_ns < 0
            or envelope.duration_ns < 0
            or envelope.uncertainty_ns < 0
            or envelope.artifact_count < 0
        ):
            raise PerfettoPlanningError("native profiler envelope values are invalid")
        if envelope.alignment_status != "partial":
            raise PerfettoPlanningError(
                "native profiler envelope must retain partial alignment"
            )
        if envelope.alignment_method != "host_api_boundary_bracket":
            raise PerfettoPlanningError(
                "native profiler envelope must use host API boundary evidence"
            )
        annotations: dict[str, AnnotationValue] = {
            "hetero.profiler_type": _required_string(
                envelope.profiler_type, "profiler_type"
            ),
            "hetero.source_role": _required_string(
                envelope.source_role, "source_role"
            ),
            "hetero.alignment_status": "partial",
            "hetero.alignment_method": "host_api_boundary_bracket",
            "hetero.host_boundary_uncertainty_ns": envelope.uncertainty_ns,
            "hetero.native_clock_domain": _required_string(
                envelope.native_clock_domain, "native_clock_domain"
            ),
            "hetero.native_timestamp_unit": _required_string(
                envelope.native_timestamp_unit, "native_timestamp_unit"
            ),
            "hetero.native_artifact_count": envelope.artifact_count,
            "hetero.unaligned_profiler_events": True,
            "hetero.native_details_emitted": False,
        }
        if envelope.profiler_type == "npu_rbln":
            annotations["hetero.rbln_pb_classification"] = (
                "perfetto_compatible_rbln_trace"
            )
            annotations["hetero.rbln_pb_structure_analysis"] = (
                "deferred_to_official_trace_processor"
            )
        slices.append(
            SliceSpec(
                track_key="profiler",
                name=f"{envelope.profiler_type} capture envelope",
                timestamp_ns=envelope.timestamp_ns,
                duration_ns=envelope.duration_ns,
                annotations=tuple(sorted(annotations.items())),
            )
        )
    return slices


def _timeline_summary_track_uuid(
    run_id: str,
    context: TimelineSummaryContext,
    track_key: str,
) -> int:
    if (
        context.mapping_version != TIMELINE_SUMMARY_MAPPING_VERSION
        or _SHA256_RE.fullmatch(context.source_identity_sha256) is None
    ):
        raise PerfettoPlanningError("timeline summary mapping/source identity is invalid")
    return _stable_uint64(
        run_id,
        (
            f"track:{context.mapping_version}:"
            f"{context.source_identity_sha256}"
        ),
        track_key,
    )


def _timeline_summary_group_tracks() -> list[TrackSpec]:
    return [
        TrackSpec(
            key="summary.root",
            uuid=0,
            name=TIMELINE_SUMMARY_ROOT_NAME,
            kind="group",
            description=(
                "Trace-native request, pipeline, KPI, quality, and resource "
                "summary. Ordering is an explicit UI hint."
            ),
            child_ordering="explicit",
        ),
        TrackSpec(
            key="summary.request_summary",
            uuid=0,
            name="Request Summary",
            kind="slice",
            description=(
                "Canonical hybrid request boundary with request-facing and "
                "pipeline KPI annotations kept distinct."
            ),
            parent_key="summary.root",
            sibling_order_rank=0,
        ),
        TrackSpec(
            key="summary.pipeline",
            uuid=0,
            name="Pipeline Stages",
            kind="group",
            description=(
                "Summary copies of evidence-backed canonical stage intervals."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=1,
        ),
        *[
            TrackSpec(
                key=timeline_summary_key,
                uuid=0,
                name=name,
                kind="slice",
                description=(
                    f"Timeline summary of the detailed {name} marker pair; "
                    "timestamp and duration are unchanged."
                ),
                parent_key="summary.pipeline",
                sibling_order_rank=rank,
            )
            for _, timeline_summary_key, name, rank in _TIMELINE_SUMMARY_STAGE_TRACKS
        ],
        TrackSpec(
            key="summary.kpi.token_throughput",
            uuid=0,
            name="Token & Throughput KPI",
            kind="group",
            description=(
                "Available request-facing, pipeline, token, and throughput "
                "scalars; unavailable values are omitted from counters."
            ),
            parent_key="summary.root",
            child_ordering="lexicographic",
            sibling_order_rank=2,
        ),
        TrackSpec(
            key="summary.kpi.transfer",
            uuid=0,
            name="Transfer KPI",
            kind="group",
            description=(
                "Available transfer scalars with canonical units and explicit "
                "calculation provenance."
            ),
            parent_key="summary.root",
            child_ordering="lexicographic",
            sibling_order_rank=3,
        ),
        TrackSpec(
            key="summary.data_quality",
            uuid=0,
            name="Data Quality",
            kind="slice",
            description=(
                "Input, marker, alignment, availability, native-profiler, and "
                "publication-validation policy."
            ),
            parent_key="summary.root",
            sibling_order_rank=4,
        ),
    ]


def _request_pair(paired: Iterable[_PairedSlice]) -> _PairedSlice:
    matches = [
        item for item in paired if item.definition.track_key == "request"
    ]
    if len(matches) != 1:
        raise PerfettoPlanningError(
            "timeline summary requires exactly one canonical request marker pair"
        )
    return matches[0]


def _pair_for_track(
    paired: Iterable[_PairedSlice],
    track_key: str,
) -> _PairedSlice:
    matches = [
        item for item in paired if item.definition.track_key == track_key
    ]
    if len(matches) != 1:
        raise PerfettoPlanningError(
            f"timeline summary requires exactly one {track_key!r} marker pair"
        )
    return matches[0]


def _kpi_by_identity(
    context: TimelineSummaryContext,
) -> dict[tuple[str, str], TimelineSummaryKpi]:
    result = {(item.section, item.name): item for item in context.kpis}
    if len(result) != len(context.kpis):
        raise PerfettoPlanningError("timeline summary KPI identities are not unique")
    return result


def _available_kpi_annotation(
    annotations: dict[str, AnnotationValue],
    key: str,
    kpi: TimelineSummaryKpi | None,
) -> None:
    if kpi is not None and kpi.available:
        if kpi.value is None:  # pragma: no cover - property invariant
            raise PerfettoPlanningError("available KPI unexpectedly has no value")
        annotations[key] = kpi.value


def _request_summary_slice(
    request: _PairedSlice,
    context: TimelineSummaryContext,
) -> SliceSpec:
    kpis = _kpi_by_identity(context)
    annotations = dict(request.spec.annotations)
    annotations.update(
        {
            "hetero.timeline_summary": True,
            "hetero.trace_mapping_version": context.mapping_version,
            "hetero.canonical_duration_role": "pipeline_e2e",
            "hetero.latency_display_rule": (
                "canonical_ns;display_ms=ns/1000000"
            ),
            "hetero.source_event_ids_json": _canonical_json(
                [request.start.event_id, request.end.event_id]
            ),
        }
    )
    data_quality = dict(context.data_quality_annotations)
    for source_key, target_key in (
        ("hetero.profiler_kind", "hetero.profiler_kind"),
        (
            "hetero.native_profiler_alignment",
            "hetero.native_profiler_alignment",
        ),
        ("hetero.clock_status", "hetero.alignment_status"),
    ):
        value = data_quality.get(source_key)
        if value is not None:
            annotations[target_key] = value
    for key, identity in (
        (
            "hetero.request_facing_e2e_ns",
            ("request_facing_latency", "latency.e2e"),
        ),
        (
            "hetero.pipeline_e2e_ns",
            ("pipeline_latency", "latency.e2e"),
        ),
        ("hetero.ttft_ns", ("request_facing_latency", "latency.ttft")),
        ("hetero.tpot_ns", ("request_facing_latency", "latency.tpot")),
        ("hetero.input_tokens", ("throughput_and_tokens", "request.input_tokens")),
        (
            "hetero.output_tokens",
            ("throughput_and_tokens", "request.output_tokens"),
        ),
        ("hetero.total_tokens", ("throughput_and_tokens", "request.total_tokens")),
    ):
        _available_kpi_annotation(annotations, key, kpis.get(identity))
    return SliceSpec(
        track_key="summary.request_summary",
        name="Hybrid Request",
        timestamp_ns=request.spec.timestamp_ns,
        duration_ns=request.spec.duration_ns,
        annotations=tuple(sorted(annotations.items())),
    )


def _pipeline_summary_slices(
    paired: Iterable[_PairedSlice],
    context: TimelineSummaryContext,
) -> list[SliceSpec]:
    summaries: list[SliceSpec] = []
    for detail_key, timeline_summary_key, _, _ in _TIMELINE_SUMMARY_STAGE_TRACKS:
        detail = _pair_for_track(paired, detail_key)
        annotations = dict(detail.spec.annotations)
        annotations.update(
            {
                "hetero.timeline_summary": True,
                "hetero.summary_of_track_key": detail_key,
                "hetero.trace_mapping_version": context.mapping_version,
            }
        )
        summaries.append(
            SliceSpec(
                track_key=timeline_summary_key,
                name=detail.spec.name,
                timestamp_ns=detail.spec.timestamp_ns,
                duration_ns=detail.spec.duration_ns,
                annotations=tuple(sorted(annotations.items())),
            )
        )
    return summaries


def _kpi_anchor(
    kpi: TimelineSummaryKpi,
    *,
    request: _PairedSlice,
    paired: tuple[_PairedSlice, ...],
) -> tuple[int, str]:
    detail_key = _KPI_ANCHOR_TRACKS.get((kpi.section, kpi.name))
    if detail_key is None:
        if kpi.name == "latency.sampling":
            sampling = [
                item
                for item in paired
                if item.definition.track_key == "sampling"
            ]
            if not sampling:
                raise PerfettoPlanningError(
                    "sampling KPI has no explicit sampling marker endpoint"
                )
            endpoint = max(sampling, key=lambda item: item.end.timestamp_ns)
            return endpoint.end.timestamp_ns, endpoint.end.event_id
        return request.end.timestamp_ns, request.end.event_id
    detail = _pair_for_track(paired, detail_key)
    return detail.end.timestamp_ns, detail.end.event_id


def _kpi_track_and_counter(
    kpi: TimelineSummaryKpi,
    *,
    context: TimelineSummaryContext,
    request: _PairedSlice,
    paired: tuple[_PairedSlice, ...],
) -> tuple[TrackSpec, CounterSpec] | None:
    if not kpi.available:
        return None
    if kpi.value is None:  # pragma: no cover - property invariant
        raise PerfettoPlanningError("available KPI unexpectedly has no value")
    track_key = f"summary.kpi:{kpi.identity}"
    timestamp_ns, anchor_event_id = _kpi_anchor(
        kpi,
        request=request,
        paired=paired,
    )
    annotations: dict[str, AnnotationValue] = {
        "hetero.kpi_identity": kpi.identity,
        "hetero.kpi_name": kpi.name,
        "hetero.availability": "available",
        "hetero.canonical_unit": kpi.canonical_unit,
        "hetero.observation_layer": kpi.observation_layer,
        "hetero.calculation_method": kpi.calculation_method,
        "hetero.display_unit": kpi.display_unit,
        "hetero.display_scale_numerator": kpi.display_scale_numerator,
        "hetero.display_scale_denominator": kpi.display_scale_denominator,
        "hetero.anchor_event_id": anchor_event_id,
        "hetero.source_event_ids_json": _canonical_json(
            list(kpi.source_event_ids)
        ),
        "hetero.trace_mapping_version": context.mapping_version,
    }
    track = TrackSpec(
        key=track_key,
        uuid=0,
        name=kpi.display_name,
        kind="counter",
        description=(
            f"{kpi.identity}; canonical unit={kpi.canonical_unit}; "
            f"display={kpi.display_unit} * "
            f"{kpi.display_scale_numerator}/{kpi.display_scale_denominator}."
        ),
        unit=kpi.canonical_unit,
        parent_key=kpi.counter_group_key,
    )
    counter = CounterSpec(
        track_key=track_key,
        timestamp_ns=timestamp_ns,
        value=kpi.value,
        annotations=tuple(sorted(annotations.items())),
    )
    return track, counter


def _resource_group(
    metric: MetricSample,
) -> tuple[str, str, int]:
    device_type = (
        metric.device_type.value if metric.device_type is not None else None
    )
    if device_type == "gpu":
        device_id = _required_string(metric.device_id, "GPU resource device_id")
        suffix = device_id.removeprefix("gpu-")
        index = (
            int(suffix)
            if suffix.isdigit()
            else int.from_bytes(
                hashlib.sha256(device_id.encode()).digest()[:2], "big"
            )
            % 90
        )
        return (
            f"telemetry.resources.gpu.{device_id}",
            "GPU" if index == 0 else f"GPU {index}",
            100 + index,
        )
    if device_type == "npu":
        device_id = _required_string(metric.device_id, "NPU resource device_id")
        suffix = device_id.removeprefix("npu-")
        index = (
            int(suffix)
            if suffix.isdigit()
            else int.from_bytes(
                hashlib.sha256(device_id.encode()).digest()[:2], "big"
            )
            % 90
        )
        return (
            f"telemetry.resources.npu.{device_id}",
            f"NPU {index}",
            200 + index,
        )
    return ("telemetry.resources.cpu_system", "CPU/System", 0)


def _group_resource_tracks(
    counter_tracks: list[TrackSpec],
    metrics: tuple[MetricSample, ...],
) -> tuple[list[TrackSpec], list[TrackSpec]]:
    metric_by_key = {
        _counter_track_key(metric): metric
        for metric in metrics
        if metric.metric_name.startswith("resource.")
        and metric.availability is Availability.AVAILABLE
    }
    groups: dict[str, TrackSpec] = {}
    grouped: list[TrackSpec] = []
    used_ranks: dict[int, str] = {}
    for track in counter_tracks:
        try:
            metric = metric_by_key[track.key]
        except KeyError as error:  # pragma: no cover - construction invariant
            raise PerfettoPlanningError(
                f"resource track {track.key!r} has no source metric"
            ) from error
        group_key, group_name, preferred_rank = _resource_group(metric)
        rank = preferred_rank
        while rank in used_ranks and used_ranks[rank] != group_key:
            rank += 1
        used_ranks[rank] = group_key
        groups.setdefault(
            group_key,
            TrackSpec(
                key=group_key,
                uuid=0,
                name=group_name,
                kind="group",
                description=(
                    f"Resource counter streams for {group_name}; samples are "
                    "not duplicated."
                ),
                parent_key="telemetry.resources",
                child_ordering="lexicographic",
                sibling_order_rank=rank,
            ),
        )
        grouped.append(replace(track, parent_key=group_key))
    return list(groups.values()), grouped


def _resource_telemetry_root_track() -> TrackSpec:
    return TrackSpec(
        key="telemetry.resources",
        uuid=0,
        name="Resource telemetry (full capture window)",
        kind="group",
        description=(
            "CPU/GPU/NPU counter streams across the full collector lifetime, "
            "including pre-request samples. Source timestamps are unchanged."
        ),
        child_ordering="explicit",
    )


def _timeline_summary_data_quality_instant(
    context: TimelineSummaryContext,
    *,
    request: _PairedSlice,
    resource_count: int,
    available_resource_count: int,
    unavailable_resource_count: int,
    resource_counters: tuple[CounterSpec, ...],
) -> InstantSpec:
    annotations = dict(context.data_quality_annotations)
    resource_timestamps = sorted(counter.timestamp_ns for counter in resource_counters)
    request_start_ns = request.start.timestamp_ns
    pre_request_timestamps = [
        timestamp for timestamp in resource_timestamps if timestamp < request_start_ns
    ]
    resource_first_timestamp_ns = (
        resource_timestamps[0] if resource_timestamps else request_start_ns
    )
    annotations.update(
        {
            "hetero.resource_metric_count": resource_count,
            "hetero.available_resource_sample_count": available_resource_count,
            "hetero.unavailable_resource_sample_count": unavailable_resource_count,
            "hetero.resource_grouping": (
                "top_level_full_capture_tracks_without_sample_duplication"
            ),
            "hetero.resource_time_scope": "full_capture_not_request_scoped",
            "hetero.resource_first_timestamp_ns": resource_first_timestamp_ns,
            "hetero.measured_request_boundary": "request_received",
            "hetero.measured_request_start_ns": request_start_ns,
            "hetero.pre_request_resource_sample_count": len(
                pre_request_timestamps
            ),
            "hetero.pre_request_resource_duration_ns": max(
                0,
                request_start_ns - resource_first_timestamp_ns,
            ),
            "hetero.warmup_interval_status": (
                "not_available_no_normalized_warmup_boundaries"
            ),
            "hetero.warmup_interval_fabricated": False,
            "hetero.flow_policy": "detail_only_explicit_correlation",
            "hetero.ordering_policy": "track_descriptor_explicit_hint",
        }
    )
    return InstantSpec(
        track_key="summary.data_quality",
        name="Data Quality status",
        timestamp_ns=request_start_ns,
        annotations=tuple(sorted(annotations.items())),
    )


def build_trace_plan(
    manifest: RunManifest,
    events: Iterable[EventRecord],
    metrics: Iterable[MetricSample],
    *,
    canonical_clock_domain_id: str,
    native_envelopes: Iterable[NativeProfileEnvelope] = (),
    timeline_summary: TimelineSummaryContext | None = None,
) -> PlanBuildResult:
    """Build a deterministic plan without mutating or inferring source records."""

    event_rows = tuple(events)
    metric_rows = tuple(metrics)
    envelope_rows = tuple(native_envelopes)
    _required_string(canonical_clock_domain_id, "canonical_clock_domain_id")
    for event in event_rows:
        if event.run_id != manifest.run_id:
            raise PerfettoPlanningError("event run_id does not match manifest")
        if event.clock_domain_id != canonical_clock_domain_id:
            raise PerfettoPlanningError(
                f"event {event.event_id!r} is not on the canonical clock"
            )
    for metric in metric_rows:
        if metric.run_id != manifest.run_id:
            raise PerfettoPlanningError("metric run_id does not match manifest")

    paired = _pair_slices(event_rows, canonical_clock_domain_id)
    paired, flows = _build_flows(manifest.run_id, paired)
    slices = [item.spec for item in paired]
    native_slices = _native_slices(envelope_rows)
    slices.extend(native_slices)

    paired_event_ids = {
        event_id
        for item in paired
        for event_id in (item.start.event_id, item.end.event_id)
    }
    instants: list[InstantSpec] = []
    response_events = [
        event for event in event_rows if event.event_name == "response_done"
    ]
    for event in response_events:
        instants.append(
            InstantSpec(
                track_key="response",
                name="Response completion",
                timestamp_ns=event.timestamp_ns,
                annotations=(
                    ("hetero.event_id", event.event_id),
                    ("hetero.correlation_id", _correlation_id(event)),
                ),
            )
        )
    for event in event_rows:
        if event.event_id in paired_event_ids:
            continue
        if event.event_type is EventType.SPAN:
            duration = event.duration_ns
            if (
                not isinstance(duration, int)
                or isinstance(duration, bool)
                or duration < 0
            ):
                raise PerfettoPlanningError(
                    f"span {event.event_id!r} has an invalid duration"
                )
            key = f"phase:{event.phase.value}"
            slices.append(
                SliceSpec(
                    track_key=key,
                    name=event.event_name,
                    timestamp_ns=event.timestamp_ns,
                    duration_ns=duration,
                    annotations=(
                        ("hetero.event_id", event.event_id),
                        ("hetero.correlation_id", _correlation_id(event)),
                    ),
                )
            )
        elif event.event_name != "response_done":
            key = f"phase:{event.phase.value}"
            instants.append(
                InstantSpec(
                    track_key=key,
                    name=event.event_name,
                    timestamp_ns=event.timestamp_ns,
                    annotations=(
                        ("hetero.event_id", event.event_id),
                        ("hetero.correlation_id", _correlation_id(event)),
                    ),
                )
            )

    (
        counter_tracks,
        counters,
        resource_count,
        available_count,
        unavailable_count,
    ) = _resource_counters(metric_rows, canonical_clock_domain_id)

    timeline_summary_tracks: list[TrackSpec] = []
    timeline_summary_slice_count = 0
    timeline_summary_kpi_counter_count = 0
    timeline_summary_unavailable_kpi_count = 0
    timeline_summary_data_quality_count = 0
    if timeline_summary is not None:
        if not isinstance(timeline_summary, TimelineSummaryContext):
            raise TypeError("timeline_summary must be a TimelineSummaryContext")
        if timeline_summary.mapping_version != TIMELINE_SUMMARY_MAPPING_VERSION:
            raise PerfettoPlanningError("unsupported timeline summary mapping version")
        request = _request_pair(paired)
        request_summary = _request_summary_slice(request, timeline_summary)
        pipeline_summaries = _pipeline_summary_slices(paired, timeline_summary)
        slices.extend((request_summary, *pipeline_summaries))
        timeline_summary_slice_count = 1 + len(pipeline_summaries)

        timeline_summary_tracks.extend(_timeline_summary_group_tracks())
        resource_groups, counter_tracks = _group_resource_tracks(
            counter_tracks,
            metric_rows,
        )
        timeline_summary_tracks.append(_resource_telemetry_root_track())
        timeline_summary_tracks.extend(resource_groups)
        resource_counters = tuple(counters)
        paired_tuple = tuple(paired)
        for kpi in timeline_summary.kpis:
            planned = _kpi_track_and_counter(
                kpi,
                context=timeline_summary,
                request=request,
                paired=paired_tuple,
            )
            if planned is None:
                timeline_summary_unavailable_kpi_count += 1
                continue
            kpi_track, kpi_counter = planned
            timeline_summary_tracks.append(kpi_track)
            counters.append(kpi_counter)
            timeline_summary_kpi_counter_count += 1
        instants.append(
            _timeline_summary_data_quality_instant(
                timeline_summary,
                request=request,
                resource_count=resource_count,
                available_resource_count=available_count,
                unavailable_resource_count=unavailable_count,
                resource_counters=resource_counters,
            )
        )
        timeline_summary_data_quality_count = 1
        counters.sort(
            key=lambda item: (
                item.timestamp_ns,
                item.track_key,
                float(item.value),
            )
        )

    timestamp_candidates = [
        *(event.timestamp_ns for event in event_rows),
        *(counter.timestamp_ns for counter in counters),
        *(envelope.timestamp_ns for envelope in envelope_rows),
    ]
    if not timestamp_candidates:
        raise PerfettoPlanningError("normalized run has no timestamped records")
    if timeline_summary is None:
        first_timestamp = min(timestamp_candidates)
        instants.append(
            InstantSpec(
                track_key="clock_metadata",
                name="Clock/alignment metadata",
                timestamp_ns=first_timestamp,
                annotations=(
                    ("hetero.canonical_clock_domain", canonical_clock_domain_id),
                    ("hetero.native_profiler_alignment", "partial_or_unaligned"),
                    ("hetero.native_details_emitted", False),
                ),
            )
        )

    used_track_keys = {
        *(item.track_key for item in slices),
        *(item.track_key for item in instants),
        *(item.track_key for item in counters),
    }
    tracks: list[TrackSpec] = []
    definition_tracks = {
        definition.track_key: (definition.track_name, _TRACK_DESCRIPTIONS[definition.track_key])
        for definition in _PAIR_DEFINITIONS
    }
    fixed_tracks = {
        **definition_tracks,
        "response": ("Token emission / response completion", _TRACK_DESCRIPTIONS["response"]),
        "profiler": ("Profiler capture boundary", _TRACK_DESCRIPTIONS["profiler"]),
        "clock_metadata": ("Clock/alignment metadata", _TRACK_DESCRIPTIONS["clock_metadata"]),
    }
    for key in sorted(used_track_keys):
        if key.startswith("counter:"):
            continue
        if key.startswith(("summary.", "telemetry.")):
            continue
        if key in fixed_tracks:
            name, description = fixed_tracks[key]
        elif key.startswith("phase:"):
            phase = key.split(":", 1)[1]
            name = f"{phase.replace('_', ' ').title()} events"
            description = "Unpaired canonical events preserved without inference."
        else:  # pragma: no cover - construction invariant
            raise PerfettoPlanningError(f"unknown track key: {key}")
        tracks.append(
            TrackSpec(
                key=key,
                uuid=0,
                name=name,
                kind="slice",
                description=description,
            )
        )
    tracks.extend(counter_tracks)
    tracks.extend(timeline_summary_tracks)
    tracks = [
        replace(
            track,
            uuid=(
                _timeline_summary_track_uuid(manifest.run_id, timeline_summary, track.key)
                if timeline_summary is not None
                and track.key.startswith(("summary.", "telemetry."))
                else _stable_uint64(manifest.run_id, "track", track.key)
            ),
        )
        for track in tracks
    ]
    if len({track.uuid for track in tracks}) != len(tracks):
        raise PerfettoPlanningError("deterministic track UUID collision")

    slices.sort(
        key=lambda item: (
            item.timestamp_ns,
            item.track_key,
            item.duration_ns,
            item.name,
            item.annotations,
        )
    )
    instants.sort(
        key=lambda item: (
            item.timestamp_ns,
            item.track_key,
            item.name,
            item.annotations,
        )
    )
    tracks.sort(key=lambda item: (item.uuid, item.key))
    plan = TracePlan(
        run_id=manifest.run_id,
        canonical_clock_domain_id=canonical_clock_domain_id,
        process_uuid=_stable_uint64(manifest.run_id, "process", "normalized-run"),
        process_id=_stable_uint31(manifest.run_id, "process-id"),
        packet_sequence_id=_stable_uint31(manifest.run_id, "packet-sequence"),
        tracks=tuple(tracks),
        slices=tuple(slices),
        instants=tuple(instants),
        counters=tuple(counters),
        flows=flows,
        trace_attributes=(
            timeline_summary.trace_attributes
            if timeline_summary is not None
            else ()
        ),
        mapping_version=(
            timeline_summary.mapping_version
            if timeline_summary is not None
            else LEGACY_MAPPING_VERSION
        ),
        source_identity_sha256=(
            timeline_summary.source_identity_sha256
            if timeline_summary is not None
            else None
        ),
    )
    metadata = PlanBuildMetadata(
        input_event_count=len(event_rows),
        input_metric_count=len(metric_rows),
        resource_metric_count=resource_count,
        available_resource_metric_count=available_count,
        unavailable_resource_metric_count=unavailable_count,
        skipped_non_resource_metric_count=len(metric_rows) - resource_count,
        emitted_track_count=len(tracks),
        emitted_slice_count=len(slices),
        emitted_instant_count=len(instants),
        emitted_counter_count=len(counters),
        emitted_flow_count=len(flows),
        native_envelope_count=len(native_slices),
        timeline_summary_track_count=sum(
            track.key.startswith("summary.") for track in tracks
        ),
        timeline_summary_slice_count=timeline_summary_slice_count,
        timeline_summary_kpi_counter_count=timeline_summary_kpi_counter_count,
        timeline_summary_unavailable_kpi_count=timeline_summary_unavailable_kpi_count,
        timeline_summary_data_quality_instant_count=timeline_summary_data_quality_count,
        resource_telemetry_track_count=sum(
            track.key.startswith("telemetry.resources") for track in tracks
        ),
    )
    return PlanBuildResult(plan=plan, metadata=metadata)
