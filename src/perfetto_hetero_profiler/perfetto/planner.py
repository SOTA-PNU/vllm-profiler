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
from ..schema.catalog import (
    PIPELINE_STAGE_ORDER,
    RESOURCE_DISPLAY_NAMES,
    RESOURCE_TRACK_ORDER,
    STAGE_DEFINITIONS,
)
from .model import (
    AnnotationValue,
    CounterSpec,
    FlowSpec,
    InstantSpec,
    SliceSpec,
    TrackSpec,
    TracePlan,
    UnclassifiedGapSpec,
)
from .timeline_summary import (
    LEGACY_MAPPING_VERSION,
    TIMELINE_SUMMARY_MAPPING_VERSION,
    TIMELINE_SUMMARY_ROOT_NAME,
    TimelineSummaryContext,
    TokenInstantEvidence,
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


_PAIR_DEFINITIONS = tuple(
    _PairDefinition(
        stage.start_event,
        stage.end_event,
        stage.track_key,
        stage.track_name,
        stage.slice_name,
        stage.phase,
        stage.end_phase,
        stage.discriminator,
    )
    for stage in STAGE_DEFINITIONS
)

_PAIR_EVENT_NAMES = {
    name
    for definition in _PAIR_DEFINITIONS
    for name in (definition.start_name, definition.end_name)
}

_TRACK_DESCRIPTIONS = {
    **{stage.track_key: stage.description for stage in STAGE_DEFINITIONS},
    "response": "Canonical response completion point.",
    "profiler": "Host API bracket for native profiler capture; partially aligned.",
    "clock_metadata": "Canonical and native clock alignment policy.",
}

_COUNTER_NAMES = RESOURCE_DISPLAY_NAMES
_RESOURCE_TRACK_ORDER = RESOURCE_TRACK_ORDER
_PIPELINE_TRACK_ORDER = PIPELINE_STAGE_ORDER
_DECODE_DETAIL_TRACK_ORDER = {"npu_decode_step": 0, "sampling": 1}
_BOUNDARY_EVENT_NAMES = frozenset(
    {
        "request_received",
        "first_token_emitted",
        "token_emitted",
        "response_done",
    }
)

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
    if start.request_id != end.request_id:
        annotations["hetero.start_request_id"] = _required_string(
            start.request_id,
            f"event {start.event_id!r} request_id",
        )
        annotations["hetero.end_request_id"] = _required_string(
            end.request_id,
            f"event {end.event_id!r} request_id",
        )
    alignment_method = _optional_string(
        start.attributes.get("hybrid.alignment_method"),
        f"event {start.event_id!r} hybrid.alignment_method",
    )
    end_alignment_method = _optional_string(
        end.attributes.get("hybrid.alignment_method"),
        f"event {end.event_id!r} hybrid.alignment_method",
    )
    if alignment_method == end_alignment_method and alignment_method is not None:
        annotations["hetero.alignment_method"] = alignment_method
    elif alignment_method is not None and end_alignment_method is not None:
        annotations["hetero.start_alignment_method"] = alignment_method
        annotations["hetero.end_alignment_method"] = end_alignment_method
    uncertainty = start.attributes.get("hybrid.alignment_uncertainty_ns")
    end_uncertainty = end.attributes.get("hybrid.alignment_uncertainty_ns")
    if uncertainty is not None:
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, int) or uncertainty < 0:
            raise PerfettoPlanningError("paired marker alignment uncertainty is invalid")
    if end_uncertainty is not None:
        if (
            isinstance(end_uncertainty, bool)
            or not isinstance(end_uncertainty, int)
            or end_uncertainty < 0
        ):
            raise PerfettoPlanningError("paired marker alignment uncertainty is invalid")
    if uncertainty == end_uncertainty and uncertainty is not None:
        annotations["hetero.alignment_uncertainty_ns"] = uncertainty
    elif uncertainty is not None and end_uncertainty is not None:
        annotations["hetero.start_alignment_uncertainty_ns"] = uncertainty
        annotations["hetero.end_alignment_uncertainty_ns"] = end_uncertainty
    source_role = _optional_string(
        start.attributes.get("hybrid.source_role"),
        f"event {start.event_id!r} hybrid.source_role",
    )
    if source_role is not None:
        annotations["hetero.source_role"] = source_role
    original_start = start.attributes.get(
        "hybrid.original_timestamp_ns", start.timestamp_ns
    )
    original_end = end.attributes.get(
        "hybrid.original_timestamp_ns", end.timestamp_ns
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (original_start, original_end)
    ):
        raise PerfettoPlanningError(
            f"{start.event_name} original timestamps are invalid"
        )
    if original_end < original_start:
        raise PerfettoPlanningError(
            f"{start.event_name} original duration is negative"
        )
    annotations.update(
        {
            "hetero.original_start_timestamp_ns": original_start,
            "hetero.original_end_timestamp_ns": original_end,
            "hetero.original_duration_ns": original_end - original_start,
        }
    )
    if discriminator == "step":
        annotations["hetero.step_index"] = _step_index(start)
    if discriminator == "transfer":
        annotations["hetero.transfer_id"] = _required_string(
            start.attributes.get("hybrid.transfer_id"),
            f"event {start.event_id!r} hybrid.transfer_id",
        )
        if start.event_name == "kv_transfer_wait_start":
            annotations["hetero.wait_observation"] = (
                "polling_incomplete_to_done"
            )
            for source, suffix in ((start, "start"), (end, "end")):
                poll_count = source.attributes.get("kv.poll_count")
                if poll_count is not None:
                    if (
                        isinstance(poll_count, bool)
                        or not isinstance(poll_count, int)
                        or poll_count < 0
                    ):
                        raise PerfettoPlanningError(
                            f"event {source.event_id!r} kv.poll_count is invalid"
                        )
                    annotations[f"hetero.{suffix}_poll_count"] = poll_count
                status = source.attributes.get("kv.transfer_status")
                if status is not None:
                    annotations[f"hetero.{suffix}_transfer_status"] = (
                        _required_string(
                            status,
                            f"event {source.event_id!r} kv.transfer_status",
                        )
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
            request_identity_changed = start.request_id != end.request_id
            if (
                not isinstance(start.request_id, str)
                or not start.request_id
                or not isinstance(end.request_id, str)
                or not end.request_id
            ):
                raise PerfettoPlanningError(
                    f"{definition.slice_name} request identity changed or is missing"
                )
            if request_identity_changed:
                if definition.track_key != "kv_handoff":
                    raise PerfettoPlanningError(
                        f"{definition.slice_name} request identity changed or is missing"
                    )
                start_suffix = _required_string(
                    start.attributes.get("hybrid.remote_request_id_suffix"),
                    f"event {start.event_id!r} hybrid.remote_request_id_suffix",
                )
                end_suffix = _required_string(
                    end.attributes.get("hybrid.remote_request_id_suffix"),
                    f"event {end.event_id!r} hybrid.remote_request_id_suffix",
                )
                if start_suffix != end_suffix:
                    raise PerfettoPlanningError(
                        "KV Handoff remote request suffix changed"
                    )
            slice_name = definition.slice_name
            if definition.discriminator == "step":
                step_index = _step_index(start)
                slice_name = (
                    f"Decode Step {step_index}"
                    if definition.track_key == "npu_decode_step"
                    else f"Sampling {step_index}"
                )
            spec = SliceSpec(
                track_key=definition.track_key,
                name=slice_name,
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


def _validate_decode_details(paired: Iterable[_PairedSlice]) -> None:
    decode = [
        item for item in paired if item.definition.track_key == "npu_decode_step"
    ]
    sampling = [
        item for item in paired if item.definition.track_key == "sampling"
    ]
    by_correlation: dict[str, dict[int, _PairedSlice]] = {}
    for item in decode:
        by_correlation.setdefault(item.correlation_id, {})[_step_index(item.start)] = item
    for correlation_id, rows in by_correlation.items():
        indices = sorted(rows)
        if indices != list(range(len(indices))):
            raise PerfettoPlanningError(
                f"decode step indices are not contiguous for {correlation_id!r}"
            )
    for item in sampling:
        index = _step_index(item.start)
        decode_row = by_correlation.get(item.correlation_id, {}).get(index)
        if decode_row is None:
            raise PerfettoPlanningError(
                "sampling has no decode step with the same correlation and index"
            )
        if item.start.timestamp_ns < decode_row.end.timestamp_ns:
            raise PerfettoPlanningError(
                "sampling begins before its decode step completes"
            )


def _boundary_instant(event: EventRecord, *, grouped: bool) -> InstantSpec:
    names = {
        "request_received": "Request Received",
        "first_token_emitted": "First Token Emitted",
        "response_done": "Response Completion",
    }
    sequence = event.attributes.get("vllm.token_sequence")
    if sequence is not None and (
        isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
    ):
        raise PerfettoPlanningError(
            f"event {event.event_id!r} token sequence is invalid"
        )
    name = names.get(event.event_name)
    if name is None:
        name = (
            f"Token Emitted {sequence}"
            if sequence is not None
            else "Token Emitted"
        )
    annotations: dict[str, AnnotationValue] = {
        "hetero.boundary_kind": event.event_name,
        "hetero.event_id": event.event_id,
        "hetero.correlation_id": _correlation_id(event),
        "hetero.clock_domain_id": event.clock_domain_id,
    }
    if event.request_id is not None:
        annotations["hetero.request_id"] = event.request_id
    if sequence is not None:
        annotations["hetero.token_sequence_index"] = sequence
    return InstantSpec(
        track_key=(
            "summary.boundaries.events"
            if grouped
            else (
                "response"
                if event.phase is Phase.RESPONSE
                else f"phase:{event.phase.value}"
            )
        ),
        name=name,
        timestamp_ns=event.timestamp_ns,
        annotations=tuple(sorted(annotations.items())),
    )


def _token_output_instant(evidence: TokenInstantEvidence) -> InstantSpec:
    if not isinstance(evidence, TokenInstantEvidence):
        raise TypeError("token evidence must be TokenInstantEvidence")
    return InstantSpec(
        track_key="summary.boundaries.events",
        name=f"Output Token {evidence.token_index}",
        timestamp_ns=evidence.timestamp_ns,
        annotations=tuple(
            sorted(
                {
                    "hetero.request_id": evidence.request_id,
                    "hetero.token_index": evidence.token_index,
                    "hetero.timestamp_source": "valid_token_timestamps_ns",
                    "hetero.original_timestamp_ns": evidence.source_timestamp_ns,
                    "hetero.original_clock_domain_id": (
                        evidence.source_clock_domain_id
                    ),
                    "hetero.aligned_clock_domain_id": (
                        evidence.target_clock_domain_id
                    ),
                    "hetero.alignment_method": evidence.alignment_method,
                    "hetero.alignment_uncertainty_ns": (
                        evidence.alignment_uncertainty_ns
                    ),
                }.items()
            )
        ),
    )


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

    for correlation_id in by_correlation:
        transitions = [
            ("request", "begin", "gpu_prefill", "begin", "request_to_prefill"),
            ("gpu_prefill", "end", "kv_export", "begin", "prefill_to_kv_export"),
        ]
        if (
            _matching_pair(updated, "kv_handoff", correlation_id) is not None
            and _matching_pair(updated, "kv_transfer_setup", correlation_id) is not None
        ):
            transitions.extend(
                (
                    ("kv_export", "end", "kv_handoff", "begin", "kv_export_to_handoff"),
                    ("kv_handoff", "end", "kv_transfer_setup", "begin", "handoff_to_setup"),
                    ("kv_transfer_setup", "end", "kv_transfer", "begin", "setup_to_transfer"),
                )
            )
        else:
            transitions.append(
                ("kv_export", "end", "kv_transfer", "begin", "kv_export_to_transfer")
            )
        transitions.append(
            ("kv_transfer", "end", "kv_transform", "begin", "transfer_to_transform")
        )
        if _matching_pair(updated, "decode_schedule_wait", correlation_id) is not None:
            transitions.extend(
                (
                    (
                        "kv_transform",
                        "end",
                        "decode_schedule_wait",
                        "begin",
                        "transform_to_decode_schedule_wait",
                    ),
                    (
                        "decode_schedule_wait",
                        "begin",
                        "npu_decode",
                        "begin",
                        "decode_schedule_wait_to_decode",
                    ),
                )
            )
        else:
            transitions.append(
                ("kv_transform", "end", "npu_decode", "begin", "transform_to_decode")
            )
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
            evidence_kind = "hybrid.correlation_id"
            evidence_id = correlation_id
            if kind == "kv_export_to_transfer":
                source_suffix = _required_string(
                    source_event.attributes.get("hybrid.remote_request_id_suffix"),
                    f"event {source_event.event_id!r} hybrid.remote_request_id_suffix",
                )
                destination_suffix = _required_string(
                    destination_event.attributes.get(
                        "hybrid.remote_request_id_suffix"
                    ),
                    f"event {destination_event.event_id!r} "
                    "hybrid.remote_request_id_suffix",
                )
                if source_suffix != destination_suffix:
                    raise PerfettoPlanningError(
                        "cross-device flow remote request suffix mismatch"
                    )
                evidence_kind = "hybrid.remote_request_id_suffix"
                evidence_id = source_suffix
            elif kind in {"kv_export_to_handoff", "handoff_to_setup"}:
                raw_source_suffix = source_event.attributes.get(
                    "hybrid.remote_request_id_suffix"
                )
                raw_destination_suffix = destination_event.attributes.get(
                    "hybrid.remote_request_id_suffix"
                )
                if (raw_source_suffix is None) != (raw_destination_suffix is None):
                    continue
                if raw_source_suffix is not None:
                    source_suffix = _required_string(
                        raw_source_suffix,
                        f"event {source_event.event_id!r} hybrid.remote_request_id_suffix",
                    )
                    destination_suffix = _required_string(
                        raw_destination_suffix,
                        f"event {destination_event.event_id!r} hybrid.remote_request_id_suffix",
                    )
                    if source_suffix != destination_suffix:
                        raise PerfettoPlanningError(
                            "cross-device flow remote request suffix mismatch"
                        )
                    evidence_kind = "hybrid.remote_request_id_suffix"
                    evidence_id = source_suffix
            elif kind == "setup_to_transfer":
                source_transfer = _required_string(
                    source_event.attributes.get("hybrid.transfer_id"),
                    f"event {source_event.event_id!r} hybrid.transfer_id",
                )
                destination_transfer = _required_string(
                    destination_event.attributes.get("hybrid.transfer_id"),
                    f"event {destination_event.event_id!r} hybrid.transfer_id",
                )
                if source_transfer != destination_transfer:
                    raise PerfettoPlanningError(
                        "setup-to-transfer flow transfer identity mismatch"
                    )
                evidence_kind = "hybrid.transfer_id"
                evidence_id = source_transfer
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
                    source_event_id=source_event.event_id,
                    destination_event_id=destination_event.event_id,
                    evidence_kind=evidence_kind,
                    evidence_id=evidence_id,
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
                interval_ns=metric.interval_ns,
                sample_role=(
                    metric.attributes.get("telemetry.sample_role")
                    if isinstance(
                        metric.attributes.get("telemetry.sample_role"), str
                    )
                    else None
                ),
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


def _processing_group_tracks(*, include_native: bool) -> list[TrackSpec]:
    tracks = [
        TrackSpec(
            key="summary.root",
            uuid=0,
            name=TIMELINE_SUMMARY_ROOT_NAME,
            kind="group",
            description=(
                "Observed request boundaries, processing stages, token-level "
                "decode work, and selected native profiler events."
            ),
            child_ordering="explicit",
        ),
        TrackSpec(
            key="summary.boundaries",
            uuid=0,
            name="Request Boundaries and Token Output",
            kind="group",
            description=(
                "Canonical request, token-arrival, and response-completion "
                "instants without prompt or token content."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=0,
        ),
        TrackSpec(
            key="summary.boundaries.events",
            uuid=0,
            name="Request and token boundaries",
            kind="slice",
            description="Observed canonical boundary instants.",
            parent_key="summary.boundaries",
            sibling_order_rank=0,
        ),
        TrackSpec(
            key="summary.pipeline",
            uuid=0,
            name="Pipeline Stages",
            kind="group",
            description=(
                "Evidence-backed canonical processing and observed wait intervals."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=1,
        ),
        TrackSpec(
            key="summary.decode_details",
            uuid=0,
            name="Decode Details",
            kind="group",
            description=(
                "Ordered token-level NPU decode and host-side sampling intervals."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=2,
        ),
    ]
    if include_native:
        tracks.append(
            TrackSpec(
                key="summary.native_details",
                uuid=0,
                name="Native Profiler Details",
                kind="group",
                description=(
                    "Selected native profiler tracks with evidence-backed clock policy."
                ),
                parent_key="summary.root",
                child_ordering="explicit",
                sibling_order_rank=3,
            )
        )
    return tracks


def _request_pair(paired: Iterable[_PairedSlice]) -> _PairedSlice:
    matches = [
        item for item in paired if item.definition.track_key == "request"
    ]
    if len(matches) != 1:
        raise PerfettoPlanningError(
            "timeline summary requires exactly one canonical request marker pair"
        )
    return matches[0]


def _unclassified_gaps(
    paired: Iterable[_PairedSlice],
) -> tuple[UnclassifiedGapSpec, ...]:
    rows = tuple(paired)
    request = _request_pair(rows)
    gaps: list[UnclassifiedGapSpec] = []

    prefill = [item for item in rows if item.definition.track_key == "gpu_prefill"]
    if len(prefill) == 1 and prefill[0].start.timestamp_ns > request.start.timestamp_ns:
        gaps.append(
            UnclassifiedGapSpec(
                start_timestamp_ns=request.start.timestamp_ns,
                end_timestamp_ns=prefill[0].start.timestamp_ns,
                duration_ns=(
                    prefill[0].start.timestamp_ns - request.start.timestamp_ns
                ),
                preceding_marker=request.start.event_name,
                following_marker=prefill[0].start.event_name,
                reason="no marker identifies work or wait within this interval",
            )
        )

    decode_details = [
        item
        for item in rows
        if item.definition.track_key in {"npu_decode_step", "sampling"}
    ]
    if decode_details:
        latest = max(
            decode_details,
            key=lambda item: (item.end.timestamp_ns, item.end.event_id),
        )
        if request.end.timestamp_ns > latest.end.timestamp_ns:
            gaps.append(
                UnclassifiedGapSpec(
                    start_timestamp_ns=latest.end.timestamp_ns,
                    end_timestamp_ns=request.end.timestamp_ns,
                    duration_ns=request.end.timestamp_ns - latest.end.timestamp_ns,
                    preceding_marker=latest.end.event_name,
                    following_marker=request.end.event_name,
                    reason="no marker identifies finalization work or wait",
                )
            )
    return tuple(gaps)


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
    group_metadata: dict[str, tuple[str, int]] = {}
    tracks_by_group: dict[str, list[tuple[TrackSpec, MetricSample]]] = {}
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
        group_metadata.setdefault(group_key, (group_name, rank))
        tracks_by_group.setdefault(group_key, []).append((track, metric))

    groups = [
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
                child_ordering="explicit",
                sibling_order_rank=rank,
        )
        for group_key, (group_name, rank) in sorted(
            group_metadata.items(), key=lambda item: (item[1][1], item[0])
        )
    ]
    grouped: list[TrackSpec] = []
    for group_key in sorted(
        tracks_by_group,
        key=lambda key: (group_metadata[key][1], key),
    ):
        ordered = sorted(
            tracks_by_group[group_key],
            key=lambda item: (
                _RESOURCE_TRACK_ORDER.get(item[1].metric_name, 100),
                item[1].metric_name,
                item[1].unit,
                item[0].key,
            ),
        )
        grouped.extend(
            replace(
                track,
                parent_key=group_key,
                sibling_order_rank=index,
            )
            for index, (track, _metric) in enumerate(ordered)
        )
    return groups, grouped


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

    grouped_timeline = timeline_summary is not None
    if grouped_timeline:
        if not isinstance(timeline_summary, TimelineSummaryContext):
            raise TypeError("timeline_summary must be a TimelineSummaryContext")
        if timeline_summary.mapping_version != TIMELINE_SUMMARY_MAPPING_VERSION:
            raise PerfettoPlanningError("unsupported timeline summary mapping version")

    paired = _pair_slices(event_rows, canonical_clock_domain_id)
    _validate_decode_details(paired)
    unclassified_gaps = (
        _unclassified_gaps(paired) if grouped_timeline else ()
    )
    paired, flows = _build_flows(manifest.run_id, paired)
    slices = [item.spec for item in paired]
    native_slices = _native_slices(envelope_rows)
    slices.extend(native_slices)

    paired_event_ids = {
        event_id
        for item in paired
        for event_id in (item.start.event_id, item.end.event_id)
    }
    token_evidence = (
        timeline_summary.token_instants
        if timeline_summary is not None
        else ()
    )
    instants: list[InstantSpec] = [
        _boundary_instant(event, grouped=grouped_timeline)
        for event in event_rows
        if event.event_name in _BOUNDARY_EVENT_NAMES
        and (
            not token_evidence
            or event.event_name in {"request_received", "response_done"}
        )
    ]
    instants.extend(_token_output_instant(item) for item in token_evidence)
    for event in event_rows:
        if event.event_name in _BOUNDARY_EVENT_NAMES:
            continue
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
        else:
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
    if grouped_timeline:
        assert timeline_summary is not None
        timeline_summary_tracks.extend(
            _processing_group_tracks(include_native=bool(envelope_rows))
        )
        resource_groups, counter_tracks = _group_resource_tracks(
            counter_tracks,
            metric_rows,
        )
        timeline_summary_tracks.append(_resource_telemetry_root_track())
        timeline_summary_tracks.extend(resource_groups)
        timeline_summary_unavailable_kpi_count = sum(
            not item.available for item in timeline_summary.kpis
        )

    timestamp_candidates = [
        *(event.timestamp_ns for event in event_rows),
        *(counter.timestamp_ns for counter in counters),
        *(envelope.timestamp_ns for envelope in envelope_rows),
    ]
    if not timestamp_candidates:
        raise PerfettoPlanningError("normalized run has no timestamped records")
    if not grouped_timeline:
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
                parent_key=(
                    None
                    if not grouped_timeline
                    else (
                        "summary.pipeline"
                        if key in _PIPELINE_TRACK_ORDER
                        else (
                            "summary.decode_details"
                            if key in _DECODE_DETAIL_TRACK_ORDER
                            else (
                                "summary.native_details"
                                if key == "profiler"
                                else "summary.boundaries"
                            )
                        )
                    )
                ),
                sibling_order_rank=(
                    _PIPELINE_TRACK_ORDER.get(
                        key,
                        _DECODE_DETAIL_TRACK_ORDER.get(
                            key,
                            (
                                0
                                if key == "profiler"
                                else (1 if key == "request" else 2)
                            ),
                        ),
                    )
                    if grouped_timeline
                    else None
                ),
            )
        )
    tracks.extend(counter_tracks)
    tracks.extend(timeline_summary_tracks)
    tracks = [
        replace(
            track,
            uuid=(
                _timeline_summary_track_uuid(manifest.run_id, timeline_summary, track.key)
                if grouped_timeline
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
            if grouped_timeline
            else None
        ),
        unclassified_gaps=unclassified_gaps,
        request_window=(
            timeline_summary.request_window
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
