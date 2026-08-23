"""Source-backed inputs for the processing timeline and Trace Attributes.

This module deliberately stops before protobuf planning. KPI values come from
the same pure calculation used by the external report. They are exported as
official Trace Attributes; the timeline itself is reserved for observed
processing events.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from ..hybrid.join import validate_marker_groups
from ..overview.calculation import calculate_overview_kpis
from ..schema import Availability
from .model import TraceAttributeSpec
from .trace_attributes import _read_source_artifact, build_performance_trace_attributes


LEGACY_MAPPING_VERSION = "legacy-unversioned-phase5-v1"
TIMELINE_SUMMARY_MAPPING_VERSION = "processing-timeline-info-stats-v1"
TIMELINE_SUMMARY_ROOT_NAME = "Heterogeneous LLM Processing"

_KPI_SECTIONS = (
    "request_facing_latency",
    "pipeline_latency",
    "throughput_and_tokens",
    "transfer",
)

_DISPLAY_NAMES = {
    ("request_facing_latency", "latency.e2e"): "Request E2E",
    ("request_facing_latency", "latency.ttft"): "TTFT",
    ("request_facing_latency", "latency.tpot"): "TPOT",
    ("pipeline_latency", "latency.e2e"): "Pipeline E2E",
    ("pipeline_latency", "latency.prefill"): "Prefill latency",
    ("pipeline_latency", "latency.kv_export"): "KV export latency",
    ("pipeline_latency", "latency.kv_transfer"): "KV transfer latency",
    ("pipeline_latency", "latency.kv_transform"): "KV transform latency",
    ("pipeline_latency", "latency.decode"): "Decode latency",
    ("pipeline_latency", "latency.sampling"): "Sampling total",
    ("pipeline_latency", "latency.wait"): "Pipeline wait",
    ("throughput_and_tokens", "request.count"): "Request count",
    ("throughput_and_tokens", "request.input_tokens"): "Input tokens",
    ("throughput_and_tokens", "request.output_tokens"): "Output tokens",
    ("throughput_and_tokens", "request.total_tokens"): "Total tokens",
    ("throughput_and_tokens", "throughput.requests"): "Requests per second",
    ("throughput_and_tokens", "throughput.input_tokens"): "Input tokens per second",
    ("throughput_and_tokens", "throughput.output_tokens"): (
        "Output tokens per second"
    ),
    ("throughput_and_tokens", "throughput.total_tokens"): (
        "Total tokens per second"
    ),
    ("transfer", "transfer.bytes"): "Transferred bytes",
    ("transfer", "transfer.duration"): "Transfer duration",
    ("transfer", "transfer.effective_bandwidth"): "Effective bandwidth",
    ("transfer", "transfer.transform_duration"): "Transform duration",
    ("transfer", "transfer.wait_duration"): "Transfer wait",
    ("transfer", "transfer.handoff_duration"): "KV handoff",
    ("transfer", "transfer.setup_duration"): "Transfer setup",
    ("transfer", "decode.schedule_wait_duration"): "Decode scheduling wait",
    ("transfer", "transfer.e2e_share"): "Transfer E2E share",
}


class TimelineSummaryInputError(ValueError):
    """Validated input cannot be mapped without inventing timeline-summary evidence."""


@dataclass(frozen=True, slots=True)
class TimelineSummaryKpi:
    """One external-report KPI prepared for a trace-native representation."""

    identity: str
    section: str
    name: str
    display_name: str
    canonical_unit: str
    value: int | float | None
    unavailable_reason: str | None
    observation_layer: str
    source_event_ids: tuple[str, ...]
    calculation_method: str
    display_unit: str
    display_scale_numerator: int
    display_scale_denominator: int

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def counter_group_key(self) -> str:
        return (
            "summary.kpi.transfer"
            if self.section == "transfer"
            else "summary.kpi.token_throughput"
        )


@dataclass(frozen=True, slots=True)
class TokenInstantEvidence:
    """One validated output-token arrival from the measured request artifact."""

    request_id: str
    token_index: int
    timestamp_ns: int
    source_timestamp_ns: int
    source_clock_domain_id: str
    target_clock_domain_id: str
    alignment_method: str
    alignment_uncertainty_ns: int


@dataclass(frozen=True, slots=True)
class TimelineSummaryContext:
    """Path-free, deterministic evidence consumed by the Perfetto planner."""

    mapping_version: str
    source_identity_sha256: str
    kpis: tuple[TimelineSummaryKpi, ...]
    data_quality_annotations: tuple[tuple[str, bool | int | float | str], ...]
    trace_attributes: tuple[TraceAttributeSpec, ...]
    token_instants: tuple[TokenInstantEvidence, ...] = ()


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
        raise TimelineSummaryInputError(
            "timeline summary evidence must be deterministic finite JSON"
        ) from error


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TimelineSummaryInputError(f"{field} must be a non-empty string")
    return value


def _non_bool_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimelineSummaryInputError(f"{field} must be a non-boolean integer")
    return value


def _finite_value(value: object, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TimelineSummaryInputError(f"{field} must be a finite non-boolean number")
    return value


def _source_identity(loaded: object) -> str:
    manifest = getattr(loaded, "manifest", None)
    run_id = _nonempty_string(getattr(manifest, "run_id", None), "run_id")
    closeout_manifest_sha256 = _nonempty_string(
        getattr(loaded, "closeout_manifest_sha256", None),
        "closeout_manifest_sha256",
    )
    roots = []
    for fingerprint in sorted(
        getattr(loaded, "root_fingerprints", ()),
        key=lambda item: getattr(item, "root_id", ""),
    ):
        roots.append(
            {
                "root_id": _nonempty_string(
                    getattr(fingerprint, "root_id", None),
                    "root fingerprint id",
                ),
                "file_count": _non_bool_int(
                    getattr(fingerprint, "file_count", None),
                    "root fingerprint file_count",
                ),
                "fingerprint_sha256": _nonempty_string(
                    getattr(fingerprint, "fingerprint_sha256", None),
                    "root fingerprint SHA-256",
                ),
            }
        )
    if not roots:
        raise TimelineSummaryInputError("validated source has no root fingerprints")
    payload = {
        "run_id": run_id,
        "closeout_manifest_sha256": closeout_manifest_sha256,
        "roots": roots,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_event_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    record_ids: set[str] = set()
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise TimelineSummaryInputError("KPI sources must be an array")
    for source in sources:
        if not isinstance(source, dict):
            raise TimelineSummaryInputError("KPI source must be an object")
        raw_ids = source.get("record_ids")
        if not isinstance(raw_ids, list):
            raise TimelineSummaryInputError("KPI source record_ids must be an array")
        for record_id in raw_ids:
            record_ids.add(_nonempty_string(record_id, "KPI source record id"))
    return tuple(sorted(record_ids))


def _kpi_from_mapping(section: str, value: Mapping[str, Any]) -> TimelineSummaryKpi:
    name = _nonempty_string(value.get("name"), f"{section} KPI name")
    try:
        display_name = _DISPLAY_NAMES[(section, name)]
    except KeyError as error:
        raise TimelineSummaryInputError(
            f"unsupported timeline summary KPI identity: {section}:{name}"
        ) from error
    unit = _nonempty_string(
        value.get("canonical_unit"),
        f"{section}:{name} canonical_unit",
    )
    availability = value.get("availability")
    raw_value = value.get("value")
    raw_reason = value.get("unavailable_reason")
    if availability == Availability.AVAILABLE.value:
        canonical_value: int | float | None = _finite_value(
            raw_value,
            f"{section}:{name} value",
        )
        if raw_reason is not None:
            raise TimelineSummaryInputError(
                f"{section}:{name} has a value and unavailable reason"
            )
        reason = None
    elif availability == Availability.NOT_AVAILABLE.value:
        if raw_value is not None:
            raise TimelineSummaryInputError(
                f"{section}:{name} unavailable value must be null"
            )
        canonical_value = None
        reason = _nonempty_string(
            raw_reason,
            f"{section}:{name} unavailable reason",
        )
    else:
        raise TimelineSummaryInputError(
            f"{section}:{name} has unsupported availability"
        )
    scope = value.get("scope")
    if not isinstance(scope, dict):
        raise TimelineSummaryInputError(f"{section}:{name} scope must be an object")
    observation_layer = _nonempty_string(
        scope.get("observation_layer"),
        f"{section}:{name} observation layer",
    )
    calculation = value.get("calculation")
    if not isinstance(calculation, dict):
        raise TimelineSummaryInputError(
            f"{section}:{name} calculation must be an object"
        )
    method = _nonempty_string(
        calculation.get("method_id"),
        f"{section}:{name} calculation method",
    )
    display = value.get("display")
    if not isinstance(display, dict):
        raise TimelineSummaryInputError(f"{section}:{name} display must be an object")
    display_unit = _nonempty_string(
        display.get("unit"),
        f"{section}:{name} display unit",
    )
    numerator = _non_bool_int(
        display.get("scale_numerator"),
        f"{section}:{name} display scale numerator",
    )
    denominator = _non_bool_int(
        display.get("scale_denominator"),
        f"{section}:{name} display scale denominator",
    )
    if denominator <= 0:
        raise TimelineSummaryInputError(
            f"{section}:{name} display scale denominator must be positive"
        )
    return TimelineSummaryKpi(
        identity=f"{section}:{name}",
        section=section,
        name=name,
        display_name=display_name,
        canonical_unit=unit,
        value=canonical_value,
        unavailable_reason=reason,
        observation_layer=observation_layer,
        source_event_ids=_source_event_ids(value),
        calculation_method=method,
        display_unit=display_unit,
        display_scale_numerator=numerator,
        display_scale_denominator=denominator,
    )


def _flatten_kpis(calculated: Mapping[str, Any]) -> tuple[TimelineSummaryKpi, ...]:
    values: list[TimelineSummaryKpi] = []
    for section in _KPI_SECTIONS:
        rows = calculated.get(section)
        if not isinstance(rows, list):
            raise TimelineSummaryInputError(
                f"calculated KPI section {section!r} must be an array"
            )
        for row in rows:
            if not isinstance(row, dict):
                raise TimelineSummaryInputError(
                    f"calculated KPI section {section!r} has a non-object row"
                )
            values.append(_kpi_from_mapping(section, row))
    identities = [item.identity for item in values]
    if len(identities) != len(set(identities)):
        raise TimelineSummaryInputError("calculated KPI identities are not unique")
    return tuple(sorted(values, key=lambda item: item.identity))


def _data_quality_annotations(
    loaded: object,
    *,
    source_identity_sha256: str,
    kpis: tuple[TimelineSummaryKpi, ...],
) -> tuple[tuple[str, bool | int | float | str], ...]:
    manifest = getattr(loaded, "manifest", None)
    events = tuple(getattr(loaded, "events", ()))
    validation = validate_marker_groups(events)
    if validation.status != "valid":
        raise TimelineSummaryInputError(
            "marker validation changed after the normalized run was loaded"
        )
    correlation_ids = {
        event.attributes.get("hybrid.correlation_id")
        for event in events
        if event.event_name
        in {
            "request_received",
            "prefill_start",
            "kv_export_start",
            "kv_transfer_start",
            "kv_transform_start",
            "decode_loop_start",
            "response_done",
        }
    }
    if (
        not correlation_ids
        or any(not isinstance(value, str) or not value for value in correlation_ids)
    ):
        raise TimelineSummaryInputError(
            "canonical request markers lack explicit correlation identity"
        )
    joined_requests = len(correlation_ids)
    unjoined_requests = 0
    attributes = getattr(manifest, "attributes", {})
    configuration = getattr(manifest, "configuration", {})
    if not isinstance(attributes, dict) or not isinstance(configuration, dict):
        raise TimelineSummaryInputError("manifest attributes/configuration are invalid")
    raw_profiler_kind = attributes.get("hybrid.phase4b2b_profile_kind")
    profiler_kind = (
        raw_profiler_kind
        if isinstance(raw_profiler_kind, str) and raw_profiler_kind
        else "unknown"
    )
    native_alignment = attributes.get(
        "hybrid.profiler_alignment_status",
        "not_applicable",
    )
    native_alignment = _nonempty_string(
        native_alignment,
        "hybrid.profiler_alignment_status",
    )
    alignment_method = _nonempty_string(
        configuration.get("alignment_method"),
        "alignment_method",
    )
    offset = attributes.get("hybrid.alignment_offset_ns")
    uncertainty = attributes.get("hybrid.alignment_uncertainty_ns")
    offset = _non_bool_int(offset, "hybrid.alignment_offset_ns")
    uncertainty = _non_bool_int(
        uncertainty,
        "hybrid.alignment_uncertainty_ns",
    )
    if uncertainty < 0:
        raise TimelineSummaryInputError("alignment uncertainty must be non-negative")
    unavailable = {
        item.identity: item.unavailable_reason
        for item in kpis
        if not item.available
    }
    roots = {
        fingerprint.root_id: fingerprint.fingerprint_sha256
        for fingerprint in sorted(
            getattr(loaded, "root_fingerprints", ()),
            key=lambda item: item.root_id,
        )
    }
    envelopes = tuple(getattr(loaded, "native_envelopes", ()))
    has_rbln = any(
        getattr(envelope, "profiler_type", None) == "npu_rbln"
        for envelope in envelopes
    )
    annotations: dict[str, bool | int | float | str] = {
        "hetero.run_status": _nonempty_string(
            getattr(getattr(manifest, "status", None), "value", None),
            "run status",
        ),
        "hetero.canonical_marker_count": len(events),
        "hetero.missing_marker_count": len(validation.missing_markers),
        "hetero.duplicate_marker_count": len(validation.duplicate_markers),
        "hetero.ordering_violation_count": len(validation.ordering_issues),
        "hetero.pairing_violation_count": len(validation.pairing_issues),
        "hetero.joined_request_count": joined_requests,
        "hetero.unjoined_request_count": unjoined_requests,
        "hetero.alignment_method": alignment_method,
        "hetero.alignment_offset_ns": offset,
        "hetero.alignment_uncertainty_ns": uncertainty,
        "hetero.clock_status": "canonical_aligned",
        "hetero.canonical_clock_domain": _nonempty_string(
            getattr(loaded, "canonical_clock_domain_id", None),
            "canonical clock domain",
        ),
        "hetero.source_artifact_validation": "passed_by_strict_input_loader",
        # This is a publication precondition, not a self-referential trace hash.
        "hetero.perfetto_validation": (
            "required_pinned_official_trace_processor_before_publication"
        ),
        "hetero.profiler_kind": profiler_kind,
        "hetero.native_profiler_alignment": native_alignment,
        "hetero.rbln_pb_state": (
            "perfetto_compatible_rbln_trace" if has_rbln else "not_applicable"
        ),
        "hetero.unavailable_kpi_count": len(unavailable),
        "hetero.unavailable_kpis_json": _canonical_json(unavailable),
        "hetero.source_fingerprints_json": _canonical_json(roots),
        "hetero.source_identity_sha256": source_identity_sha256,
        "hetero.trace_mapping_version": TIMELINE_SUMMARY_MAPPING_VERSION,
        "hetero.native_details_emitted": False,
        "hetero.rbln_pb_structure_analysis": (
            "deferred_to_official_trace_processor"
            if has_rbln
            else "not_applicable"
        ),
    }
    return tuple(sorted(annotations.items()))


def _reject_duplicate_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TimelineSummaryInputError(f"duplicate measured-request key {key!r}")
        result[key] = value
    return result


def _token_instant_evidence(loaded: object) -> tuple[TokenInstantEvidence, ...]:
    payload = _read_source_artifact(
        loaded,
        source_role="gpu",
        relative_path="raw/client/measured_requests.jsonl",
    )
    if payload is None:
        return ()
    request_ids = {
        event.request_id
        for event in getattr(loaded, "events", ())
        if event.event_name in {"request_received", "response_done"}
    }
    if len(request_ids) != 1 or None in request_ids:
        raise TimelineSummaryInputError(
            "token timestamps require one canonical request identity"
        )
    request_id = _nonempty_string(next(iter(request_ids)), "canonical request id")
    rows: list[dict[str, object]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            if not line:
                continue
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {token}")
                ),
            )
            if not isinstance(value, dict):
                raise TimelineSummaryInputError(
                    "measured-request JSONL row must be an object"
                )
            identity = value.get("request_id", value.get("client_request_id"))
            if identity == request_id:
                rows.append(value)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise TimelineSummaryInputError(
            "measured-request artifact is not valid UTF-8 JSONL"
        ) from error
    if len(rows) != 1:
        raise TimelineSummaryInputError(
            "measured-request artifact must contain one matching request row"
        )
    row = rows[0]
    raw = row.get("valid_token_timestamps_ns")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise TimelineSummaryInputError(
            "valid_token_timestamps_ns must be a non-empty array"
        )
    timestamps = tuple(
        _non_bool_int(value, "valid_token_timestamps_ns item") for value in raw
    )
    if any(value < 0 for value in timestamps):
        raise TimelineSummaryInputError("token timestamp must be non-negative")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise TimelineSummaryInputError(
            "valid_token_timestamps_ns must be strictly increasing"
        )
    output_tokens = _non_bool_int(row.get("output_tokens"), "output_tokens")
    if output_tokens != len(timestamps):
        raise TimelineSummaryInputError(
            "valid token timestamp count differs from output_tokens"
        )
    request_start = _non_bool_int(row.get("request_start_ns"), "request_start_ns")
    stream_end = _non_bool_int(row.get("stream_end_ns"), "stream_end_ns")
    if request_start > stream_end:
        raise TimelineSummaryInputError(
            "measured request interval end precedes its start"
        )
    if request_start > timestamps[0] or timestamps[-1] > stream_end:
        raise TimelineSummaryInputError(
            "token timestamps fall outside the measured request interval"
        )

    sources = [
        source
        for source in getattr(loaded, "sources", ())
        if getattr(source, "source_role", None) == "gpu"
    ]
    if len(sources) != 1:
        raise TimelineSummaryInputError("GPU source must occur exactly once")
    source_clocks = [
        clock
        for clock in getattr(sources[0], "clock_domains", ())
        if getattr(clock, "unit", None) == "ns"
        and getattr(clock, "monotonic", None) is True
    ]
    if len(source_clocks) != 1:
        raise TimelineSummaryInputError(
            "measured token timestamps require one monotonic ns GPU clock"
        )
    source_clock = _nonempty_string(
        getattr(source_clocks[0], "clock_domain_id", None),
        "GPU source clock domain",
    )
    normalized_source_clock = f"gpu:{source_clock}"
    transforms = [
        transform
        for transform in getattr(loaded, "transforms", ())
        if getattr(transform, "source_clock_domain_id", None)
        == normalized_source_clock
        and getattr(transform, "target_clock_domain_id", None)
        == getattr(loaded, "canonical_clock_domain_id", None)
    ]
    if len(transforms) != 1:
        raise TimelineSummaryInputError(
            "measured token timestamps lack one explicit canonical transform"
        )
    transform = transforms[0]
    if getattr(transform, "scale", None) != 1.0:
        raise TimelineSummaryInputError("token timestamp transform scale must be 1")
    offset = _non_bool_int(getattr(transform, "offset_ns", None), "token offset")
    uncertainty = _non_bool_int(
        getattr(transform, "uncertainty_ns", None), "token uncertainty"
    )
    if uncertainty < 0:
        raise TimelineSummaryInputError("token uncertainty must be non-negative")
    valid_from = _non_bool_int(
        getattr(transform, "valid_from_source_ns", None),
        "token transform valid_from",
    )
    valid_to = getattr(transform, "valid_to_source_ns", None)
    if valid_to is not None:
        valid_to = _non_bool_int(valid_to, "token transform valid_to")
    if request_start < valid_from or (
        valid_to is not None and stream_end > valid_to
    ):
        raise TimelineSummaryInputError(
            "measured request interval falls outside the canonical transform interval"
        )
    transform_attributes = getattr(transform, "attributes", {})
    method = _nonempty_string(
        transform_attributes.get(
            "hybrid.method",
            getattr(getattr(transform, "method", None), "value", None),
        ),
        "token alignment method",
    )
    target_clock = _nonempty_string(
        getattr(transform, "target_clock_domain_id", None),
        "token target clock domain",
    )
    canonical = tuple(value + offset for value in timestamps)
    canonical_request_start = request_start + offset
    canonical_stream_end = stream_end + offset
    if (
        canonical_request_start < 0
        or canonical_stream_end < canonical_request_start
    ):
        raise TimelineSummaryInputError(
            "token transform produces an invalid measured request interval"
        )
    if canonical[0] < 0 or any(
        right <= left for left, right in zip(canonical, canonical[1:])
    ):
        raise TimelineSummaryInputError(
            "token transform produces invalid canonical timestamps"
        )
    # Token arrivals are observed by the client, while request_received and
    # response_done are proxy-side pipeline markers.  Network delivery and SSE
    # parsing may legitimately place a client observation just outside those
    # proxy markers.  Preserve both observations and validate token arrivals
    # against their own source-backed client interval after applying the same
    # explicit clock transform.
    if (
        canonical[0] < canonical_request_start
        or canonical[-1] > canonical_stream_end
    ):
        raise TimelineSummaryInputError(
            "canonical token timestamps fall outside measured request interval"
        )
    return tuple(
        TokenInstantEvidence(
            request_id=request_id,
            token_index=index,
            timestamp_ns=timestamp,
            source_timestamp_ns=timestamps[index],
            source_clock_domain_id=normalized_source_clock,
            target_clock_domain_id=target_clock,
            alignment_method=method,
            alignment_uncertainty_ns=uncertainty,
        )
        for index, timestamp in enumerate(canonical)
    )


def build_timeline_summary_context(loaded: object) -> TimelineSummaryContext:
    """Build deterministic trace-native timeline-summary evidence from a loaded run."""

    source_identity_sha256 = _source_identity(loaded)
    calculated = calculate_overview_kpis(loaded)
    if not isinstance(calculated, dict):
        raise TimelineSummaryInputError("external KPI report calculation returned no object")
    kpis = _flatten_kpis(calculated)
    return TimelineSummaryContext(
        mapping_version=TIMELINE_SUMMARY_MAPPING_VERSION,
        source_identity_sha256=source_identity_sha256,
        kpis=kpis,
        data_quality_annotations=_data_quality_annotations(
            loaded,
            source_identity_sha256=source_identity_sha256,
            kpis=kpis,
        ),
        trace_attributes=build_performance_trace_attributes(loaded, calculated),
        token_instants=_token_instant_evidence(loaded),
    )


__all__ = [
    "LEGACY_MAPPING_VERSION",
    "TIMELINE_SUMMARY_MAPPING_VERSION",
    "TIMELINE_SUMMARY_ROOT_NAME",
    "TimelineSummaryContext",
    "TimelineSummaryInputError",
    "TimelineSummaryKpi",
    "TokenInstantEvidence",
    "build_timeline_summary_context",
]
