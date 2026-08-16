"""Source-backed inputs for the trace-native Perfetto timeline summary.

This module deliberately stops before protobuf planning.  KPI values come from
the same pure calculation used by the external KPI report, while the planner
decides how those values are represented as tracks, slices, counters, and
instants.
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
from .trace_attributes import build_performance_trace_attributes


LEGACY_MAPPING_VERSION = "legacy-unversioned-phase5-v1"
TIMELINE_SUMMARY_MAPPING_VERSION = "phase6b-timeline-summary-v2"
TIMELINE_SUMMARY_ROOT_NAME = "Heterogeneous LLM Summary"

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
class TimelineSummaryContext:
    """Path-free, deterministic evidence consumed by the Perfetto planner."""

    mapping_version: str
    source_identity_sha256: str
    kpis: tuple[TimelineSummaryKpi, ...]
    data_quality_annotations: tuple[tuple[str, bool | int | float | str], ...]
    trace_attributes: tuple[TraceAttributeSpec, ...]


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
    )


__all__ = [
    "LEGACY_MAPPING_VERSION",
    "TIMELINE_SUMMARY_MAPPING_VERSION",
    "TIMELINE_SUMMARY_ROOT_NAME",
    "TimelineSummaryContext",
    "TimelineSummaryInputError",
    "TimelineSummaryKpi",
    "build_timeline_summary_context",
]
