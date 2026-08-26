"""Strict SQL reconciliation for generated Perfetto traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import stat
from typing import Any, Final

from perfetto.trace_processor import (
    TraceProcessor,
    TraceProcessorConfig,
    TraceProcessorException,
)

from ..schema.constants import SCHEMA_VERSION
from .model import AnnotationValue, SliceSpec, TracePlan
from .tooling import (
    PERFETTO_PACKAGE_VERSION,
    PROTOBUF_PACKAGE_VERSION,
    TRACE_PROCESSOR_RPC_API_VERSION,
    TRACE_PROCESSOR_VERSION,
    ToolchainRuntime,
    resolve_toolchain,
)
from .validation_queries import (
    _TP_NATIVE_POLICY_KEYS,
    _PROCESS_SQL,
    _TRACK_SQL,
    _SLICE_SQL,
    _ANNOTATION_SQL,
    _STEP_ANNOTATION_SQL,
    _NATIVE_POLICY_SQL,
    _NATIVE_EVENT_SEMANTICS_SQL,
    _COUNTER_SQL,
    _FLOW_SQL,
    _DANGLING_FLOW_SQL,
    _IMPORT_ERROR_SQL,
    _NATIVE_TRACE_SUMMARY_SQL,
    _NATIVE_TRACE_FLOW_SQL,
    _NATIVE_TRACE_PARENT_RANGE_SQL,
    _NATIVE_TRACE_CATEGORY_SQL,
    _LEGACY_MAPPING_VERSION,
    _TIMELINE_SUMMARY_MAPPING_VERSION,
    _TIMELINE_SUMMARY_ROOT_KEY,
    _TIMELINE_SUMMARY_ROOT_NAME,
    _TIMELINE_SUMMARY_TRACK_PREFIX,
    _TIMELINE_SUMMARY_KPI_TRACK_PREFIX,
    _TIMELINE_SUMMARY_DATA_QUALITY_KEY,
    _TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX,
    _RESOURCE_TELEMETRY_ROOT_KEY,
    _REQUEST_RESOURCE_ROOT_KEY,
    _REQUEST_RESOURCE_ROOT_NAME,
    _REPORT_ROW_QUERIES,
    _TRACE_ATTRIBUTE_SQL,
    _TIMELINE_SUMMARY_HIERARCHY_SQL,
    _TIMELINE_SUMMARY_SLICE_SQL,
    _TIMELINE_SUMMARY_KPI_SQL,
    _TIMELINE_SUMMARY_DATA_QUALITY_SQL,
)


VALIDATION_RECORD_TYPE: Final = "perfetto_trace_validation"


class TraceValidationError(RuntimeError):
    """Trace Processor output does not exactly reconcile with its plan."""

    def __init__(
        self,
        message: str,
        *,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def validate_trace(
    plan: TracePlan,
    trace_path: str | Path,
    *,
    toolchain: ToolchainRuntime,
) -> dict[str, Any]:
    """Validate ``trace_path`` against ``plan`` using the pinned official TP.

    The caller must provide an already validated :class:`ToolchainRuntime`.
    Its explicit binary is revalidated before use and passed to
    :class:`TraceProcessorConfig`; latest-binary fetching is always disabled.
    The returned report contains no filesystem paths.
    """

    runtime = _validated_runtime(toolchain)
    trace_bytes, trace_sha256 = _stable_trace_bytes(Path(trace_path))
    mismatches = _plan_contract_mismatches(plan)

    config = TraceProcessorConfig(
        bin_path=os.fspath(runtime.binary_path),
        fetch_latest_trace_processor=False,
        unique_port=True,
        load_timeout=10,
    )
    try:
        with TraceProcessor(trace=io.BytesIO(trace_bytes), config=config) as processor:
            actual_queries = {
                "process": _run_query(processor, "process", _PROCESS_SQL),
                "tracks": _run_query(processor, "tracks", _TRACK_SQL),
                "slices": _run_query(processor, "slices", _SLICE_SQL),
                "annotations": _run_query(
                    processor,
                    "annotations",
                    _ANNOTATION_SQL,
                ),
                "step_annotations": _run_query(
                    processor,
                    "step_annotations",
                    _STEP_ANNOTATION_SQL,
                ),
                "counters": _run_query(processor, "counters", _COUNTER_SQL),
                "flows": _run_query(processor, "flows", _FLOW_SQL),
                "dangling_flows": _run_query(
                    processor,
                    "dangling_flows",
                    _DANGLING_FLOW_SQL,
                ),
                "import_errors": _run_query(
                    processor,
                    "import_errors",
                    _IMPORT_ERROR_SQL,
                ),
                "native_policy": _run_query(
                    processor,
                    "native_policy",
                    _NATIVE_POLICY_SQL,
                ),
            }
            if _has_native_event_specs(plan):
                actual_queries["native_event_semantics"] = _run_query(
                    processor,
                    "native_event_semantics",
                    _NATIVE_EVENT_SEMANTICS_SQL,
                )
            if plan.mapping_version != _LEGACY_MAPPING_VERSION:
                actual_queries.update(
                    {
                        "timeline_summary_hierarchy": _run_query(
                            processor,
                            "timeline_summary_hierarchy",
                            _TIMELINE_SUMMARY_HIERARCHY_SQL,
                        ),
                        "timeline_summary_slices": _run_query(
                            processor,
                            "timeline_summary_slices",
                            _TIMELINE_SUMMARY_SLICE_SQL,
                        ),
                        "timeline_summary_kpis": _run_query(
                            processor,
                            "timeline_summary_kpis",
                            _TIMELINE_SUMMARY_KPI_SQL,
                        ),
                        "timeline_summary_data_quality": _run_query(
                            processor,
                            "timeline_summary_data_quality",
                            _TIMELINE_SUMMARY_DATA_QUALITY_SQL,
                        ),
                        "trace_attributes": _run_query(
                            processor,
                            "trace_attributes",
                            _TRACE_ATTRIBUTE_SQL,
                        ),
                    }
                )
    except (TraceProcessorException, OSError) as error:
        raise TraceValidationError(
            "official Trace Processor failed to parse or query the trace"
        ) from error

    expected = _expected_rows(plan)
    query_names = (
        "process",
        "tracks",
        "slices",
        "annotations",
        "step_annotations",
        "counters",
        "flows",
        "dangling_flows",
        "import_errors",
        "native_policy",
    )
    if _has_native_event_specs(plan):
        query_names += ("native_event_semantics",)
    if plan.mapping_version != _LEGACY_MAPPING_VERSION:
        expected.update(_expected_timeline_summary_rows(plan))
        query_names += (
            "timeline_summary_hierarchy",
            "timeline_summary_slices",
            "timeline_summary_kpis",
            "timeline_summary_data_quality",
            "trace_attributes",
        )
    query_reports: list[dict[str, Any]] = []
    for name in query_names:
        actual = actual_queries[name]
        expected_rows = expected[name]
        matched = actual["rows"] == expected_rows
        if not matched:
            mismatches.append(
                f"{name} SQL rows differ: expected "
                f"{len(expected_rows)}/{_rows_sha256(expected_rows)}, got "
                f"{actual['row_count']}/{actual['rows_sha256']}"
            )
        compact_native_rows = _has_native_event_specs(plan)
        query_report = {
            key: value
            for key, value in actual.items()
            if (
                key != "rows"
                or not compact_native_rows
                or name in _REPORT_ROW_QUERIES
            )
        }
        query_reports.append(
            {
                **query_report,
                "expected_row_count": len(expected_rows),
                "expected_rows_sha256": _rows_sha256(expected_rows),
                "matched": matched,
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": VALIDATION_RECORD_TYPE,
        "valid": not mismatches,
        "run_id": plan.run_id,
        "canonical_clock_domain_id": plan.canonical_clock_domain_id,
        "trace": {
            "size_bytes": len(trace_bytes),
            "sha256": trace_sha256,
        },
        "toolchain": {
            **runtime.metadata,
            "perfetto_package_version": runtime.perfetto_package_version,
            "protobuf_package_version": runtime.protobuf_package_version,
            "trace_processor_rpc_api_version": (
                runtime.trace_processor_rpc_api_version
            ),
        },
        "counts": {
            name: actual_queries[name]["row_count"]
            for name in sorted(actual_queries)
        },
        "flow_endpoint_reconciliation": _flow_endpoint_summary(plan),
        "flow_evidence": [
            {
                "flow_id": flow.flow_id,
                "source_event_id": flow.source_event_id,
                "destination_event_id": flow.destination_event_id,
                "evidence_kind": flow.evidence_kind,
                "evidence_id": flow.evidence_id,
            }
            for flow in sorted(plan.flows, key=lambda item: item.flow_id)
        ],
        "queries": query_reports,
        "mismatches": sorted(mismatches),
    }
    _canonical_json_bytes(report)
    if mismatches:
        raise TraceValidationError(
            f"Perfetto trace validation failed with {len(mismatches)} mismatch(es)",
            report=report,
        )
    return report


def validate_native_perfetto_trace(
    trace_path: str | Path,
    *,
    toolchain: ToolchainRuntime,
    profiler_type: str,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_slice_count: int,
    expected_track_count: int,
    expected_flow_count: int,
) -> dict[str, Any]:
    """Validate an unmodified native-clock Perfetto trace with official TP."""

    runtime = _validated_runtime(toolchain)
    trace_bytes, trace_sha256 = _stable_trace_bytes(Path(trace_path))
    mismatches: list[str] = []
    if len(trace_bytes) != expected_size_bytes:
        mismatches.append("native trace size differs from immutable source")
    if trace_sha256 != expected_sha256:
        mismatches.append("native trace SHA-256 differs from immutable source")

    config = TraceProcessorConfig(
        bin_path=os.fspath(runtime.binary_path),
        fetch_latest_trace_processor=False,
        unique_port=True,
        load_timeout=10,
    )
    try:
        with TraceProcessor(
            trace=io.BytesIO(trace_bytes),
            config=config,
        ) as processor:
            summary = _run_query(
                processor,
                "native_trace_summary",
                _NATIVE_TRACE_SUMMARY_SQL,
            )
            flows = _run_query(
                processor,
                "native_trace_flows",
                _NATIVE_TRACE_FLOW_SQL,
            )
            parent_ranges = _run_query(
                processor,
                "native_trace_parent_ranges",
                _NATIVE_TRACE_PARENT_RANGE_SQL,
            )
            categories = _run_query(
                processor,
                "native_trace_categories",
                _NATIVE_TRACE_CATEGORY_SQL,
            )
            import_errors = _run_query(
                processor,
                "native_trace_import_errors",
                _IMPORT_ERROR_SQL,
            )
    except (TraceProcessorException, OSError) as error:
        raise TraceValidationError(
            "official Trace Processor failed to inspect native trace"
        ) from error

    if (
        summary["row_count"] != 1
        or flows["row_count"] != 1
        or parent_ranges["row_count"] != 1
    ):
        mismatches.append("native trace aggregate SQL did not return one row")
        counts: dict[str, object] = {}
    else:
        counts = {
            **summary["rows"][0],
            **flows["rows"][0],
            **parent_ranges["rows"][0],
        }
        if counts["slice_count"] != expected_slice_count:
            mismatches.append(
                "native trace slice count differs from protobuf evidence"
            )
        if counts["track_count"] != expected_track_count:
            mismatches.append(
                "native trace track count differs from used-track evidence"
            )
        if counts["flow_count"] != expected_flow_count:
            mismatches.append(
                "native trace flow count differs from protobuf endpoint evidence"
            )
        if counts["invalid_duration_count"] != 0:
            mismatches.append("native trace has negative slice duration")
        if (
            counts["invalid_timestamp_count"] != 0
            or counts["min_timestamp_ns"] is None
            or counts["min_timestamp_ns"] < 0
            or counts["max_end_ns"] is None
            or counts["max_end_ns"] < counts["min_timestamp_ns"]
        ):
            mismatches.append("native trace has invalid timestamp bounds")
        if counts["parent_child_range_violation_count"] != 0:
            mismatches.append("native trace has parent/child range violation")
    if import_errors["row_count"] != 0:
        mismatches.append("native trace has non-info Trace Processor stats")

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "perfetto_native_trace_validation",
        "valid": not mismatches,
        "profiler_type": profiler_type,
        "trace": {
            "size_bytes": len(trace_bytes),
            "sha256": trace_sha256,
        },
        "clock_policy": {
            "alignment_status": "partial_unaligned",
            "canonical_merge": False,
            "timestamp_rebased": False,
            "native_relative_timestamps_preserved": True,
        },
        "counts": counts,
        "category_counts": categories["rows"],
        "timestamp_fallback_count": 0,
        "fabricated_event_count": 0,
        "import_error_count": import_errors["row_count"],
        "mismatches": mismatches,
        "queries": [
            summary,
            flows,
            parent_ranges,
            categories,
            import_errors,
        ],
        "toolchain": {
            **runtime.metadata,
            "perfetto_package_version": runtime.perfetto_package_version,
            "protobuf_package_version": runtime.protobuf_package_version,
        },
    }


def validate_trace_plan(
    plan: TracePlan,
    trace_path: str | Path,
    *,
    toolchain: ToolchainRuntime,
) -> dict[str, Any]:
    """Compatibility spelling for :func:`validate_trace`."""

    return validate_trace(plan, trace_path, toolchain=toolchain)


def _validated_runtime(runtime: ToolchainRuntime) -> ToolchainRuntime:
    if not isinstance(runtime, ToolchainRuntime):
        raise TypeError("toolchain must be a validated ToolchainRuntime")
    expected = (
        PERFETTO_PACKAGE_VERSION,
        PROTOBUF_PACKAGE_VERSION,
        TRACE_PROCESSOR_VERSION,
        TRACE_PROCESSOR_RPC_API_VERSION,
    )
    actual = (
        runtime.perfetto_package_version,
        runtime.protobuf_package_version,
        runtime.trace_processor_version,
        runtime.trace_processor_rpc_api_version,
    )
    if actual != expected:
        raise TraceValidationError(
            "ToolchainRuntime metadata does not match the pinned release"
        )
    revalidated = resolve_toolchain(runtime.binary_path)
    if revalidated != runtime:
        raise TraceValidationError(
            "ToolchainRuntime changed since it was validated"
        )
    return revalidated


def _stable_trace_bytes(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise TraceValidationError("cannot inspect the trace input") from error
    if stat.S_ISLNK(before.st_mode):
        raise TraceValidationError("trace input must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise TraceValidationError("trace input must be a regular file")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise TraceValidationError("cannot read the trace input") from error
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if not stable or len(payload) != after.st_size:
        raise TraceValidationError("trace input changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _run_query(
    processor: TraceProcessor,
    name: str,
    sql: str,
) -> dict[str, Any]:
    result = processor.query(sql)
    columns = list(result.column_names)
    rows = [
        {
            column: _json_value(getattr(row, column))
            for column in columns
        }
        for row in result
    ]
    canonical_rows = _sorted_rows(rows)
    return {
        "name": name,
        "sql": sql,
        "columns": columns,
        "row_count": len(canonical_rows),
        "rows_sha256": _rows_sha256(canonical_rows),
        "rows": canonical_rows,
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise TraceValidationError("Trace Processor returned a non-finite value")
        return converted
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TraceValidationError(
        f"Trace Processor returned unsupported SQL type {type(value).__name__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TraceValidationError(
            "validation data is not deterministic finite JSON"
        ) from error


def _sorted_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    copied = [dict(row) for row in rows]
    return sorted(copied, key=_canonical_json_bytes)


def _rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(rows))).hexdigest()


def _expected_rows(plan: TracePlan) -> dict[str, list[dict[str, object]]]:
    process = [
        {
            "pid": plan.process_id,
            "name": f"perfetto-hetero-profiler:{plan.run_id}",
        }
    ]
    tracks: list[dict[str, object]] = [
        {
            "trace_uuid": plan.process_uuid,
            "name": plan.run_id,
            "type": "process_track_event",
            "description": (
                "synthetic process for canonical clock domain "
                f"{plan.canonical_clock_domain_id}"
            ),
            "unit": None,
            "pid": plan.process_id,
        }
    ]
    for track in plan.tracks:
        counter = track.kind.strip().casefold() == "counter"
        tracks.append(
            {
                "trace_uuid": track.uuid,
                "name": track.name,
                "type": (
                    "process_counter_track_event"
                    if counter
                    else "process_merged_track_event"
                ),
                "description": track.description,
                "unit": _counter_unit(track.unit) if counter else None,
                "pid": plan.process_id,
            }
        )

    slices: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    track_by_key = plan.track_by_key
    for spec in plan.slices:
        track_name = track_by_key[spec.track_key].name
        slice_row = {
            "track_name": track_name,
            "slice_name": spec.name,
            "ts": spec.timestamp_ns,
            "dur": spec.duration_ns,
        }
        slices.append(slice_row)
        annotations.extend(_annotation_rows(slice_row, spec.annotations))
    for spec in plan.instants:
        track_name = track_by_key[spec.track_key].name
        slice_row = {
            "track_name": track_name,
            "slice_name": spec.name,
            "ts": spec.timestamp_ns,
            "dur": 0,
        }
        slices.append(slice_row)
        annotations.extend(_annotation_rows(slice_row, spec.annotations))

    counters = [
        {
            "track_name": track_by_key[spec.track_key].name,
            "unit": _counter_unit(track_by_key[spec.track_key].unit),
            "ts": spec.timestamp_ns,
            "value": float(spec.value),
        }
        for spec in plan.counters
    ]
    flows = [
        {
            "flow_id": flow.flow_id,
            "source_slice_name": flow.source_slice_name,
            "destination_slice_name": flow.destination_slice_name,
            "source_correlation_id": flow.correlation_id,
            "destination_correlation_id": flow.correlation_id,
        }
        for flow in plan.flows
    ]
    step_annotations = [
        row
        for row in annotations
        if row["key"] == "debug.hetero_step_index"
    ]
    native_policy = [
        row
        for row in annotations
        if row["key"] in _TP_NATIVE_POLICY_KEYS
    ]
    native_specs = [
        spec
        for spec in (*plan.slices, *plan.instants)
        if "hetero.native_profiler" in dict(spec.annotations)
    ]
    native_event_semantics = [
        {
            "event_count": len(native_specs),
            "timestamp_fallback_violation_count": sum(
                dict(spec.annotations).get("hetero.timestamp_fallback")
                is not False
                for spec in native_specs
            ),
            "fabricated_event_violation_count": sum(
                dict(spec.annotations).get("hetero.fabricated_event")
                is not False
                for spec in native_specs
            ),
        }
    ]
    result = {
        "process": _sorted_rows(process),
        "tracks": _sorted_rows(tracks),
        "slices": _sorted_rows(slices),
        "annotations": _sorted_rows(annotations),
        "step_annotations": _sorted_rows(step_annotations),
        "counters": _sorted_rows(counters),
        "flows": _sorted_rows(flows),
        "dangling_flows": [],
        "import_errors": [],
        "native_policy": _sorted_rows(native_policy),
    }
    if native_specs:
        result["native_event_semantics"] = native_event_semantics
    return result


def _has_native_event_specs(plan: TracePlan) -> bool:
    return any(
        "hetero.native_profiler" in dict(spec.annotations)
        for spec in (*plan.slices, *plan.instants)
    )


def _expected_timeline_summary_rows(
    plan: TracePlan,
) -> dict[str, list[dict[str, object]]]:
    track_by_key = plan.track_by_key
    hierarchy: list[dict[str, object]] = []
    for track in plan.tracks:
        if track.parent_key is None:
            # Track Processor represents TrackDescriptor.parent_uuid pointing
            # at a process descriptor through upid rather than track.parent_id.
            parent_trace_uuid = None
            parent_name = None
        else:
            parent = track_by_key[track.parent_key]
            parent_trace_uuid = parent.uuid
            parent_name = parent.name
        hierarchy.append(
            {
                "trace_uuid": track.uuid,
                "name": track.name,
                "type": (
                    "process_counter_track_event"
                    if track.kind.strip().casefold() == "counter"
                    else "process_merged_track_event"
                ),
                "parent_trace_uuid": parent_trace_uuid,
                "parent_name": parent_name,
                "child_ordering": (
                    None
                    if track.child_ordering == "unknown"
                    else track.child_ordering
                ),
                "sibling_order_rank": track.sibling_order_rank,
            }
        )

    timeline_summary_slices: list[dict[str, object]] = []
    for spec in (*plan.slices, *plan.instants):
        current = track_by_key[spec.track_key]
        while (
            current.parent_key is not None
            and current.key != _TIMELINE_SUMMARY_ROOT_KEY
        ):
            current = track_by_key[current.parent_key]
        if current.key != _TIMELINE_SUMMARY_ROOT_KEY:
            continue
        track = track_by_key[spec.track_key]
        parent_name = (
            plan.run_id
            if track.parent_key is None
            else track_by_key[track.parent_key].name
        )
        timeline_summary_slices.append(
            {
                "track_name": track.name,
                "parent_track_name": parent_name,
                "slice_name": spec.name,
                "ts": spec.timestamp_ns,
                "dur": spec.duration_ns if isinstance(spec, SliceSpec) else 0,
            }
        )

    timeline_summary_kpis: list[dict[str, object]] = []
    for spec in plan.counters:
        if not spec.track_key.startswith(_TIMELINE_SUMMARY_KPI_TRACK_PREFIX):
            continue
        track = track_by_key[spec.track_key]
        annotations = dict(spec.annotations)
        parent_name = (
            plan.run_id
            if track.parent_key is None
            else track_by_key[track.parent_key].name
        )
        timeline_summary_kpis.append(
            {
                "track_name": track.name,
                "parent_track_name": parent_name,
                "kpi_identity": annotations.get("hetero.kpi_identity"),
                "unit": _counter_unit(track.unit),
                "ts": spec.timestamp_ns,
                "value": float(spec.value),
            }
        )

    data_quality: list[dict[str, object]] = []
    for spec in plan.instants:
        if spec.track_key != _TIMELINE_SUMMARY_DATA_QUALITY_KEY:
            continue
        track = track_by_key[spec.track_key]
        parent_name = (
            plan.run_id
            if track.parent_key is None
            else track_by_key[track.parent_key].name
        )
        slice_row = {
            "track_name": track.name,
            "slice_name": spec.name,
            "ts": spec.timestamp_ns,
            "dur": 0,
        }
        for row in _annotation_rows(slice_row, spec.annotations):
            data_quality.append(
                {
                    "track_name": row["track_name"],
                    "parent_track_name": parent_name,
                    "slice_name": row["slice_name"],
                    "ts": row["ts"],
                    "dur": row["dur"],
                    "key": row["key"],
                    "value_type": row["value_type"],
                    "int_value": row["int_value"],
                    "real_value": row["real_value"],
                    "string_value": row["string_value"],
                }
            )

    trace_attributes = [
        {
            "name": f"trace_attribute.{spec.key}",
            "key_type": "single",
            "int_value": spec.value if isinstance(spec.value, int) else None,
            "str_value": spec.value if isinstance(spec.value, str) else None,
        }
        for spec in plan.trace_attributes
    ]
    return {
        "timeline_summary_hierarchy": _sorted_rows(hierarchy),
        "timeline_summary_slices": _sorted_rows(timeline_summary_slices),
        "timeline_summary_kpis": _sorted_rows(timeline_summary_kpis),
        "timeline_summary_data_quality": _sorted_rows(data_quality),
        "trace_attributes": _sorted_rows(trace_attributes),
    }


def _annotation_rows(
    slice_row: Mapping[str, object],
    annotations: Sequence[tuple[str, AnnotationValue]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, value in annotations:
        row: dict[str, object] = {
            **slice_row,
            "key": _tp_debug_key(name),
            "value_type": None,
            "int_value": None,
            "real_value": None,
            "string_value": None,
        }
        if isinstance(value, bool):
            row["value_type"] = "bool"
            row["int_value"] = 1 if value else 0
        elif isinstance(value, int):
            row["value_type"] = "int"
            row["int_value"] = value
        elif isinstance(value, float):
            row["value_type"] = "real"
            row["real_value"] = value
        elif isinstance(value, str):
            row["value_type"] = "string"
            row["string_value"] = value
        else:  # pragma: no cover - writer rejects this first
            raise TraceValidationError(
                f"unsupported annotation type in plan: {type(value).__name__}"
            )
        rows.append(row)
    return rows


def _counter_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().casefold()
    if normalized in {"ns", "nanosecond", "nanoseconds", "time_ns"}:
        return "ns"
    if normalized in {"count", "counts"}:
        return "count"
    if normalized in {"byte", "bytes", "size_bytes"}:
        return "bytes"
    return unit


def _tp_debug_key(name: str) -> str:
    # Trace Processor normalizes dots in DebugAnnotation names to underscores.
    return f"debug.{name.replace('.', '_')}"


def _plan_contract_mismatches(plan: TracePlan) -> list[str]:
    mismatches: list[str] = []
    native_track_types = {
        track.key.split(".", 2)[1]
        for track in plan.tracks
        if track.key.startswith("native.")
    }
    native_details_emitted = bool(native_track_types)
    for spec in (*plan.slices, *plan.instants):
        annotations = dict(spec.annotations)
        if "hetero.native_profiler" not in annotations:
            continue
        if annotations.get("hetero.timestamp_fallback") is not False:
            mismatches.append(
                f"native event {spec.name!r} does not prove timestamp fallback=0"
            )
        if annotations.get("hetero.fabricated_event") is not False:
            mismatches.append(
                f"native event {spec.name!r} is fabricated or lacks provenance"
            )
    summary = _flow_endpoint_summary(plan)
    if not summary["matched"]:
        mismatches.append(
            "plan flow endpoint IDs do not exactly match declared FlowSpec IDs"
        )

    metadata = [
        instant
        for instant in plan.instants
        if instant.name == "Clock/alignment metadata"
    ]
    if plan.mapping_version == _LEGACY_MAPPING_VERSION and len(metadata) != 1:
        mismatches.append(
            "legacy plan must contain exactly one Clock/alignment metadata instant"
        )
    elif plan.mapping_version != _LEGACY_MAPPING_VERSION and metadata:
        mismatches.append(
            "timeline summary plan must not contain a standalone "
            "Clock/alignment metadata instant"
        )
    elif metadata:
        values = dict(metadata[0].annotations)
        required = {
            "hetero.canonical_clock_domain": plan.canonical_clock_domain_id,
            "hetero.native_profiler_alignment": "partial_or_unaligned",
            "hetero.native_details_emitted": native_details_emitted,
        }
        if any(values.get(key) != value for key, value in required.items()):
            mismatches.append("canonical/native clock policy annotations differ")

    for spec in plan.slices:
        if spec.track_key != "profiler":
            continue
        values = dict(spec.annotations)
        profiler_type = values.get("hetero.profiler_type")
        profiler_details_emitted = profiler_type in native_track_types
        required = {
            "hetero.alignment_status": "partial",
            "hetero.alignment_method": "host_api_boundary_bracket",
            "hetero.unaligned_profiler_events": not profiler_details_emitted,
            "hetero.native_details_emitted": profiler_details_emitted,
        }
        if any(values.get(key) != value for key, value in required.items()):
            mismatches.append(
                f"native profiler policy differs for slice {spec.name!r}"
            )
        for key in (
            "hetero.profiler_type",
            "hetero.source_role",
            "hetero.native_clock_domain",
            "hetero.native_timestamp_unit",
        ):
            if not isinstance(values.get(key), str) or not values[key]:
                mismatches.append(
                    f"native profiler slice {spec.name!r} lacks {key}"
                )
    if plan.mapping_version != _LEGACY_MAPPING_VERSION:
        mismatches.extend(_timeline_summary_plan_contract_mismatches(plan))
    return mismatches


def _timeline_summary_plan_contract_mismatches(plan: TracePlan) -> list[str]:
    mismatches: list[str] = []
    if plan.mapping_version != _TIMELINE_SUMMARY_MAPPING_VERSION:
        mismatches.append(
            f"unsupported processing timeline mapping version {plan.mapping_version!r}"
        )

    track_by_key = plan.track_by_key

    def is_under(track_key: str, root_key: str) -> bool:
        current = track_by_key.get(track_key)
        seen: set[str] = set()
        while current is not None:
            if current.key == root_key:
                return True
            if current.key in seen or current.parent_key is None:
                return False
            seen.add(current.key)
            current = track_by_key.get(current.parent_key)
        return False

    processing_tracks = [
        track
        for track in plan.tracks
        if track.key.startswith(_TIMELINE_SUMMARY_TRACK_PREFIX)
    ]
    roots = [track for track in processing_tracks if track.parent_key is None]
    if len(roots) != 1 or roots[0].key != _TIMELINE_SUMMARY_ROOT_KEY:
        mismatches.append("processing timeline must contain exactly one summary.root")
    else:
        root = roots[0]
        if root.name != _TIMELINE_SUMMARY_ROOT_NAME:
            mismatches.append("processing timeline root track name differs")

    for track in processing_tracks:
        if track.key == _TIMELINE_SUMMARY_ROOT_KEY:
            continue
        current = track
        seen: set[str] = set()
        while current.parent_key is not None:
            if current.key in seen:
                mismatches.append(
                    f"processing track {track.key!r} has a parent cycle"
                )
                break
            seen.add(current.key)
            parent = track_by_key.get(current.parent_key)
            if parent is None:
                mismatches.append(
                    f"processing track {track.key!r} has an unknown parent"
                )
                break
            current = parent
        else:
            if current.key != _TIMELINE_SUMMARY_ROOT_KEY:
                mismatches.append(
                    f"processing track {track.key!r} is outside summary.root"
                )

    required_groups = {
        "summary.boundaries": ("summary.root", "Request Boundaries and Token Output"),
        "summary.boundaries.events": (
            "summary.boundaries",
            "Request and token boundaries",
        ),
        "summary.pipeline": ("summary.root", "Pipeline Stages"),
        "summary.decode_details": ("summary.root", "Decode Details"),
    }
    if "profiler" in track_by_key or any(
        track.key.startswith("native.") for track in plan.tracks
    ):
        required_groups["summary.native_details"] = (
            "summary.root",
            "Native Profiler Details",
        )
    if _REQUEST_RESOURCE_ROOT_KEY in track_by_key:
        required_groups[_REQUEST_RESOURCE_ROOT_KEY] = (
            "summary.root",
            _REQUEST_RESOURCE_ROOT_NAME,
        )
    for key, (parent, name) in required_groups.items():
        track = track_by_key.get(key)
        if track is None or track.parent_key != parent or track.name != name:
            mismatches.append(f"processing group {key!r} differs")

    forbidden_track_names = {
        "Request Summary",
        "Token & Throughput KPI",
        "Transfer KPI",
        "Data Quality",
        "Clock/alignment metadata",
    }
    if any(track.name in forbidden_track_names for track in plan.tracks):
        mismatches.append("timeline contains a removed summary or quality track")
    if any(
        track.key.startswith(_TIMELINE_SUMMARY_KPI_TRACK_PREFIX)
        or track.key == _TIMELINE_SUMMARY_DATA_QUALITY_KEY
        for track in plan.tracks
    ):
        mismatches.append("timeline contains a KPI or Data Quality track")
    if any(
        spec.name in {"Hybrid Request", "Request Summary"}
        for spec in plan.slices
    ):
        mismatches.append("timeline contains a duplicate request summary slice")
    if any(
        spec.track_key.startswith(_TIMELINE_SUMMARY_KPI_TRACK_PREFIX)
        for spec in plan.counters
    ):
        mismatches.append("timeline contains KPI counters")
    if any(
        spec.name in {"Data Quality status", "Clock/alignment metadata"}
        for spec in plan.instants
    ):
        mismatches.append("timeline contains explanatory metadata instants")

    duplicated_resource_samples = [
        spec
        for spec in plan.counters
        if spec.track_key.startswith(_TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX)
    ]
    if duplicated_resource_samples:
        mismatches.append(
            "processing timeline duplicates resource counter samples"
        )

    pipeline_keys = {
        "gpu_prefill",
        "kv_export",
        "kv_handoff",
        "kv_transfer_setup",
        "kv_transfer",
        "kv_transfer_wait",
        "kv_transform",
        "decode_schedule_wait",
        "npu_decode",
    }
    for key in pipeline_keys & set(track_by_key):
        if track_by_key[key].parent_key != "summary.pipeline":
            mismatches.append(f"pipeline track {key!r} is outside Pipeline Stages")
    for key in {"npu_decode_step", "sampling"} & set(track_by_key):
        if track_by_key[key].parent_key != "summary.decode_details":
            mismatches.append(f"decode detail track {key!r} is outside Decode Details")

    boundaries = [
        (dict(spec.annotations).get("hetero.boundary_kind"), spec)
        for spec in plan.instants
        if spec.track_key == "summary.boundaries.events"
    ]
    for required in ("request_received", "response_done"):
        if sum(kind == required for kind, _ in boundaries) != 1:
            mismatches.append(f"boundary instant {required!r} count differs")
    for _, spec in boundaries:
        annotations = dict(spec.annotations)
        if any(
            sensitive in key.casefold()
            for key in annotations
            for sensitive in ("prompt", "response_text", "token_text", "sha256", "path")
        ):
            mismatches.append(f"boundary instant {spec.name!r} exposes sensitive data")

    step_rows = [
        spec for spec in plan.slices if spec.track_key == "npu_decode_step"
    ]
    step_indices = [dict(spec.annotations).get("hetero.step_index") for spec in step_rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in step_indices):
        mismatches.append("decode step index is not a non-boolean integer")
    elif sorted(step_indices) != list(range(len(step_indices))):
        mismatches.append("decode step indices are duplicated or non-contiguous")
    decode_by_index = {
        dict(spec.annotations).get("hetero.step_index"): dict(spec.annotations).get(
            "hetero.correlation_id"
        )
        for spec in step_rows
    }
    for spec in (row for row in plan.slices if row.track_key == "sampling"):
        annotations = dict(spec.annotations)
        index = annotations.get("hetero.step_index")
        if decode_by_index.get(index) != annotations.get("hetero.correlation_id"):
            mismatches.append("sampling/decode correlation or step index differs")

    wait_rows = [
        spec for spec in plan.slices if spec.track_key == "kv_transfer_wait"
    ]
    if any(
        dict(spec.annotations).get("hetero.wait_observation")
        != "polling_incomplete_to_done"
        for spec in wait_rows
    ):
        mismatches.append("KV Transfer Wait lacks polling observation evidence")

    if plan.presentation_mode:
        if any(spec.track_key in {"request", "profiler"} for spec in plan.slices):
            mismatches.append("presentation trace contains request summary or capture envelope")
        if any(
            track.key.startswith(_RESOURCE_TELEMETRY_ROOT_KEY)
            for track in plan.tracks
        ):
            mismatches.append("presentation trace contains full-window telemetry")
        if plan.request_window is None:
            mismatches.append("presentation trace lacks a canonical client request window")
        elif (
            plan.request_window.start_ns < 0
            or plan.request_window.end_ns < plan.request_window.start_ns
            or plan.request_window.target_clock_domain_id
            != plan.canonical_clock_domain_id
        ):
            mismatches.append("presentation client request window is invalid")
        if plan.counters and _REQUEST_RESOURCE_ROOT_KEY not in track_by_key:
            mismatches.append("presentation resource counters lack their group")
        if any(
            not is_under(spec.track_key, _REQUEST_RESOURCE_ROOT_KEY)
            for spec in plan.counters
        ):
            mismatches.append("presentation counter is outside request resources")
        identities = [
            (spec.track_key, spec.timestamp_ns) for spec in plan.counters
        ]
        if len(identities) != len(set(identities)):
            mismatches.append("presentation resource sample is duplicated")
        if plan.request_window is not None:
            start = plan.request_window.start_ns
            end = plan.request_window.end_ns
            by_stream_role: dict[tuple[str, str], int] = {}
            for spec in plan.counters:
                role = spec.sample_role
                if role not in {"baseline", "background", "final"}:
                    mismatches.append("presentation resource sample role is invalid")
                    continue
                key = (spec.track_key, role)
                by_stream_role[key] = by_stream_role.get(key, 0) + 1
                if role == "baseline" and spec.timestamp_ns > start:
                    mismatches.append("presentation baseline follows request start")
                elif role == "final" and spec.timestamp_ns < end:
                    mismatches.append("presentation final precedes request end")
                elif role == "background":
                    interval = spec.interval_ns
                    if (
                        interval is None
                        or isinstance(interval, bool)
                        or interval < 0
                        or spec.timestamp_ns < start
                        or spec.timestamp_ns - interval >= end
                    ):
                        mismatches.append(
                            "presentation background interval misses request window"
                        )
            if any(
                count > 1
                for (track_key, role), count in by_stream_role.items()
                if role in {"baseline", "final"}
            ):
                mismatches.append(
                    "presentation stream has multiple baseline or final samples"
                )
    elif sum(spec.track_key == "request" for spec in plan.slices) != 1:
        mismatches.append("full trace must retain one diagnostic Request lifecycle")

    for gap in plan.unclassified_gaps:
        if (
            gap.end_timestamp_ns < gap.start_timestamp_ns
            or gap.duration_ns != gap.end_timestamp_ns - gap.start_timestamp_ns
            or not gap.preceding_marker
            or not gap.following_marker
            or not gap.reason
        ):
            mismatches.append("unclassified gap metadata is invalid")

    detail_sources: list[int] = []
    detail_destinations: list[int] = []
    for spec in plan.slices:
        detail_sources.extend(spec.begin_flow_ids)
        detail_sources.extend(spec.end_flow_ids)
        detail_destinations.extend(spec.begin_terminating_flow_ids)
        detail_destinations.extend(spec.end_terminating_flow_ids)
    declared_counts = Counter(flow.flow_id for flow in plan.flows)
    expected_counts = Counter({flow_id: 1 for flow_id in declared_counts})
    if (
        Counter(detail_sources) != expected_counts
        or Counter(detail_destinations) != expected_counts
    ):
        mismatches.append("detail flow endpoint counts were not preserved")
    return mismatches


def _validate_unavailable_kpi_annotations(
    annotations: Mapping[str, AnnotationValue],
    available_identities: set[str],
    mismatches: list[str],
) -> None:
    raw_count = annotations.get("hetero.unavailable_kpi_count")
    raw_json = annotations.get("hetero.unavailable_kpis_json")
    if (
        isinstance(raw_count, bool)
        or not isinstance(raw_count, int)
        or raw_count < 0
        or not isinstance(raw_json, str)
    ):
        mismatches.append("Data Quality unavailable KPI annotations are invalid")
        return
    try:
        unavailable = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        mismatches.append("Data Quality unavailable KPI JSON is invalid")
        return
    if (
        not isinstance(unavailable, dict)
        or len(unavailable) != raw_count
        or any(
            not isinstance(identity, str)
            or not identity
            or not isinstance(reason, str)
            or not reason
            for identity, reason in unavailable.items()
        )
    ):
        mismatches.append("Data Quality unavailable KPI inventory differs")
        return
    if available_identities.intersection(unavailable):
        mismatches.append(
            "unavailable KPI was also emitted as an available counter"
        )


def _flow_endpoint_summary(plan: TracePlan) -> dict[str, object]:
    sources: list[int] = []
    destinations: list[int] = []
    for spec in plan.slices:
        sources.extend(spec.begin_flow_ids)
        sources.extend(spec.end_flow_ids)
        destinations.extend(spec.begin_terminating_flow_ids)
        destinations.extend(spec.end_terminating_flow_ids)
    declared = [flow.flow_id for flow in plan.flows]
    source_counts = Counter(sources)
    destination_counts = Counter(destinations)
    declared_counts = Counter(declared)
    expected_counts = Counter({flow_id: 1 for flow_id in declared_counts})
    matched = (
        len(declared) == len(set(declared))
        and source_counts == expected_counts
        and destination_counts == expected_counts
    )
    return {
        "matched": matched,
        "declared_flow_ids": sorted(declared),
        "source_endpoint_ids": sorted(sources),
        "destination_endpoint_ids": sorted(destinations),
    }


__all__ = [
    "TraceValidationError",
    "VALIDATION_RECORD_TYPE",
    "validate_native_perfetto_trace",
    "validate_trace",
    "validate_trace_plan",
]
