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


VALIDATION_RECORD_TYPE: Final = "perfetto_trace_validation"

_NATIVE_POLICY_KEYS: Final = frozenset(
    {
        "hetero.alignment_method",
        "hetero.alignment_status",
        "hetero.canonical_clock_domain",
        "hetero.host_boundary_uncertainty_ns",
        "hetero.native_artifact_count",
        "hetero.native_clock_domain",
        "hetero.native_details_emitted",
        "hetero.native_event_alignment",
        "hetero.native_alignment_method",
        "hetero.native_alignment_uncertainty_ns",
        "hetero.native_profiler_alignment",
        "hetero.native_timestamp_unit",
        "hetero.profiler_type",
        "hetero.rbln_pb_classification",
        "hetero.rbln_pb_structure_analysis",
        "hetero.source_role",
        "hetero.timestamp_fallback_count",
        "hetero.fabricated_event_count",
        "hetero.unaligned_profiler_events",
    }
)
_TP_NATIVE_POLICY_KEYS: Final = frozenset(
    f"debug.{key.replace('.', '_')}" for key in _NATIVE_POLICY_KEYS
)

_PROCESS_SQL: Final = """
SELECT pid, name
FROM process
WHERE upid != 0
ORDER BY pid, name
""".strip()

_TRACK_SQL: Final = """
SELECT
  CAST(EXTRACT_ARG(t.source_arg_set_id, 'trace_id') AS INT) AS trace_uuid,
  t.name,
  t.type,
  CAST(EXTRACT_ARG(t.source_arg_set_id, 'description') AS TEXT) AS description,
  pct.unit,
  p.pid
FROM track AS t
LEFT JOIN process_track AS pt ON pt.id = t.id
LEFT JOIN process_counter_track AS pct ON pct.id = t.id
JOIN process AS p ON p.upid = COALESCE(pt.upid, pct.upid)
WHERE p.upid != 0
ORDER BY trace_uuid, t.name, t.type, pct.unit
""".strip()

_SLICE_SQL: Final = """
SELECT
  t.name AS track_name,
  s.name AS slice_name,
  s.ts,
  s.dur
FROM slice AS s
JOIN process_track AS t ON t.id = s.track_id
JOIN process AS p ON p.upid = t.upid
WHERE p.upid != 0
ORDER BY s.ts, s.dur, t.name, s.name
""".strip()

_ANNOTATION_COLUMNS: Final = """
  t.name AS track_name,
  s.name AS slice_name,
  s.ts,
  s.dur,
  a.key,
  a.value_type,
  a.int_value,
  a.real_value,
  a.string_value
""".strip()

_ANNOTATION_SQL: Final = f"""
SELECT
{_ANNOTATION_COLUMNS}
FROM slice AS s
JOIN process_track AS t ON t.id = s.track_id
JOIN process AS p ON p.upid = t.upid
JOIN args AS a ON a.arg_set_id = s.arg_set_id
WHERE p.upid != 0
  AND a.key GLOB 'debug.*'
ORDER BY s.ts, s.dur, t.name, s.name, a.key
""".strip()

_STEP_ANNOTATION_SQL: Final = f"""
SELECT
{_ANNOTATION_COLUMNS}
FROM slice AS s
JOIN process_track AS t ON t.id = s.track_id
JOIN process AS p ON p.upid = t.upid
JOIN args AS a ON a.arg_set_id = s.arg_set_id
WHERE p.upid != 0
  AND a.key = 'debug.hetero_step_index'
ORDER BY s.ts, s.dur, t.name, s.name, a.key
""".strip()

_NATIVE_POLICY_KEY_SQL: Final = ", ".join(
    f"'{key}'" for key in sorted(_TP_NATIVE_POLICY_KEYS)
)
_NATIVE_POLICY_SQL: Final = f"""
SELECT
{_ANNOTATION_COLUMNS}
FROM slice AS s
JOIN process_track AS t ON t.id = s.track_id
JOIN process AS p ON p.upid = t.upid
JOIN args AS a ON a.arg_set_id = s.arg_set_id
WHERE p.upid != 0
  AND a.key IN ({_NATIVE_POLICY_KEY_SQL})
ORDER BY s.ts, s.dur, t.name, s.name, a.key
""".strip()

_NATIVE_EVENT_SEMANTICS_SQL: Final = """
WITH native_events AS (
  SELECT s.arg_set_id
  FROM slice AS s
  WHERE EXTRACT_ARG(
    s.arg_set_id,
    'debug.hetero_native_profiler'
  ) IS NOT NULL
)
SELECT
  COUNT(*) AS event_count,
  COALESCE(SUM(
    CASE
      WHEN EXTRACT_ARG(
        arg_set_id,
        'debug.hetero_timestamp_fallback'
      ) = 0 THEN 0
      ELSE 1
    END
  ), 0) AS timestamp_fallback_violation_count,
  COALESCE(SUM(
    CASE
      WHEN EXTRACT_ARG(
        arg_set_id,
        'debug.hetero_fabricated_event'
      ) = 0 THEN 0
      ELSE 1
    END
  ), 0) AS fabricated_event_violation_count
FROM native_events
""".strip()

_COUNTER_SQL: Final = """
SELECT
  t.name AS track_name,
  t.unit,
  c.ts,
  c.value
FROM counter AS c
JOIN process_counter_track AS t ON t.id = c.track_id
JOIN process AS p ON p.upid = t.upid
WHERE p.upid != 0
ORDER BY c.ts, t.name, t.unit, c.value
""".strip()

_FLOW_SQL: Final = """
SELECT
  f.trace_id AS flow_id,
  source.name AS source_slice_name,
  destination.name AS destination_slice_name,
  CAST(
    EXTRACT_ARG(source.arg_set_id, 'debug.hetero_correlation_id')
    AS TEXT
  ) AS source_correlation_id,
  CAST(
    EXTRACT_ARG(destination.arg_set_id, 'debug.hetero_correlation_id')
    AS TEXT
  ) AS destination_correlation_id
FROM flow AS f
JOIN slice AS source ON source.id = f.slice_out
JOIN slice AS destination ON destination.id = f.slice_in
ORDER BY
  f.trace_id,
  source.name,
  destination.name,
  source_correlation_id,
  destination_correlation_id
""".strip()

_DANGLING_FLOW_SQL: Final = """
SELECT name, severity, value, description
FROM stats
WHERE value != 0
  AND (
    lower(name) LIKE '%flow%'
    OR lower(description) LIKE '%flow%'
  )
ORDER BY severity, name, value, description
""".strip()

_IMPORT_ERROR_SQL: Final = """
SELECT name, severity, value, description
FROM stats
WHERE value != 0
  AND severity != 'info'
ORDER BY severity, name, value, description
""".strip()

_NATIVE_TRACE_SUMMARY_SQL: Final = """
SELECT
  COUNT(*) AS slice_count,
  COUNT(DISTINCT track_id) AS track_count,
  COUNT(DISTINCT category) AS category_count,
  MIN(ts) AS min_timestamp_ns,
  MAX(ts + dur) AS max_end_ns,
  COALESCE(SUM(CASE WHEN dur < 0 THEN 1 ELSE 0 END), 0)
    AS invalid_duration_count,
  COALESCE(SUM(CASE WHEN ts < 0 OR ts + dur < ts THEN 1 ELSE 0 END), 0)
    AS invalid_timestamp_count
FROM slice
""".strip()

_NATIVE_TRACE_FLOW_SQL: Final = """
SELECT COUNT(*) AS flow_count
FROM flow
""".strip()

_NATIVE_TRACE_PARENT_RANGE_SQL: Final = """
SELECT COUNT(*) AS parent_child_range_violation_count
FROM slice AS child
JOIN slice AS parent ON parent.id = child.parent_id
WHERE child.ts < parent.ts
   OR child.ts + child.dur > parent.ts + parent.dur
""".strip()

_NATIVE_TRACE_CATEGORY_SQL: Final = """
SELECT category, COUNT(*) AS slice_count
FROM slice
GROUP BY category
ORDER BY category
""".strip()

_LEGACY_MAPPING_VERSION: Final = "legacy-unversioned-phase5-v1"
_TIMELINE_SUMMARY_MAPPING_VERSION: Final = "phase6b-timeline-summary-v2"
_TIMELINE_SUMMARY_ROOT_KEY: Final = "summary.root"
_TIMELINE_SUMMARY_ROOT_NAME: Final = "Heterogeneous LLM Summary"
_TIMELINE_SUMMARY_TRACK_PREFIX: Final = "summary."
_TIMELINE_SUMMARY_KPI_TRACK_PREFIX: Final = "summary.kpi:"
_TIMELINE_SUMMARY_DATA_QUALITY_KEY: Final = "summary.data_quality"
_TIMELINE_SUMMARY_DATA_QUALITY_NAME: Final = "Data Quality status"
_TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX: Final = "telemetry.resources"
_RESOURCE_TELEMETRY_ROOT_KEY: Final = "telemetry.resources"
_RESOURCE_TELEMETRY_ROOT_NAME: Final = "Resource telemetry (full capture window)"
_TRACE_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "hetero.alignment_method",
        "hetero.clock_status",
        "hetero.canonical_clock_domain",
        "hetero.models",
        "hetero.native_profiler_alignment",
        "hetero.profile_kind",
        "hetero.profile_mode",
        "hetero.run_id",
        "hetero.run_mode",
        "hetero.run_status",
        "hetero.source_identity_sha256",
        "hetero.trace_mapping_version",
    }
)
_REPORT_ROW_QUERIES: Final = frozenset(
    {
        # Overview reconciliation consumes these two evidence tables. Other
        # potentially very large tables retain exact row count and SHA-256 but
        # do not duplicate every row into the JSON sidecar.
        "slices",
        "timeline_summary_slices",
    }
)

_TRACE_ATTRIBUTE_SQL: Final = """
SELECT name, key_type, int_value, str_value
FROM metadata
WHERE name GLOB 'trace_attribute.hetero.*'
ORDER BY name, int_value, str_value
""".strip()

_TIMELINE_SUMMARY_HIERARCHY_SQL: Final = """
SELECT
  CAST(EXTRACT_ARG(t.source_arg_set_id, 'trace_id') AS INT) AS trace_uuid,
  t.name,
  t.type,
  CAST(
    EXTRACT_ARG(parent.source_arg_set_id, 'trace_id')
    AS INT
  ) AS parent_trace_uuid,
  parent.name AS parent_name,
  CAST(
    EXTRACT_ARG(t.source_arg_set_id, 'child_ordering')
    AS TEXT
  ) AS child_ordering,
  CAST(
    EXTRACT_ARG(t.source_arg_set_id, 'sibling_order_rank')
    AS INT
  ) AS sibling_order_rank
FROM track AS t
LEFT JOIN track AS parent ON parent.id = t.parent_id
WHERE t.type != 'process_track_event'
ORDER BY trace_uuid, t.name, t.type
""".strip()

_TIMELINE_SUMMARY_SLICE_SQL: Final = """
WITH RECURSIVE timeline_summary_tracks(track_id) AS (
  SELECT t.id
  FROM track AS t
  WHERE t.name = 'Heterogeneous LLM Summary'
    AND t.type = 'process_merged_track_event'
  UNION
  SELECT child.id
  FROM track AS child
  JOIN timeline_summary_tracks AS parent ON child.parent_id = parent.track_id
)
SELECT
  t.name AS track_name,
  parent.name AS parent_track_name,
  s.name AS slice_name,
  s.ts,
  s.dur
FROM slice AS s
JOIN track AS t ON t.id = s.track_id
LEFT JOIN track AS parent ON parent.id = t.parent_id
JOIN timeline_summary_tracks AS summary ON summary.track_id = t.id
ORDER BY s.ts, s.dur, t.name, s.name, parent.name
""".strip()

_TIMELINE_SUMMARY_KPI_SQL: Final = """
WITH RECURSIVE timeline_summary_tracks(track_id) AS (
  SELECT t.id
  FROM track AS t
  WHERE t.name = 'Heterogeneous LLM Summary'
    AND t.type = 'process_merged_track_event'
  UNION
  SELECT child.id
  FROM track AS child
  JOIN timeline_summary_tracks AS parent ON child.parent_id = parent.track_id
)
SELECT
  counter_track.name AS track_name,
  parent.name AS parent_track_name,
  CAST(
    EXTRACT_ARG(c.arg_set_id, 'debug.hetero_kpi_identity')
    AS TEXT
  ) AS kpi_identity,
  counter_track.unit,
  c.ts,
  c.value
FROM counter AS c
JOIN process_counter_track AS counter_track
  ON counter_track.id = c.track_id
JOIN track AS descriptor_track ON descriptor_track.id = c.track_id
LEFT JOIN track AS parent ON parent.id = descriptor_track.parent_id
JOIN timeline_summary_tracks AS summary
  ON summary.track_id = descriptor_track.id
WHERE EXTRACT_ARG(
  c.arg_set_id,
  'debug.hetero_kpi_identity'
) IS NOT NULL
ORDER BY
  c.ts,
  counter_track.name,
  counter_track.unit,
  c.value,
  parent.name,
  kpi_identity
""".strip()

_TIMELINE_SUMMARY_DATA_QUALITY_SQL: Final = """
WITH RECURSIVE timeline_summary_tracks(track_id) AS (
  SELECT t.id
  FROM track AS t
  WHERE t.name = 'Heterogeneous LLM Summary'
    AND t.type = 'process_merged_track_event'
  UNION
  SELECT child.id
  FROM track AS child
  JOIN timeline_summary_tracks AS parent ON child.parent_id = parent.track_id
)
SELECT
  t.name AS track_name,
  parent.name AS parent_track_name,
  s.name AS slice_name,
  s.ts,
  s.dur,
  a.key,
  a.value_type,
  a.int_value,
  a.real_value,
  a.string_value
FROM slice AS s
JOIN track AS t ON t.id = s.track_id
LEFT JOIN track AS parent ON parent.id = t.parent_id
JOIN timeline_summary_tracks AS summary ON summary.track_id = t.id
JOIN args AS a ON a.arg_set_id = s.arg_set_id
WHERE t.name = 'Data Quality'
  AND s.name = 'Data Quality status'
  AND a.key GLOB 'debug.*'
ORDER BY s.ts, s.dur, t.name, s.name, parent.name, a.key
""".strip()


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
            f"unsupported timeline summary mapping version {plan.mapping_version!r}"
        )

    track_by_key = plan.track_by_key
    timeline_summary_tracks = [
        track
        for track in plan.tracks
        if track.key.startswith(_TIMELINE_SUMMARY_TRACK_PREFIX)
    ]
    roots = [track for track in timeline_summary_tracks if track.parent_key is None]
    if len(roots) != 1 or roots[0].key != _TIMELINE_SUMMARY_ROOT_KEY:
        mismatches.append("timeline summary must contain exactly one summary.root")
    else:
        root = roots[0]
        if root.name != _TIMELINE_SUMMARY_ROOT_NAME:
            mismatches.append("timeline summary root track name differs")

    for track in timeline_summary_tracks:
        if track.key == _TIMELINE_SUMMARY_ROOT_KEY:
            continue
        current = track
        seen: set[str] = set()
        while current.parent_key is not None:
            if current.key in seen:
                mismatches.append(
                    f"timeline summary track {track.key!r} has a parent cycle"
                )
                break
            seen.add(current.key)
            parent = track_by_key.get(current.parent_key)
            if parent is None:
                mismatches.append(
                    f"timeline summary track {track.key!r} has an unknown parent"
                )
                break
            current = parent
        else:
            if current.key != _TIMELINE_SUMMARY_ROOT_KEY:
                mismatches.append(
                    f"timeline summary track {track.key!r} is outside summary.root"
                )

    summary_pairs = (
        ("summary.request_summary", "request", "Hybrid Request"),
        ("summary.pipeline.gpu_prefill", "gpu_prefill", "GPU Prefill"),
        ("summary.pipeline.kv_export", "kv_export", "KV Export"),
        ("summary.pipeline.kv_transfer", "kv_transfer", "KV Transfer"),
        ("summary.pipeline.kv_transform", "kv_transform", "KV Transform"),
        ("summary.pipeline.npu_decode", "npu_decode", "NPU Decode"),
    )
    allowed_summary_keys = {timeline_summary_key for timeline_summary_key, _, _ in summary_pairs}
    for timeline_summary_key, detail_key, expected_name in summary_pairs:
        timeline_summary_rows = sorted(
            (spec.timestamp_ns, spec.duration_ns)
            for spec in plan.slices
            if spec.track_key == timeline_summary_key
            and spec.name == expected_name
        )
        detail_rows = sorted(
            (spec.timestamp_ns, spec.duration_ns)
            for spec in plan.slices
            if spec.track_key == detail_key
        )
        if timeline_summary_rows != detail_rows:
            mismatches.append(
                f"{timeline_summary_key} timing does not exactly match {detail_key}"
            )

    timeline_summary_slices = [
        spec
        for spec in plan.slices
        if spec.track_key.startswith(_TIMELINE_SUMMARY_TRACK_PREFIX)
    ]
    for spec in timeline_summary_slices:
        if spec.track_key not in allowed_summary_keys:
            mismatches.append(
                f"timeline summary contains non-summary slice {spec.name!r}"
            )
        if (
            spec.begin_flow_ids
            or spec.end_flow_ids
            or spec.begin_terminating_flow_ids
            or spec.end_terminating_flow_ids
        ):
            mismatches.append(
                f"timeline summary slice {spec.name!r} duplicates detail flow"
            )

    kpi_counters = [
        spec
        for spec in plan.counters
        if spec.track_key.startswith(_TIMELINE_SUMMARY_KPI_TRACK_PREFIX)
    ]
    kpi_identities: set[str] = set()
    for spec in kpi_counters:
        annotations = dict(spec.annotations)
        identity = annotations.get("hetero.kpi_identity")
        expected_identity = spec.track_key.removeprefix(
            _TIMELINE_SUMMARY_KPI_TRACK_PREFIX
        )
        if identity != expected_identity:
            mismatches.append(
                f"KPI counter {spec.track_key!r} identity annotation differs"
            )
        elif isinstance(identity, str):
            kpi_identities.add(identity)
        if annotations.get("hetero.availability") != "available":
            mismatches.append(
                f"KPI counter {spec.track_key!r} is not explicitly available"
            )

    duplicated_resource_samples = [
        spec
        for spec in plan.counters
        if spec.track_key.startswith(_TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX)
    ]
    if duplicated_resource_samples:
        mismatches.append(
            "timeline summary duplicates resource counter samples inside "
            "Heterogeneous LLM Summary"
        )

    data_quality = [
        spec
        for spec in plan.instants
        if spec.track_key == _TIMELINE_SUMMARY_DATA_QUALITY_KEY
    ]
    if (
        len(data_quality) != 1
        or data_quality[0].name != _TIMELINE_SUMMARY_DATA_QUALITY_NAME
    ):
        mismatches.append(
            "timeline summary must contain exactly one Data Quality status instant"
        )
    else:
        annotations = dict(data_quality[0].annotations)
        if (
            annotations.get("hetero.trace_mapping_version")
            != plan.mapping_version
        ):
            mismatches.append("Data Quality mapping version differs")
        if (
            annotations.get("hetero.source_identity_sha256")
            != plan.source_identity_sha256
        ):
            mismatches.append("Data Quality source identity differs")
        _validate_unavailable_kpi_annotations(
            annotations,
            kpi_identities,
            mismatches,
        )

    detail_sources: list[int] = []
    detail_destinations: list[int] = []
    for spec in plan.slices:
        if spec.track_key.startswith(_TIMELINE_SUMMARY_TRACK_PREFIX):
            continue
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
