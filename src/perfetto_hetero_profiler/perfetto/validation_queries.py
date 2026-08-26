"""Trace Processor queries and validation display constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .compatibility import LEGACY_TIMELINE_MAPPING_VERSION
from .trace_attributes import TRACE_ATTRIBUTE_NAMESPACE


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

_LEGACY_MAPPING_VERSION: Final = LEGACY_TIMELINE_MAPPING_VERSION
_TIMELINE_SUMMARY_MAPPING_VERSION: Final = "processing-timeline-info-stats-v1"
_TIMELINE_SUMMARY_ROOT_KEY: Final = "summary.root"
_TIMELINE_SUMMARY_ROOT_NAME: Final = "Heterogeneous LLM Processing"
_TIMELINE_SUMMARY_TRACK_PREFIX: Final = "summary."
_TIMELINE_SUMMARY_KPI_TRACK_PREFIX: Final = "summary.kpi:"
_TIMELINE_SUMMARY_DATA_QUALITY_KEY: Final = "summary.data_quality"
_TIMELINE_SUMMARY_DATA_QUALITY_NAME: Final = "Data Quality status"
_TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX: Final = "telemetry.resources"
_RESOURCE_TELEMETRY_ROOT_KEY: Final = "telemetry.resources"
_RESOURCE_TELEMETRY_ROOT_NAME: Final = "Resource telemetry (full capture window)"
_REQUEST_RESOURCE_ROOT_KEY: Final = "summary.request_resources"
_REQUEST_RESOURCE_ROOT_NAME: Final = "Request-window Resource Telemetry"
_REPORT_ROW_QUERIES: Final = frozenset(
    {
        # Overview reconciliation consumes these two evidence tables. Other
        # potentially very large tables retain exact row count and SHA-256 but
        # do not duplicate every row into the JSON sidecar.
        "slices",
        "timeline_summary_slices",
    }
)

_TRACE_ATTRIBUTE_SQL: Final = f"""
SELECT name, key_type, int_value, str_value
FROM metadata
WHERE name GLOB 'trace_attribute.{TRACE_ATTRIBUTE_NAMESPACE}*'
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
  WHERE t.name = 'Heterogeneous LLM Processing'
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
  WHERE t.name = 'Heterogeneous LLM Processing'
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
  WHERE t.name = 'Heterogeneous LLM Processing'
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


@dataclass(frozen=True, slots=True)
class ValidationQuery:
    """One stable Trace Processor query identity and its SQL text."""

    name: str
    sql: str


BASE_VALIDATION_QUERIES: Final = (
    ValidationQuery("process", _PROCESS_SQL),
    ValidationQuery("tracks", _TRACK_SQL),
    ValidationQuery("slices", _SLICE_SQL),
    ValidationQuery("annotations", _ANNOTATION_SQL),
    ValidationQuery("step_annotations", _STEP_ANNOTATION_SQL),
    ValidationQuery("counters", _COUNTER_SQL),
    ValidationQuery("flows", _FLOW_SQL),
    ValidationQuery("dangling_flows", _DANGLING_FLOW_SQL),
    ValidationQuery("import_errors", _IMPORT_ERROR_SQL),
    ValidationQuery("native_policy", _NATIVE_POLICY_SQL),
)
NATIVE_VALIDATION_QUERIES: Final = (
    ValidationQuery("native_event_semantics", _NATIVE_EVENT_SEMANTICS_SQL),
)
TIMELINE_VALIDATION_QUERIES: Final = (
    ValidationQuery("timeline_summary_hierarchy", _TIMELINE_SUMMARY_HIERARCHY_SQL),
    ValidationQuery("timeline_summary_slices", _TIMELINE_SUMMARY_SLICE_SQL),
    ValidationQuery("timeline_summary_kpis", _TIMELINE_SUMMARY_KPI_SQL),
    ValidationQuery("timeline_summary_data_quality", _TIMELINE_SUMMARY_DATA_QUALITY_SQL),
    ValidationQuery("trace_attributes", _TRACE_ATTRIBUTE_SQL),
)


def _validate_query_registry() -> None:
    queries = (
        *BASE_VALIDATION_QUERIES,
        *NATIVE_VALIDATION_QUERIES,
        *TIMELINE_VALIDATION_QUERIES,
    )
    names = tuple(query.name for query in queries)
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Trace Processor validation query identity")
    if any(not query.sql.strip() for query in queries):
        raise RuntimeError("Trace Processor validation query SQL must be non-empty")


_validate_query_registry()


__all__ = [
    "ValidationQuery",
    "BASE_VALIDATION_QUERIES",
    "NATIVE_VALIDATION_QUERIES",
    "TIMELINE_VALIDATION_QUERIES",
    "_NATIVE_POLICY_KEYS",
    "_TP_NATIVE_POLICY_KEYS",
    "_PROCESS_SQL",
    "_TRACK_SQL",
    "_SLICE_SQL",
    "_ANNOTATION_COLUMNS",
    "_ANNOTATION_SQL",
    "_STEP_ANNOTATION_SQL",
    "_NATIVE_POLICY_KEY_SQL",
    "_NATIVE_POLICY_SQL",
    "_NATIVE_EVENT_SEMANTICS_SQL",
    "_COUNTER_SQL",
    "_FLOW_SQL",
    "_DANGLING_FLOW_SQL",
    "_IMPORT_ERROR_SQL",
    "_NATIVE_TRACE_SUMMARY_SQL",
    "_NATIVE_TRACE_FLOW_SQL",
    "_NATIVE_TRACE_PARENT_RANGE_SQL",
    "_NATIVE_TRACE_CATEGORY_SQL",
    "_LEGACY_MAPPING_VERSION",
    "_TIMELINE_SUMMARY_MAPPING_VERSION",
    "_TIMELINE_SUMMARY_ROOT_KEY",
    "_TIMELINE_SUMMARY_ROOT_NAME",
    "_TIMELINE_SUMMARY_TRACK_PREFIX",
    "_TIMELINE_SUMMARY_KPI_TRACK_PREFIX",
    "_TIMELINE_SUMMARY_DATA_QUALITY_KEY",
    "_TIMELINE_SUMMARY_DATA_QUALITY_NAME",
    "_TIMELINE_SUMMARY_RESOURCE_TRACK_PREFIX",
    "_RESOURCE_TELEMETRY_ROOT_KEY",
    "_RESOURCE_TELEMETRY_ROOT_NAME",
    "_REQUEST_RESOURCE_ROOT_KEY",
    "_REQUEST_RESOURCE_ROOT_NAME",
    "_REPORT_ROW_QUERIES",
    "_TRACE_ATTRIBUTE_SQL",
    "_TIMELINE_SUMMARY_HIERARCHY_SQL",
    "_TIMELINE_SUMMARY_SLICE_SQL",
    "_TIMELINE_SUMMARY_KPI_SQL",
    "_TIMELINE_SUMMARY_DATA_QUALITY_SQL"
]
