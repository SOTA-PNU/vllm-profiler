"""Deterministic semantic reconciliation reports for Overview outputs."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ..perfetto.loader import LoadedHybridRun
from .loader import (
    LoadedPerfettoBundle,
    phase_duration_reconciliation,
    reconciliation_summary,
)
from .publication import canonical_json_bytes
from .schema import (
    canonical_json_bytes as canonical_model_json_bytes,
    overview_report_from_dict,
)


OVERVIEW_VALIDATION_RECORD_TYPE = "overview_validation"


class OverviewValidationError(RuntimeError):
    """A generated semantic report failed reconciliation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pipeline_kpis(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    root = report.get("kpis")
    if not isinstance(root, Mapping):
        return {}
    values = root.get("pipeline_latency")
    if not isinstance(values, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in values:
        if isinstance(item, Mapping) and isinstance(item.get("name"), str):
            result[str(item["name"])] = item
    return result


def build_overview_validation(
    report: Mapping[str, Any],
    *,
    loaded: LoadedHybridRun,
    perfetto: LoadedPerfettoBundle,
    html_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate schema, KPI/event/Perfetto reconciliation, and static HTML."""

    model = overview_report_from_dict(dict(report))
    model_bytes = canonical_model_json_bytes(model)
    phase_rows = phase_duration_reconciliation(perfetto)
    pipeline = _pipeline_kpis(report)
    mismatches: list[str] = []
    reconciled_phases: list[dict[str, Any]] = []
    aggregate_request_count = next(
        (
            row["slice_count"]
            for row in phase_rows
            if row["kpi_name"] == "latency.e2e"
            and isinstance(row.get("slice_count"), int)
            and not isinstance(row.get("slice_count"), bool)
            and row["slice_count"] > 0
        ),
        None,
    )
    for row in phase_rows:
        current = dict(row)
        kpi = pipeline.get(str(row["kpi_name"]))
        current["overview_duration_ns"] = (
            kpi.get("value") if kpi is not None else None
        )
        aggregate = (
            kpi is not None
            and kpi.get("aggregation_method")
            == "arithmetic_mean_across_measured_requests_v1"
        )
        expected_overview_duration = row["event_duration_ns"]
        if aggregate:
            if aggregate_request_count is None:
                expected_overview_duration = None
            else:
                expected_overview_duration = (
                    row["event_duration_ns"] / aggregate_request_count
                )
        current["overview_expected_duration_ns"] = expected_overview_duration
        current["overview_aggregation_method"] = (
            kpi.get("aggregation_method") if kpi is not None else None
        )
        current["overview_matched"] = (
            current["matched"] is True
            and current["overview_duration_ns"]
            == expected_overview_duration
        )
        if current["overview_matched"] is not True:
            mismatches.append(
                f"{row['kpi_name']} differs across event, Perfetto, or Overview"
            )
        reconciled_phases.append(current)

    plan = perfetto.planning
    decode_steps = sum(
        item.track_key == "npu_decode_step" for item in plan.plan.slices
    )
    sampling_steps = sum(
        item.track_key == "sampling" for item in plan.plan.slices
    )
    counts = perfetto.fresh_trace_validation["counts"]
    expected_kpi_counters = plan.metadata.timeline_summary_kpi_counter_count
    expected_total_counters = (
        plan.metadata.available_resource_metric_count + expected_kpi_counters
    )
    expected_total_flows = len(plan.plan.flows)
    actual_kpi_counters = counts.get("timeline_summary_kpis", 0)
    actual_total_counters = counts.get("counters")
    actual_resource_counters = (
        actual_total_counters - actual_kpi_counters
        if isinstance(actual_total_counters, int)
        and not isinstance(actual_total_counters, bool)
        and isinstance(actual_kpi_counters, int)
        and not isinstance(actual_kpi_counters, bool)
        else None
    )
    if counts.get("step_annotations") != decode_steps + sampling_steps:
        mismatches.append("Perfetto decode/sampling step annotations differ")
    if actual_total_counters != expected_total_counters:
        mismatches.append("Perfetto total counter count differs")
    if actual_resource_counters != plan.metadata.available_resource_metric_count:
        mismatches.append("Perfetto resource counter count differs")
    if counts.get("flows") != expected_total_flows:
        mismatches.append("Perfetto flow count differs")
    if counts.get("dangling_flows") != 0:
        mismatches.append("Perfetto contains dangling flows")
    if counts.get("import_errors") != 0:
        mismatches.append("Perfetto contains import errors")
    if html_validation.get("valid") is not True:
        issues = html_validation.get("issues")
        if isinstance(issues, list):
            mismatches.extend(f"HTML: {item}" for item in issues)
        else:
            mismatches.append("HTML offline validation failed")

    source = {
        "valid": True,
        "closeout_manifest_sha256": loaded.closeout_manifest_sha256,
        "closeout_artifact_count": loaded.closeout_artifact_count,
        "roots": [
            item.metadata
            for item in sorted(
                loaded.root_fingerprints,
                key=lambda fingerprint: fingerprint.root_id,
            )
        ],
    }
    report_run = report.get("run")
    run_id = report_run.get("run_id") if isinstance(report_run, Mapping) else None
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": OVERVIEW_VALIDATION_RECORD_TYPE,
        "run_id": run_id,
        "valid": not mismatches,
        "schema_validation": {
            "valid": True,
            "schema_name": "overview_report.schema.json",
            "model_version": "1.0.0",
            "overview_sha256": _sha256(model_bytes),
        },
        "source_reconciliation": source,
        "perfetto_input_identity": perfetto.identity.metadata,
        "perfetto_reconciliation": reconciliation_summary(perfetto),
        "phase_duration_reconciliation": reconciled_phases,
        "step_reconciliation": {
            "decode_step_count": decode_steps,
            "sampling_step_count": sampling_steps,
            "perfetto_step_annotation_count": counts.get("step_annotations"),
            "matched": counts.get("step_annotations")
            == decode_steps + sampling_steps,
        },
        "resource_reconciliation": {
            "normalized_total": plan.metadata.resource_metric_count,
            "normalized_available": plan.metadata.available_resource_metric_count,
            "normalized_unavailable": (
                plan.metadata.unavailable_resource_metric_count
            ),
            "perfetto_resource_counter_count": actual_resource_counters,
            "perfetto_kpi_counter_count": actual_kpi_counters,
            "perfetto_total_counter_count": actual_total_counters,
            "expected_kpi_counter_count": expected_kpi_counters,
            "matched": (
                actual_resource_counters
                == plan.metadata.available_resource_metric_count
                and actual_kpi_counters == expected_kpi_counters
                and actual_total_counters == expected_total_counters
            ),
        },
        "flow_reconciliation": {
            "expected": expected_total_flows,
            "actual": counts.get("flows"),
            "dangling": counts.get("dangling_flows"),
            "matched": (
                counts.get("flows") == expected_total_flows
                and counts.get("dangling_flows") == 0
            ),
        },
        "numeric_policy": {
            "integer_marker_tolerance_ns": 0,
            "floating_reconciliation": "exact_same_integer_window_arithmetic",
            "relative_tolerance": 0,
            "absolute_tolerance": 0,
        },
        "html_validation": dict(html_validation),
        "publication_policy": {
            "semantic_payload_count": 3,
            "published_file_count": 5,
            "overwrite": False,
            "atomic_no_replace": True,
            "detached_manifest_self_reference": False,
        },
        "mismatches": sorted(set(mismatches)),
    }
    # Reject accidental non-finite/non-JSON values before publication.
    canonical_json_bytes(result)
    if result["valid"] is not True:
        raise OverviewValidationError(
            f"Overview validation found {len(result['mismatches'])} mismatch(es)"
        )
    return result


__all__ = [
    "OVERVIEW_VALIDATION_RECORD_TYPE",
    "OverviewValidationError",
    "build_overview_validation",
]
