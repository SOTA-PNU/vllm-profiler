"""Deterministic, evidence-aware comparison of Overview report dictionaries."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from perfetto_hetero_profiler.schema.constants import SCHEMA_VERSION
from perfetto_hetero_profiler.schema.catalog import METRIC_CATALOG


COMPARISON_RECORD_TYPE = "overview_comparison"
_AVAILABLE = "available"
_UNAVAILABLE = "not_available"
_CONCLUSION_WORD_RE = re.compile(r"\b(?:winner|fastest|best)\b", re.IGNORECASE)


class OverviewComparisonError(ValueError):
    """Raised when Overview reports cannot form a well-defined comparison."""


def _plain_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OverviewComparisonError(f"{field} must be an object")
    return value


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OverviewComparisonError(f"{field} must be a non-empty string")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OverviewComparisonError(
            "Overview reports must contain finite JSON values"
        ) from exc
    return text.encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sorted_objects(value: object, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OverviewComparisonError(f"{field} must be an array")
    objects = [
        _plain_mapping(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    return sorted(objects, key=_canonical_json)


def _run(report: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = report.get("run")
    if isinstance(nested, Mapping):
        return nested
    # Read the first internal Overview model as a compatibility aid. Generated
    # reports use the nested ``run`` contract.
    return {
        "run_id": report.get("run_id"),
        "mode": report.get("run_mode"),
        "profile_mode": report.get("profile_mode"),
        "status": report.get("run_status"),
        "profiler_kind": report.get("profile_kind"),
        "canonical_clock_domain_id": report.get("canonical_clock_domain_id"),
    }


def _models(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = report.get("models", report.get("model_identity", []))
    return _sorted_objects(value, field="models")


def _hardware(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = report.get("hardware", report.get("hardware_inventory", []))
    return _sorted_objects(value, field="hardware")


def _workload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _plain_mapping(report.get("workload", {}), field="workload")


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _request_count(report: Mapping[str, Any]) -> int:
    value = _workload(report).get("request_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OverviewComparisonError(
            "workload.request_count must be a non-negative integer"
        )
    return value


def _token_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    workload = _workload(report)
    return {
        key: workload.get(key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def _profile_kind(report: Mapping[str, Any]) -> str:
    value = _run(report).get("profiler_kind", "unknown")
    return str(value) if value is not None else "unknown"


def _profile_mode(report: Mapping[str, Any]) -> str:
    value = _run(report).get("profile_mode", "unknown")
    return str(value) if value is not None else "unknown"


def _run_mode(report: Mapping[str, Any]) -> str:
    value = _run(report).get("mode", "unknown")
    return str(value) if value is not None else "unknown"


def _clock_status(report: Mapping[str, Any]) -> str:
    quality = report.get("data_quality", {})
    if not isinstance(quality, Mapping):
        return "unknown"
    for key in (
        "canonical_clock_alignment",
        "clock_alignment",
        "alignment",
    ):
        candidate = quality.get(key)
        if isinstance(candidate, Mapping):
            status = candidate.get("status", candidate.get("alignment_status"))
            if isinstance(status, str) and status:
                return status
        elif isinstance(candidate, str) and candidate:
            return candidate
    for key in ("clock_alignment_status", "alignment_status"):
        value = quality.get(key)
        if isinstance(value, str) and value:
            return value
    perfetto = report.get("perfetto", {})
    if isinstance(perfetto, Mapping):
        value = perfetto.get("clock_alignment_status")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _canonical_clock(report: Mapping[str, Any]) -> str:
    value = _run(report).get("canonical_clock_domain_id")
    return value if isinstance(value, str) and value else "unknown"


def _validation_evidence(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        return None

    decisions: list[bool] = []
    if "valid" in value:
        decisions.append(value.get("valid") is True)
    if "fresh" in value:
        decisions.append(value.get("fresh") is True)
    if "matched" in value:
        decisions.append(value.get("matched") is True)
    if "source_match" in value:
        decisions.append(value.get("source_match") is True)
    if "source_run_match" in value:
        decisions.append(value.get("source_run_match") is True)
    if "source_fingerprint_match" in value:
        decisions.append(value.get("source_fingerprint_match") is True)
    for key in (
        "mismatch_count",
        "missing_count",
        "duplicate_count",
        "pairing_violation_count",
        "order_violation_count",
        "error_count",
    ):
        if key in value:
            raw = value.get(key)
            decisions.append(
                isinstance(raw, int) and not isinstance(raw, bool) and raw == 0
            )
    for key in ("mismatches", "errors"):
        if key in value:
            decisions.append(value.get(key) == [])
    status = value.get("status")
    if isinstance(status, str):
        decisions.append(
            status in {"succeeded", "valid", "matched", "fresh", "complete"}
        )
    if not decisions:
        return None
    return all(decisions)


def _source_integrity_valid(report: Mapping[str, Any]) -> bool:
    quality = report.get("data_quality", {})
    perfetto = report.get("perfetto", {})
    candidates: list[object] = []
    if isinstance(quality, Mapping):
        candidates.extend(
            quality.get(key)
            for key in (
                "source_artifact_validation",
                "source_integrity",
                "source_integrity_valid",
            )
            if key in quality
        )
        if "perfetto_sql_validation" in quality:
            candidates.append(quality.get("perfetto_sql_validation"))
        marker = quality.get("marker_validation")
        if isinstance(marker, Mapping):
            candidates.append(marker)
        candidates.extend(
            quality.get(key)
            for key in ("cleanup_complete", "per_sample_stream_preserved")
            if key in quality
        )
        if "run_status" in quality:
            candidates.append(
                quality.get("run_status") == _run(report).get("status")
                and quality.get("run_status") == "succeeded"
            )
        profiler = quality.get("profiler")
        if isinstance(profiler, Mapping) and "kind" in profiler:
            candidates.append(
                profiler.get("kind") == _run(report).get("profiler_kind")
            )
    if isinstance(perfetto, Mapping):
        if "valid" in perfetto:
            candidates.append(perfetto.get("valid"))
        candidates.extend(
            perfetto.get(key)
            for key in (
                "source_validation",
                "source_match",
                "trace_validation",
                "artifact_validation",
                "sql_validation",
                "validation",
            )
            if key in perfetto
        )
    decisions = [
        decision
        for decision in (_validation_evidence(item) for item in candidates)
        if decision is not None
    ]
    trace_sha256 = (
        quality.get("trace_sha256") if isinstance(quality, Mapping) else None
    )
    trace_identity_valid = (
        isinstance(trace_sha256, str)
        and len(trace_sha256) == 64
        and all(character in "0123456789abcdef" for character in trace_sha256)
    )
    return bool(decisions) and all(decisions) and trace_identity_valid


def _quality_warnings(report: Mapping[str, Any]) -> list[str]:
    warnings: set[str] = set()
    quality = report.get("data_quality", {})
    if isinstance(quality, Mapping):
        for key in ("warnings", "quality_warnings", "sample_limitations"):
            raw = quality.get(key, [])
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                warnings.update(
                    _neutralize_conclusion_wording(item) for item in raw
                )
    interpretation = report.get("interpretation", {})
    if isinstance(interpretation, Mapping):
        raw = interpretation.get("limitations", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            warnings.update(
                _neutralize_conclusion_wording(item) for item in raw
            )
    return sorted(warnings)


def _neutralize_conclusion_wording(value: object) -> str:
    """Keep source cautions while avoiding automated ranking vocabulary."""

    return _CONCLUSION_WORD_RE.sub("ranking conclusion", str(value))


def _native_alignment_is_limited(report: Mapping[str, Any]) -> bool:
    quality = report.get("data_quality", {})
    if isinstance(quality, Mapping):
        profiler = quality.get("profiler", {})
        if isinstance(profiler, Mapping):
            value = profiler.get("native_alignment_status")
            if value in {"partial", "unaligned", "unknown", "not_available"}:
                return True
    native = report.get("native_profiles", [])
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        return True
    for item in native:
        if not isinstance(item, Mapping):
            return True
        value = item.get("alignment_status", item.get("status"))
        if value in {"partial", "unaligned", "unknown", "not_available"}:
            return True
    return False


def _iter_kpis(
    report: Mapping[str, Any],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    root = report.get("kpis", {})
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes)):
        for item in root:
            if isinstance(item, Mapping):
                yield "uncategorized", item
        return
    if not isinstance(root, Mapping):
        raise OverviewComparisonError("kpis must be an object")
    for category in sorted(root):
        value = root[category]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if not isinstance(item, Mapping):
                    raise OverviewComparisonError(
                        f"kpis.{category} entries must be objects"
                    )
                yield str(category), item
        elif isinstance(value, Mapping) and "name" in value:
            yield str(category), value
        elif isinstance(value, Mapping):
            for name in sorted(value):
                item = value[name]
                if not isinstance(item, Mapping):
                    raise OverviewComparisonError(
                        f"kpis.{category}.{name} must be an object"
                    )
                if "name" not in item:
                    item = dict(item)
                    item["name"] = str(name)
                yield str(category), item
        else:
            raise OverviewComparisonError(
                f"kpis.{category} must contain KPI objects"
            )


def _metric_key(
    category: str, kpi: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    name = _non_empty_string(kpi.get("name"), field="KPI name")
    unit = _non_empty_string(
        kpi.get("canonical_unit"), field=f"{name}.canonical_unit"
    )
    definition = METRIC_CATALOG.get(name)
    if definition is None:
        raise OverviewComparisonError(f"{name} is not an official KPI")
    if unit != definition.unit:
        raise OverviewComparisonError(
            f"{name}.canonical_unit does not match the metric catalog"
        )
    scope = kpi.get("scope", {})
    layer = (
        str(scope.get("observation_layer", "unspecified"))
        if isinstance(scope, Mapping)
        else "unspecified"
    )
    return category, layer, name, unit


def _kpi_map(
    report: Mapping[str, Any],
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for category, kpi in _iter_kpis(report):
        key = _metric_key(category, kpi)
        if key in result:
            raise OverviewComparisonError(
                "a report contains duplicate KPI identity "
                f"{key[0]}/{key[1]}/{key[2]}/{key[3]}"
            )
        availability = kpi.get("availability")
        if availability not in {
            "available",
            "not_available",
            "not_collected",
            "error",
        }:
            raise OverviewComparisonError(
                f"{key[2]}.availability is invalid"
            )
        if availability == _AVAILABLE:
            if _finite_number(kpi.get("value")) is None:
                raise OverviewComparisonError(
                    f"available KPI {key[2]} must have a finite non-bool value"
                )
        elif kpi.get("value") is not None:
            raise OverviewComparisonError(
                f"unavailable KPI {key[2]} must have value=null"
            )
        result[key] = kpi
    return result


def _direction(name: str) -> str:
    if name.startswith("latency.") or name in {
        "transfer.duration",
        "transfer.transform_duration",
        "transfer.handoff_duration",
        "transfer.setup_duration",
        "transfer.wait_duration",
        "decode.schedule_wait_duration",
    }:
        return "lower_is_preferred"
    if name.startswith("throughput.") or name == "transfer.effective_bandwidth":
        return "higher_is_preferred"
    return "neutral"


def _comparison_value(
    run_id: str, kpi: Mapping[str, Any] | None
) -> dict[str, Any]:
    if kpi is None:
        return {
            "run_id": run_id,
            "availability": _UNAVAILABLE,
            "value": None,
            "unavailable_reason": "KPI is absent from this Overview report",
            "sample_count": 0,
        }
    availability = str(kpi.get("availability", _UNAVAILABLE))
    value = kpi.get("value") if availability == _AVAILABLE else None
    count = kpi.get("sample_count", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise OverviewComparisonError(
            f"{kpi.get('name', 'KPI')}.sample_count must be a non-negative integer"
        )
    reason = (
        None
        if availability == _AVAILABLE
        else (
            kpi.get("unavailable_reason")
            or "KPI is not available in this Overview report"
        )
    )
    if availability == _AVAILABLE and count == 0:
        raise OverviewComparisonError(
            f"{kpi.get('name', 'KPI')}.sample_count must be positive when available"
        )
    return {
        "run_id": run_id,
        "availability": availability,
        "value": value,
        "unavailable_reason": reason,
        "sample_count": count,
    }


def _unavailable_delta(reason: str) -> dict[str, Any]:
    return {
        "availability": _UNAVAILABLE,
        "value": None,
        "unavailable_reason": reason,
    }


def _delta(
    value: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    baseline_run_id: str,
) -> dict[str, Any]:
    run_id = str(value["run_id"])
    baseline_value = _finite_number(baseline.get("value"))
    current_value = _finite_number(value.get("value"))
    if baseline.get("availability") != _AVAILABLE or baseline_value is None:
        reason = "baseline KPI is not available"
        absolute = percentage = _unavailable_delta(reason)
    elif baseline_value == 0:
        reason = "baseline KPI is zero; deltas are undefined by policy"
        absolute = percentage = _unavailable_delta(reason)
    elif value.get("availability") != _AVAILABLE or current_value is None:
        reason = "run KPI is not available"
        absolute = percentage = _unavailable_delta(reason)
    else:
        difference = current_value - baseline_value
        absolute = {
            "availability": _AVAILABLE,
            "value": difference,
            "unavailable_reason": None,
        }
        percentage = {
            "availability": _AVAILABLE,
            "value": difference / baseline_value * 100.0,
            "unavailable_reason": None,
        }
    return {
        "run_id": run_id,
        "baseline_run_id": baseline_run_id,
        "absolute": absolute,
        "percentage": percentage,
    }


def _identity_reasons(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    reference = reports[0]
    reference_id = str(_run(reference)["run_id"])
    critical: set[str] = set()
    diagnostic: set[str] = set()

    reference_models = _sha256(_models(reference))
    reference_hardware = _sha256(_hardware(reference))
    reference_workload = _sha256(_workload(reference))
    reference_tokens = _sha256(_token_identity(reference))
    reference_count = _request_count(reference)
    reference_mode = _run_mode(reference)
    reference_clock = _canonical_clock(reference)
    reference_clock_status = _clock_status(reference)

    if not _source_integrity_valid(reference):
        critical.add(f"source integrity evidence is invalid for {reference_id}")
    for report in reports[1:]:
        run_id = str(_run(report)["run_id"])
        if _sha256(_models(report)) != reference_models:
            critical.add(f"model identity differs for {run_id}")
        if _sha256(_hardware(report)) != reference_hardware:
            critical.add(f"hardware identity differs for {run_id}")
        if _sha256(_workload(report)) != reference_workload:
            critical.add(f"workload configuration differs for {run_id}")
        if _sha256(_token_identity(report)) != reference_tokens:
            critical.add(f"input/output token counts differ for {run_id}")
        if _request_count(report) != reference_count:
            critical.add(f"request count differs for {run_id}")
        if _run_mode(report) != reference_mode:
            critical.add(f"run mode differs for {run_id}")
        if _canonical_clock(report) != reference_clock:
            critical.add(f"canonical clock domain differs for {run_id}")
        if _clock_status(report) != reference_clock_status:
            critical.add(f"canonical clock alignment differs for {run_id}")
        if not _source_integrity_valid(report):
            critical.add(f"source integrity evidence is invalid for {run_id}")

    if reference_clock == "unknown" or reference_clock_status in {
        "unknown",
        "unaligned",
        "invalid",
        "not_available",
        "partial",
    }:
        critical.add("canonical clock alignment evidence is incomplete")
    if reference_count <= 1:
        diagnostic.add(
            "request sample count is one; observations are not a repeated benchmark"
        )
    if len({_profile_mode(report) for report in reports}) > 1:
        diagnostic.add("profiler modes differ across runs")
    if len({_profile_kind(report) for report in reports}) > 1:
        diagnostic.add("profiler kinds differ across runs")
    if "unknown" in {_profile_kind(report) for report in reports}:
        diagnostic.add("one or more profiler kinds are unknown")
    if any(_native_alignment_is_limited(report) for report in reports):
        diagnostic.add(
            "one or more native profiler clocks are partial or unaligned"
        )

    availability_sets: list[set[tuple[tuple[str, str, str, str], str]]] = []
    for report in reports:
        availability_sets.append(
            {
                (key, str(kpi.get("availability", _UNAVAILABLE)))
                for key, kpi in _kpi_map(report).items()
            }
        )
    if any(items != availability_sets[0] for items in availability_sets[1:]):
        diagnostic.add("KPI availability differs across runs")
    return sorted(critical), sorted(diagnostic)


def _run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    run = _run(report)
    run_id = _non_empty_string(run.get("run_id"), field="run.run_id")
    return {
        "run_id": run_id,
        "run_mode": _run_mode(report),
        "profile_mode": _profile_mode(report),
        "profile_kind": _profile_kind(report),
        "overview_sha256": _sha256(report),
        "request_sample_count": _request_count(report),
        "model_identity_sha256": _sha256(_models(report)),
        "hardware_identity_sha256": _sha256(_hardware(report)),
        "workload_identity_sha256": _sha256(_workload(report)),
        "canonical_clock_domain_id": _canonical_clock(report),
        "clock_alignment_status": _clock_status(report),
        "source_integrity_valid": _source_integrity_valid(report),
        "quality_warnings": _quality_warnings(report),
    }


def _select_baseline(
    reports: Sequence[Mapping[str, Any]], baseline_run_id: str | None
) -> str:
    run_ids = [str(_run(report)["run_id"]) for report in reports]
    if baseline_run_id is not None:
        _non_empty_string(baseline_run_id, field="baseline_run_id")
        if baseline_run_id not in run_ids:
            raise OverviewComparisonError(
                f"baseline run {baseline_run_id!r} is not present"
            )
        return baseline_run_id
    controls = [
        str(_run(report)["run_id"])
        for report in reports
        if _profile_kind(report) == "control"
    ]
    if len(controls) != 1:
        raise OverviewComparisonError(
            "comparison requires an explicit baseline or exactly one control run"
        )
    return controls[0]


def build_comparison(
    reports: Sequence[Mapping[str, Any]],
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic comparison from validated plain Overview dicts.

    Input order never affects the result.  The selected baseline supplies delta
    denominators, while the first run in sorted order supplies only identity
    comparison reference values.
    """

    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise OverviewComparisonError("reports must be an array")
    if len(reports) < 2:
        raise OverviewComparisonError("at least two Overview reports are required")

    checked: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(reports):
        report = _plain_mapping(candidate, field=f"reports[{index}]")
        record_type = report.get("record_type")
        if record_type not in {None, "overview_report"}:
            raise OverviewComparisonError(
                f"reports[{index}] is not an overview_report"
            )
        run_id = _non_empty_string(
            _run(report).get("run_id"), field=f"reports[{index}].run.run_id"
        )
        if run_id in seen:
            raise OverviewComparisonError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        _canonical_json(report)
        _kpi_map(report)
        checked.append(report)
    checked.sort(key=lambda item: str(_run(item)["run_id"]))

    baseline = _select_baseline(checked, baseline_run_id)
    critical, diagnostic = _identity_reasons(checked)
    if critical:
        status = "not_comparable"
        reasons = sorted(set(critical + diagnostic))
    elif diagnostic:
        status = "diagnostic_only"
        reasons = sorted(set(diagnostic))
    else:
        status = "comparable"
        reasons = [
            "model, hardware, workload, request, clock, and integrity evidence match"
        ]

    run_maps = {
        str(_run(report)["run_id"]): _kpi_map(report) for report in checked
    }
    all_keys = sorted({key for mapping in run_maps.values() for key in mapping})
    metrics: list[dict[str, Any]] = []
    for category, layer, name, unit in all_keys:
        values = [
            _comparison_value(run_id, run_maps[run_id].get((category, layer, name, unit)))
            for run_id in sorted(run_maps)
        ]
        baseline_value = next(
            value for value in values if value["run_id"] == baseline
        )
        warnings: set[str] = set()
        for mapping in run_maps.values():
            kpi = mapping.get((category, layer, name, unit))
            if kpi is not None:
                raw = kpi.get("quality_warnings", [])
                if isinstance(raw, Sequence) and not isinstance(
                    raw, (str, bytes)
                ):
                    warnings.update(str(item) for item in raw)
        metrics.append(
            {
                "section": category,
                "observation_layer": layer,
                "name": name,
                "canonical_unit": unit,
                "direction": _direction(name),
                "values": values,
                "deltas": (
                    []
                    if status == "not_comparable"
                    else [
                        _delta(value, baseline_value, baseline_run_id=baseline)
                        for value in values
                        if value["run_id"] != baseline
                    ]
                ),
                "quality_warnings": sorted(warnings),
            }
        )

    limitations: set[str] = {
        (
            "This comparison reports observed captures and does not establish "
            "general hardware or configuration performance."
        )
    }
    for report in checked:
        interpretation = report.get("interpretation", {})
        if isinstance(interpretation, Mapping):
            raw = interpretation.get("limitations", [])
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                limitations.update(str(item) for item in raw)
    if status == "diagnostic_only":
        limitations.add(
            "Diagnostic-only results must not be presented as a general benchmark."
        )
    limitations = {
        _neutralize_conclusion_wording(item) for item in limitations
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": COMPARISON_RECORD_TYPE,
        "comparison": {
            "comparability": status,
            "comparability_reasons": reasons,
            "baseline_run_id": baseline,
        },
        "runs": [_run_summary(report) for report in checked],
        "metrics": metrics,
        "limitations": sorted(limitations),
    }
    # This final serialization check also rejects NaN/Infinity introduced by
    # delta arithmetic before the comparison reaches an output publisher.
    _canonical_json(result)
    return result


__all__ = [
    "COMPARISON_RECORD_TYPE",
    "OverviewComparisonError",
    "build_comparison",
]
