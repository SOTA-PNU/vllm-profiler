"""Fresh evaluation validation and independent client reconciliation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .accuracy import client_latency_accuracy, exact_count_accuracy, exact_marker_accuracy


class TrialValidationError(RuntimeError):
    pass


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrialValidationError(f"cannot read JSON {path}: {error}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TrialValidationError(f"{path}:{number} is not an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrialValidationError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrialValidationError(f"{field} must be finite")
    return result


def _mean(values: list[float], field: str) -> float:
    if not values:
        raise TrialValidationError(f"{field} is empty")
    return math.fsum(values) / len(values)


def _paths(attempt: Path, attempt_id: str) -> dict[str, Path]:
    run_root = attempt / "runs"
    return {
        "run_root": run_root,
        "hybrid": run_root / attempt_id,
        "gpu": run_root / f"{attempt_id}-gpu",
        "npu": run_root / f"{attempt_id}-npu",
        "coordinator": run_root / f"{attempt_id}-coordinator",
        "perfetto": run_root / f"{attempt_id}-perfetto",
        "focused": run_root / f"{attempt_id}-perfetto-request-focused",
        "overview": run_root / f"{attempt_id}-overview",
        "recovery": run_root / f"{attempt_id}-closeout-recovery",
        "publication": run_root / f"{attempt_id}-publication",
    }


def validate_trial(
    attempt: Path,
    *,
    attempt_id: str,
    condition: str,
    expected_requests: int = 10,
    expected_input_tokens: int = 5,
    expected_output_tokens: int = 8,
) -> dict[str, object]:
    roots = _paths(attempt, attempt_id)
    if any(path.is_symlink() for path in attempt.rglob("*")):
        raise TrialValidationError("trial contains a symlink")
    required = [
        roots["hybrid"] / "manifest.json",
        roots["gpu"] / "manifest.json",
        roots["npu"] / "manifest.json",
        roots["coordinator"] / "result.json",
        roots["coordinator"] / "requests.json",
        roots["coordinator"] / "cleanup.json",
        roots["coordinator"] / "source_fingerprint.json",
        roots["perfetto"] / "trace_validation.json",
        roots["focused"] / "trace.request-focused.validation.json",
        roots["overview"] / "overview_validation.json",
        roots["recovery"] / "artifact_manifest_validation.json",
        roots["publication"] / "determinism.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise TrialValidationError(f"required artifact missing: {missing[0]}")
    manifests = [_json(roots[name] / "manifest.json") for name in ("hybrid", "gpu", "npu")]
    if any(item.get("status") != "succeeded" for item in manifests):
        raise TrialValidationError("one or more source/hybrid manifests did not succeed")
    result = _json(roots["coordinator"] / "result.json")
    if result.get("status") != "succeeded":
        raise TrialValidationError("hybrid runner result did not succeed")
    requests = _json(roots["coordinator"] / "requests.json")
    raw_rows = _jsonl(roots["gpu"] / "raw/client/measured_requests.jsonl")
    metric_rows = _jsonl(roots["gpu"] / "metrics/metrics.jsonl")
    if requests.get("stores_prompt_or_generated_text") is not False:
        raise TrialValidationError("request content retention policy is invalid")
    if requests.get("clock") != "CLOCK_MONOTONIC_NS":
        raise TrialValidationError("independent client clock is not explicit")
    if len(raw_rows) != expected_requests:
        raise TrialValidationError("independent measured request count mismatch")

    per_request: dict[tuple[str, str], dict[str, Any]] = {}
    run_metrics: dict[str, dict[str, Any]] = {}
    resources: dict[str, list[float]] = {}
    for row in metric_rows:
        name = row.get("metric_name")
        if not isinstance(name, str):
            continue
        if row.get("availability") == "available":
            value = _finite(row.get("value"), name)
            if name.startswith("resource."):
                resources.setdefault(name, []).append(value)
            request_id = row.get("request_id")
            if isinstance(request_id, str):
                per_request[(request_id, name)] = row
            elif name.startswith("throughput."):
                run_metrics[name] = row

    checks: list[dict[str, object]] = []
    e2e_values: list[float] = []
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    request_ids: list[str] = []
    input_total = output_total = total_total = 0
    for index, row in enumerate(raw_rows):
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in request_ids:
            raise TrialValidationError("request IDs must be explicit and unique")
        request_ids.append(request_id)
        start = row.get("request_start_ns")
        end = row.get("stream_end_ns")
        token_times = row.get("valid_token_timestamps_ns")
        if (
            isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
            or not isinstance(token_times, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in token_times)
        ):
            raise TrialValidationError("independent client timestamps are malformed")
        if len(token_times) != expected_output_tokens:
            raise TrialValidationError("valid token timestamp count mismatch")
        reference = {
            "latency.e2e": end - start,
            "latency.ttft": token_times[0] - start,
            "latency.tpot": (token_times[-1] - token_times[0]) / (len(token_times) - 1),
        }
        for name, value in reference.items():
            metric = per_request.get((request_id, name))
            if metric is None:
                raise TrialValidationError(f"normalized metric missing: {request_id} {name}")
            check = client_latency_accuracy(
                name,
                reference_ns=value,
                observed_ns=_finite(metric.get("value"), name),
            )
            checks.append(check.to_dict())
            if not check.passed:
                raise TrialValidationError(f"accuracy tolerance failed: {request_id} {name}")
        e2e_values.append(float(reference["latency.e2e"]))
        ttft_values.append(float(reference["latency.ttft"]))
        tpot_values.append(float(reference["latency.tpot"]))
        for field, expected, metric_name in (
            ("input_tokens", expected_input_tokens, "request.input_tokens"),
            ("output_tokens", expected_output_tokens, "request.output_tokens"),
            ("total_tokens", expected_input_tokens + expected_output_tokens, "request.total_tokens"),
        ):
            actual = row.get(field)
            metric = per_request.get((request_id, metric_name))
            if metric is None:
                raise TrialValidationError(f"normalized token metric missing: {metric_name}")
            checks.append(exact_count_accuracy(field, reference=expected, observed=actual).to_dict())
            checks.append(exact_count_accuracy(metric_name, reference=actual, observed=metric.get("value")).to_dict())
        input_total += row["input_tokens"]
        output_total += row["output_tokens"]
        total_total += row["total_tokens"]

    summary = _json(roots["hybrid"] / "summary/hybrid_summary.json")
    joins = summary.get("joins")
    if not isinstance(joins, list) or len(joins) != expected_requests:
        raise TrialValidationError("marker pairing count mismatch")
    for join in joins:
        if (
            join.get("status") != "joined"
            or join.get("join_method") != "correlation_id"
            or join.get("missing_markers")
            or join.get("duplicate_markers")
            or join.get("ordering_violations")
            or join.get("pairing_issues")
        ):
            raise TrialValidationError("marker or correlation reconciliation failed")
    joined_ids = sorted(join.get("request_id") for join in joins)
    if joined_ids != sorted(request_ids):
        raise TrialValidationError("request/correlation ID set mismatch")

    source_events = _jsonl(roots["gpu"] / "events/events.jsonl") + _jsonl(
        roots["npu"] / "events/events.jsonl"
    )
    merged_events = _jsonl(roots["hybrid"] / "events/events.jsonl")
    merged_by_original = {
        (row.get("attributes") or {}).get("hybrid.original_event_id", row.get("event_id", "").split(":", 1)[-1]): row
        for row in merged_events
    }
    exact_marker_checks = 0
    for source in source_events:
        event_id = source.get("event_id")
        merged = merged_by_original.get(event_id)
        if merged is None:
            raise TrialValidationError(f"merged marker missing: {event_id}")
        checks.append(exact_marker_accuracy(
            "marker.timestamp_ns",
            reference_ns=source["timestamp_ns"],
            observed_ns=merged["timestamp_ns"],
        ).to_dict())
        if source.get("duration_ns") is not None:
            checks.append(exact_marker_accuracy(
                "marker.duration_ns",
                reference_ns=source["duration_ns"],
                observed_ns=merged["duration_ns"],
            ).to_dict())
        exact_marker_checks += 1

    for path in (
        roots["perfetto"] / "trace_validation.json",
        roots["focused"] / "trace.request-focused.validation.json",
        roots["overview"] / "overview_validation.json",
        roots["recovery"] / "artifact_manifest_validation.json",
    ):
        value = _json(path)
        if value.get("valid") is not True or value.get("mismatches"):
            raise TrialValidationError(f"fresh validation failed: {path}")
    fingerprints = _json(roots["coordinator"] / "source_fingerprint.json")
    if fingerprints.get("unchanged") is not True:
        raise TrialValidationError("source fingerprint changed")
    cleanup = _json(roots["coordinator"] / "cleanup.json")
    if any(item.get("killed") is True or item.get("terminated") is not True for item in cleanup.values()):
        raise TrialValidationError("process cleanup is incomplete")
    determinism = _json(roots["publication"] / "determinism.json")
    if any(
        determinism.get(name) is not True
        for name in (
            "perfetto_byte_identical",
            "request_focused_perfetto_byte_identical",
            "overview_byte_identical",
        )
    ):
        raise TrialValidationError("derived output is not deterministic")

    throughput = {
        name: _finite(run_metrics[name].get("value"), name)
        for name in (
            "throughput.requests",
            "throughput.input_tokens",
            "throughput.output_tokens",
            "throughput.total_tokens",
        )
    }
    resource_summary = {
        name: {
            "sample_count": len(values),
            "mean": _mean(values, name),
            "maximum": max(values),
        }
        for name, values in sorted(resources.items())
    }
    if condition == "reference" and resource_summary:
        raise TrialValidationError("reference condition unexpectedly collected resource telemetry")
    if condition != "reference" and not resource_summary:
        raise TrialValidationError("instrumented condition is missing resource telemetry")
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "condition": condition,
        "valid": True,
        "accuracy": {
            "method_id": "independent_streaming_client_v1",
            "checks": checks,
            "all_passed": all(item["passed"] for item in checks),
        },
        "exact_reconciliation": {
            "measured_requests": len(raw_rows),
            "successful_requests": len(raw_rows),
            "input_tokens": input_total,
            "output_tokens": output_total,
            "total_tokens": total_total,
            "request_ids": request_ids,
            "marker_pairings": len(joins),
            "marker_events": exact_marker_checks,
        },
        "metrics": {
            "latency.e2e": _mean(e2e_values, "latency.e2e"),
            "latency.ttft": _mean(ttft_values, "latency.ttft"),
            "latency.tpot": _mean(tpot_values, "latency.tpot"),
            **throughput,
        },
        "resources": resource_summary,
        "limitations": {
            "reference_runtime_markers_remain_enabled": condition == "reference",
        },
        "paths": {name: str(path) for name, path in roots.items()},
    }


__all__ = ["TrialValidationError", "validate_trial"]
