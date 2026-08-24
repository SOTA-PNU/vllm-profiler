"""Deterministic profiler experiment JSON/HTML report generation."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from .checkpoint import AttemptStatus, ExperimentCheckpoint
from .limitations import limitation_inventory
from .schedule import Condition, ExperimentSchedule, TrialKind, schedule_by_logical_id
from .statistics import OverheadDirection, paired_overhead, summarize_distribution


METRICS = (
    "latency.e2e",
    "latency.ttft",
    "latency.tpot",
    "throughput.requests",
    "throughput.output_tokens",
    "throughput.total_tokens",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _npu_resources(attempt_root: Path, attempt_id: str) -> dict[str, dict[str, object]]:
    path = attempt_root / "runs" / f"{attempt_id}-npu" / "metrics" / "metrics.jsonl"
    if not path.is_file():
        return {}
    samples: dict[str, list[float]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object: {path}:{number}")
        name = row.get("metric_name")
        if not isinstance(name, str) or not name.startswith("resource.npu."):
            continue
        if row.get("availability") != "available":
            continue
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"NPU resource metric is not numeric: {path}:{number}")
        sample = float(value)
        if not math.isfinite(sample):
            raise ValueError(f"NPU resource metric is not finite: {path}:{number}")
        samples.setdefault(name, []).append(sample)
    return {
        name: {
            "sample_count": len(values),
            "mean": math.fsum(values) / len(values),
            "maximum": max(values),
        }
        for name, values in sorted(samples.items())
    }


def build_report(
    *,
    root: Path,
    config: dict[str, object],
    schedule: ExperimentSchedule,
    checkpoint: ExperimentCheckpoint,
) -> dict[str, object]:
    trials = schedule_by_logical_id(schedule)
    successful: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for attempt in checkpoint.attempts:
        attempts.append(attempt.to_dict())
        if attempt.status is AttemptStatus.SUCCEEDED:
            attempt_root = root / "trials" / attempt.relative_directory
            validation = _json(attempt_root / "validation.json")
            validation = dict(validation)
            resources = dict(validation.get("resources", {}))
            resources.update(_npu_resources(attempt_root, attempt.attempt_id))
            validation["resources"] = resources
            successful[attempt.logical_trial_id] = validation
        elif attempt.status is AttemptStatus.FAILED:
            failures.append(attempt.to_dict())

    formal: dict[str, dict[int, dict[str, Any]]] = {condition.value: {} for condition in Condition}
    pilot: dict[str, dict[str, Any]] = {}
    for logical_id, validation in successful.items():
        trial = trials[logical_id]
        if trial.phase is TrialKind.PILOT:
            pilot[trial.condition.value] = validation
        else:
            formal[trial.condition.value][trial.round_index] = validation

    repeatability: dict[str, object] = {}
    for condition, rounds in formal.items():
        repeatability[condition] = {
            metric: summarize_distribution(
                [rounds[index]["metrics"][metric] for index in sorted(rounds)]
            ).to_dict()
            for metric in METRICS
        } if len(rounds) == 5 else {
            "availability": "not_available",
            "reason": f"formal condition has {len(rounds)} of 5 rounds",
        }

    def metric_rounds(condition: str, metric: str) -> dict[int, float]:
        return {
            index: float(value["metrics"][metric])
            for index, value in formal[condition].items()
        }

    comparisons = {
        "monitor_vs_reference": ("reference", "monitor"),
        "gpu_torch_vs_monitor": ("monitor", "gpu_torch"),
        "gpu_nsys_vs_monitor": ("monitor", "gpu_nsys"),
        "npu_torch_vs_monitor": ("monitor", "npu_torch"),
        "npu_rbln_vs_monitor": ("monitor", "npu_rbln"),
        "gpu_torch_vs_reference": ("reference", "gpu_torch"),
        "gpu_nsys_vs_reference": ("reference", "gpu_nsys"),
        "npu_torch_vs_reference": ("reference", "npu_torch"),
        "npu_rbln_vs_reference": ("reference", "npu_rbln"),
    }
    overhead: dict[str, object] = {}
    complete = all(len(rounds) == 5 for rounds in formal.values())
    if complete:
        for name, (baseline, observed) in comparisons.items():
            overhead[name] = {}
            for metric in METRICS:
                direction = (
                    OverheadDirection.THROUGHPUT_DEGRADATION
                    if metric.startswith("throughput.")
                    else OverheadDirection.INCREASE
                )
                overhead[name][metric] = paired_overhead(
                    metric_rounds(baseline, metric),
                    metric_rounds(observed, metric),
                    direction=direction,
                    expected_pair_count=5,
                ).to_dict()
    else:
        overhead = {"availability": "not_available", "reason": "formal trials are incomplete"}

    resource_summary: dict[str, object] = {}
    for condition, rounds in formal.items():
        names = sorted({name for value in rounds.values() for name in value.get("resources", {})})
        resource_summary[condition] = {
            name: {
                "trial_mean_distribution": summarize_distribution([
                    rounds[index]["resources"][name]["mean"]
                    for index in sorted(rounds)
                    if name in rounds[index].get("resources", {})
                ]).to_dict(),
                "trial_maximum_distribution": summarize_distribution([
                    rounds[index]["resources"][name]["maximum"]
                    for index in sorted(rounds)
                    if name in rounds[index].get("resources", {})
                ]).to_dict(),
            }
            for name in names
        }

    limitations = list(limitation_inventory())
    capability = {
        "monitor": "correlated requests and resource observation",
        "gpu_torch": "PyTorch/ATen and framework-level GPU prefill analysis",
        "gpu_nsys": "CUDA API, kernel, memcpy, and proven correlation analysis",
        "npu_torch": "host-side PyTorch/ATen work in the NPU decode server",
        "npu_rbln": "RBLN Neural Engine and DMA analysis in a native-relative trace",
    }
    comparison_for = {
        "monitor": "monitor_vs_reference",
        "gpu_torch": "gpu_torch_vs_monitor",
        "gpu_nsys": "gpu_nsys_vs_monitor",
        "npu_torch": "npu_torch_vs_monitor",
        "npu_rbln": "npu_rbln_vs_monitor",
    }
    recommendations = {
        condition: {
            "use_for": detail,
            "measured_e2e_overhead": (
                overhead[comparison_for[condition]]["latency.e2e"]
                if complete
                else {"availability": "not_available", "reason": "formal trials are incomplete"}
            ),
        }
        for condition, detail in capability.items()
    }
    return {
        "schema_version": "1.0",
        "report_type": "profiler_repeatability_overhead",
        "config": config,
        "schedule_sha256": schedule.sha256,
        "policy": {
            "pilot_in_formal_statistics": False,
            "outlier_exclusion": False,
            "latency_accuracy": "max(2ms,2%) for E2E/TTFT; max(1ms,5%) for TPOT",
            "paired_by": "formal_round_index",
        },
        "progress": {
            "logical_trials": len(schedule.trials),
            "successful_logical_trials": len(successful),
            "hardware_attempts": len(checkpoint.attempts),
            "failures": len(failures),
            "retries": sum(1 for item in checkpoint.attempts if item.attempt_number == 2),
        },
        "pilot": pilot,
        "formal_trials": formal,
        "formal_repeatability": repeatability,
        "paired_overhead": overhead,
        "resources": resource_summary,
        "attempts": attempts,
        "failures": failures,
        "recommendations": recommendations,
        "limitations": limitations,
    }


def render_report_html(report: dict[str, object]) -> bytes:
    progress = report["progress"]
    repeatability = report["formal_repeatability"]
    rows: list[str] = []
    for condition, metrics in repeatability.items():
        if not isinstance(metrics, dict) or "availability" in metrics:
            continue
        e2e = metrics["latency.e2e"]
        rows.append(
            "<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.4f}</td></tr>".format(
                html.escape(condition),
                e2e["mean"] / 1_000_000,
                e2e["median"] / 1_000_000,
                e2e["coefficient_of_variation"],
            )
        )
    payload = html.escape(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Hybrid Profiler Repeatability and Overhead Validation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem;overflow:auto}}</style></head>
<body><h1>Hybrid Profiler Repeatability and Overhead Validation</h1>
<p>Independent result dashboard. This is not a Perfetto built-in Overview or UI plugin.</p>
<p>Successful logical trials: {progress['successful_logical_trials']} / {progress['logical_trials']}; hardware attempts: {progress['hardware_attempts']}.</p>
<h2>Formal E2E repeatability</h2><table><thead><tr><th>Condition</th><th>Mean (ms)</th><th>Median (ms)</th><th>CV</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Complete machine-readable report</h2><pre>{payload}</pre></body></html>"""
    return document.encode("utf-8")


__all__ = ["METRICS", "build_report", "canonical_json", "render_report_html"]
