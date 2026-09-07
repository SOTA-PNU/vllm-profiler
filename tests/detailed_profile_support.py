"""Repository-only helpers for detailed-profile contract tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Literal

from perfetto_hetero_profiler.hybrid.detailed_profile import (
    DetailedProfileValidationError,
)
from perfetto_hetero_profiler.hybrid.layout import HybridRunLayout
from perfetto_hetero_profiler.schema import MetricSample, read_jsonl, write_jsonl
from perfetto_hetero_profiler.support.json_io import write_jsonl_exclusive


HybridProfileKind = Literal[
    "control",
    "gpu_torch",
    "gpu_nsys",
    "npu_vllm",
    "npu_rbln",
]
SUPPORTED_PROFILE_KINDS = frozenset(
    {"control", "gpu_torch", "gpu_nsys", "npu_vllm", "npu_rbln"}
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")


@dataclass(frozen=True, slots=True)
class HybridDetailedProfileConfig:
    run_root: Path
    run_id: str
    profile_kind: HybridProfileKind

    def __post_init__(self) -> None:
        if not self.run_root.is_absolute():
            raise ValueError("run_root must be absolute")
        if _RUN_ID_RE.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a safe single path component")
        if self.profile_kind not in SUPPORTED_PROFILE_KINDS:
            raise ValueError(f"unsupported profile_kind: {self.profile_kind!r}")
        bundle = HybridRunLayout(self.run_root, self.run_id).bundle
        if bundle.exists():
            raise FileExistsError(f"run output already exists: {bundle}")

    @property
    def output_roots(self) -> dict[str, Path]:
        layout = HybridRunLayout(self.run_root, self.run_id)
        return {
            "hybrid": layout.hybrid,
            "gpu": layout.gpu,
            "npu": layout.npu,
            "coordinator": layout.coordinator,
            "recovery": layout.recovery,
        }

    @property
    def detailed_target(self) -> str | None:
        if self.profile_kind.startswith("gpu_"):
            return "gpu"
        if self.profile_kind in {"npu_vllm", "npu_rbln"}:
            return "npu"
        return None


def select_profile_kind(enabled: Sequence[str]) -> HybridProfileKind:
    values = tuple(enabled)
    if len(values) != 1:
        raise ValueError("exactly one profile kind must be selected")
    value = values[0]
    if value not in SUPPORTED_PROFILE_KINDS:
        raise ValueError(f"unsupported profile kind: {value!r}")
    return value  # type: ignore[return-value]


def summarize_metrics(metrics: Sequence[MetricSample]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    total: dict[str, int] = defaultdict(int)
    unavailable: dict[str, int] = defaultdict(int)
    for metric in metrics:
        key = ":".join(
            part for part in (metric.metric_name, metric.device_id) if part
        )
        total[key] += 1
        if metric.value is None:
            unavailable[key] += 1
        else:
            grouped[key].append(float(metric.value))
    return {
        key: {
            "sample_count": total[key],
            "available_count": len(grouped[key]),
            "unavailable_count": unavailable[key],
            "average": statistics.fmean(grouped[key]) if grouped[key] else None,
            "peak": max(grouped[key]) if grouped[key] else None,
        }
        for key in sorted(total)
    }


def persist_per_sample_streams(
    *,
    gpu_root: Path,
    npu_root: Path,
    gpu_metrics: Sequence[MetricSample],
    npu_metrics: Sequence[MetricSample],
    system_metrics: Sequence[MetricSample],
    collector_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    streams = {
        "gpu": tuple(gpu_metrics),
        "npu": tuple(npu_metrics),
        "system": tuple(system_metrics),
    }
    for name, values in streams.items():
        if not values:
            raise DetailedProfileValidationError(
                f"{name} per-sample metric stream is empty"
            )
    if not collector_samples:
        raise DetailedProfileValidationError("raw collector sample stream is empty")
    timestamps = [int(item["monotonic_ns"]) for item in collector_samples]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise DetailedProfileValidationError(
            "collector sample timestamps must be strictly increasing"
        )

    paths = {
        "gpu": gpu_root / "raw/telemetry/gpu_metrics.jsonl",
        "system": gpu_root / "raw/telemetry/system_metrics.jsonl",
        "npu": npu_root / "raw/telemetry/npu_metrics.jsonl",
        "gpu_collector": gpu_root / "raw/telemetry/collector_samples.jsonl",
        "npu_collector": npu_root / "raw/telemetry/collector_samples.jsonl",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["gpu"], streams["gpu"])
    write_jsonl(paths["system"], streams["system"])
    write_jsonl(paths["npu"], streams["npu"])
    write_jsonl_exclusive(paths["gpu_collector"], collector_samples)
    write_jsonl_exclusive(paths["npu_collector"], collector_samples)

    persisted = {
        name: tuple(read_jsonl(paths[name])) for name in ("gpu", "npu", "system")
    }
    if persisted != streams:
        raise DetailedProfileValidationError(
            "persisted metric samples differ from in-memory samples"
        )
    persisted_gpu_samples = [
        json.loads(line)
        for line in paths["gpu_collector"].read_text(encoding="utf-8").splitlines()
    ]
    persisted_npu_samples = [
        json.loads(line)
        for line in paths["npu_collector"].read_text(encoding="utf-8").splitlines()
    ]
    if {
        name: summarize_metrics(values) for name, values in persisted.items()
    } != {
        name: summarize_metrics(values) for name, values in streams.items()
    }:
        raise DetailedProfileValidationError(
            "persisted metric aggregates differ from in-memory samples"
        )
    expected_samples = list(collector_samples)
    if (
        persisted_gpu_samples != expected_samples
        or persisted_npu_samples != expected_samples
    ):
        raise DetailedProfileValidationError(
            "persisted collector samples differ from in-memory samples"
        )

    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "metric_counts": {key: len(values) for key, values in streams.items()},
        "collector_sample_count": len(collector_samples),
        "actual_interval_ns": {
            "count": len(intervals),
            "average": statistics.fmean(intervals) if intervals else None,
            "minimum": min(intervals) if intervals else None,
            "maximum": max(intervals) if intervals else None,
        },
        "aggregates": {
            key: summarize_metrics(values) for key, values in streams.items()
        },
    }


def validate_owned_wrapper_child_leader(
    *,
    wrapper_pid: int,
    target_pid: int,
    target_pgid: int,
    parent_by_pid: Mapping[int, int],
) -> int:
    if min(wrapper_pid, target_pid, target_pgid) <= 0:
        raise DetailedProfileValidationError(
            "profiler wrapper and target identities must be positive"
        )
    if target_pid != target_pgid:
        raise DetailedProfileValidationError(
            "profiler target must lead its own process group"
        )
    current = target_pid
    visited: set[int] = set()
    while current != wrapper_pid:
        if current in visited:
            raise DetailedProfileValidationError(
                "profiler target ancestry contains a cycle"
            )
        visited.add(current)
        parent = parent_by_pid.get(current)
        if parent is None or parent <= 0:
            raise DetailedProfileValidationError(
                "profiler target is not an owned wrapper descendant"
            )
        current = parent
    return target_pgid


def validate_proxy_marker_stats(marker_root: Path) -> dict[str, Any]:
    marker_paths = sorted(marker_root.glob("runtime-markers-*.jsonl"))
    if not marker_paths:
        raise DetailedProfileValidationError("runtime marker JSONL is missing")
    matched: list[dict[str, Any]] = []
    for marker_path in marker_paths:
        stats_path = marker_path.with_suffix(".stats.json")
        if not stats_path.is_file():
            raise DetailedProfileValidationError(
                f"runtime marker stats are missing: {stats_path}"
            )
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DetailedProfileValidationError(
                f"invalid runtime marker stats: {stats_path}"
            ) from error
        records = len(marker_path.read_text(encoding="utf-8").splitlines())
        size = marker_path.stat().st_size
        if stats.get("records") != records or stats.get("bytes") != size:
            raise DetailedProfileValidationError(
                f"runtime marker stats mismatch: {stats_path}"
            )
        if int(stats.get("dropped", 0)) != 0:
            raise DetailedProfileValidationError(
                f"runtime marker stats report dropped records: {stats_path}"
            )
        if int(stats.get("duplicates", 0)) != 0:
            raise DetailedProfileValidationError(
                f"runtime marker stats report duplicate records: {stats_path}"
            )
        matched.append(
            {
                "marker_path": str(marker_path),
                "stats_path": str(stats_path),
                "records": records,
                "bytes": size,
                "average_write_ns": stats.get("average_write_ns"),
                "max_write_ns": stats.get("max_write_ns"),
                "dropped": stats.get("dropped"),
                "duplicates": stats.get("duplicates"),
            }
        )
    marker_stats = {path.with_suffix(".stats.json") for path in marker_paths}
    extra = []
    for stats_path in sorted(marker_root.glob("runtime-markers-*.stats.json")):
        if stats_path in marker_stats:
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("records") != 0:
            raise DetailedProfileValidationError(
                f"unmatched non-empty marker stats: {stats_path}"
            )
        extra.append(str(stats_path))
    return {
        "coverage": "complete",
        "marker_file_count": len(marker_paths),
        "matched_stats_count": len(matched),
        "matched": matched,
        "extra_zero_record_stats": extra,
        "records": sum(item["records"] for item in matched),
        "bytes": sum(item["bytes"] for item in matched),
        "dropped": sum(int(item["dropped"] or 0) for item in matched),
        "duplicates": sum(int(item["duplicates"] or 0) for item in matched),
    }


def compare_overhead(
    control: Mapping[str, int | float | None],
    profiled: Mapping[str, int | float | None],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(control) | set(profiled)):
        baseline = control.get(name)
        observed = profiled.get(name)
        numeric = (
            isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
            and math.isfinite(float(baseline))
            and isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
        )
        if not numeric:
            result[name] = {
                "control": baseline,
                "profiled": observed,
                "absolute_delta": None,
                "relative_delta": None,
                "reason": "control or profiled value is unavailable",
            }
            continue
        absolute = observed - baseline
        result[name] = {
            "control": baseline,
            "profiled": observed,
            "absolute_delta": absolute,
            "relative_delta": absolute / baseline if baseline != 0 else None,
            "reason": (
                None
                if baseline != 0
                else "relative delta is unavailable because control is zero"
            ),
        }
    return result
