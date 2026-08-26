"""Validation helpers for independent hybrid detailed-profiler smokes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Literal

from ..schema import (
    ClockDomain,
    ClockType,
    MetricSample,
    read_jsonl,
    validate_record,
    write_jsonl,
)
from ..support.files import sha256_file
from ..support.json_io import write_jsonl_exclusive


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


class DetailedProfileValidationError(RuntimeError):
    """A detailed-profile input or artifact violates the smoke contract."""


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
        existing = [path for path in self.output_roots.values() if path.exists()]
        if existing:
            raise FileExistsError(f"run output already exists: {existing[0]}")

    @property
    def output_roots(self) -> dict[str, Path]:
        return {
            "hybrid": self.run_root / self.run_id,
            "gpu": self.run_root / f"{self.run_id}-gpu",
            "npu": self.run_root / f"{self.run_id}-npu",
            "coordinator": self.run_root / f"{self.run_id}-coordinator",
            "recovery": self.run_root / f"{self.run_id}-closeout-recovery",
        }

    @property
    def detailed_target(self) -> str | None:
        if self.profile_kind.startswith("gpu_"):
            return "gpu"
        if self.profile_kind in {"npu_vllm", "npu_rbln"}:
            return "npu"
        return None


def select_profile_kind(enabled: Sequence[str]) -> HybridProfileKind:
    """Select exactly one supported mode and reject combined profilers."""

    values = tuple(enabled)
    if len(values) != 1:
        raise ValueError("exactly one profile kind must be selected")
    value = values[0]
    if value not in SUPPORTED_PROFILE_KINDS:
        raise ValueError(f"unsupported profile kind: {value!r}")
    return value  # type: ignore[return-value]


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _write_plain_jsonl_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl_exclusive(path, rows)


def summarize_metrics(metrics: Sequence[MetricSample]) -> dict[str, Any]:
    """Recompute availability and numeric aggregates from persisted samples."""

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
            "average": (
                statistics.fmean(grouped[key]) if grouped[key] else None
            ),
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
    """Persist normalized and raw per-sample streams before postprocessing."""

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
        raise DetailedProfileValidationError(
            "raw collector sample stream is empty"
        )
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
    _write_plain_jsonl_exclusive(paths["gpu_collector"], collector_samples)
    _write_plain_jsonl_exclusive(paths["npu_collector"], collector_samples)

    persisted = {
        name: tuple(read_jsonl(paths[name]))
        for name in ("gpu", "npu", "system")
    }
    if persisted != streams:
        raise DetailedProfileValidationError(
            "persisted metric samples differ from in-memory samples"
        )
    persisted_gpu_samples = [
        json.loads(line)
        for line in paths["gpu_collector"].read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    persisted_npu_samples = [
        json.loads(line)
        for line in paths["npu_collector"].read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if {
        name: summarize_metrics(values)
        for name, values in persisted.items()
    } != {
        name: summarize_metrics(values)
        for name, values in streams.items()
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

    intervals = [
        right - left for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "metric_counts": {
            key: len(values) for key, values in streams.items()
        },
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


def validate_fresh_artifacts(
    paths: Sequence[Path],
    *,
    capture_started_unix_ns: int,
    run_root: Path | None = None,
    expected_suffixes: Sequence[str] = (),
    preexisting_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Reject missing, linked, stale, or empty mandatory profiler artifacts."""

    if not paths:
        raise DetailedProfileValidationError("profiler artifact is missing")
    normalized_root = run_root.resolve() if run_root is not None else None
    normalized_preexisting = {
        path.resolve(strict=False) for path in preexisting_paths
    }
    normalized_paths = [path.resolve(strict=False) for path in paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise DetailedProfileValidationError(
            "profiler artifact list contains duplicate paths"
        )
    records: list[dict[str, Any]] = []
    for path, normalized_path in sorted(
        zip(paths, normalized_paths),
        key=lambda item: str(item[1]),
    ):
        if normalized_root is not None and not normalized_path.is_relative_to(
            normalized_root
        ):
            raise DetailedProfileValidationError(
                f"profiler artifact is outside the owned run root: {path}"
            )
        if normalized_path in normalized_preexisting:
            raise DetailedProfileValidationError(
                f"profiler artifact existed before capture: {path}"
            )
        if expected_suffixes and not any(
            str(path).endswith(suffix) for suffix in expected_suffixes
        ):
            raise DetailedProfileValidationError(
                f"profiler artifact has an unexpected suffix: {path}"
            )
        if not path.is_file():
            raise DetailedProfileValidationError(
                f"profiler artifact is missing: {path}"
            )
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise DetailedProfileValidationError(
                f"profiler artifact must be an owned regular file: {path}"
            )
        before = path.stat()
        if before.st_size <= 0:
            raise DetailedProfileValidationError(
                f"profiler artifact is empty: {path}"
            )
        if before.st_mtime_ns < capture_started_unix_ns:
            raise DetailedProfileValidationError(
                f"profiler artifact is stale: {path}"
            )
        digest = _sha256_file(path)
        after = path.stat()
        if (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DetailedProfileValidationError(
                f"profiler artifact changed while hashing: {path}"
            )
        records.append(
            {
                "path": (
                    str(normalized_path.relative_to(normalized_root))
                    if normalized_root is not None
                    else str(path)
                ),
                "size_bytes": before.st_size,
                "sha256": digest,
                "mtime_ns": before.st_mtime_ns,
                "inode": before.st_ino,
            }
        )
    return records


def _read_trace(path: Path) -> tuple[dict[str, Any], str]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            text = stream.read()
        value = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DetailedProfileValidationError(
            f"malformed torch trace {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DetailedProfileValidationError(
            f"torch trace root must be an object: {path}"
        )
    return value, text


def validate_torch_traces(
    paths: Sequence[Path],
    *,
    target: Literal["gpu", "npu"],
    capture_started_unix_ns: int,
    forbidden_text: Sequence[str] = (),
    run_root: Path | None = None,
    preexisting_paths: Sequence[Path] = (),
    capture_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate Chrome trace structure and target-appropriate event coverage."""

    if target not in {"gpu", "npu"}:
        raise DetailedProfileValidationError(
            f"unsupported torch profiler target: {target!r}"
        )
    file_records = validate_fresh_artifacts(
        paths,
        capture_started_unix_ns=capture_started_unix_ns,
        run_root=run_root,
        expected_suffixes=(".pt.trace.json", ".pt.trace.json.gz"),
        preexisting_paths=preexisting_paths,
    )
    total_events = 0
    activity_events = 0
    cpu_events = 0
    cuda_runtime_events = 0
    cuda_kernel_events = 0
    cuda_memory_events = 0
    native_min: float | None = None
    native_max: float | None = None
    base_times: set[int] = set()
    sensitive_matches: list[str] = []
    names: set[str] = set()
    display_units: set[str] = set()
    for path in paths:
        value, text = _read_trace(path)
        events = value.get("traceEvents")
        if not isinstance(events, list) or not events:
            raise DetailedProfileValidationError(
                f"torch trace has no traceEvents: {path}"
            )
        base_time = value.get("baseTimeNanoseconds")
        if isinstance(base_time, int) and not isinstance(base_time, bool):
            base_times.add(base_time)
        display_time_unit = value.get("displayTimeUnit")
        if isinstance(display_time_unit, str):
            display_units.add(display_time_unit)
        for forbidden in forbidden_text:
            if forbidden and forbidden in text:
                sensitive_matches.append(
                    hashlib.sha256(forbidden.encode("utf-8")).hexdigest()
                )
        for event in events:
            if not isinstance(event, dict):
                continue
            total_events += 1
            name = str(event.get("name", ""))
            category = str(event.get("cat", ""))
            names.add(name)
            timestamp = event.get("ts")
            duration = event.get("dur", 0)
            timestamp_is_finite = (
                isinstance(timestamp, (int, float))
                and not isinstance(timestamp, bool)
                and math.isfinite(float(timestamp))
            )
            duration_is_finite = (
                isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and math.isfinite(float(duration))
                and float(duration) >= 0
            )
            is_activity = (
                event.get("ph") == "X"
                and timestamp_is_finite
                and duration_is_finite
            )
            if is_activity:
                activity_events += 1
                if category == "cpu_op":
                    cpu_events += 1
                elif category in {"cuda_runtime", "cuda_driver"}:
                    cuda_runtime_events += 1
                elif category == "kernel":
                    cuda_kernel_events += 1
                elif category in {"gpu_memcpy", "gpu_memset"}:
                    cuda_memory_events += 1
            if timestamp_is_finite:
                native_min = (
                    float(timestamp)
                    if native_min is None
                    else min(native_min, float(timestamp))
                )
                event_end = float(timestamp) + (
                    float(duration) if duration_is_finite else 0.0
                )
                native_max = (
                    event_end
                    if native_max is None
                    else max(native_max, event_end)
                )
    if sensitive_matches:
        raise DetailedProfileValidationError(
            "sensitive payload found in torch trace "
            f"(match_count={len(sensitive_matches)}, hashes={sensitive_matches})"
        )
    if len(base_times) > 1:
        raise DetailedProfileValidationError(
            "torch traces contain inconsistent baseTimeNanoseconds values"
        )
    if cpu_events == 0:
        raise DetailedProfileValidationError(
            "torch trace has no timestamped CPU operation events"
        )
    if target == "gpu" and (
        cuda_runtime_events == 0 or cuda_kernel_events == 0
    ):
        raise DetailedProfileValidationError(
            "GPU torch trace lacks CUDA runtime or kernel activity"
        )
    measured_scope = _validate_capture_boundary(capture_boundary)
    return {
        "files": file_records,
        "trace_count": len(paths),
        "event_count": total_events,
        "activity_event_count": activity_events,
        "cpu_event_count": cpu_events,
        "cuda_event_count": (
            cuda_runtime_events + cuda_kernel_events + cuda_memory_events
        ),
        "cuda_runtime_event_count": cuda_runtime_events,
        "cuda_kernel_event_count": cuda_kernel_events,
        "cuda_memory_event_count": cuda_memory_events,
        "event_names": sorted(names),
        "native_timestamp_unit": "chrome_trace_microseconds",
        "native_timestamp_min": native_min,
        "native_timestamp_max": native_max,
        "base_time_nanoseconds": sorted(base_times),
        "display_time_units": sorted(display_units),
        "measured_scope": measured_scope,
        "sensitive_matches": [],
    }


def _validate_capture_boundary(
    boundary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if boundary is None:
        raise DetailedProfileValidationError(
            "capture API boundary evidence was not supplied"
        )
    required = (
        "start_before_monotonic_ns",
        "start_after_monotonic_ns",
        "request_start_monotonic_ns",
        "request_end_monotonic_ns",
        "stop_before_monotonic_ns",
        "stop_after_monotonic_ns",
        "start_http_status",
        "stop_http_status",
    )
    missing = [name for name in required if name not in boundary]
    if missing:
        raise DetailedProfileValidationError(
            f"capture boundary is missing fields: {missing}"
        )
    points = [boundary[name] for name in required[:6]]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in points
    ):
        raise DetailedProfileValidationError(
            "capture boundary timestamps must be non-negative integers"
        )
    if points != sorted(points):
        raise DetailedProfileValidationError(
            "capture boundary does not bracket the measured request"
        )
    if boundary["start_http_status"] != 200 or boundary["stop_http_status"] != 200:
        raise DetailedProfileValidationError(
            "profiler start/stop API did not both return HTTP 200"
        )
    return {
        "status": "bracketed",
        "method": "vllm_start_stop_api_boundary",
        "start_api_rtt_ns": points[1] - points[0],
        "stop_api_rtt_ns": points[5] - points[4],
        "valid_interval_monotonic_ns": [points[0], points[5]],
    }


def validate_nsys_report(
    report_path: Path,
    *,
    capture_started_unix_ns: int,
    run_root: Path,
    preexisting_paths: Sequence[Path],
    stats: Mapping[str, Mapping[str, Any]],
    capture_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a fresh Nsight report and official summary report evidence."""

    files = validate_fresh_artifacts(
        (report_path,),
        capture_started_unix_ns=capture_started_unix_ns,
        run_root=run_root,
        expected_suffixes=(".nsys-rep",),
        preexisting_paths=preexisting_paths,
    )
    required = ("cuda_api_sum", "cuda_gpu_kern_sum", "osrt_sum")
    evidence: dict[str, Any] = {}
    for name in (*required, "nvtx_sum"):
        result = stats.get(name)
        if result is None:
            if name in required:
                raise DetailedProfileValidationError(
                    f"nsys stats report is missing: {name}"
                )
            continue
        returncode = result.get("returncode")
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        skipped = "SKIPPED:" in stdout or "SKIPPED:" in stderr
        data_row_count = _nsys_csv_data_row_count(stdout)
        has_rows = data_row_count > 0
        if name in required and (
            returncode != 0 or skipped or not has_rows
        ):
            raise DetailedProfileValidationError(
                f"required nsys stats evidence is unavailable: {name}"
            )
        evidence[name] = {
            "returncode": returncode,
            "available": returncode == 0 and not skipped and has_rows,
            "skipped": skipped,
            "data_row_count": data_row_count,
            "stdout_sha256": hashlib.sha256(
                stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                stderr.encode("utf-8")
            ).hexdigest(),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
        }
    return {
        "files": files,
        "official_reader": "nsys stats",
        "reports": evidence,
        "capture_scope": _validate_capture_boundary(capture_boundary),
        "native_clock_domain": "nsight-systems-native",
        "native_timestamp_unit": "nsight-report-native",
    }


def validate_rbln_reports(
    paths: Sequence[Path],
    *,
    capture_started_unix_ns: int,
    run_root: Path,
    preexisting_paths: Sequence[Path],
    strings_results: Mapping[str, Mapping[str, Any]],
    capture_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fresh RBLN reports and bounded host/device timing evidence."""

    files = validate_fresh_artifacts(
        paths,
        capture_started_unix_ns=capture_started_unix_ns,
        run_root=run_root,
        expected_suffixes=(".pb",),
        preexisting_paths=preexisting_paths,
    )
    device_categories = (
        "Device",
        "Neural Engine Clusters",
        "Task DMA",
        "uDMA",
    )
    evidence: list[dict[str, Any]] = []
    host_timing_present = False
    device_timing_present = False
    for path in paths:
        result = strings_results.get(str(path))
        if result is None or result.get("returncode") != 0:
            raise DetailedProfileValidationError(
                f"RBLN report reader failed: {path}"
            )
        stdout = str(result.get("stdout", ""))
        matched = [
            category for category in device_categories if category in stdout
        ]
        has_host = bool(re.search(r"\bHost\b", stdout))
        has_device = bool(matched)
        host_timing_present |= has_host
        device_timing_present |= has_device
        evidence.append(
            {
                "path": str(path.relative_to(run_root)),
                "reader": "strings",
                "returncode": result["returncode"],
                "stdout_sha256": hashlib.sha256(
                    stdout.encode("utf-8")
                ).hexdigest(),
                "stdout_bytes": len(stdout.encode("utf-8")),
                "host_timing_present": has_host,
                "device_timing_present": has_device,
                "device_categories": matched,
            }
        )
    if not host_timing_present or not device_timing_present:
        raise DetailedProfileValidationError(
            "RBLN reports lack required host/device timing evidence"
        )
    return {
        "files": files,
        "format": "perfetto_trace_protobuf",
        "official_capture_api": "rebel.profiler.profile",
        "bounded_reader": "strings",
        "structural_parse": "deferred_to_official_trace_processor",
        "structural_parse_reason": (
            "capture closeout validates freshness and bounded timing evidence; "
            "native-detail conversion performs official Trace Processor validation"
        ),
        "reader_evidence": evidence,
        "host_timing_present": host_timing_present,
        "device_timing_present": device_timing_present,
        "capture_scope": _validate_capture_boundary(capture_boundary),
        "native_clock_domain": "rbln-profiler-native",
        "native_timestamp_unit": "rbln_report_native",
    }


def _nsys_csv_data_row_count(stdout: str) -> int:
    lines = [
        line
        for line in stdout.splitlines()
        if line.strip()
        and not line.startswith(("Processing [", "Generating ", "SKIPPED:"))
    ]
    rows = list(csv.reader(lines))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(name in row for name in ("Name", "Range"))
            and any(name in row for name in ("Num Calls", "Instances"))
        ),
        None,
    )
    if header_index is None:
        return 0
    header = rows[header_index]
    count_name = next(
        (name for name in ("Num Calls", "Instances") if name in header),
        None,
    )
    if count_name is None:
        return 0
    calls_index = header.index(count_name)
    identity_name = next(
        (name for name in ("Name", "Range") if name in header),
        None,
    )
    if identity_name is None:
        return 0
    name_index = header.index(identity_name)
    count = 0
    for row in rows[header_index + 1 :]:
        if len(row) <= max(calls_index, name_index) or not row[name_index].strip():
            continue
        try:
            calls = int(row[calls_index])
        except ValueError:
            continue
        if calls > 0:
            count += 1
    return count


def build_profiler_alignment(
    *,
    profiler_type: str,
    native_clock_domain: str,
    native_timestamp_unit: str,
    canonical_clock_domain: str,
    anchors: Sequence[Mapping[str, Any]],
    native_capture_start: int | float | None,
    native_capture_end: int | float | None,
    same_clock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe only evidence-backed alignment; preserve unaligned native clocks."""

    if native_capture_start is not None and (
        not isinstance(native_capture_start, (int, float))
        or isinstance(native_capture_start, bool)
        or not math.isfinite(float(native_capture_start))
    ):
        raise DetailedProfileValidationError(
            "native capture start must be finite when available"
        )
    if native_capture_end is not None and (
        not isinstance(native_capture_end, (int, float))
        or isinstance(native_capture_end, bool)
        or not math.isfinite(float(native_capture_end))
    ):
        raise DetailedProfileValidationError(
            "native capture end must be finite when available"
        )
    if (
        native_capture_start is not None
        and native_capture_end is not None
        and native_capture_end < native_capture_start
    ):
        raise DetailedProfileValidationError(
            "native capture interval is reversed"
        )
    capture_boundary = _validate_alignment_anchors(anchors)
    if same_clock_evidence and native_clock_domain != canonical_clock_domain:
        raise DetailedProfileValidationError(
            "different profiler and canonical clocks cannot use same-clock alignment"
        )
    if same_clock_evidence:
        if (
            same_clock_evidence.get("method") != "clock_descriptor_identity"
            or same_clock_evidence.get("clock_domain_id")
            != canonical_clock_domain
            or canonical_clock_domain != "host-monotonic"
        ):
            raise DetailedProfileValidationError(
                "same-clock alignment lacks canonical clock descriptor evidence"
            )
        status = "aligned"
        method = "same_clock_domain"
        offset_ns: int | None = 0
        uncertainty_ns: int | None = 0
        reason = None
    else:
        status = "partial"
        method = "host_api_boundary_bracket"
        offset_ns = None
        uncertainty_ns = None
        reason = (
            "host start/stop boundaries bracket capture, but the profiler "
            "native clock has no proven direct transform to CLOCK_MONOTONIC"
        )
    return {
        "profiler_type": profiler_type,
        "native_clock_domain": native_clock_domain,
        "native_timestamp_unit": native_timestamp_unit,
        "canonical_clock_domain": canonical_clock_domain,
        "alignment_status": status,
        "alignment_method": method,
        "offset_ns": offset_ns,
        "uncertainty_ns": uncertainty_ns,
        "anchors": [dict(item) for item in anchors],
        "anchor_count": len(anchors),
        "native_capture_start": native_capture_start,
        "native_capture_end": native_capture_end,
        "valid_interval_monotonic_ns": capture_boundary[
            "valid_interval_monotonic_ns"
        ],
        "host_boundary_uncertainty_ns": max(
            capture_boundary["start_api_rtt_ns"],
            capture_boundary["stop_api_rtt_ns"],
        ),
        "timestamp_fallback": False,
        "unaligned_profiler_events": same_clock_evidence is None,
        "reason": reason,
    }


def build_profiler_clock_domain(
    *,
    run_id: str,
    clock_domain_id: str,
    host_id: str,
    clock_type: ClockType,
    profile_kind: str,
    native_timestamp_unit: str,
    alignment_status: str,
) -> ClockDomain:
    """Build a v1 clock descriptor without discarding the artifact's raw unit."""

    if not native_timestamp_unit:
        raise DetailedProfileValidationError(
            "profiler native timestamp unit must be non-empty"
        )
    if alignment_status not in {"aligned", "partial"}:
        raise DetailedProfileValidationError(
            f"unsupported profiler alignment status: {alignment_status!r}"
        )
    record = ClockDomain(
        run_id=run_id,
        clock_domain_id=clock_domain_id,
        host_id=host_id,
        clock_type=clock_type,
        unit="ns",
        monotonic=True,
        adjustable=False,
        attributes={
            "hybrid.profile_kind": profile_kind,
            "hybrid.alignment_status": alignment_status,
            "hybrid.raw_timestamp_preserved": True,
            "hybrid.native_timestamp_unit": native_timestamp_unit,
        },
    )
    validate_record(record)
    return record


def validate_owned_wrapper_child_leader(
    *,
    wrapper_pid: int,
    target_pid: int,
    target_pgid: int,
    parent_by_pid: Mapping[int, int],
) -> int:
    """Require an externally wrapped server to be an owned process-group leader."""

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


def _validate_alignment_anchors(
    anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(anchors) != 2:
        raise DetailedProfileValidationError(
            "alignment requires exactly one profiler start and stop API anchor"
        )
    by_kind = {str(item.get("kind")): item for item in anchors}
    if set(by_kind) != {"profiler_start_api", "profiler_stop_api"}:
        raise DetailedProfileValidationError(
            "alignment requires exactly one profiler start and stop API anchor"
        )
    start = by_kind["profiler_start_api"]
    stop = by_kind["profiler_stop_api"]
    boundary = {
        "start_before_monotonic_ns": start.get("before_monotonic_ns"),
        "start_after_monotonic_ns": start.get("after_monotonic_ns"),
        "request_start_monotonic_ns": start.get("request_start_monotonic_ns"),
        "request_end_monotonic_ns": stop.get("request_end_monotonic_ns"),
        "stop_before_monotonic_ns": stop.get("before_monotonic_ns"),
        "stop_after_monotonic_ns": stop.get("after_monotonic_ns"),
        "start_http_status": start.get("http_status"),
        "stop_http_status": stop.get("http_status"),
    }
    return _validate_capture_boundary(boundary)


def validate_proxy_marker_stats(marker_root: Path) -> dict[str, Any]:
    """Require matching finalized stats for every marker JSONL sink."""

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
    """Compute single-request absolute/relative deltas without invented zeros."""

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
            "relative_delta": (
                absolute / baseline if baseline != 0 else None
            ),
            "reason": (
                None
                if baseline != 0
                else "relative delta is unavailable because control is zero"
            ),
        }
    return result
