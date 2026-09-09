"""Schema v1 fixtures shared by parity and runtime validation tests."""

from __future__ import annotations

from copy import deepcopy
import math


def valid_records() -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    manifest_minimum = {
        "schema_version": "1.0.0",
        "record_type": "run_manifest",
        "run_id": "run-minimum",
        "mode": "gpu_only",
        "profile_mode": "monitor",
        "status": "succeeded",
        "created_at_unix_ns": 0,
        "models": [{"role": "served", "model_id": "model", "revision": None, "tokenizer_id": None, "dtype": None}],
        "workload": {"request_count": None, "concurrency": None, "request_rate_per_s": None, "input_tokens": None, "output_tokens": None, "max_model_len": None, "warmup_requests": None},
        "hosts": [{"host_id": "host", "role": "local", "hostname": "host", "operating_system": "linux", "architecture": "x86_64"}],
        "software": [],
        "devices": [{"host_id": "host", "device_type": "gpu", "device_id": "gpu0", "vendor": "vendor", "model": "device", "status": "available", "memory_total_bytes": None, "attributes": {}}],
        "configuration": {},
        "attributes": {},
    }
    manifest_full = deepcopy(manifest_minimum)
    manifest_full.update({"run_id": "run-full", "mode": "hybrid", "created_at_unix_ns": 10})
    manifest_full["models"] = [
        {"role": "prefill", "model_id": "model", "revision": "rev", "tokenizer_id": "tok", "dtype": "float16"},
        {"role": "decode", "model_id": "model", "revision": "rev", "tokenizer_id": "tok", "dtype": "float16"},
    ]
    manifest_full["devices"] = [
        manifest_minimum["devices"][0],
        {"host_id": "host", "device_type": "npu", "device_id": "npu0", "vendor": "vendor", "model": "device", "status": "available", "memory_total_bytes": 1024, "attributes": {"vendor.partition": 0}},
    ]
    event_minimum = {
        "schema_version": "1.0.0", "record_type": "event", "run_id": "run-minimum",
        "event_id": "event-1", "event_name": "request_received", "event_type": "instant",
        "phase": "request", "host_id": "host", "clock_domain_id": "clock", "timestamp_ns": 0,
        "attributes": {},
    }
    event_full = {**event_minimum, "run_id": "run-full", "event_id": "event-2", "event_name": "response_done", "event_type": "span", "duration_ns": 10, "request_id": "request-1", "parent_event_id": "event-1", "process_id": 1, "thread_id": 2, "device_type": "gpu", "device_id": "gpu0", "attributes": {"test.detail": True}}
    metric_minimum = {
        "schema_version": "1.0.0", "record_type": "metric", "run_id": "run-minimum",
        "metric_name": "resource.gpu.utilization", "metric_kind": "gauge", "scope": "device",
        "host_id": "host", "clock_domain_id": "clock", "timestamp_ns": 0,
        "availability": "available", "origin": "measured", "unit": "percent", "value": 0,
        "dimensions": {}, "attributes": {},
    }
    metric_full = {**metric_minimum, "run_id": "run-full", "timestamp_ns": 10, "value": 50.5, "request_id": "request-1", "phase": "prefill", "device_type": "gpu", "device_id": "gpu0", "interval_ns": 5, "reason": None, "source_event_ids": ["event-1"], "dimensions": {"worker": 0}, "attributes": {"test.detail": True}}
    artifact_minimum = {
        "schema_version": "1.0.0", "record_type": "artifact", "run_id": "run-minimum",
        "artifact_id": "artifact-1", "artifact_kind": "raw_log", "relative_path": "logs/server.log",
        "format": "text", "producer": "collector", "created_at_unix_ns": 0, "attributes": {},
    }
    artifact_full = {**artifact_minimum, "run_id": "run-full", "host_id": "host", "request_id": "request-1", "size_bytes": 10, "sha256": "a" * 64, "clock_domain_id": "clock", "attributes": {"test.detail": True}}
    clock_minimum = {"schema_version": "1.0.0", "record_type": "clock_domain", "run_id": "run-minimum", "clock_domain_id": "clock", "host_id": "host", "clock_type": "monotonic", "unit": "ns", "monotonic": True, "adjustable": False, "attributes": {}}
    sync_minimum = {"schema_version": "1.0.0", "record_type": "sync_point", "run_id": "run-minimum", "sync_point_id": "sync", "source_clock_domain_id": "clock-a", "target_clock_domain_id": "clock-b", "source_timestamp_ns": 0, "target_timestamp_ns": 1, "method": "shared_event", "uncertainty_ns": 0, "attributes": {}}
    transform_minimum = {"schema_version": "1.0.0", "record_type": "clock_transform", "run_id": "run-minimum", "transform_id": "transform", "source_clock_domain_id": "clock-a", "target_clock_domain_id": "clock-b", "scale": 1.0, "offset_ns": 0, "uncertainty_ns": 0, "method": "shared_event", "valid_from_source_ns": 0, "valid_to_source_ns": None, "attributes": {}}
    return {
        "run_manifest": (manifest_minimum, manifest_full),
        "event": (event_minimum, event_full),
        "metric": (metric_minimum, metric_full),
        "artifact": (artifact_minimum, artifact_full),
        "clock_domain": (clock_minimum, {**clock_minimum, "run_id": "run-full", "attributes": {"test.detail": True}}),
        "sync_point": (sync_minimum, {**sync_minimum, "run_id": "run-full", "uncertainty_ns": 5, "attributes": {"test.detail": True}}),
        "clock_transform": (transform_minimum, {**transform_minimum, "run_id": "run-full", "scale": 1.25, "offset_ns": -5, "valid_to_source_ns": 10, "attributes": {"test.detail": True}}),
    }


def parity_cases() -> list[tuple[str, bool, dict[str, object]]]:
    records = valid_records()
    cases: list[tuple[str, bool, dict[str, object]]] = []
    enum_field = {"run_manifest": "mode", "event": "event_type", "metric": "scope", "artifact": "artifact_kind", "clock_domain": "clock_type", "sync_point": "method", "clock_transform": "method"}
    integer_field = {"run_manifest": "created_at_unix_ns", "event": "timestamp_ns", "metric": "timestamp_ns", "artifact": "created_at_unix_ns", "sync_point": "uncertainty_ns", "clock_transform": "offset_ns"}
    semantic = {
        "run_manifest": ("mode", "npu_only"),
        "event": ("event_name", "not_namespaced"),
        "metric": ("metric_name", "queue.depth"),
        "artifact": ("relative_path", "../escape"),
        "clock_domain": ("run_id", "../escape"),
        "sync_point": ("target_clock_domain_id", "clock-a"),
        "clock_transform": ("target_clock_domain_id", "clock-a"),
    }
    for record_type, (minimum, full) in records.items():
        cases.extend(((f"{record_type}:valid_minimum", True, deepcopy(minimum)), (f"{record_type}:valid_full", True, deepcopy(full))))
        required = next(key for key in minimum if key not in {"schema_version", "record_type"})
        value = deepcopy(minimum); del value[required]; cases.append((f"{record_type}:missing_required", False, value))
        value = deepcopy(minimum); value["unknown"] = True; cases.append((f"{record_type}:unknown_field", False, value))
        value = deepcopy(minimum); value[enum_field[record_type]] = "invalid"; cases.append((f"{record_type}:invalid_enum", False, value))
        value = deepcopy(minimum); value["schema_version"] = "invalid"; cases.append((f"{record_type}:invalid_pattern", False, value))
        value = deepcopy(minimum); value["attributes"] = {"not_namespaced": True}; cases.append((f"{record_type}:invalid_attributes", False, value))
        value = deepcopy(minimum); value["attributes"] = {"test.nonfinite": math.inf}; cases.append((f"{record_type}:nonfinite", False, value))
        field, invalid = semantic[record_type]; value = deepcopy(minimum); value[field] = invalid; cases.append((f"{record_type}:semantic_only", False, value))
        if record_type in integer_field:
            field = integer_field[record_type]
            value = deepcopy(minimum); value[field] = "1"; cases.append((f"{record_type}:wrong_primitive", False, value))
            value = deepcopy(minimum); value[field] = True; cases.append((f"{record_type}:bool_as_integer", False, value))
        if record_type == "run_manifest":
            value = deepcopy(minimum); value["models"][0]["unknown"] = True; cases.append(("run_manifest:nested_unknown", False, value))
            value = deepcopy(minimum); del value["models"][0]["model_id"]; cases.append(("run_manifest:nested_missing", False, value))
        if record_type == "event":
            value = deepcopy(minimum); value["timestamp_ns"] = -1; cases.append(("event:negative_bound", False, value))
            value = deepcopy(minimum); value["duration_ns"] = 1; cases.append(("event:cross_field", False, value))
            value = deepcopy(minimum); value["request_id"] = None; cases.append(("event:nullable", True, value))
        elif record_type == "metric":
            value = deepcopy(minimum); value["timestamp_ns"] = -1; cases.append(("metric:negative_bound", False, value))
            value = deepcopy(minimum); value.update({"availability": "not_available", "value": None}); cases.append(("metric:cross_field", False, value))
            value = deepcopy(minimum); value["reason"] = None; cases.append(("metric:nullable", True, value))
        elif record_type == "artifact":
            value = deepcopy(minimum); value["created_at_unix_ns"] = -1; cases.append(("artifact:negative_bound", False, value))
            value = deepcopy(minimum); value["sha256"] = "bad"; cases.append(("artifact:invalid_sha_pattern", False, value))
            value = deepcopy(minimum); value["host_id"] = None; cases.append(("artifact:nullable", True, value))
        elif record_type == "sync_point":
            value = deepcopy(minimum); value["uncertainty_ns"] = -1; cases.append(("sync_point:negative_bound", False, value))
        elif record_type == "clock_transform":
            value = deepcopy(minimum); value["scale"] = -1; cases.append(("clock_transform:negative_bound", False, value))
            value = deepcopy(minimum); value.update({"valid_from_source_ns": 2, "valid_to_source_ns": 1}); cases.append(("clock_transform:cross_field", False, value))
    return cases
