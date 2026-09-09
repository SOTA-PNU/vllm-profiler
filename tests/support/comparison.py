"""Shared dictionary fixtures for evaluation comparisons."""

from __future__ import annotations

def kpi(
    name: str,
    value: int | float | None,
    *,
    unit: str = "ns",
    layer: str = "request_facing_client",
    reason: str | None = None,
) -> dict[str, object]:
    available = value is not None
    return {
        "name": name,
        "canonical_unit": unit,
        "availability": "available" if available else "not_available",
        "value": value,
        "unavailable_reason": None if available else (reason or "not measured"),
        "aggregation_method": "single_request_v1",
        "sample_count": 1 if available else 0,
        "sources": [
            {
                "source_kind": "normalized_metric",
                "record_ids": [],
                "metric_names": [name],
                "root_id": "metrics",
                "relative_path": "metrics/metrics.jsonl",
                "details": {},
            }
        ],
        "scope": {
            "run_id": "placeholder",
            "scope_type": "request",
            "observation_layer": layer,
            "request_id": "request-0",
            "host_id": "host",
            "device_type": None,
            "device_id": None,
            "phase": None,
            "window": "measured_request",
        },
        "calculation": {
            "method_id": "single_request_v1",
            "formula": "end_timestamp_ns - start_timestamp_ns",
        },
        "clock": {
            "domain_ids": ["hybrid-canonical"],
            "alignment_status": "aligned",
            "alignment_method": "same_clock_domain",
            "offset_ns": 0,
            "uncertainty_ns": 0,
        },
        "quality_warnings": [],
        "display": {
            "unit": "ms" if unit == "ns" else unit,
            "scale_numerator": 1,
            "scale_denominator": 1_000_000 if unit == "ns" else 1,
            "decimal_places": 3,
            "rounding": "half_even",
        },
    }

def report(
    run_id: str,
    *,
    profile_mode: str = "monitor",
    profiler_kind: str = "control",
    request_count: int = 4,
    request_e2e: int | float | None = 100,
    pipeline_e2e: int | float | None = 90,
    throughput: int | float | None = 10,
    run_mode: str = "hybrid",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "record_type": "overview_report",
        "run": {
            "run_id": run_id,
            "mode": run_mode,
            "profile_mode": profile_mode,
            "status": "succeeded",
            "profiler_kind": profiler_kind,
            "canonical_clock_domain_id": "hybrid-canonical",
        },
        "workload": {
            "request_count": request_count,
            "input_tokens": 5,
            "output_tokens": 8,
            "total_tokens": 13,
            "concurrency": 1,
            "request_rate_per_s": None,
            "warmup_requests": 1,
            "max_output_tokens": 8,
            "temperature": 0,
            "retry_count": 0,
            "prompt_sha256": "1" * 64,
            "request_body_sha256": "2" * 64,
            "offline": True,
            "max_model_len": 512,
            "block_size": 512,
        },
        "models": [
            {
                "role": "decode",
                "model_id": "Qwen3-0.6B",
                "revision": None,
                "dtype": None,
            },
            {
                "role": "prefill",
                "model_id": "Qwen3-0.6B",
                "revision": None,
                "dtype": None,
            },
        ],
        "hardware": [
            {
                "device_type": "npu",
                "device_id": "npu-0",
                "vendor": "Rebellions",
                "model": "RBLN-CA22",
                "memory_total_bytes": 16_877_879_296,
            },
            {
                "device_type": "gpu",
                "device_id": "gpu-0",
                "vendor": "NVIDIA",
                "model": "RTX PRO 6000",
                "memory_total_bytes": 102_641_958_912,
            },
        ],
        "kpis": {
            "request_facing_latency": [
                kpi(
                    "latency.e2e",
                    request_e2e,
                    layer="request_facing_client",
                )
            ],
            "pipeline_latency": [
                kpi("latency.e2e", pipeline_e2e, layer="hybrid_pipeline")
            ],
            "throughput_and_tokens": [
                kpi(
                    "throughput.requests",
                    throughput,
                    unit="requests/s",
                    layer="run",
                )
            ],
            "transfer": [
                kpi(
                    "transfer.wait_duration",
                    None,
                    layer="hybrid_pipeline",
                    reason="no classified wait interval",
                )
            ],
        },
        "resources": [],
        "data_quality": {
            "run_status": "succeeded",
            "canonical_marker_count": 44,
            "marker_validation": {
                "status": "valid",
                "missing_count": 0,
                "duplicate_count": 0,
                "pairing_violation_count": 0,
                "order_violation_count": 0,
            },
            "request_join": {
                "joined_count": 1,
                "unjoined_count": 0,
                "method": "explicit_correlation",
            },
            "alignment": {
                "status": "aligned",
                "method": "same_clock_domain",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
            "resource_samples": {
                "total": 10,
                "available": 9,
                "unavailable": 1,
            },
            "profiler": {
                "kind": profiler_kind,
                "native_alignment_status": "not_applicable",
            },
            "source_artifact_validation": {
                "valid": True,
                "closeout_artifact_count": 69,
                "closeout_manifest_sha256": "3" * 64,
                "roots": [],
            },
            "perfetto_sql_validation": {
                "valid": True,
                "query_count": 10,
                "mismatches": [],
            },
            "trace_sha256": "4" * 64,
            "per_sample_stream_preserved": True,
            "cleanup_complete": True,
            "rbln_pb_policy": "perfetto_compatible_separate_unaligned",
            "sample_limitations": [],
        },
        "perfetto": {
            "trace_validation": {"valid": True, "mismatches": []},
            "source_match": True,
        },
        "native_profiles": [],
        "interpretation": {
            "comparison_scope": "same-workload capture diagnostics",
            "limitations": ["No randomized repeated trial was performed."],
            "policies": [],
        },
    }
