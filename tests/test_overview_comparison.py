"""Deterministic and policy-focused tests for Overview comparisons."""

from __future__ import annotations

import copy
import json
import math
import unittest

from tools.evaluation.overview import (
    OverviewComparisonError,
    build_comparison,
)
from perfetto_hetero_profiler.overview.schema import (
    overview_document_from_json,
)


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


class ComparisonDeterminismTests(unittest.TestCase):
    def test_input_order_is_irrelevant_and_control_is_implicit_baseline(self):
        control = report("run-control")
        detailed = report(
            "run-torch",
            profile_mode="detailed_profile",
            profiler_kind="gpu_torch",
            request_e2e=110,
        )
        forward = build_comparison([control, detailed])
        reverse = build_comparison([detailed, control])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["comparison"]["baseline_run_id"], "run-control")
        self.assertEqual(
            [item["run_id"] for item in forward["runs"]],
            ["run-control", "run-torch"],
        )

    def test_explicit_baseline_and_unique_run_ids_are_enforced(self):
        first = report("a", profiler_kind="gpu_torch")
        second = report("b", profiler_kind="gpu_nsys")
        result = build_comparison([second, first], baseline_run_id="b")
        self.assertEqual(result["comparison"]["baseline_run_id"], "b")
        with self.assertRaisesRegex(OverviewComparisonError, "not present"):
            build_comparison([first, second], baseline_run_id="missing")
        with self.assertRaisesRegex(OverviewComparisonError, "duplicate run_id"):
            build_comparison([first, copy.deepcopy(first)], baseline_run_id="a")
        with self.assertRaisesRegex(OverviewComparisonError, "exactly one control"):
            build_comparison([first, second])

    def test_report_inputs_are_not_mutated(self):
        reports = [report("control"), report("other", profiler_kind="gpu_torch")]
        before = copy.deepcopy(reports)
        build_comparison(reports)
        self.assertEqual(reports, before)

    def test_plain_comparison_matches_the_versioned_schema(self):
        value = build_comparison(
            [report("control"), report("candidate")],
            baseline_run_id="control",
        )
        parsed = overview_document_from_json(
            json.dumps(value, allow_nan=False, sort_keys=True)
        )
        self.assertEqual(parsed.record_type, "overview_comparison")


class ComparisonPolicyTests(unittest.TestCase):
    def test_matching_repeated_controls_are_comparable(self):
        baseline = report("control")
        candidate = report("candidate", profiler_kind="control")
        result = build_comparison(
            [candidate, baseline], baseline_run_id="control"
        )
        self.assertEqual(result["comparison"]["comparability"], "comparable")

    def test_one_request_and_different_profiler_are_diagnostic_only(self):
        control = report("control", request_count=1)
        profiler = report(
            "profiler",
            request_count=1,
            profile_mode="detailed_profile",
            profiler_kind="npu_rbln",
        )
        profiler["data_quality"]["profiler"]["native_alignment_status"] = "partial"
        result = build_comparison([profiler, control])
        self.assertEqual(
            result["comparison"]["comparability"], "diagnostic_only"
        )
        reasons = " ".join(result["comparison"]["comparability_reasons"])
        self.assertIn("request sample count is one", reasons)
        self.assertIn("profiler kinds differ", reasons)
        self.assertIn("partial", reasons)

    def test_core_identity_mismatches_are_not_comparable(self):
        mutators = {
            "model": lambda item: item["models"][0].update(model_id="other"),
            "hardware": lambda item: item["hardware"][0].update(model="other"),
            "workload": lambda item: item["workload"].update(temperature=1),
            "tokens": lambda item: item["workload"].update(output_tokens=9),
            "requests": lambda item: item["workload"].update(request_count=5),
            "clock": lambda item: item["run"].update(
                canonical_clock_domain_id="other-clock"
            ),
            "alignment": lambda item: item["data_quality"]["alignment"].update(
                status="unaligned"
            ),
            "mode": lambda item: item["run"].update(mode="gpu_only"),
        }
        for label, mutate in mutators.items():
            with self.subTest(label=label):
                control = report("control")
                candidate = report("candidate")
                mutate(candidate)
                result = build_comparison(
                    [candidate, control], baseline_run_id="control"
                )
                self.assertEqual(
                    result["comparison"]["comparability"], "not_comparable"
                )
                parsed = overview_document_from_json(
                    json.dumps(result, allow_nan=False, sort_keys=True)
                )
                self.assertEqual(parsed.metrics[0].deltas, ())

    def test_invalid_integrity_is_not_comparable(self):
        control = report("control")
        candidate = report("candidate")
        candidate["data_quality"]["perfetto_sql_validation"]["mismatches"] = [
            "counter mismatch"
        ]
        result = build_comparison(
            [control, candidate], baseline_run_id="control"
        )
        self.assertEqual(
            result["comparison"]["comparability"], "not_comparable"
        )
        self.assertIn(
            False,
            [item["source_integrity_valid"] for item in result["runs"]],
        )

    def test_kpi_availability_difference_is_diagnostic(self):
        control = report("control")
        candidate = report("candidate", request_e2e=None)
        result = build_comparison(
            [control, candidate], baseline_run_id="control"
        )
        self.assertEqual(
            result["comparison"]["comparability"], "diagnostic_only"
        )
        self.assertIn(
            "KPI availability differs across runs",
            result["comparison"]["comparability_reasons"],
        )


class ComparisonDeltaTests(unittest.TestCase):
    def _metric(
        self, result: dict[str, object], category: str, name: str
    ) -> dict[str, object]:
        return next(
            item
            for item in result["metrics"]
            if item["section"] == category and item["name"] == name
        )

    def test_request_and_pipeline_e2e_remain_separate(self):
        result = build_comparison(
            [report("control"), report("candidate")],
            baseline_run_id="control",
        )
        e2e = [
            metric
            for metric in result["metrics"]
            if metric["name"] == "latency.e2e"
        ]
        self.assertEqual(len(e2e), 2)
        self.assertEqual(
            {item["observation_layer"] for item in e2e},
            {"request_facing_client", "hybrid_pipeline"},
        )

    def test_available_nonzero_baseline_has_absolute_and_percent_delta(self):
        result = build_comparison(
            [
                report("control", request_e2e=100, throughput=10),
                report("candidate", request_e2e=125, throughput=12),
            ],
            baseline_run_id="control",
        )
        latency = self._metric(result, "request_facing_latency", "latency.e2e")
        throughput = self._metric(
            result, "throughput_and_tokens", "throughput.requests"
        )
        latency_delta = next(
            item for item in latency["deltas"] if item["run_id"] == "candidate"
        )
        throughput_delta = next(
            item
            for item in throughput["deltas"]
            if item["run_id"] == "candidate"
        )
        self.assertEqual(latency["direction"], "lower_is_preferred")
        self.assertEqual(throughput["direction"], "higher_is_preferred")
        self.assertEqual(latency_delta["absolute"]["value"], 25)
        self.assertEqual(latency_delta["percentage"]["value"], 25)
        self.assertEqual(throughput_delta["absolute"]["value"], 2)
        self.assertEqual(throughput_delta["percentage"]["value"], 20)

    def test_zero_baseline_makes_both_deltas_unavailable(self):
        result = build_comparison(
            [
                report("control", request_e2e=0),
                report("candidate", request_e2e=1),
            ],
            baseline_run_id="control",
        )
        metric = self._metric(result, "request_facing_latency", "latency.e2e")
        for delta in metric["deltas"]:
            self.assertEqual(delta["absolute"]["availability"], "not_available")
            self.assertEqual(
                delta["percentage"]["availability"], "not_available"
            )
            self.assertIn("baseline KPI is zero", delta["absolute"]["unavailable_reason"])

    def test_unavailable_target_has_no_delta(self):
        result = build_comparison(
            [
                report("control", request_e2e=100),
                report("candidate", request_e2e=None),
            ],
            baseline_run_id="control",
        )
        metric = self._metric(result, "request_facing_latency", "latency.e2e")
        delta = next(
            item for item in metric["deltas"] if item["run_id"] == "candidate"
        )
        self.assertEqual(delta["absolute"]["availability"], "not_available")
        self.assertEqual(delta["percentage"]["availability"], "not_available")

    def test_bool_nan_and_infinity_available_values_are_rejected(self):
        for bad_value in (True, math.nan, math.inf, -math.inf):
            with self.subTest(value=bad_value):
                candidate = report("candidate")
                candidate["kpis"]["request_facing_latency"][0]["value"] = bad_value
                with self.assertRaisesRegex(
                    OverviewComparisonError, "finite"
                ):
                    build_comparison(
                        [report("control"), candidate],
                        baseline_run_id="control",
                    )

    def test_metric_catalog_unit_mismatch_is_rejected(self):
        candidate = report("candidate")
        candidate["kpis"]["request_facing_latency"][0][
            "canonical_unit"
        ] = "ms"
        with self.assertRaisesRegex(
            OverviewComparisonError, "metric catalog"
        ):
            build_comparison(
                [report("control"), candidate],
                baseline_run_id="control",
            )


if __name__ == "__main__":
    unittest.main()
