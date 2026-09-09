"""Deterministic and policy-focused tests for Overview comparisons."""

from __future__ import annotations

import copy
import json
import math
import unittest

from tools.evaluation.overview import (
    OverviewComparisonError,
    build_comparison,
    overview_document_from_json,
)
from tests.support.comparison import kpi, report



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
