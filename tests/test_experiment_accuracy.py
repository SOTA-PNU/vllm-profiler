"""Tests for fixed experiment accuracy policies."""

from __future__ import annotations

import math
import unittest

from tools.evaluation.accuracy import (
    AccuracyError,
    ClientLatencyMetric,
    client_latency_accuracy,
    exact_count_accuracy,
    exact_marker_accuracy,
)


class ClientLatencyAccuracyTests(unittest.TestCase):
    def test_e2e_ttft_floor_and_relative_tolerances(self) -> None:
        floor_boundary = client_latency_accuracy(
            "e2e",
            reference_ns=100_000_000,
            observed_ns=102_000_000,
        )
        self.assertEqual(floor_boundary.tolerance, 2_000_000)
        self.assertTrue(floor_boundary.passed)
        self.assertFalse(
            client_latency_accuracy(
                ClientLatencyMetric.TTFT,
                reference_ns=100_000_000,
                observed_ns=102_000_001,
            ).passed
        )

        relative_boundary = client_latency_accuracy(
            "latency.e2e",
            reference_ns=200_000_000,
            observed_ns=204_000_000,
        )
        self.assertEqual(relative_boundary.tolerance, 4_000_000)
        self.assertTrue(relative_boundary.passed)

    def test_tpot_floor_relative_and_boundary(self) -> None:
        floor_boundary = client_latency_accuracy(
            "tpot",
            reference_ns=10_000_000,
            observed_ns=11_000_000,
        )
        self.assertEqual(floor_boundary.tolerance, 1_000_000)
        self.assertTrue(floor_boundary.passed)
        self.assertFalse(
            client_latency_accuracy(
                "latency.tpot",
                reference_ns=10_000_000,
                observed_ns=11_000_001,
            ).passed
        )
        relative = client_latency_accuracy(
            "tpot",
            reference_ns=100_000_000,
            observed_ns=105_000_000,
        )
        self.assertEqual(relative.tolerance, 5_000_000)
        self.assertTrue(relative.passed)

    def test_bad_metric_bool_nonfinite_and_negative_are_rejected(self) -> None:
        with self.assertRaises(AccuracyError):
            client_latency_accuracy(
                "unknown",
                reference_ns=1,
                observed_ns=1,
            )
        for value in (True, math.nan, math.inf, -math.inf, -1):
            with self.subTest(value=value):
                with self.assertRaises(AccuracyError):
                    client_latency_accuracy(
                        "e2e",
                        reference_ns=value,
                        observed_ns=1,
                    )


class ExactAccuracyTests(unittest.TestCase):
    def test_request_token_and_marker_values_require_exact_equality(self) -> None:
        request_count = exact_count_accuracy(
            "request.count",
            reference=10,
            observed=10,
        )
        self.assertTrue(request_count.passed)
        self.assertEqual(request_count.tolerance, 0)
        self.assertEqual(request_count.absolute_error, 0)

        token_count = exact_count_accuracy(
            "request.output_tokens",
            reference=80,
            observed=79,
        )
        self.assertFalse(token_count.passed)
        self.assertEqual(token_count.absolute_error, 1)

        marker = exact_marker_accuracy(
            "latency.kv_transfer",
            reference_ns=123,
            observed_ns=123,
        )
        self.assertTrue(marker.passed)
        self.assertEqual(marker.canonical_unit, "ns")
        self.assertEqual(marker.method_id, "exact_same_clock_integer_ns_v1")

    def test_exact_checks_reject_bool_float_and_negative(self) -> None:
        for value in (True, 1.0, -1):
            with self.subTest(value=value):
                with self.assertRaises(AccuracyError):
                    exact_count_accuracy(
                        "request.count",
                        reference=value,  # type: ignore[arg-type]
                        observed=1,
                    )


if __name__ == "__main__":
    unittest.main()
