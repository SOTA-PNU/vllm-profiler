"""Tests for deterministic Phase 7 statistics and paired overhead."""

from __future__ import annotations

import math
import unittest

from perfetto_hetero_profiler.phase7.statistics import (
    OverheadDirection,
    StatisticsError,
    paired_overhead,
    percentile_r7,
    summarize_distribution,
)


class DistributionTests(unittest.TestCase):
    def test_phase7_descriptive_statistics_contract(self) -> None:
        summary = summarize_distribution([5, 1, 4, 2, 3])
        self.assertEqual(summary.sample_count, 5)
        self.assertEqual(summary.mean, 3.0)
        self.assertEqual(summary.median, 3.0)
        self.assertEqual(summary.minimum, 1)
        self.assertEqual(summary.maximum, 5)
        self.assertAlmostEqual(
            summary.sample_standard_deviation or 0,
            math.sqrt(2.5),
        )
        self.assertAlmostEqual(
            summary.coefficient_of_variation or 0,
            math.sqrt(2.5) / 3,
        )
        self.assertEqual(summary.median_absolute_deviation, 1.0)
        self.assertEqual(summary.p50, 3.0)
        self.assertAlmostEqual(summary.p95, 4.8)
        self.assertEqual(percentile_r7([5, 1, 4, 2, 3], 0.95), 4.8)

    def test_small_sample_and_zero_mean_are_explicitly_unavailable(self) -> None:
        singleton = summarize_distribution([7])
        self.assertIsNone(singleton.sample_standard_deviation)
        self.assertIn(
            "at least two",
            singleton.sample_standard_deviation_unavailable_reason or "",
        )
        self.assertIsNone(singleton.coefficient_of_variation)

        zero_mean = summarize_distribution([-1, 0, 1])
        self.assertIsNotNone(zero_mean.sample_standard_deviation)
        self.assertIsNone(zero_mean.coefficient_of_variation)
        self.assertIn(
            "zero mean",
            zero_mean.coefficient_of_variation_unavailable_reason or "",
        )

    def test_bool_nonfinite_empty_and_bad_probability_are_rejected(self) -> None:
        for value in (True, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(StatisticsError):
                    summarize_distribution([value])
        with self.assertRaises(StatisticsError):
            summarize_distribution([])
        with self.assertRaises(StatisticsError):
            percentile_r7([1], 1.1)
        with self.assertRaises(StatisticsError):
            percentile_r7([1], True)


class PairedOverheadTests(unittest.TestCase):
    def test_latency_overhead_is_same_round_and_exactly_five_pairs(self) -> None:
        reference = {4: 100, 2: 100, 0: 100, 3: 100, 1: 100}
        observed = {0: 110, 1: 120, 2: 90, 3: 100, 4: 130}
        result = paired_overhead(reference, observed)
        self.assertEqual(
            [item.round_index for item in result.pairs],
            list(range(5)),
        )
        self.assertEqual(
            [item.absolute_delta for item in result.pairs],
            [10, 20, -10, 0, 30],
        )
        self.assertEqual(
            [item.overhead_ratio for item in result.pairs],
            [0.1, 0.2, -0.1, 0.0, 0.3],
        )
        self.assertEqual(result.absolute_delta_summary.median, 10.0)
        self.assertEqual(result.overhead_ratio_summary.median, 0.1)

    def test_zero_reference_preserves_absolute_and_disables_ratio_summary(self) -> None:
        result = paired_overhead(
            {0: 0},
            {0: 5},
            expected_pair_count=1,
        )
        self.assertEqual(result.pairs[0].absolute_delta, 5)
        self.assertIsNone(result.pairs[0].overhead_ratio)
        self.assertIsNone(result.overhead_ratio_summary)
        self.assertIn(
            "non-zero reference",
            result.overhead_ratio_summary_unavailable_reason or "",
        )

    def test_throughput_degradation_has_positive_degradation_sign(self) -> None:
        degraded = paired_overhead(
            {0: 100},
            {0: 90},
            direction=OverheadDirection.THROUGHPUT_DEGRADATION,
            expected_pair_count=1,
        )
        improved = paired_overhead(
            {0: 100},
            {0: 110},
            direction="throughput_degradation",
            expected_pair_count=1,
        )
        self.assertAlmostEqual(degraded.pairs[0].overhead_ratio or 0, 0.1)
        self.assertAlmostEqual(improved.pairs[0].overhead_ratio or 0, -0.1)
        self.assertEqual(degraded.pairs[0].absolute_delta, -10)

    def test_pair_count_round_identity_and_inputs_are_strict(self) -> None:
        with self.assertRaisesRegex(StatisticsError, "exactly 5"):
            paired_overhead({0: 1}, {0: 2})
        configured = paired_overhead(
            {0: 1, 1: 2, 2: 3},
            {0: 2, 1: 3, 2: 4},
            expected_pair_count=3,
        )
        self.assertEqual(len(configured.pairs), 3)
        with self.assertRaisesRegex(StatisticsError, "identical round sets"):
            paired_overhead(
                {0: 1},
                {1: 2},
                expected_pair_count=1,
            )
        with self.assertRaises(StatisticsError):
            paired_overhead(
                {False: 1},  # type: ignore[dict-item]
                {False: 2},  # type: ignore[dict-item]
                expected_pair_count=1,
            )
        with self.assertRaises(StatisticsError):
            paired_overhead(
                {0: math.inf},
                {0: 2},
                expected_pair_count=1,
            )
        with self.assertRaises(StatisticsError):
            paired_overhead(
                {0: 1},
                {0: 2},
                expected_pair_count=True,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
