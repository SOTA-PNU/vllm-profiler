"""Clock probe and estimator tests for hybrid time alignment."""

import unittest

from perfetto_hetero_profiler.hybrid.clock_sync import (
    ClockProbeSample,
    ClockSyncError,
    FakeClockProbeTransport,
    LocalClockProbeTransport,
    collect_probe_samples,
    estimate_clock,
    probe_clock,
    same_clock_estimate,
)


class ClockProbeSampleTests(unittest.TestCase):
    def test_fixed_positive_offset(self):
        sample = ClockProbeSample(100, 160, 170, 220)
        self.assertEqual(sample.offset_ns, 5)

    def test_fixed_negative_offset(self):
        sample = ClockProbeSample(100, 60, 70, 220)
        self.assertEqual(sample.offset_ns, -95)

    def test_round_trip_excludes_remote_processing(self):
        sample = ClockProbeSample(100, 150, 170, 240)
        self.assertEqual(sample.round_trip_ns, 120)

    def test_negative_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ClockProbeSample(-1, 0, 0, 0)

    def test_remote_reverse_rejected(self):
        with self.assertRaisesRegex(ValueError, "target send"):
            ClockProbeSample(0, 10, 9, 20)

    def test_negative_round_trip_rejected(self):
        with self.assertRaisesRegex(ValueError, "processing"):
            ClockProbeSample(0, 10, 100, 20)


class ClockEstimatorTests(unittest.TestCase):
    def test_same_clock_has_zero_uncertainty(self):
        estimate = same_clock_estimate()
        self.assertEqual((estimate.offset_ns, estimate.uncertainty_ns), (0, 0))
        self.assertEqual(estimate.method, "same_clock_domain")

    def test_fake_positive_offset(self):
        estimate = probe_clock(
            FakeClockProbeTransport(offset_ns=50_000_000), count=7
        )
        self.assertEqual(estimate.offset_ns, 50_000_000)

    def test_fake_negative_offset(self):
        estimate = probe_clock(
            FakeClockProbeTransport(offset_ns=-50_000_000), count=7
        )
        self.assertEqual(estimate.offset_ns, -50_000_000)

    def test_asymmetric_delay_affects_estimate(self):
        estimate = probe_clock(
            FakeClockProbeTransport(offset_ns=10_000, asymmetry_ns=1_000),
            count=7,
        )
        self.assertEqual(estimate.offset_ns, 11_000)

    def test_jitter_records_nonzero_uncertainty(self):
        estimate = probe_clock(
            FakeClockProbeTransport(jitter_ns=10_000), count=7
        )
        self.assertGreater(estimate.uncertainty_ns, 0)

    def test_minimum_rtt_sample_selected(self):
        samples = (
            ClockProbeSample(0, 20, 20, 100),
            ClockProbeSample(200, 220, 220, 240),
            ClockProbeSample(300, 320, 320, 360),
            ClockProbeSample(400, 420, 420, 460),
            ClockProbeSample(500, 520, 520, 560),
        )
        self.assertEqual(estimate_clock(samples).selected_index, 1)

    def test_offset_outlier_does_not_win_by_rtt(self):
        normal = [
            ClockProbeSample(i * 1000, i * 1000 + 100, i * 1000 + 100, i * 1000 + 200)
            for i in range(5)
        ]
        outlier = ClockProbeSample(10_000, 1_010_000, 1_010_000, 10_010)
        estimate = estimate_clock(tuple([*normal, outlier]), minimum_samples=5)
        self.assertNotEqual(estimate.selected_index, 5)

    def test_timeout_can_be_tolerated(self):
        samples = collect_probe_samples(
            FakeClockProbeTransport(timeout_indices=(0, 1)),
            count=7,
            minimum_samples=5,
        )
        self.assertEqual(len(samples), 5)

    def test_malformed_can_be_tolerated(self):
        samples = collect_probe_samples(
            FakeClockProbeTransport(malformed_indices=(2,)),
            count=6,
            minimum_samples=5,
        )
        self.assertEqual(len(samples), 5)

    def test_insufficient_samples_fail(self):
        with self.assertRaises(ClockSyncError):
            collect_probe_samples(
                FakeClockProbeTransport(timeout_indices=(0, 1, 2)),
                count=5,
                minimum_samples=5,
            )

    def test_integer_nanosecond_precision(self):
        sample = ClockProbeSample(10**18, 10**18 + 3, 10**18 + 4, 10**18 + 9)
        self.assertIsInstance(sample.offset_ns, int)

    def test_local_transport_is_monotonic(self):
        values = iter((10, 11, 12, 13))
        sample = LocalClockProbeTransport(lambda: next(values)).probe()
        self.assertEqual((sample.t0_ns, sample.t3_ns), (10, 13))

    def test_actual_local_probe_produces_estimate(self):
        estimate = probe_clock(
            LocalClockProbeTransport(), count=5, minimum_samples=5
        )
        self.assertEqual(estimate.sample_count, 5)
        self.assertGreaterEqual(estimate.uncertainty_ns, 0)


if __name__ == "__main__":
    unittest.main()
