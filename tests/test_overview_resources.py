from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from perfetto_hetero_profiler.overview.resources import (
    ResourceCalculationError,
    percentile_r7,
    summarize_resources,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceType,
    MetricSample,
    ValueOrigin,
)
from perfetto_hetero_profiler.schema.metric_catalog import METRIC_CATALOG


RUN_ID = "resource-fixture"
CLOCK_ID = "canonical"
ALIGNMENT = {
    "hybrid.alignment_method": "same_clock_domain",
    "hybrid.alignment_uncertainty_ns": 0,
}


def _resource(
    name: str,
    value: int | float | None,
    timestamp_ns: int,
    *,
    interval_ns: int | None,
    availability: Availability = Availability.AVAILABLE,
    device_type: DeviceType | None = None,
    device_id: str | None = None,
    dimensions: dict[str, object] | None = None,
    attributes: dict[str, object] | None = None,
) -> MetricSample:
    definition = METRIC_CATALOG[name]
    scope = definition.allowed_scopes[-1]
    if name == "resource.system.memory_used":
        scope = definition.allowed_scopes[0]
    return MetricSample(
        run_id=RUN_ID,
        metric_name=name,
        metric_kind=definition.kind,
        scope=scope,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=timestamp_ns,
        availability=availability,
        origin=ValueOrigin.MEASURED,
        unit=definition.unit,
        value=value,
        dimensions=dimensions or {},
        attributes={**ALIGNMENT, **(attributes or {})},
        device_type=device_type,
        device_id=device_id,
        interval_ns=interval_ns,
        reason=("not reported" if availability != Availability.AVAILABLE else None),
    )


def _loaded(metrics: list[MetricSample]) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            run_id=RUN_ID,
            attributes={"hybrid.alignment_offset_ns": 0},
        ),
        canonical_clock_domain_id=CLOCK_ID,
        metrics=tuple(metrics),
    )


def _aggregate(summary: dict[str, object], suffix: str) -> dict[str, object]:
    return next(
        item for item in summary["aggregates"] if item["name"].endswith(suffix)
    )


class OverviewResourceTests(unittest.TestCase):
    def test_percentile_uses_hyndman_fan_r7(self) -> None:
        values = [0, 10, 20, 30]
        self.assertEqual(percentile_r7(values, 0.50), 15.0)
        self.assertAlmostEqual(percentile_r7(values, 0.95), 28.5)
        self.assertEqual(percentile_r7([7], 0.95), 7.0)
        with self.assertRaisesRegex(ResourceCalculationError, "at least one"):
            percentile_r7([], 0.5)

    def test_devices_are_separate_and_available_zero_is_preserved(self) -> None:
        metrics = [
            _resource(
                "resource.gpu.utilization",
                0,
                0,
                interval_ns=500,
                device_type=DeviceType.GPU,
                device_id="gpu-0",
            ),
            _resource(
                "resource.gpu.utilization",
                10,
                10,
                interval_ns=10,
                device_type=DeviceType.GPU,
                device_id="gpu-0",
            ),
            _resource(
                "resource.gpu.utilization",
                20,
                20,
                interval_ns=10,
                device_type=DeviceType.GPU,
                device_id="gpu-0",
            ),
            _resource(
                "resource.gpu.utilization",
                100,
                0,
                interval_ns=500,
                device_type=DeviceType.GPU,
                device_id="gpu-1",
            ),
            _resource(
                "resource.gpu.utilization",
                None,
                10,
                interval_ns=10,
                availability=Availability.NOT_AVAILABLE,
                device_type=DeviceType.GPU,
                device_id="gpu-1",
            ),
            _resource(
                "resource.gpu.utilization",
                100,
                20,
                interval_ns=10,
                device_type=DeviceType.GPU,
                device_id="gpu-1",
            ),
        ]
        summaries = summarize_resources(_loaded(metrics))
        self.assertEqual(len(summaries), 2)
        by_device = {summary["scope"]["device_id"]: summary for summary in summaries}
        gpu0 = by_device["gpu-0"]
        gpu1 = by_device["gpu-1"]

        self.assertEqual(_aggregate(gpu0, ".min")["value"], 0)
        self.assertEqual(_aggregate(gpu0, ".mean")["value"], 10)
        self.assertEqual(
            _aggregate(gpu0, ".time_weighted_mean")["value"], 15
        )
        self.assertEqual(
            _aggregate(gpu0, ".time_weighted_mean")["sample_count"], 2
        )
        self.assertEqual(gpu0["coverage_ns"], 20)
        self.assertEqual(gpu0["availability_ratio"], 1.0)

        self.assertEqual(gpu1["available_sample_count"], 2)
        self.assertEqual(gpu1["unavailable_sample_count"], 1)
        self.assertEqual(gpu1["availability_ratio"], 2 / 3)
        weighted = _aggregate(gpu1, ".time_weighted_mean")
        self.assertEqual(weighted["availability"], "not_available")
        self.assertIn("unavailable", weighted["unavailable_reason"])

    def test_first_synthetic_interval_is_not_time_weighted(self) -> None:
        metrics = [
            _resource(
                "resource.npu.power",
                1,
                0,
                interval_ns=999,
                device_type=DeviceType.NPU,
                device_id="npu-0",
            ),
            _resource(
                "resource.npu.power",
                3,
                10,
                interval_ns=10,
                device_type=DeviceType.NPU,
                device_id="npu-0",
            ),
            _resource(
                "resource.npu.power",
                7,
                30,
                interval_ns=20,
                device_type=DeviceType.NPU,
                device_id="npu-0",
            ),
        ]
        summary = summarize_resources(_loaded(metrics))[0]
        weighted = _aggregate(summary, ".time_weighted_mean")
        self.assertEqual(weighted["value"], (3 * 10 + 7 * 20) / 30)
        self.assertIn("first sample interval", weighted["quality_warnings"][0])

    def test_incomplete_intervals_do_not_fabricate_time_weighting(self) -> None:
        base = [
            _resource(
                "resource.system.memory_used",
                100,
                0,
                interval_ns=None,
            ),
            _resource(
                "resource.system.memory_used",
                200,
                10,
                interval_ns=9,
            ),
        ]
        summary = summarize_resources(_loaded(base))[0]
        weighted = _aggregate(summary, ".time_weighted_mean")
        self.assertEqual(weighted["availability"], "not_available")
        self.assertIn("exactly tile", weighted["unavailable_reason"])
        self.assertEqual(_aggregate(summary, ".mean")["value"], 150)

        one = summarize_resources(_loaded(base[:1]))[0]
        self.assertEqual(one["coverage_ns"], 0)
        self.assertEqual(
            _aggregate(one, ".time_weighted_mean")["availability"],
            "not_available",
        )

    def test_unavailable_only_stream_has_no_numeric_aggregates(self) -> None:
        sample = _resource(
            "resource.cpu.utilization",
            None,
            5,
            interval_ns=None,
            availability=Availability.NOT_COLLECTED,
        )
        summary = summarize_resources(_loaded([sample]))[0]
        self.assertEqual(summary["available_sample_count"], 0)
        self.assertEqual(summary["availability_ratio"], 0.0)
        for aggregate in summary["aggregates"]:
            self.assertEqual(aggregate["availability"], "not_available")
            self.assertIsNone(aggregate["value"])

    def test_bool_nonfinite_and_unit_mismatch_are_rejected(self) -> None:
        good = _resource(
            "resource.gpu.power",
            10,
            0,
            interval_ns=None,
            device_type=DeviceType.GPU,
            device_id="gpu-0",
        )
        for value, message in (
            (True, "non-bool"),
            (float("nan"), "finite"),
            (float("inf"), "finite"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ResourceCalculationError, message):
                    summarize_resources(_loaded([replace(good, value=value)]))
        with self.assertRaisesRegex(ResourceCalculationError, "unit mismatch"):
            summarize_resources(_loaded([replace(good, unit="mW")]))
        with self.assertRaisesRegex(
            ResourceCalculationError, "unavailable sample must have value=null"
        ):
            summarize_resources(
                _loaded(
                    [
                        replace(
                            good,
                            availability=Availability.NOT_AVAILABLE,
                            value=0,
                        )
                    ]
                )
            )
        with self.assertRaisesRegex(ResourceCalculationError, "invalid availability"):
            summarize_resources(_loaded([replace(good, availability="unknown")]))

    def test_duplicate_timestamps_and_invalid_dimensions_are_rejected(self) -> None:
        first = _resource(
            "resource.npu.memory_used",
            0,
            0,
            interval_ns=None,
            device_type=DeviceType.NPU,
            device_id="npu-0",
        )
        with self.assertRaisesRegex(ResourceCalculationError, "duplicate timestamps"):
            summarize_resources(_loaded([first, replace(first, value=1)]))
        with self.assertRaisesRegex(ResourceCalculationError, "canonical JSON"):
            summarize_resources(
                _loaded([replace(first, dimensions={"bad": float("nan")})])
            )

    def test_dimensions_participate_in_stream_identity_deterministically(self) -> None:
        samples = [
            _resource(
                "resource.cpu.utilization",
                1,
                0,
                interval_ns=None,
                dimensions={"window": "capture", "z": 1},
            ),
            _resource(
                "resource.cpu.utilization",
                2,
                0,
                interval_ns=None,
                dimensions={"z": 2, "window": "request"},
            ),
        ]
        first = summarize_resources(_loaded(samples))
        second = summarize_resources(_loaded(list(reversed(samples))))
        self.assertEqual(first, second)
        self.assertEqual(
            [summary["scope"]["window"] for summary in first],
            ["capture", "request"],
        )

    def test_clock_nulls_propagate_without_assuming_zero(self) -> None:
        sample = _resource(
            "resource.gpu.memory_used",
            0,
            0,
            interval_ns=None,
            device_type=DeviceType.GPU,
            device_id="gpu-0",
        )
        sample = replace(sample, attributes={})
        summary = summarize_resources(_loaded([sample]))[0]
        self.assertEqual(summary["clock"]["alignment_status"], "unknown")
        self.assertIsNone(summary["clock"]["alignment_method"])
        self.assertIsNone(summary["clock"]["offset_ns"])
        self.assertIsNone(summary["clock"]["uncertainty_ns"])


if __name__ == "__main__":
    unittest.main()
