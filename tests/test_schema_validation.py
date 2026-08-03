"""Metric availability, catalog, and time validation tests."""

import math
import unittest

from perfetto_hetero_profiler.schema import (
    Availability,
    ClockDomain,
    ClockTransform,
    ClockType,
    DeviceType,
    MetricKind,
    MetricSample,
    MetricScope,
    SchemaValidationError,
    SyncMethod,
    SyncPoint,
    ValueOrigin,
    validate_record,
)


def metric(**changes):
    values = {
        "run_id": "run-1",
        "metric_name": "resource.gpu.utilization",
        "metric_kind": MetricKind.GAUGE,
        "scope": MetricScope.DEVICE,
        "host_id": "host-1",
        "clock_domain_id": "clock-1",
        "timestamp_ns": 100,
        "availability": Availability.AVAILABLE,
        "origin": ValueOrigin.MEASURED,
        "unit": "percent",
        "value": 50,
        "dimensions": {},
        "attributes": {},
        "device_type": DeviceType.GPU,
        "device_id": "gpu0",
    }
    values.update(changes)
    return MetricSample(**values)


def transform(**changes):
    values = {
        "run_id": "run-1",
        "transform_id": "transform-1",
        "source_clock_domain_id": "clock-a",
        "target_clock_domain_id": "clock-b",
        "scale": 1.0,
        "offset_ns": -10,
        "uncertainty_ns": 5,
        "method": SyncMethod.RPC_MIDPOINT,
        "valid_from_source_ns": 0,
        "valid_to_source_ns": None,
        "attributes": {},
    }
    values.update(changes)
    return ClockTransform(**values)


class MetricValidationTests(unittest.TestCase):
    def test_available_value(self) -> None:
        validate_record(metric())

    def test_available_null_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "metric.value"):
            validate_record(metric(value=None))

    def test_unavailable_null_with_reason(self) -> None:
        validate_record(
            metric(
                availability=Availability.NOT_AVAILABLE,
                value=None,
                reason="source does not expose this field",
            )
        )

    def test_unavailable_numeric_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "metric.value"):
            validate_record(
                metric(
                    availability=Availability.ERROR,
                    value=1,
                    reason="collector failed",
                )
            )

    def test_unavailable_without_reason_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "metric.reason"):
            validate_record(
                metric(availability=Availability.NOT_COLLECTED, value=None)
            )

    def test_nan_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "finite"):
            validate_record(metric(value=math.nan))

    def test_infinity_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "finite"):
            validate_record(metric(value=math.inf))

    def test_actual_zero_allowed(self) -> None:
        validate_record(metric(value=0))

    def test_percent_range(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "<= 100"):
            validate_record(metric(value=100.1))

    def test_ratio_range(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "<= 1"):
            validate_record(
                metric(
                    metric_name="transfer.e2e_share",
                    metric_kind=MetricKind.RATIO,
                    scope=MetricScope.TRANSFER,
                    unit="ratio",
                    value=1.1,
                    device_type=None,
                    device_id=None,
                )
            )

    def test_negative_bytes_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, ">= 0"):
            validate_record(
                metric(
                    metric_name="transfer.bytes",
                    metric_kind=MetricKind.COUNT,
                    scope=MetricScope.TRANSFER,
                    unit="bytes",
                    value=-1,
                    device_type=None,
                    device_id=None,
                )
            )

    def test_catalog_unit_mismatch(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "metric.unit"):
            validate_record(metric(unit="ratio"))

    def test_unknown_metric_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "metric.metric_name"):
            validate_record(metric(metric_name="queue.depth"))

    def test_vendor_metric_allowed(self) -> None:
        validate_record(
            metric(
                metric_name="vendor.queue_depth",
                unit="count",
                value=3,
            )
        )


class TimeValidationTests(unittest.TestCase):
    def test_clock_domain(self) -> None:
        validate_record(
            ClockDomain(
                run_id="run-1",
                clock_domain_id="clock-1",
                host_id="host-1",
                clock_type=ClockType.MONOTONIC,
                unit="ns",
                monotonic=True,
                adjustable=False,
                attributes={},
            )
        )

    def test_sync_point_negative_uncertainty_rejected(self) -> None:
        point = SyncPoint(
            run_id="run-1",
            sync_point_id="sync-1",
            source_clock_domain_id="clock-a",
            target_clock_domain_id="clock-b",
            source_timestamp_ns=1,
            target_timestamp_ns=2,
            method=SyncMethod.SHARED_EVENT,
            uncertainty_ns=-1,
            attributes={},
        )
        with self.assertRaisesRegex(SchemaValidationError, "uncertainty_ns"):
            validate_record(point)

    def test_transform_zero_scale_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "scale"):
            validate_record(transform(scale=0))

    def test_transform_negative_scale_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "scale"):
            validate_record(transform(scale=-1))

    def test_transform_formula_fields(self) -> None:
        record = transform(scale=1.5, offset_ns=-5)
        validate_record(record)
        self.assertEqual(int(10 * record.scale + record.offset_ns), 10)

    def test_clock_domains_not_implicitly_equal(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "must differ"):
            validate_record(
                transform(
                    source_clock_domain_id="same-clock",
                    target_clock_domain_id="same-clock",
                )
            )
