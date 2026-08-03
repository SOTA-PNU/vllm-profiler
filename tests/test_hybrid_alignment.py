"""Canonical timestamp conversion tests."""

from dataclasses import replace
import unittest

from perfetto_hetero_profiler.hybrid.alignment import (
    AlignmentError,
    TimestampTransform,
    align_event,
    align_event_stream,
    align_metric,
    align_metric_stream,
    align_timestamp,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceType,
    MetricKind,
    MetricSample,
    MetricScope,
    ValueOrigin,
    validate_record,
)

from tests.hybrid_fixtures import event


def transform(**overrides):
    values = {
        "source_clock_domain_id": "source",
        "target_clock_domain_id": "canonical",
        "offset_ns": -50,
        "uncertainty_ns": 7,
        "method": "ntp_style_four_timestamp",
    }
    values.update(overrides)
    return TimestampTransform(**values)


class TimestampAlignmentTests(unittest.TestCase):
    def test_gpu_timestamp_conversion(self):
        aligned = align_timestamp(100, "source", transform(offset_ns=0))
        self.assertEqual(aligned.timestamp_ns, 100)

    def test_npu_timestamp_conversion(self):
        aligned = align_timestamp(100, "source", transform())
        self.assertEqual(aligned.timestamp_ns, 50)

    def test_original_timestamp_preserved(self):
        aligned = align_timestamp(100, "source", transform())
        self.assertEqual(aligned.original_timestamp_ns, 100)

    def test_clock_domain_changes(self):
        aligned = align_timestamp(100, "source", transform())
        self.assertEqual(aligned.target_clock_domain_id, "canonical")

    def test_negative_target_rejected(self):
        with self.assertRaisesRegex(AlignmentError, "negative"):
            align_timestamp(10, "source", transform(offset_ns=-20))

    def test_wrong_domain_rejected(self):
        with self.assertRaisesRegex(AlignmentError, "does not match"):
            align_timestamp(10, "wrong", transform())

    def test_unavailable_alignment_rejected(self):
        unavailable = transform(available=False, reason="no clock probes")
        with self.assertRaisesRegex(AlignmentError, "no clock probes"):
            align_timestamp(100, "source", unavailable)

    def test_unavailable_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            transform(available=False)


class RecordAlignmentTests(unittest.TestCase):
    def source_event(self, timestamp=100):
        return event(
            run_id="gpu-run",
            event_name="request_received",
            timestamp_ns=timestamp,
            host_id="gpu-host",
            clock_domain_id="source",
            device_type=DeviceType.GPU,
        )

    def test_event_provenance_and_run_id(self):
        aligned = align_event(
            self.source_event(),
            hybrid_run_id="hybrid",
            source_role="gpu",
            transform=transform(offset_ns=0),
        )
        self.assertEqual(aligned.run_id, "hybrid")
        self.assertEqual(aligned.event_id, "gpu:request_received-100")
        self.assertEqual(aligned.attributes["hybrid.original_timestamp_ns"], 100)
        self.assertEqual(
            aligned.attributes["hybrid.original_clock_domain_id"], "source"
        )
        validate_record(aligned)

    def test_parent_and_source_event_ids_are_namespaced(self):
        source = replace(self.source_event(), parent_event_id="parent")
        aligned = align_event(
            source,
            hybrid_run_id="hybrid",
            source_role="gpu",
            transform=transform(offset_ns=0),
        )
        self.assertEqual(aligned.parent_event_id, "gpu:parent")

    def test_metric_preserves_device_and_uncertainty(self):
        metric = MetricSample(
            run_id="gpu-run",
            metric_name="resource.gpu.utilization",
            metric_kind=MetricKind.GAUGE,
            scope=MetricScope.DEVICE,
            host_id="gpu-host",
            clock_domain_id="source",
            timestamp_ns=100,
            availability=Availability.AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit="percent",
            value=10,
            device_type=DeviceType.GPU,
            device_id="gpu-0",
            source_event_ids=["event"],
            dimensions={},
            attributes={},
        )
        aligned = align_metric(
            metric,
            hybrid_run_id="hybrid",
            source_role="gpu",
            transform=transform(offset_ns=0),
        )
        self.assertEqual((aligned.device_type, aligned.device_id), (DeviceType.GPU, "gpu-0"))
        self.assertEqual(aligned.source_event_ids, ["gpu:event"])
        self.assertEqual(aligned.attributes["hybrid.alignment_uncertainty_ns"], 7)
        validate_record(aligned)

    def test_stream_order_is_preserved(self):
        rows = [self.source_event(100), self.source_event(200)]
        aligned = align_event_stream(
            rows,
            hybrid_run_id="hybrid",
            source_role="gpu",
            transforms={"source": transform()},
        )
        self.assertEqual([row.timestamp_ns for row in aligned], [50, 150])

    def test_decreasing_source_event_stream_is_rejected(self):
        rows = [self.source_event(200), self.source_event(100)]
        with self.assertRaisesRegex(AlignmentError, "event stream"):
            align_event_stream(
                rows,
                hybrid_run_id="hybrid",
                source_role="gpu",
                transforms={"source": transform()},
            )

    def test_decreasing_source_metric_stream_is_rejected(self):
        rows = [
            MetricSample(
                run_id="gpu-run",
                metric_name="resource.gpu.utilization",
                metric_kind=MetricKind.GAUGE,
                scope=MetricScope.DEVICE,
                host_id="gpu-host",
                clock_domain_id="source",
                timestamp_ns=timestamp,
                availability=Availability.AVAILABLE,
                origin=ValueOrigin.MEASURED,
                unit="percent",
                value=10,
                device_type=DeviceType.GPU,
                device_id="gpu-0",
                dimensions={},
                attributes={},
            )
            for timestamp in (200, 100)
        ]
        with self.assertRaisesRegex(AlignmentError, "metric stream"):
            align_metric_stream(
                rows,
                hybrid_run_id="hybrid",
                source_role="gpu",
                transforms={"source": transform()},
            )

    def test_missing_transform_rejected(self):
        with self.assertRaisesRegex(AlignmentError, "no transform"):
            align_event_stream(
                [self.source_event()],
                hybrid_run_id="hybrid",
                source_role="gpu",
                transforms={},
            )


if __name__ == "__main__":
    unittest.main()
