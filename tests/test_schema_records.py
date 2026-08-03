"""Enum, envelope, ID, and event record tests."""

from dataclasses import replace
import unittest

from perfetto_hetero_profiler.schema import (
    EventRecord,
    EventType,
    Phase,
    RecordType,
    SCHEMA_VERSION,
    SchemaValidationError,
    record_to_dict,
    validate_record,
)


def event(**changes):
    values = {
        "run_id": "run-1",
        "event_id": "event-1",
        "event_name": "request_received",
        "event_type": EventType.INSTANT,
        "phase": Phase.REQUEST,
        "host_id": "host-1",
        "clock_domain_id": "clock-1",
        "timestamp_ns": 10,
        "attributes": {},
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
    }
    values.update(changes)
    return EventRecord(**values)


class SchemaRecordTests(unittest.TestCase):
    def test_enum_values_serialize_as_strings(self) -> None:
        data = record_to_dict(event())
        self.assertEqual(data["event_type"], "instant")
        self.assertEqual(data["phase"], "request")

    def test_envelope_defaults(self) -> None:
        record = event()
        self.assertEqual(record.schema_version, SCHEMA_VERSION)
        self.assertIs(record.record_type, RecordType.EVENT)

    def test_empty_run_id_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.run_id"):
            validate_record(event(run_id=""))

    def test_normal_uuid_request_id_allowed(self) -> None:
        validate_record(event())

    def test_instant_duration_null(self) -> None:
        validate_record(event(duration_ns=None))

    def test_instant_duration_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.duration_ns"):
            validate_record(event(duration_ns=1))

    def test_span_nonnegative_duration(self) -> None:
        validate_record(event(event_type=EventType.SPAN, duration_ns=0))

    def test_span_negative_duration_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.duration_ns"):
            validate_record(event(event_type=EventType.SPAN, duration_ns=-1))

    def test_negative_timestamp_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.timestamp_ns"):
            validate_record(event(timestamp_ns=-1))

    def test_canonical_event_name_allowed(self) -> None:
        validate_record(event(event_name="first_token_emitted"))

    def test_namespaced_custom_event_allowed(self) -> None:
        validate_record(event(event_name="rbln.runtime_dispatch"))

    def test_collector_custom_event_allowed(self) -> None:
        validate_record(event(event_name="collector.child_process_start"))

    def test_unnamespaced_custom_event_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.event_name"):
            validate_record(event(event_name="runtime_dispatch"))

    def test_unapproved_event_namespace_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "event.event_name"):
            validate_record(event(event_name="typo.child_process_start"))

    def test_parent_cannot_reference_self(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "parent_event_id"):
            validate_record(replace(event(), parent_event_id="event-1"))

    def test_process_and_thread_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "process_id"):
            validate_record(event(process_id=-1))
