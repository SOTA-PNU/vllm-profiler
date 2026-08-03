"""JSON and JSONL round-trip tests."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.schema import (
    EventRecord,
    EventType,
    Phase,
    SchemaValidationError,
    read_json,
    read_jsonl,
    record_from_dict,
    record_from_json,
    record_to_dict,
    record_to_json,
    write_json,
    write_jsonl,
)


def event(event_id: str = "event-1") -> EventRecord:
    return EventRecord(
        run_id="run-1",
        event_id=event_id,
        event_name="request_received",
        event_type=EventType.INSTANT,
        phase=Phase.REQUEST,
        host_id="host-1",
        clock_domain_id="clock-1",
        timestamp_ns=1,
        attributes={},
    )


class SerializationTests(unittest.TestCase):
    def test_dict_round_trip(self) -> None:
        original = event()
        self.assertEqual(record_from_dict(record_to_dict(original)), original)

    def test_json_round_trip(self) -> None:
        original = event()
        self.assertEqual(record_from_json(record_to_json(original)), original)

    def test_jsonl_multiple_round_trip_and_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            expected = [event("event-1"), event("event-2")]
            write_jsonl(path, expected)
            self.assertEqual(read_jsonl(path), expected)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_json_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            write_json(path, event())
            self.assertEqual(read_json(path), event())

    def test_unknown_record_type_rejected(self) -> None:
        data = record_to_dict(event())
        data["record_type"] = "unknown"
        with self.assertRaisesRegex(SchemaValidationError, "record_type"):
            record_from_dict(data)

    def test_unknown_major_version_rejected(self) -> None:
        data = record_to_dict(event())
        data["schema_version"] = "2.0.0"
        with self.assertRaisesRegex(SchemaValidationError, "unsupported schema major"):
            record_from_dict(data)

    def test_same_major_minor_version_accepted(self) -> None:
        data = record_to_dict(event())
        data["schema_version"] = "1.7.3"
        self.assertEqual(record_from_dict(data).schema_version, "1.7.3")

    def test_writer_rejects_non_current_same_major_version(self) -> None:
        record = event()
        record.schema_version = "1.7.3"
        with self.assertRaisesRegex(SchemaValidationError, "writer only supports"):
            record_to_dict(record)

    def test_nan_json_input_rejected(self) -> None:
        text = record_to_json(event()).replace('"timestamp_ns":1', '"timestamp_ns":NaN')
        with self.assertRaisesRegex(SchemaValidationError, "non-finite"):
            record_from_json(text)

    def test_nan_attribute_output_rejected(self) -> None:
        record = event()
        record.attributes["vendor.bad"] = math.nan
        with self.assertRaisesRegex(SchemaValidationError, "NaN"):
            record_to_json(record)

    def test_exclusive_write_rejects_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            write_json(path, event())
            with self.assertRaises(FileExistsError):
                write_json(path, event())

    def test_duplicate_event_id_in_jsonl_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            line = record_to_json(event())
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaValidationError, "unique within a run"):
                read_jsonl(path)

    def test_unknown_top_level_field_rejected(self) -> None:
        data = record_to_dict(event())
        data["surprise"] = True
        with self.assertRaisesRegex(SchemaValidationError, "unknown field"):
            record_from_dict(data)

    def test_missing_schema_version_rejected(self) -> None:
        data = record_to_dict(event())
        del data["schema_version"]
        with self.assertRaisesRegex(SchemaValidationError, "schema_version"):
            record_from_dict(data)
