"""Validation tests for checked-in schema examples and JSON Schema docs."""

from dataclasses import fields
import json
from pathlib import Path
import unittest

from perfetto_hetero_profiler.schema import (
    ArtifactReference,
    Availability,
    ClockDomain,
    ClockTransform,
    DeviceType,
    EventRecord,
    MetricSample,
    RunManifest,
    RunMode,
    SyncPoint,
    read_json,
    read_jsonl,
)


EXAMPLES = Path("examples/schema_v1")
SCHEMAS = Path("src/perfetto_hetero_profiler/schema/json/v1")


class SchemaExampleTests(unittest.TestCase):
    def test_all_json_examples_validate(self) -> None:
        for path in sorted(EXAMPLES.glob("*.json")):
            with self.subTest(path=path):
                read_json(path)

    def test_all_jsonl_examples_validate(self) -> None:
        for path in sorted(EXAMPLES.glob("*.jsonl")):
            with self.subTest(path=path):
                self.assertGreater(len(read_jsonl(path)), 0)

    def test_gpu_only_manifest(self) -> None:
        manifest = read_json(EXAMPLES / "manifest_gpu_only.json")
        self.assertIsInstance(manifest, RunManifest)
        self.assertIs(manifest.mode, RunMode.GPU_ONLY)
        self.assertEqual({device.device_type for device in manifest.devices}, {DeviceType.GPU})

    def test_npu_only_manifest(self) -> None:
        manifest = read_json(EXAMPLES / "manifest_npu_only.json")
        self.assertIs(manifest.mode, RunMode.NPU_ONLY)
        self.assertEqual({device.device_type for device in manifest.devices}, {DeviceType.NPU})

    def test_hybrid_manifest(self) -> None:
        manifest = read_json(EXAMPLES / "manifest_hybrid.json")
        self.assertIs(manifest.mode, RunMode.HYBRID)
        self.assertEqual(
            {device.device_type for device in manifest.devices},
            {DeviceType.GPU, DeviceType.NPU},
        )
        self.assertTrue({"served", "prefill", "decode"}.issubset({m.role for m in manifest.models}))

    def test_event_flow_uses_one_request_id(self) -> None:
        records = read_jsonl(EXAMPLES / "events.jsonl")
        request_ids = {record.request_id for record in records}
        self.assertEqual(request_ids, {"123e4567-e89b-12d3-a456-426614174000"})
        self.assertEqual(len(records), 14)

    def test_metrics_include_unavailable_and_real_zero(self) -> None:
        records = read_jsonl(EXAMPLES / "metrics.jsonl")
        self.assertTrue(
            any(record.availability is Availability.NOT_AVAILABLE for record in records)
        )
        self.assertTrue(
            any(
                record.availability is Availability.AVAILABLE and record.value == 0
                for record in records
            )
        )

    def test_example_record_counts(self) -> None:
        expected = {
            "events.jsonl": 14,
            "metrics.jsonl": 8,
            "clocks.jsonl": 3,
            "artifacts.jsonl": 2,
        }
        self.assertEqual(
            {name: len(read_jsonl(EXAMPLES / name)) for name in expected},
            expected,
        )

    def test_json_schema_draft_and_dataclass_fields_match(self) -> None:
        mappings = {
            "run_manifest.schema.json": RunManifest,
            "event_record.schema.json": EventRecord,
            "metric_sample.schema.json": MetricSample,
            "artifact_reference.schema.json": ArtifactReference,
            "clock_domain.schema.json": ClockDomain,
            "sync_point.schema.json": SyncPoint,
            "clock_transform.schema.json": ClockTransform,
        }
        for filename, record_class in mappings.items():
            with self.subTest(filename=filename):
                data = json.loads((SCHEMAS / filename).read_text(encoding="utf-8"))
                self.assertEqual(
                    data["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertEqual(
                    set(data["properties"]),
                    {field.name for field in fields(record_class)},
                )
