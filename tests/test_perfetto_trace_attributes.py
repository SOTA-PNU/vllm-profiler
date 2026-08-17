"""Official trace-level performance attribute export contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.overview.calculation import calculate_overview_kpis
from perfetto_hetero_profiler.perfetto.loader import load_hybrid_run
from perfetto_hetero_profiler.perfetto.model import TraceAttributeSpec
from perfetto_hetero_profiler.perfetto.planner import build_trace_plan
from perfetto_hetero_profiler.perfetto.timeline_summary import (
    build_timeline_summary_context,
)
from perfetto_hetero_profiler.perfetto.trace_attributes import (
    LEGACY_TRACE_ATTRIBUTE_NAMESPACE,
    TRACE_ATTRIBUTE_NAMESPACE,
    TRACE_ATTRIBUTE_SCHEMA_VERSION,
    TraceAttributeExportError,
    build_performance_trace_attributes,
    fixed_point_half_even,
    trace_attribute_validation_report,
)
from perfetto_hetero_profiler.perfetto.writer import build_trace, serialize_trace
from perfetto_hetero_profiler.schema import DeviceType

from tests.test_perfetto_conversion import _build_monitor_family


def _attribute_map(attributes):
    return {item.key: item.value for item in attributes}


class PerfettoTraceAttributeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.family = _build_monitor_family(
            Path(cls._temporary.name),
            overview_metrics=True,
        )
        cls.loaded = load_hybrid_run(cls.family["hybrid"])
        cls.calculated = calculate_overview_kpis(cls.loaded)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_allowlist_namespace_types_sorting_and_privacy(self) -> None:
        attributes = build_performance_trace_attributes(
            self.loaded,
            self.calculated,
        )
        keys = [item.key for item in attributes]
        values = _attribute_map(attributes)
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key.startswith(TRACE_ATTRIBUTE_NAMESPACE) for key in keys))
        self.assertEqual(TRACE_ATTRIBUTE_NAMESPACE, "vllm_profiler.")
        self.assertFalse(
            any(key.startswith(LEGACY_TRACE_ATTRIBUTE_NAMESPACE) for key in keys)
        )
        self.assertTrue(
            all(isinstance(item.value, (int, str)) and not isinstance(item.value, bool)
                for item in attributes)
        )
        self.assertEqual(
            values[f"{TRACE_ATTRIBUTE_NAMESPACE}schema_version"],
            TRACE_ATTRIBUTE_SCHEMA_VERSION,
        )
        self.assertIn(
            f"{TRACE_ATTRIBUTE_NAMESPACE}kpi.latency.e2e.value_ns",
            values,
        )
        self.assertFalse(any(key.endswith(".availability") for key in keys))
        required_bases = (
            "kpi.latency.e2e",
            "kpi.latency.ttft",
            "kpi.latency.tpot",
            "kpi.latency.prefill",
            "kpi.latency.decode",
            "kpi.throughput.requests",
            "kpi.throughput.output_tokens",
            "transfer.kv_export_duration",
            "transfer.kv_transfer_duration",
            "transfer.kv_transform_duration",
            "transfer.bytes",
            "transfer.effective_bandwidth",
            "transfer.e2e_share",
            "resource.prefill.cpu.utilization_mean",
            "resource.prefill.cpu.utilization_peak",
            "resource.prefill.gpu_0.utilization_mean",
            "resource.prefill.gpu_0.utilization_peak",
            "resource.prefill.gpu_0.memory_peak",
            "resource.prefill.system_memory_peak",
            "resource.decode.cpu.utilization_mean",
            "resource.decode.cpu.utilization_peak",
            "resource.decode.npu_0.utilization_mean",
            "resource.decode.npu_0.utilization_peak",
            "resource.decode.npu_0.memory_peak",
            "resource.decode.system_memory_peak",
        )
        for base in required_bases:
            self.assertTrue(
                any(
                    key.startswith(f"{TRACE_ATTRIBUTE_NAMESPACE}{base}.value_")
                    for key in values
                ),
                base,
            )
        exported = json.dumps(values, sort_keys=True)
        for forbidden in (
            "/home/",
            "prompt",
            "response",
            "sha256",
            "artifact",
            "synthetic-hybrid",
            "host-0",
        ):
            self.assertNotIn(forbidden, exported)
        self.assertFalse(any(key.startswith("hetero.") for key in keys))

    def test_fatal_shutdown_is_explicitly_labeled_as_demo_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                fatal_shutdown=True,
            )
            loaded = load_hybrid_run(family["hybrid"])
            values = _attribute_map(
                build_performance_trace_attributes(
                    loaded,
                    calculate_overview_kpis(loaded),
                )
            )
        self.assertEqual(values[f"{TRACE_ATTRIBUTE_NAMESPACE}demo_only"], "true")
        self.assertEqual(
            values[f"{TRACE_ATTRIBUTE_NAMESPACE}source.inference_status"],
            "succeeded",
        )
        self.assertEqual(
            values[f"{TRACE_ATTRIBUTE_NAMESPACE}source.shutdown_integrity"],
            "invalid",
        )
        self.assertEqual(
            values[f"{TRACE_ATTRIBUTE_NAMESPACE}source.shutdown_reason"],
            "native_sigsegv_rtnl_tc_unregister",
        )

    def test_fixed_point_uses_decimal_half_even(self) -> None:
        self.assertEqual(
            fixed_point_half_even(42.497, multiplier=1_000, field="rate"),
            42_497,
        )
        self.assertEqual(
            fixed_point_half_even(42.4965, multiplier=1_000, field="rate"),
            42_496,
        )
        self.assertEqual(
            fixed_point_half_even(42.4975, multiplier=1_000, field="rate"),
            42_498,
        )

    def test_unavailable_uses_value_key_but_real_zero_is_preserved(self) -> None:
        unavailable = copy.deepcopy(self.calculated)
        transfer = next(
            item for item in unavailable["transfer"] if item["name"] == "transfer.bytes"
        )
        transfer.update(
            availability="not_available",
            value=None,
            unavailable_reason="canonical transfer size is unavailable",
            sample_count=3,
        )
        values = _attribute_map(
            build_performance_trace_attributes(self.loaded, unavailable)
        )
        base = f"{TRACE_ATTRIBUTE_NAMESPACE}transfer.bytes"
        self.assertNotIn(f"{base}.availability", values)
        self.assertEqual(values[f"{base}.value_bytes"], "not_available")
        self.assertIn(f"{base}.reason", values)
        self.assertEqual(values[f"{base}.sample_count"], 3)
        self.assertEqual(
            values[f"{base}.aggregation"],
            transfer["aggregation_method"],
        )

        zero = copy.deepcopy(self.calculated)
        transfer = next(
            item for item in zero["transfer"] if item["name"] == "transfer.bytes"
        )
        transfer.update(
            availability="available",
            value=0,
            unavailable_reason=None,
            sample_count=1,
        )
        values = _attribute_map(build_performance_trace_attributes(self.loaded, zero))
        self.assertNotIn(f"{base}.availability", values)
        self.assertEqual(values[f"{base}.value_bytes"], 0)
        self.assertNotIn(f"{base}.reason", values)

    def test_canonical_unit_mismatch_is_rejected(self) -> None:
        calculated = copy.deepcopy(self.calculated)
        transfer = next(
            item
            for item in calculated["transfer"]
            if item["name"] == "transfer.bytes"
        )
        transfer["canonical_unit"] = "KiB"
        with self.assertRaisesRegex(
            TraceAttributeExportError,
            "contract differs",
        ):
            build_performance_trace_attributes(self.loaded, calculated)

    def test_stage_windows_and_multiple_npu_devices_are_separate(self) -> None:
        second_npu = replace(
            self.loaded.manifest.devices[-1],
            device_type=DeviceType.NPU,
            device_id="npu-9",
        )
        loaded = replace(
            self.loaded,
            manifest=replace(
                self.loaded.manifest,
                devices=tuple(self.loaded.manifest.devices) + (second_npu,),
            ),
        )
        calculated = copy.deepcopy(self.calculated)
        calculated["resource_summaries"].extend(
            (
                {
                    "metric_name": "resource.gpu.utilization",
                    "scope": {
                        "phase": "prefill",
                        "window": "prefill",
                        "device_type": "gpu",
                        "device_id": "gpu-0",
                    },
                    "aggregates": [
                        {
                            "name": "resource.gpu.utilization.mean",
                            "canonical_unit": "percent",
                            "availability": "available",
                            "value": 12.3455,
                            "unavailable_reason": None,
                            "aggregation_method": "arithmetic_mean_v1",
                            "sample_count": 4,
                        }
                    ],
                },
                {
                    "metric_name": "resource.npu.memory_used",
                    "scope": {
                        "phase": "decode",
                        "window": "decode",
                        "device_type": "npu",
                        "device_id": "npu-9",
                    },
                    "aggregates": [
                        {
                            "name": "resource.npu.memory_used.max",
                            "canonical_unit": "bytes",
                            "availability": "available",
                            "value": 4096,
                            "unavailable_reason": None,
                            "aggregation_method": "maximum_v1",
                            "sample_count": 3,
                        }
                    ],
                },
            )
        )
        values = _attribute_map(build_performance_trace_attributes(loaded, calculated))
        gpu = f"{TRACE_ATTRIBUTE_NAMESPACE}resource.prefill.gpu_0.utilization_mean"
        self.assertEqual(values[f"{gpu}.value_milli_percent"], 12_346)
        self.assertEqual(values[f"{gpu}.sample_count"], 4)
        npu = f"{TRACE_ATTRIBUTE_NAMESPACE}resource.decode.npu_1.memory_peak"
        self.assertEqual(values[f"{npu}.value_bytes"], 4096)
        self.assertEqual(values[f"{npu}.sample_count"], 3)
        npu0 = f"{TRACE_ATTRIBUTE_NAMESPACE}resource.decode.npu_0.memory_peak"
        self.assertNotIn(f"{npu0}.availability", values)
        self.assertEqual(values[f"{npu0}.value_bytes"], "not_available")

    def test_official_packet_is_sorted_and_duplicate_keys_are_rejected(self) -> None:
        context = build_timeline_summary_context(self.loaded)
        plan = build_trace_plan(
            self.loaded.manifest,
            self.loaded.events,
            self.loaded.metrics,
            canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
            native_envelopes=self.loaded.native_envelopes,
            timeline_summary=context,
        ).plan
        trace = build_trace(plan)
        packets = [packet for packet in trace.packet if packet.HasField("trace_attributes")]
        self.assertEqual(len(packets), 1)
        keys = [item.key for item in packets[0].trace_attributes.attribute]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(serialize_trace(plan), serialize_trace(plan))

        duplicate = replace(
            plan,
            trace_attributes=plan.trace_attributes + (plan.trace_attributes[0],),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            serialize_trace(duplicate)

        same_key_different_oneof = replace(
            plan,
            trace_attributes=(
                TraceAttributeSpec(
                    key=f"{TRACE_ATTRIBUTE_NAMESPACE}test.value_ns",
                    value=0,
                ),
                TraceAttributeSpec(
                    key=f"{TRACE_ATTRIBUTE_NAMESPACE}test.value_ns",
                    value="not_available",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            serialize_trace(same_key_different_oneof)

        for invalid in (True, 1.5, float("nan"), float("inf")):
            invalid_type = replace(
                plan,
                trace_attributes=(
                    TraceAttributeSpec(
                        key=f"{TRACE_ATTRIBUTE_NAMESPACE}test.value_ns",
                        value=invalid,  # type: ignore[arg-type]
                    ),
                ),
            )
            with self.assertRaisesRegex(TypeError, "integer or string"):
                serialize_trace(invalid_type)

    def test_detached_validation_counts_exact_sql_result(self) -> None:
        attributes = build_performance_trace_attributes(
            self.loaded,
            self.calculated,
        )
        report = trace_attribute_validation_report(
            attributes,
            {
                "queries": [
                    {
                        "name": "trace_attributes",
                        "matched": True,
                        "row_count": len(attributes),
                        "rows_sha256": "a" * 64,
                    }
                ]
            },
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["attribute_count"], len(attributes))
        self.assertEqual(
            report["integer_count"] + report["string_count"],
            len(attributes),
        )

    def test_legacy_trace_without_attributes_remains_valid(self) -> None:
        report = trace_attribute_validation_report((), {"queries": []})
        self.assertTrue(report["valid"])
        self.assertEqual(report["attribute_count"], 0)
        self.assertEqual(
            report["trace_processor_query_matched"],
            "not_applicable_legacy_trace",
        )

    def test_schema_1_0_availability_rows_remain_validation_compatible(self) -> None:
        attributes = tuple(
            sorted(
                (
                    TraceAttributeSpec(
                        key=f"{TRACE_ATTRIBUTE_NAMESPACE}schema_version",
                        value="1.0.0",
                    ),
                    TraceAttributeSpec(
                        key=(
                            f"{TRACE_ATTRIBUTE_NAMESPACE}"
                            "kpi.latency.e2e.availability"
                        ),
                        value="available",
                    ),
                    TraceAttributeSpec(
                        key=(
                            f"{TRACE_ATTRIBUTE_NAMESPACE}"
                            "kpi.latency.e2e.value_ns"
                        ),
                        value=1,
                    ),
                ),
                key=lambda item: item.key,
            ),
        )
        report = trace_attribute_validation_report(
            attributes,
            {
                "queries": [
                    {
                        "name": "trace_attributes",
                        "matched": True,
                        "row_count": len(attributes),
                        "rows_sha256": "a" * 64,
                    }
                ]
            },
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["schema_version"], "1.0.0")

    def test_legacy_namespace_schema_1_0_remains_validation_compatible(self) -> None:
        attributes = tuple(sorted((
            TraceAttributeSpec(
                key=f"{LEGACY_TRACE_ATTRIBUTE_NAMESPACE}schema_version",
                value="1.0.0",
            ),
            TraceAttributeSpec(
                key=(
                    f"{LEGACY_TRACE_ATTRIBUTE_NAMESPACE}"
                    "kpi.latency.e2e.availability"
                ),
                value="available",
            ),
        ), key=lambda item: item.key))
        report = trace_attribute_validation_report(
            attributes,
            {
                "queries": [
                    {
                        "name": "trace_attributes",
                        "matched": True,
                        "row_count": len(attributes),
                        "rows_sha256": "a" * 64,
                    }
                ]
            },
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["namespace"], LEGACY_TRACE_ATTRIBUTE_NAMESPACE)

    def test_schema_1_1_rejects_availability_rows(self) -> None:
        attributes = (
            TraceAttributeSpec(
                key=f"{TRACE_ATTRIBUTE_NAMESPACE}schema_version",
                value=TRACE_ATTRIBUTE_SCHEMA_VERSION,
            ),
            TraceAttributeSpec(
                key=f"{TRACE_ATTRIBUTE_NAMESPACE}kpi.latency.e2e.availability",
                value="available",
            ),
        )
        report = trace_attribute_validation_report(
            attributes,
            {
                "queries": [
                    {
                        "name": "trace_attributes",
                        "matched": True,
                        "row_count": len(attributes),
                        "rows_sha256": "a" * 64,
                    }
                ]
            },
        )
        self.assertFalse(report["valid"])
        self.assertIn(
            "schema 1.1.0 must not contain availability keys",
            report["mismatches"],
        )


if __name__ == "__main__":
    unittest.main()
