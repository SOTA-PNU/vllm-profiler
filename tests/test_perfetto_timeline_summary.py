"""Perfetto processing-timeline planning and protobuf contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor

from perfetto_hetero_profiler.perfetto.loader import load_hybrid_run
from perfetto_hetero_profiler.perfetto.model import RequestWindowSpec
from perfetto_hetero_profiler.perfetto.native_details import request_focused_plan
from perfetto_hetero_profiler.perfetto.planner import (
    PerfettoPlanningError,
    build_trace_plan,
)
from perfetto_hetero_profiler.perfetto.timeline_summary import (
    TIMELINE_SUMMARY_MAPPING_VERSION,
    TIMELINE_SUMMARY_ROOT_NAME,
    build_timeline_summary_context,
    TimelineSummaryInputError,
)
from perfetto_hetero_profiler.perfetto.writer import build_trace, serialize_trace
from perfetto_hetero_profiler.schema import Availability, MetricScope
from tests.test_perfetto_conversion import _build_monitor_family


class PerfettoTimelineSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.family = _build_monitor_family(
            Path(cls._temporary.name),
            overview_metrics=True,
        )
        cls.loaded = load_hybrid_run(cls.family["hybrid"])
        cls.context = build_timeline_summary_context(cls.loaded)
        cls.result = build_trace_plan(
            cls.loaded.manifest,
            cls.loaded.events,
            cls.loaded.metrics,
            canonical_clock_domain_id=cls.loaded.canonical_clock_domain_id,
            native_envelopes=cls.loaded.native_envelopes,
            timeline_summary=cls.context,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_processing_hierarchy_and_versioned_uuid_are_deterministic(self):
        plan = self.result.plan
        tracks = plan.track_by_key
        root = tracks["summary.root"]
        self.assertEqual(plan.mapping_version, TIMELINE_SUMMARY_MAPPING_VERSION)
        self.assertEqual(root.name, TIMELINE_SUMMARY_ROOT_NAME)
        self.assertIsNone(root.parent_key)
        self.assertEqual(root.child_ordering, "explicit")
        self.assertEqual(
            [
                (tracks[key].parent_key, tracks[key].sibling_order_rank)
                for key in (
                    "summary.boundaries",
                    "summary.pipeline",
                    "summary.decode_details",
                )
            ],
            [("summary.root", index) for index in range(3)],
        )
        self.assertEqual(
            [
                tracks[key].sibling_order_rank
                for key in (
                    "gpu_prefill",
                    "kv_export",
                    "kv_transfer",
                    "kv_transform",
                    "npu_decode",
                )
            ],
            [0, 1, 4, 6, 8],
        )
        rebuilt = build_trace_plan(
            self.loaded.manifest,
            reversed(self.loaded.events),
            reversed(self.loaded.metrics),
            canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
            native_envelopes=self.loaded.native_envelopes,
            timeline_summary=self.context,
        )
        self.assertEqual(serialize_trace(plan), serialize_trace(rebuilt.plan))
        self.assertEqual(root.uuid, rebuilt.plan.track_by_key[root.key].uuid)

        changed = build_trace_plan(
            self.loaded.manifest,
            self.loaded.events,
            self.loaded.metrics,
            canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
            native_envelopes=self.loaded.native_envelopes,
            timeline_summary=replace(
                self.context,
                source_identity_sha256="f" * 64,
            ),
        ).plan
        self.assertNotEqual(root.uuid, changed.track_by_key[root.key].uuid)
        self.assertEqual(
            tracks["gpu_prefill"].uuid,
            changed.track_by_key["gpu_prefill"].uuid,
        )

    def test_timeline_contains_each_observed_processing_interval_once(self):
        plan = self.result.plan
        expected = {
            "GPU Prefill": "gpu_prefill",
            "KV Export": "kv_export",
            "KV Transfer": "kv_transfer",
            "KV Transform": "kv_transform",
            "NPU Decode": "npu_decode",
            "Decode Step 0": "npu_decode_step",
            "Sampling 0": "sampling",
        }
        for name, track_key in expected.items():
            rows = [row for row in plan.slices if row.name == name]
            self.assertEqual(len(rows), 1, name)
            self.assertEqual(rows[0].track_key, track_key)
        self.assertFalse(
            any(
                row.name in {"Hybrid Request", "Request Summary"}
                for row in plan.slices
            )
        )
        self.assertFalse(
            any(row.track_key.startswith("summary.kpi") for row in plan.counters)
        )
        self.assertFalse(any("Data Quality" in row.name for row in plan.instants))
        self.assertEqual(len(plan.flows), 5)

    def test_boundary_instants_are_evidence_only_and_non_sensitive(self):
        plan = self.result.plan
        boundaries = {
            dict(row.annotations)["hetero.boundary_kind"]: row
            for row in plan.instants
            if row.track_key == "summary.boundaries.events"
        }
        self.assertEqual(set(boundaries), {"request_received", "response_done"})
        self.assertEqual(boundaries["request_received"].name, "Request Received")
        self.assertEqual(boundaries["response_done"].name, "Response Completion")
        for row in boundaries.values():
            annotations = dict(row.annotations)
            self.assertEqual(
                annotations["hetero.correlation_id"],
                "correlation-1",
            )
            self.assertNotIn("prompt", annotations)
            self.assertNotIn("response", annotations)
            self.assertNotIn("token_text", annotations)

    def test_unclassified_gaps_are_metadata_not_fabricated_slices(self):
        gaps = self.result.plan.unclassified_gaps
        self.assertEqual(len(gaps), 2)
        self.assertEqual(
            [(gap.preceding_marker, gap.following_marker) for gap in gaps],
            [
                ("request_received", "prefill_start"),
                ("sampling_end", "response_done"),
            ],
        )
        self.assertTrue(all(gap.duration_ns > 0 and gap.reason for gap in gaps))
        self.assertFalse(
            any("Unclassified" in row.name for row in self.result.plan.slices)
        )

    def test_kpis_stay_in_official_trace_attributes(self):
        plan = self.result.plan
        self.assertTrue(plan.trace_attributes)
        self.assertEqual(
            plan.trace_attributes,
            tuple(sorted(plan.trace_attributes, key=lambda row: row.key)),
        )
        self.assertEqual(
            self.result.metadata.timeline_summary_kpi_counter_count,
            0,
        )
        self.assertFalse(
            any(row.track_key.startswith("summary.kpi") for row in plan.counters)
        )

    def test_resources_are_full_capture_diagnostics_without_copies(self):
        plan = self.result.plan
        resource_counters = [
            row for row in plan.counters if row.track_key.startswith("counter:")
        ]
        self.assertEqual(
            len(resource_counters),
            self.result.metadata.available_resource_metric_count,
        )
        self.assertEqual(len(plan.counters), len(resource_counters))
        self.assertEqual(
            {
                plan.track_by_key[row.track_key].parent_key
                for row in resource_counters
            },
            {
                "telemetry.resources.gpu.gpu-0",
                "telemetry.resources.npu.npu-0",
            },
        )
        self.assertIsNone(plan.track_by_key["telemetry.resources"].parent_key)

    def test_resource_tracks_use_explicit_memory_power_utilization_order(self):
        gpu = next(
            row
            for row in self.loaded.metrics
            if row.metric_name == "resource.gpu.utilization"
        )
        npu = next(
            row
            for row in self.loaded.metrics
            if row.metric_name == "resource.npu.utilization"
        )
        metrics = (
            replace(
                gpu,
                metric_name="resource.system.memory_used",
                scope=MetricScope.HOST,
                device_type=None,
                device_id=None,
                unit="bytes",
                value=1,
            ),
            replace(
                gpu,
                metric_name="resource.cpu.utilization",
                scope=MetricScope.HOST,
                device_type=None,
                device_id=None,
                unit="percent",
                value=2,
            ),
            replace(
                gpu,
                metric_name="resource.gpu.memory_used",
                unit="bytes",
                value=3,
            ),
            replace(gpu, metric_name="resource.gpu.power", unit="W", value=4),
            replace(gpu, value=5),
            replace(
                gpu,
                metric_name="resource.gpu.memory_used",
                device_id="gpu-1",
                unit="bytes",
                value=6,
            ),
            replace(
                npu,
                metric_name="resource.npu.memory_used",
                unit="bytes",
                value=7,
            ),
            replace(npu, metric_name="resource.npu.power", unit="W", value=8),
            replace(npu, value=9),
            replace(
                npu,
                metric_name="resource.npu.memory_used",
                device_id="npu-1",
                unit="bytes",
                value=10,
            ),
        )
        plan = build_trace_plan(
            self.loaded.manifest,
            self.loaded.events,
            metrics,
            canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
            timeline_summary=self.context,
        ).plan
        tracks = plan.track_by_key
        root_children = sorted(
            (
                track.sibling_order_rank,
                track.key,
            )
            for track in plan.tracks
            if track.parent_key == "telemetry.resources"
        )
        self.assertEqual(
            root_children,
            [
                (0, "telemetry.resources.cpu_system"),
                (100, "telemetry.resources.gpu.gpu-0"),
                (101, "telemetry.resources.gpu.gpu-1"),
                (200, "telemetry.resources.npu.npu-0"),
                (201, "telemetry.resources.npu.npu-1"),
            ],
        )
        expected = {
            "telemetry.resources.cpu_system": [
                "System memory [host-0]",
                "CPU utilization [host-0]",
            ],
            "telemetry.resources.gpu.gpu-0": [
                "GPU memory [gpu-0]",
                "GPU power [gpu-0]",
                "GPU utilization [gpu-0]",
            ],
            "telemetry.resources.npu.npu-0": [
                "NPU memory [npu-0]",
                "NPU power [npu-0]",
                "NPU utilization [npu-0]",
            ],
        }
        for group_key, names in expected.items():
            self.assertEqual(tracks[group_key].child_ordering, "explicit")
            children = sorted(
                (
                    track.sibling_order_rank,
                    track.name,
                )
                for track in plan.tracks
                if track.parent_key == group_key
            )
            self.assertEqual(children, list(enumerate(names)))

    def test_official_descriptors_encode_processing_parent_chain(self):
        plan = self.result.plan
        trace = build_trace(plan)
        descriptors = {
            packet.track_descriptor.uuid: packet.track_descriptor
            for packet in trace.packet
            if packet.HasField("track_descriptor")
        }
        root = plan.track_by_key["summary.root"]
        pipeline = plan.track_by_key["summary.pipeline"]
        stage = plan.track_by_key["gpu_prefill"]
        self.assertEqual(descriptors[root.uuid].parent_uuid, plan.process_uuid)
        self.assertEqual(
            descriptors[root.uuid].child_ordering,
            TrackDescriptor.EXPLICIT,
        )
        self.assertEqual(descriptors[pipeline.uuid].parent_uuid, root.uuid)
        self.assertEqual(descriptors[pipeline.uuid].sibling_order_rank, 1)
        self.assertEqual(descriptors[stage.uuid].parent_uuid, pipeline.uuid)
        self.assertEqual(descriptors[stage.uuid].sibling_order_rank, 0)

    def test_unknown_ui_mapping_is_rejected(self):
        with self.assertRaisesRegex(
            PerfettoPlanningError,
            "unsupported timeline summary mapping version",
        ):
            build_trace_plan(
                self.loaded.manifest,
                self.loaded.events,
                self.loaded.metrics,
                canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
                native_envelopes=self.loaded.native_envelopes,
                timeline_summary=replace(
                    self.context,
                    mapping_version="unknown-mapping",
                ),
            )


class MeasuredTokenInstantTests(unittest.TestCase):
    def test_valid_measured_token_timestamps_create_indexed_instants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                measured_token_timestamps=(2_250_000, 2_350_000),
            )
            measured = (
                family["gpu"] / "raw/client/measured_requests.jsonl"
            )
            before = (
                measured.stat().st_size,
                measured.stat().st_mtime_ns,
                measured.read_bytes(),
            )
            loaded = load_hybrid_run(family["hybrid"])
            context = build_timeline_summary_context(loaded)
            plan = build_trace_plan(
                loaded.manifest,
                loaded.events,
                loaded.metrics,
                canonical_clock_domain_id=loaded.canonical_clock_domain_id,
                timeline_summary=context,
            ).plan
            after = (
                measured.stat().st_size,
                measured.stat().st_mtime_ns,
                measured.read_bytes(),
            )
        tokens = [row for row in plan.instants if row.name.startswith("Output Token ")]
        self.assertEqual([row.name for row in tokens], ["Output Token 0", "Output Token 1"])
        self.assertEqual([row.timestamp_ns for row in tokens], [2_250_000, 2_350_000])
        self.assertTrue(
            all(
                dict(row.annotations)["hetero.timestamp_source"]
                == "valid_token_timestamps_ns"
                for row in tokens
            )
        )
        self.assertEqual(before, after)

    def test_client_token_instants_may_extend_past_proxy_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                measured_token_timestamps=(2_250_000, 2_350_000),
            )
            loaded = load_hybrid_run(family["hybrid"])
            client_observation = (
                json.dumps(
                    {
                        "request_id": "request-1",
                        "request_start_ns": 900_000,
                        "stream_end_ns": 2_600_000,
                        "output_tokens": 2,
                        "valid_token_timestamps_ns": [950_000, 2_550_000],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            with mock.patch(
                "perfetto_hetero_profiler.perfetto.timeline_summary."
                "_read_source_artifact",
                return_value=client_observation,
            ):
                context = build_timeline_summary_context(loaded)

        self.assertEqual(
            [item.timestamp_ns for item in context.token_instants],
            [950_000, 2_550_000],
        )

    def test_focused_plan_preserves_processing_tokens_flows_and_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                measured_token_timestamps=(2_250_000, 2_350_000),
            )
            loaded = load_hybrid_run(family["hybrid"])
            context = build_timeline_summary_context(loaded)
            full = build_trace_plan(
                loaded.manifest,
                loaded.events,
                loaded.metrics,
                canonical_clock_domain_id=loaded.canonical_clock_domain_id,
                timeline_summary=context,
            ).plan
            focused = request_focused_plan(full)

        expected_slices = tuple(
            row
            for row in full.slices
            if row.track_key not in {"request", "profiler"}
            and row.name not in {"Hybrid Request", "Request Summary"}
        )
        self.assertEqual(
            [
                (
                    row.track_key,
                    row.name,
                    row.timestamp_ns,
                    row.duration_ns,
                    row.annotations,
                )
                for row in focused.slices
            ],
            [
                (
                    row.track_key,
                    row.name,
                    row.timestamp_ns,
                    row.duration_ns,
                    row.annotations,
                )
                for row in expected_slices
            ],
        )
        self.assertEqual(focused.instants, full.instants)
        self.assertEqual(
            focused.flows,
            tuple(
                flow
                for flow in full.flows
                if flow.source_slice_name != "Request"
            ),
        )
        self.assertEqual(focused.trace_attributes, full.trace_attributes)
        attributes = {row.key: row.value for row in focused.trace_attributes}
        self.assertEqual(attributes["vllm_profiler.schema_version"], "1.1.0")
        self.assertFalse(any(key.endswith(".availability") for key in attributes))
        self.assertIsNotNone(focused.request_window)

    def test_unavailable_resource_sample_is_not_fabricated_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
            )
            loaded = load_hybrid_run(family["hybrid"])
            context = replace(
                build_timeline_summary_context(loaded),
                request_window=RequestWindowSpec(
                    request_id="request-1",
                    start_ns=1_000_000,
                    end_ns=2_500_000,
                    source_clock_domain_id="gpu:host-monotonic",
                    target_clock_domain_id=loaded.canonical_clock_domain_id,
                    alignment_method="same_clock_domain",
                    alignment_uncertainty_ns=0,
                ),
            )
            resource = next(
                row
                for row in loaded.metrics
                if row.metric_name.startswith("resource.")
            )
            unavailable = replace(
                resource,
                availability=Availability.NOT_AVAILABLE,
                value=None,
                reason="not reported",
                attributes={"telemetry.sample_role": "background"},
                interval_ns=10,
            )
            full = build_trace_plan(
                loaded.manifest,
                loaded.events,
                (unavailable,),
                canonical_clock_domain_id=loaded.canonical_clock_domain_id,
                timeline_summary=context,
            ).plan
            focused = request_focused_plan(full)

        self.assertFalse(full.counters)
        self.assertFalse(focused.counters)
        self.assertNotIn(
            "summary.request_resources",
            focused.track_by_key,
        )

    def test_non_monotonic_measured_token_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                measured_token_timestamps=(2_350_000, 2_250_000),
            )
            loaded = load_hybrid_run(family["hybrid"])
            with self.assertRaisesRegex(
                TimelineSummaryInputError,
                "strictly increasing",
            ):
                build_timeline_summary_context(loaded)


if __name__ == "__main__":
    unittest.main()
