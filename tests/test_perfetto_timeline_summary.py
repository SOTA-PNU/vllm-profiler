"""Trace-native Perfetto timeline-summary planning and protobuf contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor

from perfetto_hetero_profiler.perfetto.loader import load_hybrid_run
from perfetto_hetero_profiler.perfetto.planner import (
    PerfettoPlanningError,
    build_trace_plan,
)
from perfetto_hetero_profiler.perfetto.timeline_summary import (
    TIMELINE_SUMMARY_MAPPING_VERSION,
    TIMELINE_SUMMARY_ROOT_NAME,
    build_timeline_summary_context,
)
from perfetto_hetero_profiler.perfetto.writer import (
    build_trace,
    serialize_trace,
)
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

    def test_root_hierarchy_ordering_and_versioned_uuid_are_deterministic(self):
        plan = self.result.plan
        tracks = plan.track_by_key
        root = tracks["summary.root"]
        self.assertEqual(plan.mapping_version, TIMELINE_SUMMARY_MAPPING_VERSION)
        self.assertEqual(root.name, TIMELINE_SUMMARY_ROOT_NAME)
        self.assertIsNone(root.parent_key)
        self.assertEqual(root.child_ordering, "explicit")
        self.assertEqual(
            [
                (
                    tracks[key].parent_key,
                    tracks[key].sibling_order_rank,
                )
                for key in (
                    "summary.request_summary",
                    "summary.pipeline",
                    "summary.kpi.token_throughput",
                    "summary.kpi.transfer",
                    "summary.data_quality",
                )
            ],
            [("summary.root", index) for index in range(5)],
        )
        self.assertEqual(
            [
                tracks[f"summary.pipeline.{name}"].sibling_order_rank
                for name in (
                    "gpu_prefill",
                    "kv_export",
                    "kv_transfer",
                    "kv_transform",
                    "npu_decode",
                )
            ],
            list(range(5)),
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

        changed_identity = replace(
            self.context,
            source_identity_sha256="f" * 64,
        )
        changed = build_trace_plan(
            self.loaded.manifest,
            self.loaded.events,
            self.loaded.metrics,
            canonical_clock_domain_id=self.loaded.canonical_clock_domain_id,
            native_envelopes=self.loaded.native_envelopes,
            timeline_summary=changed_identity,
        ).plan
        self.assertNotEqual(
            root.uuid,
            changed.track_by_key["summary.root"].uuid,
        )
        self.assertEqual(
            tracks["gpu_prefill"].uuid,
            changed.track_by_key["gpu_prefill"].uuid,
        )

    def test_summary_intervals_copy_detail_and_never_copy_flow(self):
        plan = self.result.plan
        summary_to_detail = {
            "summary.request_summary": "request",
            "summary.pipeline.gpu_prefill": "gpu_prefill",
            "summary.pipeline.kv_export": "kv_export",
            "summary.pipeline.kv_transfer": "kv_transfer",
            "summary.pipeline.kv_transform": "kv_transform",
            "summary.pipeline.npu_decode": "npu_decode",
        }
        for summary_key, detail_key in summary_to_detail.items():
            summary = [row for row in plan.slices if row.track_key == summary_key]
            detail = [row for row in plan.slices if row.track_key == detail_key]
            self.assertEqual(len(summary), 1)
            self.assertEqual(len(detail), 1)
            self.assertEqual(
                (summary[0].timestamp_ns, summary[0].duration_ns),
                (detail[0].timestamp_ns, detail[0].duration_ns),
            )
            self.assertFalse(summary[0].begin_flow_ids)
            self.assertFalse(summary[0].end_flow_ids)
            self.assertFalse(summary[0].begin_terminating_flow_ids)
            self.assertFalse(summary[0].end_terminating_flow_ids)

        request = next(
            row
            for row in plan.slices
            if row.track_key == "summary.request_summary"
        )
        annotations = dict(request.annotations)
        self.assertIn("hetero.request_facing_e2e_ns", annotations)
        self.assertIn("hetero.pipeline_e2e_ns", annotations)
        self.assertEqual(
            request.duration_ns,
            annotations["hetero.pipeline_e2e_ns"],
        )
        self.assertIn("hetero.ttft_ns", annotations)
        tpot = next(
            item
            for item in self.context.kpis
            if item.identity == "request_facing_latency:latency.tpot"
        )
        if tpot.available:
            self.assertEqual(annotations["hetero.tpot_ns"], tpot.value)
        else:
            self.assertNotIn("hetero.tpot_ns", annotations)
        self.assertIn("hetero.request_id", annotations)
        self.assertIn("hetero.correlation_id", annotations)
        self.assertEqual(
            len(json.loads(annotations["hetero.source_event_ids_json"])),
            2,
        )
        self.assertFalse(
            any(
                row.name in {"TPOT", "Sampling total"}
                for row in plan.slices
            )
        )
        self.assertEqual(len(plan.flows), 5)
        self.assertFalse(
            any(
                row.track_key.startswith("summary.")
                and (
                    row.begin_flow_ids
                    or row.end_flow_ids
                    or row.begin_terminating_flow_ids
                    or row.end_terminating_flow_ids
                )
                for row in plan.slices
            )
        )

    def test_kpis_omit_unavailable_values_and_preserve_units(self):
        plan = self.result.plan
        kpi_counters = [
            row
            for row in plan.counters
            if row.track_key.startswith("summary.kpi:")
        ]
        available = [item for item in self.context.kpis if item.available]
        unavailable = [item for item in self.context.kpis if not item.available]
        self.assertEqual(len(kpi_counters), len(available))
        self.assertEqual(
            {row.track_key.removeprefix("summary.kpi:") for row in kpi_counters},
            {item.identity for item in available},
        )
        for row in kpi_counters:
            self.assertNotIsInstance(row.value, bool)
            self.assertTrue(math.isfinite(float(row.value)))
            annotations = dict(row.annotations)
            self.assertEqual(annotations["hetero.availability"], "available")
            self.assertEqual(
                annotations["hetero.canonical_unit"],
                plan.track_by_key[row.track_key].unit,
            )
            self.assertIn("hetero.anchor_event_id", annotations)
            self.assertIn("hetero.source_event_ids_json", annotations)

        quality = next(
            row
            for row in plan.instants
            if row.track_key == "summary.data_quality"
        )
        quality_annotations = dict(quality.annotations)
        reasons = json.loads(
            quality_annotations["hetero.unavailable_kpis_json"]
        )
        self.assertEqual(
            quality_annotations["hetero.unavailable_kpi_count"],
            len(unavailable),
        )
        self.assertEqual(set(reasons), {item.identity for item in unavailable})
        self.assertTrue(all(reasons.values()))
        self.assertEqual(
            quality_annotations["hetero.perfetto_validation"],
            "required_pinned_official_trace_processor_before_publication",
        )

    def test_resources_are_reparented_without_copying_samples(self):
        plan = self.result.plan
        resource_counters = [
            row
            for row in plan.counters
            if row.track_key.startswith("counter:")
        ]
        self.assertEqual(
            len(resource_counters),
            self.result.metadata.available_resource_metric_count,
        )
        self.assertEqual(
            len(plan.counters),
            len(resource_counters)
            + self.result.metadata.timeline_summary_kpi_counter_count,
        )
        parents = {
            plan.track_by_key[row.track_key].parent_key
            for row in resource_counters
        }
        self.assertEqual(
            parents,
            {
                "telemetry.resources.gpu.gpu-0",
                "telemetry.resources.npu.npu-0",
            },
        )
        self.assertEqual(
            plan.track_by_key["telemetry.resources.gpu.gpu-0"].name,
            "GPU",
        )
        self.assertEqual(
            plan.track_by_key["telemetry.resources.npu.npu-0"].name,
            "NPU 0",
        )
        self.assertIsNone(
            plan.track_by_key["telemetry.resources"].parent_key,
        )

    def test_official_descriptors_encode_parent_chain_and_counter_units(self):
        plan = self.result.plan
        trace = build_trace(plan)
        descriptors = {
            packet.track_descriptor.uuid: packet.track_descriptor
            for packet in trace.packet
            if packet.HasField("track_descriptor")
        }
        root = plan.track_by_key["summary.root"]
        pipeline = plan.track_by_key["summary.pipeline"]
        stage = plan.track_by_key["summary.pipeline.gpu_prefill"]
        self.assertEqual(descriptors[root.uuid].parent_uuid, plan.process_uuid)
        self.assertEqual(
            descriptors[root.uuid].child_ordering,
            TrackDescriptor.EXPLICIT,
        )
        self.assertEqual(descriptors[pipeline.uuid].parent_uuid, root.uuid)
        self.assertEqual(descriptors[pipeline.uuid].sibling_order_rank, 1)
        self.assertEqual(descriptors[stage.uuid].parent_uuid, pipeline.uuid)
        self.assertEqual(descriptors[stage.uuid].sibling_order_rank, 0)
        kpi = plan.track_by_key[
            "summary.kpi:request_facing_latency:latency.e2e"
        ]
        self.assertEqual(descriptors[kpi.uuid].counter.unit_name, "")
        self.assertNotEqual(descriptors[kpi.uuid].counter.unit, 0)

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


if __name__ == "__main__":
    unittest.main()
