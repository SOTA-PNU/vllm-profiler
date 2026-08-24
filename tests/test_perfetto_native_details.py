"""CPU-only contracts for native-detail conversion."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from decimal import Decimal
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

from perfetto_hetero_profiler.cli import build_parser
from perfetto_hetero_profiler.perfetto.converter import _native_profile_metadata
from perfetto_hetero_profiler.perfetto.model import (
    CounterSpec,
    FlowSpec,
    InstantSpec,
    RequestWindowSpec,
    SliceSpec,
    TrackSpec,
    TracePlan,
)
from perfetto_hetero_profiler.perfetto.native_details import (
    NativeDetailError,
    NativeDetailResult,
    NativeDetailSummary,
    _NSYS_GLOBAL_PID_MASK,
    _NativeSlice,
    _attach_explicit_flows,
    _chrome_category,
    _microseconds_to_ns,
    _nsys_api_category,
    _rbln_native_only_result,
    augment_trace_plan,
    native_validation_metadata,
    request_focused_plan,
)
from perfetto_hetero_profiler.perfetto.planner import NativeProfileEnvelope
from perfetto_hetero_profiler.perfetto.validation import _expected_rows
from perfetto_hetero_profiler.perfetto.writer import (
    _pending_events,
    serialize_trace,
)


def _request_window(start: int = 100, end: int = 200) -> RequestWindowSpec:
    return RequestWindowSpec(
        request_id="one",
        start_ns=start,
        end_ns=end,
        source_clock_domain_id="gpu:mono",
        target_clock_domain_id="mono",
        alignment_method="same_clock_domain",
        alignment_uncertainty_ns=0,
    )


def _track(
    key: str,
    uuid: int,
    *,
    parent: str | None = None,
    rank: int | None = None,
    ordering: str = "unknown",
    kind: str = "slice",
) -> TrackSpec:
    return TrackSpec(
        key=key,
        uuid=uuid,
        name=key,
        kind=kind,
        description=key,
        parent_key=parent,
        sibling_order_rank=rank,
        child_ordering=ordering,
        unit="count" if kind == "counter" else None,
    )


def _summary(*, event_count: int = 1) -> NativeDetailSummary:
    return NativeDetailSummary(
        profiler_type="gpu_torch",
        source_role="gpu",
        support_status="converted",
        alignment_status="partial_derived",
        alignment_method="evidence",
        native_clock_domain="gpu-native",
        native_timestamp_unit="chrome_trace_microseconds",
        emitted_event_count=event_count,
        emitted_slice_count=event_count,
        emitted_instant_count=0,
        emitted_flow_count=0,
        metadata_only_event_count=0,
        skipped_event_count=0,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=3,
        clock_offset_ns=4,
        observed_offset_half_range_ns=1,
        native_epoch_base_ns=5,
        clock_sample_offsets_ns=(4,),
        canonical_transform_offset_ns=0,
        clock_formula="canonical_ns = native_ns - offset_ns",
        alignment_valid_interval_ns=(0, 100),
        mapped_event_interval_ns=(10, 20),
        event_counts=(("operators", event_count),),
        artifact_count=1,
        artifact_sha256=("a" * 64,),
    )


class NativeTimestampTests(unittest.TestCase):
    def test_decimal_microseconds_preserve_integer_nanoseconds_exactly(self):
        self.assertEqual(
            _microseconds_to_ns(Decimal("2035071083668.508")),
            2_035_071_083_668_508,
        )
        self.assertEqual(_microseconds_to_ns(Decimal("1")), 1_000)
        with self.assertRaisesRegex(NativeDetailError, "finer"):
            _microseconds_to_ns(Decimal("1.0001"))

    def test_same_boundary_nested_slices_keep_stack_order(self):
        track = _track("native.test.lane", 2)
        plan = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=(track,),
            slices=(
                SliceSpec(
                    track_key=track.key,
                    name="z-outer-same-start",
                    timestamp_ns=0,
                    duration_ns=10,
                ),
                SliceSpec(
                    track_key=track.key,
                    name="a-inner-same-start",
                    timestamp_ns=0,
                    duration_ns=5,
                ),
                SliceSpec(
                    track_key=track.key,
                    name="a-outer-same-end",
                    timestamp_ns=20,
                    duration_ns=10,
                ),
                SliceSpec(
                    track_key=track.key,
                    name="z-inner-same-end",
                    timestamp_ns=25,
                    duration_ns=5,
                ),
            ),
            instants=(),
            counters=(),
            flows=(),
        )
        pending = _pending_events(plan, plan.track_by_key)
        self.assertEqual(
            [
                item.name
                for item in pending
                if item.timestamp_ns == 0
                and item.event_type == TrackEvent.TYPE_SLICE_BEGIN
            ],
            ["z-outer-same-start", "a-inner-same-start"],
        )
        self.assertEqual(
            [
                item.name
                for item in pending
                if item.timestamp_ns == 30
                and item.event_type == TrackEvent.TYPE_SLICE_END
            ],
            ["z-inner-same-end", "a-outer-same-end"],
        )

    def test_unrepresentable_same_lane_intervals_fail_closed(self):
        track = _track("native.test.lane", 2)

        def plan_with(slices: tuple[SliceSpec, ...]) -> TracePlan:
            return TracePlan(
                run_id="run",
                canonical_clock_domain_id="mono",
                process_uuid=1,
                process_id=1,
                packet_sequence_id=1,
                tracks=(track,),
                slices=slices,
                instants=(),
                counters=(),
                flows=(),
            )

        identical = (
            SliceSpec(
                track_key=track.key,
                name="first",
                timestamp_ns=0,
                duration_ns=10,
            ),
            SliceSpec(
                track_key=track.key,
                name="second",
                timestamp_ns=0,
                duration_ns=10,
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate slice intervals"):
            serialize_trace(plan_with(identical))

        crossing = (
            SliceSpec(
                track_key=track.key,
                name="first",
                timestamp_ns=0,
                duration_ns=10,
            ),
            SliceSpec(
                track_key=track.key,
                name="second",
                timestamp_ns=5,
                duration_ns=10,
            ),
        )
        with self.assertRaisesRegex(ValueError, "crossing slice intervals"):
            serialize_trace(plan_with(crossing))

    def test_cli_help_discloses_native_and_request_focused_modes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(["convert", "perfetto", "--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--include-native-details", stdout.getvalue())
        self.assertIn("--request-focused", stdout.getvalue())


class NativeFlowTests(unittest.TestCase):
    def test_native_category_classification_fails_closed(self):
        self.assertEqual(_nsys_api_category(0), "CUDA Runtime API")
        self.assertEqual(_nsys_api_category(1), "CUDA Driver API")
        with self.assertRaisesRegex(NativeDetailError, "eventClass"):
            _nsys_api_category(2)
        with self.assertRaisesRegex(NativeDetailError, "GPU/device"):
            _chrome_category("npu_vllm", "kernel", "device work", "X")
        self.assertEqual(
            _chrome_category(
                "npu_vllm",
                "vendor_unknown",
                "unknown work",
                "X",
            ),
            (
                "Other NPU vLLM events (device identity unverified)",
                "unknown",
            ),
        )

    def test_nsight_global_tid_uses_official_global_pid_scope_mask(self):
        global_pid = 328_326_543_572_992
        global_tid = 328_326_546_365_563
        self.assertEqual(global_tid & _NSYS_GLOBAL_PID_MASK, global_pid)

    def test_only_unique_explicit_correlations_become_flows(self):
        def row(
            name: str,
            timestamp: int,
            correlation: int,
            endpoint: str,
            scope: str | None = None,
        ) -> _NativeSlice:
            return _NativeSlice(
                spec=SliceSpec(
                    track_key=f"native.test.{endpoint}",
                    name=name,
                    timestamp_ns=timestamp,
                    duration_ns=10,
                    annotations=(("native.correlation_id", correlation),),
                ),
                category=endpoint,
                correlation_id=correlation,
                endpoint_kind=endpoint,
                correlation_scope=scope,
            )

        rows = [
            row("api-1", 10, 1, "host_api"),
            row("kernel-1", 20, 1, "device"),
            row("api-2a", 30, 2, "host_api"),
            row("api-2b", 31, 2, "host_api"),
            row("kernel-2", 40, 2, "device"),
            row("api-sentinel-negative", 50, -1, "host_api"),
            row("kernel-sentinel-negative", 60, -1, "device"),
            row("api-sentinel-zero", 70, 0, "host_api"),
            row("kernel-sentinel-zero", 80, 0, "device"),
        ]
        converted, flows = _attach_explicit_flows(
            "run",
            "gpu_torch",
            rows,
        )
        self.assertEqual(len(flows), 1)
        self.assertEqual(
            flows[0].correlation_id,
            "gpu_torch:single-artifact:1",
        )
        flow_id = flows[0].flow_id
        self.assertEqual(converted[0].spec.begin_flow_ids, (flow_id,))
        self.assertEqual(
            converted[1].spec.begin_terminating_flow_ids,
            (flow_id,),
        )
        self.assertFalse(converted[2].spec.begin_flow_ids)
        self.assertFalse(converted[4].spec.begin_terminating_flow_ids)
        self.assertTrue(
            all(
                not item.spec.begin_flow_ids
                and not item.spec.begin_terminating_flow_ids
                for item in converted[5:]
            )
        )

    def test_correlation_ids_do_not_cross_artifact_boundaries(self):
        source = _NativeSlice(
            spec=SliceSpec(
                track_key="native.test.host",
                name="api",
                timestamp_ns=10,
                duration_ns=10,
            ),
            category="host",
            correlation_id=7,
            endpoint_kind="host_api",
            correlation_scope="artifact:0",
        )
        destination = _NativeSlice(
            spec=SliceSpec(
                track_key="native.test.device",
                name="kernel",
                timestamp_ns=20,
                duration_ns=10,
            ),
            category="device",
            correlation_id=7,
            endpoint_kind="device",
            correlation_scope="artifact:1",
        )
        converted, flows = _attach_explicit_flows(
            "run",
            "gpu_torch",
            (source, destination),
        )
        self.assertFalse(flows)
        self.assertFalse(converted[0].spec.begin_flow_ids)
        self.assertFalse(converted[1].spec.begin_terminating_flow_ids)


class NativeMetadataTests(unittest.TestCase):
    def test_unaligned_rbln_envelope_does_not_claim_zero_uncertainty(self):
        plan = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=(_track("profiler", 2),),
            slices=(
                SliceSpec(
                    track_key="profiler",
                    name="RBLN profiler capture",
                    timestamp_ns=1,
                    duration_ns=2,
                    annotations=(
                        ("hetero.profiler_type", "npu_rbln"),
                    ),
                ),
            ),
            instants=(),
            counters=(),
            flows=(),
        )
        summary = replace(
            _summary(event_count=0),
            profiler_type="npu_rbln",
            support_status="separate_native_perfetto_trace_unaligned",
            alignment_status="partial_unaligned",
            alignment_method="none_no_clock_snapshot_or_shared_anchor",
            alignment_uncertainty_ns=None,
            clock_offset_ns=None,
            observed_offset_half_range_ns=None,
            native_epoch_base_ns=None,
            clock_sample_offsets_ns=(),
            canonical_transform_offset_ns=None,
            clock_formula=None,
            alignment_valid_interval_ns=None,
            mapped_event_interval_ns=None,
        )
        augmented = augment_trace_plan(
            plan,
            NativeDetailResult(summaries=(summary,)),
        )
        annotations = dict(augmented.slices[0].annotations)
        self.assertNotIn(
            "hetero.native_alignment_uncertainty_ns",
            annotations,
        )
        envelope = NativeProfileEnvelope(
            profiler_type="npu_rbln",
            source_role="npu",
            timestamp_ns=1,
            duration_ns=2,
            alignment_status="partial",
            alignment_method="host_api_boundary_bracket",
            uncertainty_ns=3,
            native_clock_domain="rbln-native",
            native_timestamp_unit="ns",
            artifact_count=1,
            opaque_rbln_pb=True,
        )
        loaded = SimpleNamespace(
            native_envelopes=(envelope,),
            source_by_role={
                "npu": SimpleNamespace(
                    artifacts=(
                        SimpleNamespace(
                            relative_path="raw/report.pb",
                            clock_domain_id="rbln-native",
                            format="rbln_report",
                            size_bytes=10,
                            sha256="a" * 64,
                        ),
                    )
                )
            },
        )
        metadata = _native_profile_metadata(
            loaded,
            NativeDetailResult(summaries=(summary,)),
        )
        self.assertFalse(metadata[0]["opaque_rbln_pb"])
        self.assertEqual(
            metadata[0]["rbln_pb_classification"],
            "perfetto_compatible_rbln_trace",
        )
        envelope_only = _native_profile_metadata(
            loaded,
            NativeDetailResult(),
        )
        self.assertFalse(envelope_only[0]["opaque_rbln_pb"])
        self.assertEqual(
            envelope_only[0]["rbln_pb_classification"],
            "perfetto_compatible_rbln_trace",
        )
        self.assertEqual(
            envelope_only[0]["rbln_pb_structure_analysis"],
            "deferred_to_official_trace_processor",
        )

    def test_emitted_alignment_and_sql_lineage_override_envelope_defaults(self):
        artifacts = (
            SimpleNamespace(
                relative_path="raw/report.nsys-rep",
                clock_domain_id="gpu-native",
                format="nsys-rep",
                size_bytes=10,
                sha256="a" * 64,
            ),
            SimpleNamespace(
                relative_path="raw/report.sqlite",
                clock_domain_id="host-monotonic",
                format="sqlite",
                size_bytes=20,
                sha256="b" * 64,
            ),
        )
        envelope = NativeProfileEnvelope(
            profiler_type="gpu_nsys",
            source_role="gpu",
            timestamp_ns=1,
            duration_ns=2,
            alignment_status="partial",
            alignment_method="host_api_boundary_bracket",
            uncertainty_ns=3,
            native_clock_domain="gpu-native",
            native_timestamp_unit="nsight-report-native",
            artifact_count=1,
        )
        summary = replace(
            _summary(),
            profiler_type="gpu_nsys",
            native_timestamp_unit="nsight-report-native",
            artifact_count=2,
            artifact_sha256=("a" * 64, "b" * 64),
        )
        loaded = SimpleNamespace(
            native_envelopes=(envelope,),
            source_by_role={
                "gpu": SimpleNamespace(artifacts=artifacts)
            },
        )
        metadata = _native_profile_metadata(
            loaded,
            NativeDetailResult(summaries=(summary,)),
        )
        self.assertEqual(
            metadata[0]["native_event_alignment"],
            "partial_derived",
        )
        self.assertTrue(metadata[0]["native_details_emitted"])
        self.assertEqual(
            {
                item["relative_path"]
                for item in metadata[0]["artifact_references"]
            },
            {"raw/report.nsys-rep", "raw/report.sqlite"},
        )

    def test_equal_numeric_ids_in_distinct_artifacts_do_not_collide(self):
        rows = []
        for scope, start in (("artifact:0", 10), ("artifact:1", 30)):
            for endpoint, timestamp in (
                ("host_api", start),
                ("device", start + 10),
            ):
                rows.append(
                    _NativeSlice(
                        spec=SliceSpec(
                            track_key=f"native.test.{endpoint}",
                            name=f"{scope}-{endpoint}",
                            timestamp_ns=timestamp,
                            duration_ns=5,
                        ),
                        category=endpoint,
                        correlation_id=7,
                        endpoint_kind=endpoint,
                        correlation_scope=scope,
                    )
                )
        _, flows = _attach_explicit_flows("run", "gpu_torch", rows)
        self.assertCountEqual(
            [flow.correlation_id for flow in flows],
            [
                "gpu_torch:artifact:0:7",
                "gpu_torch:artifact:1:7",
            ],
        )


class RequestFocusedTests(unittest.TestCase):
    @staticmethod
    def _resource_plan(counters: tuple[CounterSpec, ...]) -> TracePlan:
        tracks = (
            _track("summary.root", 2, ordering="explicit"),
            _track(
                "summary.pipeline",
                3,
                parent="summary.root",
                rank=1,
                ordering="explicit",
            ),
            _track("gpu_prefill", 4, parent="summary.pipeline", rank=0),
            _track("telemetry.resources", 5, ordering="explicit"),
            _track(
                "telemetry.resources.cpu_system",
                6,
                parent="telemetry.resources",
                rank=0,
                ordering="explicit",
            ),
            _track(
                "telemetry.resources.gpu.gpu-0",
                7,
                parent="telemetry.resources",
                rank=100,
                ordering="explicit",
            ),
            _track(
                "telemetry.resources.npu.npu-0",
                8,
                parent="telemetry.resources",
                rank=200,
                ordering="explicit",
            ),
            _track(
                "counter.system.memory",
                9,
                parent="telemetry.resources.cpu_system",
                rank=0,
                kind="counter",
            ),
            _track(
                "counter.gpu.power",
                10,
                parent="telemetry.resources.gpu.gpu-0",
                rank=1,
                kind="counter",
            ),
            _track(
                "counter.npu.utilization",
                11,
                parent="telemetry.resources.npu.npu-0",
                rank=2,
                kind="counter",
            ),
        )
        return TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=tracks,
            slices=(
                SliceSpec(
                    track_key="gpu_prefill",
                    name="GPU Prefill",
                    timestamp_ns=120,
                    duration_ns=20,
                ),
            ),
            instants=(),
            counters=counters,
            flows=(),
            request_window=_request_window(),
        )

    def test_request_resource_selection_uses_exact_client_window_rules(self):
        def sample(
            timestamp: int,
            value: int,
            role: str,
            interval: int | None,
        ) -> CounterSpec:
            return CounterSpec(
                track_key="counter.system.memory",
                timestamp_ns=timestamp,
                value=value,
                interval_ns=interval,
                sample_role=role,
            )

        duplicate = sample(150, 5, "background", 50)
        plan = self._resource_plan(
            (
                sample(50, 1, "background", 10),
                sample(80, 2, "baseline", 10),
                sample(90, 3, "baseline", 10),
                sample(100, 4, "background", 10),
                duplicate,
                duplicate,
                sample(200, 6, "final", 10),
                sample(205, 7, "final", 10),
                sample(210, 8, "background", 20),
                sample(220, 9, "background", 20),
            )
        )
        focused = request_focused_plan(plan)
        self.assertEqual(
            [
                (row.timestamp_ns, row.value, row.interval_ns, row.sample_role)
                for row in focused.counters
            ],
            [
                (90, 3, 10, "baseline"),
                (100, 4, 10, "background"),
                (150, 5, 50, "background"),
                (200, 6, 10, "final"),
                (210, 8, 20, "background"),
            ],
        )
        self.assertNotIn(50, {row.timestamp_ns for row in focused.counters})
        self.assertNotIn(220, {row.timestamp_ns for row in focused.counters})

    def test_resource_streams_are_independent_and_boundaries_are_not_synthesized(self):
        plan = self._resource_plan(
            (
                CounterSpec(
                    "counter.gpu.power",
                    150,
                    1.5,
                    interval_ns=30,
                    sample_role="background",
                ),
                CounterSpec(
                    "counter.npu.utilization",
                    90,
                    0.0,
                    interval_ns=20,
                    sample_role="baseline",
                ),
                CounterSpec(
                    "counter.npu.utilization",
                    205,
                    0.0,
                    interval_ns=20,
                    sample_role="final",
                ),
            )
        )
        focused = request_focused_plan(plan)
        by_track = {
            key: [row.sample_role for row in focused.counters if row.track_key == key]
            for key in {row.track_key for row in focused.counters}
        }
        self.assertEqual(by_track["counter.gpu.power"], ["background"])
        self.assertEqual(
            by_track["counter.npu.utilization"],
            ["baseline", "final"],
        )

    def test_request_resource_tracks_keep_explicit_semantic_order(self):
        plan = self._resource_plan(
            (
                CounterSpec(
                    "counter.system.memory", 150, 1,
                    interval_ns=20, sample_role="background",
                ),
                CounterSpec(
                    "counter.gpu.power", 150, 2,
                    interval_ns=20, sample_role="background",
                ),
                CounterSpec(
                    "counter.npu.utilization", 150, 3,
                    interval_ns=20, sample_role="background",
                ),
            )
        )
        focused = request_focused_plan(plan)
        tracks = focused.track_by_key
        root = tracks["summary.request_resources"]
        self.assertEqual(root.name, "Request-window Resource Telemetry")
        self.assertEqual(root.parent_key, "summary.root")
        self.assertEqual(root.sibling_order_rank, 3)
        self.assertEqual(
            [
                (track.name, track.sibling_order_rank)
                for track in focused.tracks
                if track.parent_key == root.key
            ],
            [
                ("telemetry.resources.cpu_system", 0),
                ("telemetry.resources.gpu.gpu-0", 100),
                ("telemetry.resources.npu.npu-0", 200),
            ],
        )

    def test_focus_preserves_timestamps_and_prunes_telemetry_and_half_flows(self):
        tracks = (
            _track("summary.root", 2, ordering="explicit"),
            _track("summary.boundaries", 3, parent="summary.root", rank=0),
            _track(
                "summary.boundaries.events",
                4,
                parent="summary.boundaries",
            ),
            _track("native.test", 5, parent="summary.root", rank=3),
            _track("native.test.lane", 6, parent="native.test"),
            _track("telemetry.resources", 7),
            _track(
                "counter.cpu",
                8,
                parent="telemetry.resources",
                kind="counter",
            ),
        )
        flow_id = 99
        inside = SliceSpec(
            track_key="native.test.lane",
            name="inside",
            timestamp_ns=120,
            duration_ns=10,
            begin_flow_ids=(flow_id,),
        )
        outside = SliceSpec(
            track_key="native.test.lane",
            name="outside",
            timestamp_ns=250,
            duration_ns=10,
            begin_terminating_flow_ids=(flow_id,),
        )
        plan = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=tracks,
            slices=(inside, outside),
            instants=(
                InstantSpec(
                    track_key="summary.boundaries.events",
                    name="Request Received",
                    timestamp_ns=100,
                    annotations=(
                        ("hetero.boundary_kind", "request_received"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
                InstantSpec(
                    track_key="summary.boundaries.events",
                    name="Response Completion",
                    timestamp_ns=200,
                    annotations=(
                        ("hetero.boundary_kind", "response_done"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
            ),
            counters=(
                CounterSpec(
                    track_key="counter.cpu",
                    timestamp_ns=110,
                    value=1,
                ),
            ),
            flows=(
                FlowSpec(
                    flow_id=flow_id,
                    source_slice_name="inside",
                    destination_slice_name="outside",
                    correlation_id="one",
                ),
            ),
            request_window=_request_window(),
        )
        focused = request_focused_plan(plan)
        self.assertEqual(
            [(item.name, item.timestamp_ns) for item in focused.slices],
            [("inside", 120)],
        )
        self.assertTrue(focused.presentation_mode)
        self.assertEqual(len(focused.instants), 2)
        self.assertFalse(focused.counters)
        self.assertFalse(focused.flows)
        retained_inside = next(
            item for item in focused.slices if item.name == "inside"
        )
        self.assertFalse(retained_inside.begin_flow_ids)
        self.assertNotIn(
            "telemetry.resources",
            {item.key for item in focused.tracks},
        )

    def test_touching_boundary_and_profiler_envelope_are_excluded(self):
        tracks = (
            _track("summary.root", 2, ordering="explicit"),
            _track("summary.boundaries", 3, parent="summary.root"),
            _track(
                "summary.boundaries.events",
                4,
                parent="summary.boundaries",
            ),
            _track("profiler", 5),
            _track("native.gpu_torch", 6, parent="summary.root"),
            _track("native.gpu_torch.lane", 7, parent="native.gpu_torch"),
        )
        plan = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=tracks,
            slices=(
                SliceSpec(
                    track_key="profiler",
                    name="GPU profiler capture",
                    timestamp_ns=1,
                    duration_ns=300,
                    annotations=(
                        ("hetero.profiler_type", "gpu_torch"),
                        ("hetero.native_details_emitted", True),
                        ("hetero.unaligned_profiler_events", False),
                    ),
                ),
                SliceSpec(
                    track_key="native.gpu_torch.lane",
                    name="touches request end",
                    timestamp_ns=200,
                    duration_ns=10,
                ),
            ),
            instants=(
                InstantSpec(
                    track_key="summary.boundaries.events",
                    name="Request Received",
                    timestamp_ns=100,
                    annotations=(
                        ("hetero.boundary_kind", "request_received"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
                InstantSpec(
                    track_key="summary.boundaries.events",
                    name="Response Completion",
                    timestamp_ns=200,
                    annotations=(
                        ("hetero.boundary_kind", "response_done"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
                InstantSpec(
                    track_key="native.gpu_torch.lane",
                    name="instant touches request end",
                    timestamp_ns=200,
                ),
            ),
            counters=(),
            flows=(),
            request_window=_request_window(),
        )
        focused = request_focused_plan(plan)
        self.assertNotIn(
            "native.gpu_torch",
            {item.key for item in focused.tracks},
        )
        self.assertNotIn(
            "instant touches request end",
            {item.name for item in focused.instants},
        )
        self.assertNotIn("profiler", {item.key for item in focused.tracks})
        self.assertEqual(
            {item.name for item in focused.instants},
            {"Request Received", "Response Completion"},
        )


class NativeValidationTests(unittest.TestCase):
    def test_native_sql_semantics_query_is_added_only_for_native_events(self):
        track = _track("work", 2)
        plain = SliceSpec(
            track_key=track.key,
            name="plain",
            timestamp_ns=10,
            duration_ns=10,
        )
        base = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=(track,),
            slices=(plain,),
            instants=(),
            counters=(),
            flows=(),
        )
        self.assertNotIn("native_event_semantics", _expected_rows(base))

        native = replace(
            base,
            slices=(
                replace(
                    plain,
                    annotations=(
                        ("hetero.native_profiler", "gpu_torch"),
                        ("hetero.timestamp_fallback", False),
                        ("hetero.fabricated_event", False),
                    ),
                ),
            ),
        )
        self.assertIn("native_event_semantics", _expected_rows(native))

    def test_full_counts_are_exact_and_parent_range_is_unavailable(self):
        root = _track("native.gpu_torch", 2)
        lane = _track(
            "native.gpu_torch.lane",
            3,
            parent="native.gpu_torch",
        )
        event = SliceSpec(
            track_key=lane.key,
            name="operator",
            timestamp_ns=10,
            duration_ns=10,
            annotations=(
                ("hetero.native_profiler", "gpu_torch"),
                ("hetero.timestamp_fallback", False),
                ("hetero.fabricated_event", False),
            ),
        )
        native = NativeDetailResult(
            tracks=(root, lane),
            slices=(event,),
            summaries=(_summary(),),
        )
        plan = TracePlan(
            run_id="run",
            canonical_clock_domain_id="mono",
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=(root, lane),
            slices=(event,),
            instants=(),
            counters=(),
            flows=(),
        )
        report = native_validation_metadata(plan, native)
        self.assertTrue(report["valid"])
        self.assertTrue(report["native_counts_reconciled"])
        self.assertTrue(report["native_identity_reconciled"])
        evidence = report["clock_alignment_evidence"][0]
        self.assertEqual(
            evidence["formula"],
            "canonical_ns = native_ns - offset_ns",
        )
        self.assertFalse(evidence["clock_error_bound_proven"])
        self.assertEqual(evidence["mapped_event_interval_ns"], [10, 20])
        self.assertIsNone(report["parent_child_range_violation_count"])
        self.assertEqual(
            report["parent_child_range_status"],
            "not_available_no_explicit_native_parent_id",
        )

        missing = replace(plan, slices=())
        report = native_validation_metadata(missing, native)
        self.assertFalse(report["valid"])
        self.assertFalse(report["native_counts_reconciled"])
        boundary_group = _track("summary.boundaries", 4)
        boundary_track = _track(
            "summary.boundaries.events",
            5,
            parent="summary.boundaries",
        )
        missing = replace(
            missing,
            tracks=(*missing.tracks, boundary_group, boundary_track),
            instants=(
                InstantSpec(
                    track_key=boundary_track.key,
                    name="Request Received",
                    timestamp_ns=0,
                    annotations=(
                        ("hetero.boundary_kind", "request_received"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
                InstantSpec(
                    track_key=boundary_track.key,
                    name="Response Completion",
                    timestamp_ns=100,
                    annotations=(
                        ("hetero.boundary_kind", "response_done"),
                        ("hetero.correlation_id", "one"),
                    ),
                ),
            ),
            request_window=_request_window(0, 100),
        )
        filtered = native_validation_metadata(
            missing,
            native,
            filtered_subset=True,
        )
        self.assertFalse(filtered["valid"])
        self.assertFalse(filtered["native_identity_reconciled"])


class RblnPerfettoDetectionTests(unittest.TestCase):
    def test_standard_perfetto_pb_is_kept_as_unaligned_native_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report.pb"
            trace = Trace()
            descriptor = trace.packet.add().track_descriptor
            descriptor.uuid = 7
            descriptor.name = "RBLN lane"
            begin = trace.packet.add()
            begin.timestamp = 10
            begin.trusted_packet_sequence_id = 1
            begin.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
            begin.track_event.track_uuid = 7
            begin.track_event.name = "work"
            end = trace.packet.add()
            end.timestamp = 20
            end.trusted_packet_sequence_id = 1
            end.track_event.type = TrackEvent.TYPE_SLICE_END
            end.track_event.track_uuid = 7
            # Flow direction is timestamp-derived and IDs are trace-global.
            # Deliberately serialize these endpoints out of timestamp order,
            # across packet sequences, with explicit ID reuse.
            for timestamp, sequence_id, terminating in (
                (30, 1, False),
                (20, 2, True),
                (40, 3, True),
                (10, 4, False),
            ):
                packet = trace.packet.add()
                packet.timestamp = timestamp
                packet.trusted_packet_sequence_id = sequence_id
                packet.track_event.type = TrackEvent.TYPE_INSTANT
                packet.track_event.track_uuid = 7
                packet.track_event.name = f"flow-{timestamp}"
                target = (
                    packet.track_event.terminating_flow_ids
                    if terminating
                    else packet.track_event.flow_ids
                )
                target.append(99)
            path.write_bytes(trace.SerializeToString(deterministic=True))
            payload = path.read_bytes()
            artifact = SimpleNamespace(
                relative_path="report.pb",
                clock_domain_id="rbln-native",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            source = SimpleNamespace(
                root=root,
                source_role="npu",
                artifacts=(artifact,),
            )
            result = _rbln_native_only_result(
                source,
                native_clock_domain="rbln-native",
                native_timestamp_unit="ns",
            )
            summary = result.summaries[0]
            self.assertEqual(
                summary.support_status,
                "separate_native_perfetto_trace_unaligned",
            )
            self.assertEqual(
                dict(summary.event_counts)["aggregate_perfetto_slice_count"],
                5,
            )
            self.assertEqual(summary.emitted_event_count, 0)
            self.assertEqual(summary.fabricated_event_count, 0)
            self.assertEqual(len(result.separate_traces), 1)
            view = result.separate_traces[0]
            self.assertEqual(view.output_name, "trace.rbln-native.pftrace")
            self.assertEqual(view.expected_slice_count, 5)
            self.assertEqual(view.expected_track_count, 1)
            self.assertEqual(view.expected_flow_count, 2)
            self.assertEqual(view.payload, payload)
            self.assertEqual(view.size_bytes, len(payload))
            self.assertEqual(
                view.sha256,
                hashlib.sha256(payload).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
