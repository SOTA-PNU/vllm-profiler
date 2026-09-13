"""CPU-only regression tests for the Perfetto conversion core."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from perfetto_hetero_profiler.hybrid.runtime_markers import (
    CANONICAL_MARKER_PHASES,
)
from perfetto_hetero_profiler.perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    ArtifactInventoryError,
    build_manifest,
    validate_manifest,
    verify_stored_sidecar,
    write_json_exclusive,
)
from perfetto_hetero_profiler.perfetto.model import (
    CounterSpec,
    TracePlan,
    TrackSpec,
)
from perfetto_hetero_profiler.perfetto.planner import (
    PerfettoPlanningError,
    build_trace_plan,
)
from perfetto_hetero_profiler.perfetto import tooling
from perfetto_hetero_profiler.perfetto.validation import (
    _rows_sha256,
    _run_query,
    summarize_trace_validation,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceDescriptor,
    DeviceType,
    EventRecord,
    EventType,
    HostDescriptor,
    MetricKind,
    MetricSample,
    MetricScope,
    ModelDescriptor,
    Phase,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
    ValueOrigin,
    WorkloadDescriptor,
)
from tests.support.toolchain import (
    require_trace_processor,
    trace_processor_test_class,
)


try:
    from perfetto_hetero_profiler.perfetto.writer import serialize_trace
except (ImportError, ModuleNotFoundError) as error:
    serialize_trace = None
    _WRITER_IMPORT_ERROR = str(error)
else:
    _WRITER_IMPORT_ERROR = ""


RUN_ID = "perfetto-synthetic"
CLOCK_ID = "hybrid-canonical"
REQUEST_ID = "request-1"
CORRELATION_ID = "correlation-1"
REMOTE_SUFFIX = "_kv_xfer_params"


def synthetic_manifest() -> RunManifest:
    return RunManifest(
        run_id=RUN_ID,
        mode=RunMode.HYBRID,
        profile_mode=ProfileMode.MONITOR,
        status=RunStatus.SUCCEEDED,
        created_at_unix_ns=1,
        models=[
            ModelDescriptor(
                role="prefill_decode",
                model_id="synthetic-model",
                revision=None,
                tokenizer_id=None,
                dtype=None,
            )
        ],
        workload=WorkloadDescriptor(
            request_count=1,
            concurrency=1,
            request_rate_per_s=None,
            input_tokens=8,
            output_tokens=2,
            max_model_len=512,
            warmup_requests=0,
        ),
        hosts=[
            HostDescriptor(
                host_id="host-0",
                role="hybrid",
                hostname="synthetic",
                operating_system="linux",
                architecture="x86_64",
            )
        ],
        software=[
            SoftwareDescriptor(
                name="test",
                version="1",
                role="test",
                path=None,
            )
        ],
        devices=[
            DeviceDescriptor(
                host_id="host-0",
                device_type=DeviceType.GPU,
                device_id="gpu-0",
                vendor="synthetic",
                model="synthetic-gpu",
                status="available",
                memory_total_bytes=None,
            ),
            DeviceDescriptor(
                host_id="host-0",
                device_type=DeviceType.NPU,
                device_id="npu-0",
                vendor="synthetic",
                model="synthetic-npu",
                status="available",
                memory_total_bytes=None,
            ),
        ],
        configuration={"canonical_clock_domain_id": CLOCK_ID},
        attributes={},
    )



def instant_event(
    name: str,
    timestamp_ns: int,
    *,
    correlation_id: str = CORRELATION_ID,
    request_id: str = REQUEST_ID,
    step_index: int | bool | None = None,
    transfer_id: str | None = None,
    remote_suffix: str | None = None,
    source_role: str | None = None,
    token_sequence: int | bool | None = None,
) -> EventRecord:
    attributes: dict[str, object] = {
        "hybrid.correlation_id": correlation_id,
    }
    if step_index is not None:
        attributes["decode.step_index"] = step_index
    if transfer_id is not None:
        attributes["hybrid.transfer_id"] = transfer_id
    if remote_suffix is not None:
        attributes["hybrid.remote_request_id_suffix"] = remote_suffix
    if source_role is not None:
        attributes["hybrid.source_role"] = source_role
    if token_sequence is not None:
        attributes["vllm.token_sequence"] = token_sequence
    discriminator = (
        f"-step-{step_index}" if step_index is not None else ""
    )
    return EventRecord(
        run_id=RUN_ID,
        event_id=f"{name}-{timestamp_ns}{discriminator}-{correlation_id}",
        event_name=name,
        event_type=EventType.INSTANT,
        phase=CANONICAL_MARKER_PHASES[name],
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=timestamp_ns,
        request_id=request_id,
        attributes=attributes,
    )


def canonical_events() -> tuple[EventRecord, ...]:
    return (
        instant_event("request_received", 100),
        instant_event("prefill_start", 120, source_role="gpu"),
        instant_event("prefill_end", 200, source_role="gpu"),
        instant_event("kv_export_start", 210, source_role="gpu"),
        instant_event(
            "kv_export_end",
            250,
            remote_suffix=REMOTE_SUFFIX,
            source_role="gpu",
        ),
        instant_event(
            "kv_transfer_start",
            260,
            transfer_id="transfer-1",
            remote_suffix=REMOTE_SUFFIX,
            source_role="gpu",
        ),
        instant_event(
            "kv_transfer_end",
            320,
            transfer_id="transfer-1",
            source_role="npu",
        ),
        instant_event("kv_transform_start", 330, source_role="npu"),
        instant_event("kv_transform_end", 380, source_role="npu"),
        instant_event("decode_loop_start", 400, source_role="npu"),
        instant_event(
            "decode_step_start",
            420,
            step_index=0,
            source_role="npu",
        ),
        instant_event(
            "decode_step_end",
            500,
            step_index=0,
            source_role="npu",
        ),
        instant_event(
            "decode_step_start",
            520,
            step_index=1,
            source_role="npu",
        ),
        instant_event(
            "decode_step_end",
            600,
            step_index=1,
            source_role="npu",
        ),
        instant_event(
            "sampling_start",
            610,
            step_index=1,
            source_role="npu",
        ),
        instant_event(
            "sampling_end",
            650,
            step_index=1,
            source_role="npu",
        ),
        instant_event("decode_loop_end", 800, source_role="npu"),
        instant_event("response_done", 900),
    )


def resource_metric(
    value: int | float | None,
    *,
    availability: Availability = Availability.AVAILABLE,
    timestamp_ns: int = 300,
) -> MetricSample:
    return MetricSample(
        run_id=RUN_ID,
        metric_name="resource.gpu.utilization",
        metric_kind=MetricKind.GAUGE,
        scope=MetricScope.DEVICE,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=timestamp_ns,
        availability=availability,
        origin=ValueOrigin.MEASURED,
        unit="percent",
        value=value,
        dimensions={},
        attributes={},
        device_type=DeviceType.GPU,
        device_id="gpu-0",
        reason=(
            None
            if availability is Availability.AVAILABLE
            else "synthetic unavailable"
        ),
    )


def non_resource_metric() -> MetricSample:
    return MetricSample(
        run_id=RUN_ID,
        metric_name="latency.e2e",
        metric_kind=MetricKind.DURATION,
        scope=MetricScope.REQUEST,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        timestamp_ns=900,
        availability=Availability.AVAILABLE,
        origin=ValueOrigin.DERIVED,
        unit="ns",
        value=800,
        dimensions={},
        attributes={},
        request_id=REQUEST_ID,
    )


class PlannerTests(unittest.TestCase):
    def test_crossing_concurrent_requests_use_deterministic_lanes(self):
        events: list[EventRecord] = []
        for request_index in range(4):
            correlation_id = f"correlation-{request_index + 1}"
            for event in canonical_events():
                attributes = dict(event.attributes)
                attributes["hybrid.correlation_id"] = correlation_id
                if "hybrid.transfer_id" in attributes:
                    attributes["hybrid.transfer_id"] = (
                        f"{attributes['hybrid.transfer_id']}-{correlation_id}"
                    )
                events.append(
                    replace(
                        event,
                        event_id=f"{event.event_id}-{correlation_id}",
                        timestamp_ns=event.timestamp_ns + 10 * request_index,
                        request_id=f"request-{request_index + 1}",
                        attributes=attributes,
                    )
                )

        first = build_trace_plan(
            synthetic_manifest(),
            events,
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        reordered = build_trace_plan(
            synthetic_manifest(),
            tuple(reversed(events)),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )

        self.assertEqual(first.plan, reordered.plan)
        for base in ("request", "gpu_prefill", "npu_decode_step"):
            for lane in range(4):
                key = base if lane == 0 else f"{base}:lane:{lane}"
                self.assertIn(key, first.plan.track_by_key)
        by_track: dict[str, list[tuple[int, int]]] = {}
        for item in first.plan.slices:
            by_track.setdefault(item.track_key, []).append(
                (item.timestamp_ns, item.timestamp_ns + item.duration_ns)
            )
        for intervals in by_track.values():
            previous_end = None
            for start_ns, end_ns in sorted(intervals):
                if previous_end is not None:
                    self.assertGreaterEqual(start_ns, previous_end)
                previous_end = end_ns
        if serialize_trace is not None:
            serialize_trace(first.plan)

    def test_observed_transfer_and_schedule_intervals_get_distinct_tracks(self):
        details = (
            instant_event(
                "kv_handoff_start", 251, transfer_id="request-1-handoff"
            ),
            instant_event(
                "kv_handoff_end", 255, transfer_id="request-1-handoff"
            ),
            instant_event(
                "kv_transfer_setup_start", 256, transfer_id="transfer-1"
            ),
            instant_event(
                "kv_transfer_setup_end", 259, transfer_id="transfer-1"
            ),
            instant_event(
                "kv_transfer_wait_start", 270, transfer_id="transfer-1"
            ),
            instant_event(
                "kv_transfer_wait_end", 315, transfer_id="transfer-1"
            ),
            instant_event(
                "kv_device_sync_start", 381, transfer_id="transfer-1"
            ),
            instant_event(
                "kv_device_sync_end", 385, transfer_id="transfer-1"
            ),
            instant_event(
                "decode_schedule_wait_start",
                386,
                transfer_id="request-1-decode-ready",
            ),
            instant_event(
                "decode_schedule_wait_end",
                399,
                transfer_id="request-1-decode-ready",
            ),
        )
        first = build_trace_plan(
            synthetic_manifest(),
            (*canonical_events(), *details),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        second = build_trace_plan(
            synthetic_manifest(),
            tuple(reversed((*canonical_events(), *details))),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        slices = {item.name: item for item in first.plan.slices}
        self.assertEqual(slices["KV Handoff"].duration_ns, 4)
        self.assertEqual(slices["KV Transfer Setup"].duration_ns, 3)
        self.assertEqual(slices["KV Transfer Wait"].duration_ns, 45)
        self.assertEqual(slices["KV Device Sync"].duration_ns, 4)
        self.assertEqual(slices["Decode Scheduling Wait"].duration_ns, 13)
        self.assertEqual(first.plan, second.plan)

    def test_token_boundaries_are_instants_and_never_summary_durations(self):
        details = (
            instant_event("first_token_emitted", 640, token_sequence=0),
            instant_event("token_emitted", 650, token_sequence=0),
            instant_event("token_emitted", 700, token_sequence=1),
        )
        result = build_trace_plan(
            synthetic_manifest(),
            (*canonical_events(), *details),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        boundaries = [
            row
            for row in result.plan.instants
            if dict(row.annotations).get("hetero.boundary_kind")
            in {
                "request_received",
                "first_token_emitted",
                "token_emitted",
                "response_done",
            }
        ]
        self.assertEqual(
            [row.name for row in boundaries],
            [
                "Request Received",
                "First Token Emitted",
                "Token Emitted 0",
                "Token Emitted 1",
                "Response Completion",
            ],
        )
        self.assertFalse(
            any("Token Emitted" in row.name for row in result.plan.slices)
        )
        self.assertEqual(
            [
                dict(row.annotations).get("hetero.token_sequence_index")
                for row in boundaries[1:4]
            ],
            [0, 0, 1],
        )

    def test_missing_optional_wait_markers_do_not_fabricate_wait_slices(self):
        result = build_trace_plan(
            synthetic_manifest(),
            canonical_events(),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        self.assertFalse(
            {
                "KV Handoff",
                "KV Transfer Setup",
                "KV Transfer Wait",
                "Decode Scheduling Wait",
            }
            & {row.name for row in result.plan.slices}
        )

    def test_canonical_markers_form_nested_slices_and_preserve_step_index(self):
        result = build_trace_plan(
            synthetic_manifest(),
            canonical_events(),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        slices_by_name: dict[str, list] = {}
        for item in result.plan.slices:
            slices_by_name.setdefault(item.name, []).append(item)

        request = slices_by_name["Request"][0]
        prefill = slices_by_name["GPU Prefill"][0]
        decode = slices_by_name["NPU Decode"][0]
        steps = [
            item for item in result.plan.slices
            if item.track_key == "npu_decode_step"
        ]
        self.assertEqual([item.name for item in steps], ["Decode Step 0", "Decode Step 1"])
        self.assertEqual((request.timestamp_ns, request.duration_ns), (100, 800))
        self.assertGreaterEqual(prefill.timestamp_ns, request.timestamp_ns)
        self.assertLessEqual(
            prefill.timestamp_ns + prefill.duration_ns,
            request.timestamp_ns + request.duration_ns,
        )
        self.assertEqual(
            [dict(item.annotations)["hetero.step_index"] for item in steps],
            [0, 1],
        )
        for item in steps:
            self.assertGreaterEqual(item.timestamp_ns, decode.timestamp_ns)
            self.assertLessEqual(
                item.timestamp_ns + item.duration_ns,
                decode.timestamp_ns + decode.duration_ns,
            )
        self.assertEqual(result.metadata.emitted_slice_count, 9)

    def test_flows_use_explicit_correlation_and_never_timestamp_fallback(self):
        explicit = build_trace_plan(
            synthetic_manifest(),
            canonical_events(),
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        self.assertEqual(len(explicit.plan.flows), 5)
        self.assertEqual(
            {flow.correlation_id for flow in explicit.plan.flows},
            {CORRELATION_ID},
        )
        self.assertEqual(
            {
                (flow.source_slice_name, flow.destination_slice_name)
                for flow in explicit.plan.flows
            },
            {
                ("Request", "GPU Prefill"),
                ("GPU Prefill", "KV Export"),
                ("KV Export", "KV Transfer"),
                ("KV Transfer", "KV Transform"),
                ("KV Transform", "NPU Decode"),
            },
        )

        unrelated = (
            instant_event(
                "request_received",
                100,
                correlation_id="request-correlation",
                request_id="request-a",
            ),
            instant_event(
                "response_done",
                500,
                correlation_id="request-correlation",
                request_id="request-a",
            ),
            instant_event(
                "prefill_start",
                100,
                correlation_id="prefill-correlation",
                request_id="request-b",
            ),
            instant_event(
                "prefill_end",
                500,
                correlation_id="prefill-correlation",
                request_id="request-b",
            ),
        )
        no_fallback = build_trace_plan(
            synthetic_manifest(),
            unrelated,
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        self.assertEqual(no_fallback.plan.flows, ())

        no_explicit_correlation = tuple(
            replace(
                event,
                attributes={
                    key: value
                    for key, value in event.attributes.items()
                    if key != "hybrid.correlation_id"
                },
            )
            for event in canonical_events()
        )
        request_id_only = build_trace_plan(
            synthetic_manifest(),
            no_explicit_correlation,
            (),
            canonical_clock_domain_id=CLOCK_ID,
        )
        self.assertEqual(request_id_only.plan.flows, ())

    def test_unavailable_resource_samples_are_omitted_not_zero_filled(self):
        event = instant_event("token_emitted", 100)
        result = build_trace_plan(
            synthetic_manifest(),
            (event,),
            (
                resource_metric(55.0, timestamp_ns=110),
                resource_metric(
                    None,
                    availability=Availability.NOT_COLLECTED,
                    timestamp_ns=120,
                ),
                non_resource_metric(),
            ),
            canonical_clock_domain_id=CLOCK_ID,
        )
        self.assertEqual([counter.value for counter in result.plan.counters], [55.0])
        self.assertEqual(result.metadata.resource_metric_count, 2)
        self.assertEqual(result.metadata.available_resource_metric_count, 1)
        self.assertEqual(result.metadata.unavailable_resource_metric_count, 1)
        self.assertEqual(result.metadata.skipped_non_resource_metric_count, 1)

    def test_planner_rejects_boolean_nan_and_infinite_resource_values(self):
        event = instant_event("token_emitted", 100)
        for value in (True, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    PerfettoPlanningError,
                    "finite non-boolean",
                ):
                    build_trace_plan(
                        synthetic_manifest(),
                        (event,),
                        (resource_metric(value),),
                        canonical_clock_domain_id=CLOCK_ID,
                    )

    def test_bad_pairing_step_index_and_order_are_rejected(self):
        cases = (
            (
                (
                    instant_event("prefill_start", 100),
                ),
                "incomplete GPU Prefill pairing",
            ),
            (
                (
                    instant_event("prefill_start", 200),
                    instant_event("prefill_end", 100),
                ),
                "negative duration",
            ),
            (
                (
                    instant_event(
                        "decode_step_start",
                        100,
                        step_index=True,
                    ),
                    instant_event(
                        "decode_step_end",
                        200,
                        step_index=True,
                    ),
                ),
                "decode.step_index",
            ),
        )
        for events, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(PerfettoPlanningError, pattern):
                    build_trace_plan(
                        synthetic_manifest(),
                        events,
                        (),
                        canonical_clock_domain_id=CLOCK_ID,
                    )

    def test_event_without_a_canonical_clock_transform_is_rejected(self):
        native_event = replace(
            instant_event("token_emitted", 100),
            clock_domain_id="gpu-native-clock",
        )
        with self.assertRaisesRegex(
            PerfettoPlanningError,
            "not on the canonical clock",
        ):
            build_trace_plan(
                synthetic_manifest(),
                (native_event,),
                (),
                canonical_clock_domain_id=CLOCK_ID,
            )

    def test_cross_device_flow_requires_matching_explicit_suffix(self):
        events = list(canonical_events())
        index = next(
            index
            for index, event in enumerate(events)
            if event.event_name == "kv_transfer_start"
        )
        events[index] = replace(
            events[index],
            attributes={
                **events[index].attributes,
                "hybrid.remote_request_id_suffix": "_wrong",
            },
        )
        with self.assertRaisesRegex(
            PerfettoPlanningError,
            "remote request suffix mismatch",
        ):
            build_trace_plan(
                synthetic_manifest(),
                events,
                (),
                canonical_clock_domain_id=CLOCK_ID,
            )


class ModelTests(unittest.TestCase):
    def test_track_lookup_is_by_stable_key(self):
        tracks = (
            TrackSpec("request", 11, "Request", "slice", "request"),
            TrackSpec("counter", 12, "Counter", "counter", "counter", "count"),
        )
        plan = TracePlan(
            run_id=RUN_ID,
            canonical_clock_domain_id=CLOCK_ID,
            process_uuid=1,
            process_id=1,
            packet_sequence_id=1,
            tracks=tracks,
            slices=(),
            instants=(),
            counters=(),
            flows=(),
        )
        self.assertIs(plan.track_by_key["request"], tracks[0])
        self.assertIs(plan.track_by_key["counter"], tracks[1])


@trace_processor_test_class(
    require_binary=False,
    require_socket=False,
    dependency_error=_WRITER_IMPORT_ERROR or None,
)
class WriterTests(unittest.TestCase):
    def test_writer_bytes_are_deterministic_for_reordered_input(self):
        events = canonical_events()
        metrics = (
            resource_metric(55, timestamp_ns=300),
            resource_metric(56, timestamp_ns=301),
        )
        first = build_trace_plan(
            synthetic_manifest(),
            events,
            metrics,
            canonical_clock_domain_id=CLOCK_ID,
        ).plan
        second = build_trace_plan(
            synthetic_manifest(),
            reversed(events),
            reversed(metrics),
            canonical_clock_domain_id=CLOCK_ID,
        ).plan

        first_bytes = serialize_trace(first)
        self.assertEqual(first_bytes, serialize_trace(first))
        self.assertEqual(first_bytes, serialize_trace(second))
        self.assertGreater(len(first_bytes), 0)

    def test_writer_rejects_boolean_nan_and_infinite_counter_values(self):
        valid = build_trace_plan(
            synthetic_manifest(),
            (instant_event("token_emitted", 100),),
            (resource_metric(1),),
            canonical_clock_domain_id=CLOCK_ID,
        ).plan
        counter = valid.counters[0]
        for value, error_type in (
            (True, TypeError),
            (math.nan, ValueError),
            (math.inf, ValueError),
            (-math.inf, ValueError),
        ):
            with self.subTest(value=value):
                invalid = replace(
                    valid,
                    counters=(replace(counter, value=value),),
                )
                with self.assertRaises(error_type):
                    serialize_trace(invalid)


class ValidationSummaryTests(unittest.TestCase):
    def test_persisted_summary_removes_only_query_rows(self) -> None:
        report = {
            "valid": True,
            "trace": {"sha256": "0" * 64},
            "queries": [
                {
                    "name": "slices",
                    "sql": "SELECT * FROM slice",
                    "columns": ["name"],
                    "row_count": 1,
                    "rows_sha256": "1" * 64,
                    "rows": [{"name": "work"}],
                    "expected_row_count": 1,
                    "expected_rows_sha256": "1" * 64,
                    "matched": True,
                }
            ],
            "native_details": {"rows": ["metadata is not a SQL query row"]},
        }
        summarized = summarize_trace_validation(report)
        self.assertNotIn("rows", summarized["queries"][0])
        self.assertEqual(
            summarized["native_details"], report["native_details"]
        )
        self.assertIn("rows", report["queries"][0])


def make_artifact_roots(base: Path) -> tuple[Path, Path]:
    source = base / "source"
    output = base / "output"
    source.mkdir()
    output.mkdir()
    (source / "manifest.json").write_text(
        '{"run_id":"source"}\n',
        encoding="utf-8",
    )
    (output / "trace.pftrace").write_bytes(b"trace")
    write_json_exclusive(
        output / "conversion_manifest.json",
        {"status": "succeeded"},
    )
    write_json_exclusive(
        output / "trace_validation.json",
        {"valid": True},
    )
    return source, output


def artifact_required() -> tuple[tuple[str, str], ...]:
    return (
        ("source", "manifest.json"),
        ("output", "trace.pftrace"),
        ("output", "conversion_manifest.json"),
        ("output", "trace_validation.json"),
    )


class ArtifactTests(unittest.TestCase):
    def test_manifest_and_sidecar_are_detached_and_verify_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = make_artifact_roots(Path(directory))
            roots = {"source": source, "output": output}
            manifest = build_manifest(
                roots,
                output_root_id="output",
                required_artifacts=artifact_required(),
            )
            manifest_path = output / ARTIFACT_MANIFEST_NAME
            sidecar_path = output / ARTIFACT_VALIDATION_NAME
            write_json_exclusive(manifest_path, manifest)
            report = validate_manifest(
                manifest_path,
                roots,
                output_root_id="output",
            )
            write_json_exclusive(sidecar_path, report)

            entries = {
                (item["root_id"], item["relative_path"])
                for item in manifest["artifacts"]
            }
            self.assertIn(("output", "trace.pftrace"), entries)
            self.assertIn(("output", "conversion_manifest.json"), entries)
            self.assertIn(("output", "trace_validation.json"), entries)
            self.assertNotIn(("output", ARTIFACT_MANIFEST_NAME), entries)
            self.assertNotIn(("output", ARTIFACT_VALIDATION_NAME), entries)
            before = {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in output.iterdir()
            }
            self.assertEqual(
                verify_stored_sidecar(
                    manifest_path,
                    roots,
                    output_root_id="output",
                ),
                report,
            )
            after = {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in output.iterdir()
            }
            self.assertEqual(before, after)

    def test_existing_json_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json_exclusive(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"value": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_symlinks_and_self_reference_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = make_artifact_roots(base)
            target = base / "target"
            target.write_text("target", encoding="utf-8")
            os.symlink(target, source / "linked")
            with self.assertRaisesRegex(
                ArtifactInventoryError,
                "symlink",
            ):
                build_manifest(
                    {"source": source, "output": output},
                    output_root_id="output",
                    required_artifacts=artifact_required(),
                )

        with tempfile.TemporaryDirectory() as directory:
            source, output = make_artifact_roots(Path(directory))
            with self.assertRaisesRegex(
                ArtifactInventoryError,
                "cannot be required",
            ):
                build_manifest(
                    {"source": source, "output": output},
                    output_root_id="output",
                    required_artifacts=(
                        *artifact_required(),
                        ("output", ARTIFACT_MANIFEST_NAME),
                    ),
                )


class ToolingTests(unittest.TestCase):
    def test_manifest_metadata_is_path_free(self):
        private_path = Path("/private/cache/trace_processor_shell")
        runtime = tooling.ToolchainRuntime(
            binary_path=private_path,
            perfetto_package_version=tooling.PERFETTO_PACKAGE_VERSION,
            protobuf_package_version=tooling.PROTOBUF_PACKAGE_VERSION,
            trace_processor_version=tooling.TRACE_PROCESSOR_VERSION,
            trace_processor_version_output=tooling.TRACE_PROCESSOR_VERSION_OUTPUT,
            trace_processor_rpc_api_version=(
                tooling.TRACE_PROCESSOR_RPC_API_VERSION
            ),
        )
        metadata = runtime.to_manifest()
        serialized = json.dumps(metadata, sort_keys=True)
        self.assertNotIn(str(private_path), serialized)
        self.assertNotIn("binary_path", metadata)
        self.assertEqual(
            set(metadata),
            {"filename", "version", "sha256", "source"},
        )

    def test_wrong_binary_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trace_processor_shell"
            binary.write_bytes(b"not the pinned binary")
            binary.chmod(0o755)
            with (
                mock.patch.object(
                    tooling,
                    "_installed_version",
                    return_value="pinned",
                ),
                mock.patch.object(tooling, "_validate_official_manifest"),
            ):
                with self.assertRaisesRegex(
                    tooling.ToolchainValidationError,
                    "size mismatch",
                ):
                    tooling.resolve_toolchain(binary)

    def test_dedicated_environment_resolves_pinned_toolchain(self):
        binary = require_trace_processor(self, require_socket=False)
        runtime = tooling.resolve_toolchain(binary)
        self.assertEqual(
            runtime.trace_processor_version,
            tooling.TRACE_PROCESSOR_VERSION,
        )
        self.assertEqual(
            runtime.binary_path,
            binary.absolute(),
        )


class TraceProcessorQueryPagingTests(unittest.TestCase):
    def test_large_query_is_paged_without_changing_complete_row_evidence(self):
        rows = [
            {"identity": index, "name": f"row-{index:05d}"}
            for index in range(200_003)
        ]

        class FakeResult:
            column_names = ("identity", "name")

            def __init__(self, values):
                self._values = values

            def __iter__(self):
                return iter(SimpleNamespace(**value) for value in self._values)

        class FakeProcessor:
            def __init__(self):
                self.queries = []

            def query(self, sql):
                self.queries.append(sql)
                if sql.startswith("CREATE TEMP TABLE") or sql.startswith(
                    "DROP TABLE"
                ):
                    return FakeResult([])
                match = re.search(r"LIMIT (\d+) OFFSET (\d+)$", sql)
                if match is None:
                    raise AssertionError("query was not paged")
                limit, offset = map(int, match.groups())
                return FakeResult(rows[offset : offset + limit])

        processor = FakeProcessor()
        report = _run_query(
            processor,
            "large_native_slices",
            "SELECT identity, name FROM synthetic ORDER BY identity",
            expected_rows=rows,
        )

        expected_rows = sorted(
            rows,
            key=lambda row: json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )
        self.assertEqual(report["row_count"], len(rows))
        self.assertEqual(report["rows"], expected_rows)
        self.assertEqual(report["rows_sha256"], _rows_sha256(expected_rows))
        self.assertEqual(report["expected_row_count"], len(rows))
        self.assertEqual(report["expected_rows_sha256"], _rows_sha256(expected_rows))
        self.assertTrue(report["matched"])
        self.assertEqual(len(processor.queries), 5)
        self.assertTrue(processor.queries[0].startswith("CREATE TEMP TABLE"))
        self.assertTrue(processor.queries[-1].startswith("DROP TABLE"))

        mismatch = _run_query(
            processor,
            "large_native_slices_mismatch",
            "SELECT identity, name FROM synthetic ORDER BY identity",
            expected_rows=rows[:-1],
        )
        self.assertFalse(mismatch["matched"])
        self.assertEqual(mismatch["expected_row_count"], len(rows) - 1)


if __name__ == "__main__":
    unittest.main()
