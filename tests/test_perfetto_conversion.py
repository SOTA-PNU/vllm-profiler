"""End-to-end Perfetto conversion tests with immutable synthetic inputs."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.hybrid import (
    AlignmentMethod,
    HybridBundleMerger,
    HybridMergeConfig,
)
from perfetto_hetero_profiler.hybrid.detailed_profile import (
    build_profiler_alignment,
    build_profiler_clock_domain,
)
from perfetto_hetero_profiler.perfetto.converter import (
    CONVERSION_MANIFEST_NAME,
    RBLN_NATIVE_TRACE_NAME,
    RBLN_NATIVE_VALIDATION_NAME,
    TRACE_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
    TRACE_VALIDATION_NAME,
    PerfettoConversionConfig,
    convert_perfetto,
    plan_perfetto_conversion,
)
from perfetto_hetero_profiler.perfetto.loader import (
    PerfettoInputError,
    load_hybrid_run,
)
from perfetto_hetero_profiler.perfetto.tooling import (
    TRACE_PROCESSOR_FILENAME,
    TRACE_PROCESSOR_RELEASE,
)
from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    ArtifactReference,
    Availability,
    ClockType,
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
    DeviceType,
    MetricKind,
    MetricSample,
    MetricScope,
    ProfileMode,
    RunStatus,
    ValueOrigin,
    build_detached_artifact_manifest,
    create_detached_recovery,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

from tests.hybrid_fixtures import (
    GPU_MARKERS,
    NPU_MARKERS,
    build_source_bundle,
)


RUN_ID = "synthetic-hybrid"
GPU_RUN_ID = "synthetic-gpu"
NPU_RUN_ID = "synthetic-npu"
CLOCK_ID = "host-monotonic"
CANONICAL_CLOCK_ID = "hybrid-canonical"
REMOTE_SUFFIX = "_kv_transfer"
CORRELATION_ID = "correlation-1"
NATIVE_CLOCK_ID = "rbln-profiler-native"

_CLOSEOUT_REQUIRED = (
    ("coordinator", "result.json"),
    ("gpu", "manifest.json"),
    ("hybrid", "artifacts/artifacts.jsonl"),
    ("hybrid", "clocks/clock_domains.jsonl"),
    ("hybrid", "clocks/transforms.jsonl"),
    ("hybrid", "events/events.jsonl"),
    ("hybrid", "manifest.json"),
    ("hybrid", "metrics/metrics.jsonl"),
    ("npu", "manifest.json"),
)

_OUTPUT_NAMES = {
    TRACE_NAME,
    CONVERSION_MANIFEST_NAME,
    TRACE_VALIDATION_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
}


def _trace_processor_path() -> Path:
    return (
        Path(sys.prefix)
        / "bin"
        / f"{TRACE_PROCESSOR_FILENAME}-{TRACE_PROCESSOR_RELEASE}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_state(roots: tuple[Path, ...]) -> dict[str, tuple[int, int, str]]:
    state: dict[str, tuple[int, int, str]] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                key = f"{root.name}/{path.relative_to(root).as_posix()}"
                file_stat = path.stat()
                state[key] = (
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    _sha256(path),
                )
    return state


def _write_closeout(
    runs: Path,
    *,
    hybrid: Path,
    gpu: Path,
    npu: Path,
    run_id: str = RUN_ID,
) -> tuple[Path, Path]:
    coordinator = runs / f"{run_id}-coordinator"
    coordinator.mkdir()
    (coordinator / "result.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "succeeded",
                "hardware_rerun": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    recovery = runs / f"{run_id}-closeout-recovery"
    create_detached_recovery(
        recovery,
        {
            "coordinator": coordinator,
            "gpu": gpu,
            "hybrid": hybrid,
            "npu": npu,
        },
        {
            "schema_version": "1.0.0",
            "record_type": "closeout_recovery_result",
            "source_run_id": run_id,
            "success": True,
            "hardware_rerun": False,
            "postprocess_only": True,
        },
        required_artifacts=_CLOSEOUT_REQUIRED,
    )
    return coordinator, recovery


def _add_overview_metrics(hybrid: Path, *, run_id: str) -> None:
    """Add the explicit measured-smoke contract required by Overview tests."""

    metrics_path = hybrid / "metrics/metrics.jsonl"
    interval_ns = 1_000_000_000
    counts = {
        "request.count": ("requests", MetricKind.COUNT, 1),
        "request.input_tokens": ("tokens", MetricKind.COUNT, 4),
        "request.output_tokens": ("tokens", MetricKind.COUNT, 2),
        "request.total_tokens": ("tokens", MetricKind.COUNT, 6),
    }
    rates = {
        "throughput.requests": ("requests/s", 1.0),
        "throughput.input_tokens": ("tokens/s", 4.0),
        "throughput.output_tokens": ("tokens/s", 2.0),
        "throughput.total_tokens": ("tokens/s", 6.0),
    }
    measured = [
        MetricSample(
            run_id=run_id,
            metric_name=name,
            metric_kind=kind,
            scope=MetricScope.RUN,
            host_id="host-0",
            clock_domain_id=CANONICAL_CLOCK_ID,
            timestamp_ns=2_500_000,
            interval_ns=interval_ns,
            availability=Availability.AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit=unit,
            value=value,
            dimensions={"window": "measured_smoke"},
            attributes={"vllm.measurement_window": "measured_smoke"},
        )
        for name, (unit, kind, value) in counts.items()
    ]
    measured.extend(
        MetricSample(
            run_id=run_id,
            metric_name=name,
            metric_kind=MetricKind.RATE,
            scope=MetricScope.RUN,
            host_id="host-0",
            clock_domain_id=CANONICAL_CLOCK_ID,
            timestamp_ns=2_500_000,
            interval_ns=interval_ns,
            availability=Availability.AVAILABLE,
            origin=ValueOrigin.DERIVED,
            unit=unit,
            value=value,
            dimensions={"window": "measured_smoke"},
            attributes={"vllm.measurement_window": "measured_smoke"},
        )
        for name, (unit, value) in rates.items()
    )
    measured.extend(
        MetricSample(
            run_id=run_id,
            metric_name=name,
            metric_kind=MetricKind.DURATION,
            scope=MetricScope.REQUEST,
            host_id="host-0",
            clock_domain_id=CANONICAL_CLOCK_ID,
            timestamp_ns=2_500_000,
            interval_ns=value,
            availability=Availability.AVAILABLE,
            origin=ValueOrigin.MEASURED,
            unit="ns",
            value=value,
            request_id="request-1",
            dimensions={"window": "measured_smoke"},
            attributes={"vllm.measurement_window": "measured_smoke"},
        )
        for name, value in (
            ("latency.ttft", 1_000_000),
            ("latency.tpot", 500_000),
        )
    )
    write_jsonl(
        metrics_path,
        [*read_jsonl(metrics_path), *measured],
        overwrite=True,
    )


def _artifact(
    root: Path,
    *,
    artifact_id: str,
    artifact_kind: ArtifactKind,
    relative_path: str,
    format_name: str,
    clock_domain_id: str | None,
) -> ArtifactReference:
    path = root / relative_path
    return ArtifactReference(
        run_id=root.name,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        relative_path=relative_path,
        format=format_name,
        producer="synthetic-conversion-test",
        created_at_unix_ns=1,
        host_id="host-0",
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        clock_domain_id=clock_domain_id,
        attributes={},
    )


def _add_rbln_profile(npu: Path) -> None:
    manifest_path = npu / "manifest.json"
    manifest = read_json(manifest_path)
    write_json(
        manifest_path,
        replace(manifest, profile_mode=ProfileMode.DETAILED_PROFILE),
        overwrite=True,
    )

    native_clock = build_profiler_clock_domain(
        run_id=NPU_RUN_ID,
        clock_domain_id=NATIVE_CLOCK_ID,
        host_id="host-0",
        clock_type=ClockType.RBLN,
        profile_kind="npu_rbln",
        native_timestamp_unit="rbln_report_native",
        alignment_status="partial",
    )
    clock_path = npu / "clocks/clock_domains.jsonl"
    write_jsonl(
        clock_path,
        [*read_jsonl(clock_path), native_clock],
        overwrite=True,
    )

    report = npu / "raw/profiler/report.pb"
    report.parent.mkdir(parents=True, exist_ok=True)
    trace = Trace()
    descriptor = trace.packet.add()
    descriptor.track_descriptor.uuid = 7
    descriptor.track_descriptor.name = "RBLN device"
    begin = trace.packet.add()
    begin.timestamp = 10
    begin.trusted_packet_sequence_id = 1
    begin.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
    begin.track_event.track_uuid = 7
    begin.track_event.name = "RBLN work"
    begin.track_event.flow_ids.append(99)
    end = trace.packet.add()
    end.timestamp = 20
    end.trusted_packet_sequence_id = 1
    end.track_event.type = TrackEvent.TYPE_SLICE_END
    end.track_event.track_uuid = 7
    end.track_event.flow_ids.append(99)
    report.write_bytes(trace.SerializeToString(deterministic=True))
    report_stat = report.stat()
    report_relative = report.relative_to(npu).as_posix()
    report_reference = _artifact(
        npu,
        artifact_id="rbln-report",
        artifact_kind=ArtifactKind.RBLN_REPORT,
        relative_path=report_relative,
        format_name="vendor-rbln-pb",
        clock_domain_id=NATIVE_CLOCK_ID,
    )

    alignment_path = npu / "clocks/profiler_alignment.json"
    alignment_path.write_text(
        json.dumps(
            build_profiler_alignment(
                profiler_type="npu_rbln",
                native_clock_domain=NATIVE_CLOCK_ID,
                native_timestamp_unit="rbln_report_native",
                canonical_clock_domain=CLOCK_ID,
                anchors=(
                    {
                        "kind": "profiler_start_api",
                        "before_monotonic_ns": 1_750_000,
                        "after_monotonic_ns": 1_800_000,
                        "request_start_monotonic_ns": 1_850_000,
                        "http_status": 200,
                    },
                    {
                        "kind": "profiler_stop_api",
                        "request_end_monotonic_ns": 2_550_000,
                        "before_monotonic_ns": 2_600_000,
                        "after_monotonic_ns": 2_650_000,
                        "http_status": 200,
                    },
                ),
                native_capture_start=None,
                native_capture_end=None,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_reference = _artifact(
        npu,
        artifact_id="profiler-alignment",
        artifact_kind=ArtifactKind.OTHER,
        relative_path="clocks/profiler_alignment.json",
        format_name="json",
        clock_domain_id=CLOCK_ID,
    )

    detail_path = npu / "summary/detailed_profile.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(
        json.dumps(
            {
                "kind": "npu_rbln",
                "enabled": True,
                "format": "vendor_rbln_pb",
                "structural_parse": "not_available",
                "report_count": 1,
                "files": [
                    {
                        "inode": report_stat.st_ino,
                        "mtime_ns": report_stat.st_mtime_ns,
                        "path": report_relative,
                        "sha256": report_reference.sha256,
                        "size_bytes": report_reference.size_bytes,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    detail_reference = _artifact(
        npu,
        artifact_id="detailed-profile-summary",
        artifact_kind=ArtifactKind.OTHER,
        relative_path="summary/detailed_profile.json",
        format_name="json",
        clock_domain_id=CLOCK_ID,
    )
    artifacts_path = npu / "artifacts/artifacts.jsonl"
    write_jsonl(
        artifacts_path,
        [
            *read_jsonl(artifacts_path),
            alignment_reference,
            detail_reference,
            report_reference,
        ],
        overwrite=True,
    )


def _build_monitor_family(
    base: Path,
    *,
    rbln_profile: bool = False,
    drop_hybrid_marker: str | None = None,
    overview_metrics: bool = False,
    measured_token_timestamps: tuple[int, ...] | None = None,
    fatal_shutdown: bool = False,
    run_id: str = RUN_ID,
    gpu_run_id: str = GPU_RUN_ID,
    npu_run_id: str = NPU_RUN_ID,
) -> dict[str, Path]:
    runs = base / "runs"
    runs.mkdir()
    gpu_attributes = {
        name: {"hybrid.correlation_id": CORRELATION_ID}
        for name in GPU_MARKERS
    }
    gpu_attributes["kv_export_end"][
        "hybrid.remote_request_id_suffix"
    ] = REMOTE_SUFFIX
    gpu_attributes["kv_transfer_start"].update(
        {
            "hybrid.remote_request_id_suffix": REMOTE_SUFFIX,
            "hybrid.transfer_id": "transfer-1",
        }
    )
    gpu_attributes["kv_transfer_end"]["hybrid.transfer_id"] = "transfer-1"
    gpu = build_source_bundle(
        runs / gpu_run_id,
        device_type=DeviceType.GPU,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        markers=GPU_MARKERS,
        marker_attributes=gpu_attributes,
        include_artifact=True,
    )
    npu = build_source_bundle(
        runs / npu_run_id,
        device_type=DeviceType.NPU,
        host_id="host-0",
        clock_domain_id=CLOCK_ID,
        markers=NPU_MARKERS,
        timestamps=tuple(
            1_900_000 + index * 100_000
            for index in range(len(NPU_MARKERS))
        ),
        include_artifact=True,
        marker_attributes={
            name: {"hybrid.correlation_id": CORRELATION_ID}
            for name in NPU_MARKERS
        },
    )
    if measured_token_timestamps is not None:
        measured = gpu / "raw/client/measured_requests.jsonl"
        measured.parent.mkdir(parents=True, exist_ok=True)
        measured.write_text(
            json.dumps(
                {
                    "request_id": "request-1",
                    "request_start_ns": 1_000_000,
                    "stream_end_ns": 2_500_000,
                    "output_tokens": len(measured_token_timestamps),
                    "valid_token_timestamps_ns": list(measured_token_timestamps),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts_path = gpu / "artifacts/artifacts.jsonl"
        write_jsonl(
            artifacts_path,
            [
                *read_jsonl(artifacts_path),
                _artifact(
                    gpu,
                    artifact_id="measured-requests",
                    artifact_kind=ArtifactKind.RAW_LOG,
                    relative_path="raw/client/measured_requests.jsonl",
                    format_name="jsonl",
                    clock_domain_id=CLOCK_ID,
                ),
            ],
            overwrite=True,
        )
    if fatal_shutdown:
        stderr = npu / "raw/server/decode.stderr.log"
        stderr.parent.mkdir(parents=True, exist_ok=True)
        stderr.write_text(
            "Segfault encountered\nrtnl_tc_unregister\n",
            encoding="utf-8",
        )
        artifacts_path = npu / "artifacts/artifacts.jsonl"
        write_jsonl(
            artifacts_path,
            [
                *read_jsonl(artifacts_path),
                _artifact(
                    npu,
                    artifact_id="decode-stderr",
                    artifact_kind=ArtifactKind.RAW_LOG,
                    relative_path="raw/server/decode.stderr.log",
                    format_name="text",
                    clock_domain_id=None,
                ),
            ],
            overwrite=True,
        )
    if rbln_profile:
        _add_rbln_profile(npu)
    result = HybridBundleMerger(
        HybridMergeConfig(
            run_root=runs,
            run_id=run_id,
            gpu_run=gpu,
            npu_run=npu,
            alignment_method=(
                AlignmentMethod.FAKE
                if rbln_profile
                else AlignmentMethod.SAME_CLOCK_DOMAIN
            ),
            coordinator_host_id="host-0",
        ),
        unix_time_ns=lambda: 1,
    ).merge()
    if result.status is not RunStatus.SUCCEEDED:
        raise AssertionError(f"synthetic hybrid merge failed: {result.reasons}")
    hybrid = result.run_directory

    if rbln_profile:
        # Native profiler timestamps remain deliberately unaligned.  The
        # normalized hybrid stream therefore retains only the source host
        # clock transform; the native clock stays in the immutable NPU source.
        native_hybrid_clock = f"npu:{NATIVE_CLOCK_ID}"
        hybrid_clocks = hybrid / "clocks/clock_domains.jsonl"
        write_jsonl(
            hybrid_clocks,
            [
                clock
                for clock in read_jsonl(hybrid_clocks)
                if clock.clock_domain_id != native_hybrid_clock
            ],
            overwrite=True,
        )
        hybrid_transforms = hybrid / "clocks/transforms.jsonl"
        write_jsonl(
            hybrid_transforms,
            [
                transform
                for transform in read_jsonl(hybrid_transforms)
                if transform.source_clock_domain_id != native_hybrid_clock
            ],
            overwrite=True,
        )
    if drop_hybrid_marker is not None:
        hybrid_events = hybrid / "events/events.jsonl"
        write_jsonl(
            hybrid_events,
            [
                event
                for event in read_jsonl(hybrid_events)
                if event.event_name != drop_hybrid_marker
            ],
            overwrite=True,
        )
    if overview_metrics:
        _add_overview_metrics(hybrid, run_id=run_id)

    # Conversion requires the finalized closeout policy marker. The generic
    # merger predates that marker, so the synthetic closeout fixture records
    # the evidence before it is detached and fingerprinted.
    manifest_path = hybrid / "manifest.json"
    manifest = read_json(manifest_path)
    write_json(
        manifest_path,
        replace(
            manifest,
            attributes={
                **manifest.attributes,
                "hybrid.profiler_alignment_status": (
                    "partial" if rbln_profile else "not_applicable"
                ),
            },
        ),
        overwrite=True,
    )
    coordinator, recovery = _write_closeout(
        runs,
        hybrid=hybrid,
        gpu=gpu,
        npu=npu,
        run_id=run_id,
    )
    return {
        "runs": runs,
        "hybrid": hybrid,
        "gpu": gpu,
        "npu": npu,
        "coordinator": coordinator,
        "recovery": recovery,
    }


@unittest.skipUnless(
    _trace_processor_path().is_file(),
    "dedicated pinned Trace Processor binary is unavailable",
)
class PerfettoConversionIntegrationTests(unittest.TestCase):
    def test_default_output_is_a_new_source_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            expected = family["hybrid"].with_name(
                f"{family['hybrid'].name}-perfetto"
            )
            plan = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    trace_processor_path=_trace_processor_path(),
                )
            )
            self.assertEqual(plan["output_directory"], str(expected))
            self.assertFalse(expected.exists())

    def test_dry_run_conversion_is_deterministic_and_sql_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            input_roots = (
                family["hybrid"],
                family["gpu"],
                family["npu"],
                family["coordinator"],
                family["recovery"],
            )
            before = _tree_state(input_roots)
            first_output = family["runs"] / "perfetto-a"
            second_output = family["runs"] / "perfetto-b"
            dry_output = family["runs"] / "perfetto-dry"
            common = {
                "run_directory": family["hybrid"],
                "trace_processor_path": _trace_processor_path(),
            }

            plan = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    **common,
                    output_directory=dry_output,
                )
            )
            self.assertEqual(plan["status"], "planned")
            self.assertTrue(plan["dry_run"])
            self.assertFalse(plan["hardware_execution"])
            self.assertFalse(dry_output.exists())
            self.assertEqual(_tree_state(input_roots), before)

            first = convert_perfetto(
                PerfettoConversionConfig(
                    **common,
                    output_directory=first_output,
                )
            )
            second = convert_perfetto(
                PerfettoConversionConfig(
                    **common,
                    output_directory=second_output,
                )
            )
            self.assertEqual(first["status"], "succeeded")
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(
                {path.name for path in first_output.iterdir()},
                _OUTPUT_NAMES,
            )
            self.assertEqual(
                {path.name for path in second_output.iterdir()},
                _OUTPUT_NAMES,
            )
            self.assertEqual(_tree_state(input_roots), before)

            for name in (
                TRACE_NAME,
                CONVERSION_MANIFEST_NAME,
                TRACE_VALIDATION_NAME,
                TRACE_ATTRIBUTE_VALIDATION_NAME,
            ):
                self.assertEqual(
                    (first_output / name).read_bytes(),
                    (second_output / name).read_bytes(),
                    name,
                )

            validation = json.loads(
                (first_output / TRACE_VALIDATION_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["mismatches"], [])
            self.assertTrue(
                all(query["matched"] for query in validation["queries"])
            )
            self.assertGreater(validation["counts"]["slices"], 0)
            self.assertGreater(validation["counts"]["step_annotations"], 0)
            self.assertGreater(validation["counts"]["counters"], 0)
            self.assertGreater(validation["counts"]["flows"], 0)
            self.assertGreater(
                validation["counts"]["timeline_summary_hierarchy"],
                0,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_slices"],
                10,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_kpis"],
                0,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_data_quality"],
                0,
            )
            self.assertEqual(validation["counts"]["dangling_flows"], 0)
            self.assertEqual(validation["counts"]["import_errors"], 0)
            validation_text = json.dumps(validation, sort_keys=True)
            self.assertNotIn(str(_trace_processor_path()), validation_text)
            self.assertNotIn(str(first_output), validation_text)

            manifest_text = (
                first_output / CONVERSION_MANIFEST_NAME
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(_trace_processor_path()), manifest_text)
            self.assertNotIn(str(first_output), manifest_text)
            self.assertNotIn(str(family["hybrid"]), manifest_text)
            manifest = json.loads(
                manifest_text
            )
            self.assertFalse(
                manifest["flow_policy"]["timestamp_proximity_fallback"]
            )
            self.assertEqual(
                manifest["trace_mapping"]["mapping_version"],
                "processing-timeline-info-stats-v1",
            )
            self.assertEqual(
                manifest["trace_mapping"]["root_track"]["name"],
                "Heterogeneous LLM Processing",
            )
            self.assertEqual(
                manifest["trace_mapping"]["kpi_presentation"],
                "info_and_stats_trace_attributes_only",
            )
            self.assertEqual(
                manifest["trace_mapping"]["kpi_counter_mapping"],
                [],
            )
            self.assertEqual(
                manifest["trace_mapping"]["flow_policy"][
                    "representative_location"
                ],
                "detailed_tracks_only",
            )
            self.assertEqual(
                manifest["trace_mapping"]["unavailable_handling"][
                    "timeline_counter_policy"
                ],
                "not_emitted",
            )
            self.assertFalse(
                manifest["trace_mapping"]["resource_grouping"][
                    "counter_samples_copied"
                ]
            )
            self.assertEqual(
                manifest["trace"]["sha256"],
                _sha256(first_output / TRACE_NAME),
            )

    def test_overwrite_and_input_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            output = family["runs"] / "existing-output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            config = PerfettoConversionConfig(
                run_directory=family["hybrid"],
                output_directory=output,
                trace_processor_path=_trace_processor_path(),
            )
            with self.assertRaises(FileExistsError):
                plan_perfetto_conversion(config)
            with self.assertRaises(FileExistsError):
                convert_perfetto(config)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

            linked = family["runs"] / "linked-hybrid"
            os.symlink(family["hybrid"], linked)
            with self.assertRaisesRegex(PerfettoInputError, "symlink"):
                plan_perfetto_conversion(
                    replace(
                        config,
                        run_directory=linked,
                        output_directory=family["runs"] / "unused",
                    )
                )

    def test_cli_dry_run_success_conversion_and_failure_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            dry_output = family["runs"] / "cli-dry"
            output = family["runs"] / "cli-output"
            base_args = [
                "convert",
                "perfetto",
                "--run",
                str(family["hybrid"]),
                "--trace-processor",
                str(_trace_processor_path()),
            ]

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main(
                    [
                        *base_args,
                        "--output",
                        str(dry_output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")
            self.assertFalse(dry_output.exists())

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main([*base_args, "--output", str(output)])
            self.assertEqual(code, 0, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "succeeded")
            self.assertTrue(output.is_dir())

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main([*base_args, "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("conversion error:", stderr.getvalue())

    def test_rbln_native_profile_is_a_separate_unaligned_perfetto_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                rbln_profile=True,
            )
            output = family["runs"] / "perfetto-rbln"
            loaded = load_hybrid_run(family["hybrid"])
            self.assertEqual(len(loaded.native_envelopes), 1)
            envelope = loaded.native_envelopes[0]
            self.assertEqual(envelope.profiler_type, "npu_rbln")
            self.assertEqual(envelope.source_role, "npu")
            self.assertEqual(envelope.alignment_status, "partial")
            self.assertFalse(envelope.opaque_rbln_pb)

            deferred = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=family["runs"] / "perfetto-rbln-plan",
                    trace_processor_path=_trace_processor_path(),
                )
            )
            deferred_native = deferred["native_profiles"][0]
            self.assertFalse(deferred_native["opaque_rbln_pb"])
            self.assertEqual(
                deferred_native["rbln_pb_classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                deferred_native["rbln_pb_structure_analysis"],
                "deferred_to_official_trace_processor",
            )
            self.assertEqual(deferred["separate_native_traces"], [])

            result = convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=output,
                    trace_processor_path=_trace_processor_path(),
                    include_native_details=True,
                )
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["clock_alignment_status"], "partial")
            self.assertEqual(len(result["native_profiles"]), 1)

            manifest = json.loads(
                (output / CONVERSION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            native = manifest["native_profiles"][0]
            self.assertEqual(native["profiler_type"], "npu_rbln")
            self.assertEqual(native["alignment_status"], "partial_unaligned")
            self.assertEqual(
                native["rbln_pb_classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                native["rbln_pb_structure_analysis"],
                "official_perfetto_protobuf_schema",
            )
            self.assertFalse(native["opaque_rbln_pb"])
            self.assertFalse(native["rbln_pb_raw_bytes_embedded"])
            self.assertFalse(native["native_details_emitted"])
            self.assertTrue(native["separate_native_trace_published"])
            self.assertEqual(
                manifest["rbln_pb_policy"]["classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                manifest["rbln_pb_policy"]["structure_analysis"],
                "official_perfetto_protobuf_schema",
            )
            self.assertFalse(manifest["rbln_pb_policy"]["canonical_merge"])
            self.assertTrue(
                manifest["rbln_pb_policy"]["separate_native_trace_published"]
            )
            self.assertEqual(
                native["artifact_references"][0]["root_id"],
                "npu",
            )
            self.assertEqual(
                native["artifact_references"][0]["relative_path"],
                "raw/profiler/report.pb",
            )
            source_payload = (
                family["npu"] / "raw/profiler/report.pb"
            ).read_bytes()
            self.assertEqual(
                (output / RBLN_NATIVE_TRACE_NAME).read_bytes(),
                source_payload,
            )
            self.assertNotEqual(
                (output / TRACE_NAME).read_bytes(),
                source_payload,
            )

            native_validation = json.loads(
                (output / RBLN_NATIVE_VALIDATION_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(native_validation["valid"])
            self.assertEqual(native_validation["counts"]["slice_count"], 1)
            self.assertEqual(native_validation["counts"]["track_count"], 1)
            self.assertEqual(native_validation["counts"]["flow_count"], 1)
            self.assertFalse(
                native_validation["clock_policy"]["timestamp_rebased"]
            )
            self.assertFalse(
                native_validation["clock_policy"]["canonical_merge"]
            )
            self.assertEqual(len(result["separate_native_traces"]), 1)

            validation = json.loads(
                (output / TRACE_VALIDATION_NAME).read_text(encoding="utf-8")
            )
            self.assertTrue(validation["valid"])
            self.assertGreater(validation["counts"]["native_policy"], 0)
            native_query = next(
                query
                for query in validation["queries"]
                if query["name"] == "native_policy"
            )
            self.assertTrue(native_query["matched"])


class DetachedFamilyContractTests(unittest.TestCase):
    def test_synthetic_family_has_fresh_detached_closeout_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            loaded = load_hybrid_run(family["hybrid"])
            self.assertEqual(loaded.manifest.run_id, RUN_ID)
            self.assertGreater(loaded.closeout_artifact_count, 0)
            self.assertEqual(
                {item.root_id for item in loaded.root_fingerprints},
                {"coordinator", "gpu", "hybrid", "npu", "recovery"},
            )
            closeout_manifest = json.loads(
                (family["recovery"] / DETACHED_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            rebuilt = build_detached_artifact_manifest(
                {
                    "coordinator": family["coordinator"],
                    "gpu": family["gpu"],
                    "hybrid": family["hybrid"],
                    "npu": family["npu"],
                    "recovery": family["recovery"],
                },
                required_artifacts=_CLOSEOUT_REQUIRED,
            )
            self.assertEqual(closeout_manifest, rebuilt)

    def test_fresh_closeout_cannot_bless_an_incomplete_marker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                drop_hybrid_marker="prefill_end",
            )
            with self.assertRaisesRegex(
                PerfettoInputError,
                "canonical marker contract is not valid",
            ):
                load_hybrid_run(family["hybrid"])


if __name__ == "__main__":
    unittest.main()
