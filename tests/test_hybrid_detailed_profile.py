"""CPU-only guards for independent hybrid detailed-profiler captures."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.hybrid.detailed_profile import (
    DetailedProfileValidationError,
    HybridDetailedProfileConfig,
    build_profiler_alignment,
    build_profiler_clock_domain,
    compare_overhead,
    persist_per_sample_streams,
    select_profile_kind,
    validate_fresh_artifacts,
    validate_nsys_report,
    validate_owned_wrapper_child_leader,
    validate_proxy_marker_stats,
    validate_rbln_reports,
    validate_torch_traces,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    ClockType,
    MetricKind,
    MetricSample,
    MetricScope,
    ValueOrigin,
)


def metric(run_id: str, name: str, value: float | None) -> MetricSample:
    return MetricSample(
        run_id=run_id,
        metric_name=name,
        metric_kind=MetricKind.GAUGE,
        scope=MetricScope.DEVICE,
        host_id="host",
        clock_domain_id="host-monotonic",
        timestamp_ns=100,
        availability=(
            Availability.AVAILABLE
            if value is not None
            else Availability.NOT_AVAILABLE
        ),
        origin=ValueOrigin.MEASURED,
        unit="percent",
        value=value,
        dimensions={},
        attributes={},
        reason=None if value is not None else "unsupported",
    )


def capture_boundary() -> dict[str, int]:
    return {
        "start_before_monotonic_ns": 10,
        "start_after_monotonic_ns": 20,
        "request_start_monotonic_ns": 30,
        "request_end_monotonic_ns": 40,
        "stop_before_monotonic_ns": 50,
        "stop_after_monotonic_ns": 60,
        "start_http_status": 200,
        "stop_http_status": 200,
    }


def alignment_anchors() -> tuple[dict[str, int | str], ...]:
    return (
        {
            "kind": "profiler_start_api",
            "before_monotonic_ns": 10,
            "after_monotonic_ns": 20,
            "request_start_monotonic_ns": 30,
            "http_status": 200,
        },
        {
            "kind": "profiler_stop_api",
            "request_end_monotonic_ns": 40,
            "before_monotonic_ns": 50,
            "after_monotonic_ns": 60,
            "http_status": 200,
        },
    )


class DetailedProfileConfigTests(unittest.TestCase):
    def test_each_supported_profile_has_unique_output_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for kind in (
                "control",
                "gpu_torch",
                "gpu_nsys",
                "npu_vllm",
                "npu_rbln",
            ):
                config = HybridDetailedProfileConfig(
                    run_root=root,
                    run_id=f"run-{kind}",
                    profile_kind=kind,
                )
                self.assertEqual(len(set(config.output_roots.values())), 5)

    def test_combined_profilers_are_rejected_and_rbln_is_supported(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_profile_kind(("gpu_torch", "gpu_nsys"))
        self.assertEqual(select_profile_kind(("npu_rbln",)), "npu_rbln")

    def test_existing_output_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "run-gpu").mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                HybridDetailedProfileConfig(
                    run_root=root,
                    run_id="run",
                    profile_kind="gpu_torch",
                )


class DetailedArtifactTests(unittest.TestCase):
    def test_zero_and_stale_profiler_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json"
            path.touch()
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "empty"
            ):
                validate_fresh_artifacts(
                    [path],
                    capture_started_unix_ns=0,
                )
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "stale"
            ):
                validate_fresh_artifacts(
                    [path],
                    capture_started_unix_ns=path.stat().st_mtime_ns + 1,
                )

    def test_malformed_and_empty_torch_traces_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "malformed"
            ):
                validate_torch_traces(
                    [path],
                    target="npu",
                    capture_started_unix_ns=0,
                )
            path.write_text('{"traceEvents":[]}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "no traceEvents"
            ):
                validate_torch_traces(
                    [path],
                    target="npu",
                    capture_started_unix_ns=0,
                )

    def test_gpu_trace_requires_cuda_and_preserves_native_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(
                    {
                        "baseTimeNanoseconds": 123,
                        "traceEvents": [
                            {
                                "name": "Forward",
                                "cat": "cpu_op",
                                "ph": "X",
                                "ts": 1.5,
                                "dur": 0.5,
                            },
                            {
                                "name": "cudaLaunchKernel",
                                "cat": "cuda_runtime",
                                "ph": "X",
                                "ts": 2.5,
                                "dur": 0.25,
                            },
                            {
                                "name": "real kernel",
                                "cat": "kernel",
                                "ph": "X",
                                "ts": 3.0,
                                "dur": 1.0,
                            },
                        ],
                    },
                    stream,
                )
            result = validate_torch_traces(
                [path],
                target="gpu",
                capture_started_unix_ns=0,
                capture_boundary=capture_boundary(),
            )
            self.assertEqual(result["event_count"], 3)
            self.assertEqual(result["cuda_kernel_event_count"], 1)
            self.assertEqual(result["native_timestamp_min"], 1.5)
            self.assertEqual(result["base_time_nanoseconds"], [123])
            self.assertEqual(result["measured_scope"]["status"], "bracketed")

    def test_metadata_and_named_fake_kernel_do_not_count_as_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json"
            path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"name": "process_name", "ph": "M", "ts": 1},
                            {"name": "kernel", "ph": "X", "ts": 2, "dur": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "CPU operation"
            ):
                validate_torch_traces(
                    [path],
                    target="gpu",
                    capture_started_unix_ns=0,
                )

    def test_sensitive_prompt_is_rejected_from_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json"
            path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "Capital of South Korea is",
                                "cat": "cpu_op",
                                "ph": "X",
                                "ts": 1,
                                "dur": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "sensitive"
            ) as caught:
                validate_torch_traces(
                    [path],
                    target="npu",
                    capture_started_unix_ns=0,
                    forbidden_text=("Capital of South Korea is",),
                )
            self.assertNotIn(
                "Capital of South Korea is", str(caught.exception)
            )

    def test_run_local_and_preexisting_artifact_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = root / "owned"
            owned.mkdir()
            outside = root / "outside.nsys-rep"
            outside.write_bytes(b"report")
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "outside"
            ):
                validate_fresh_artifacts(
                    [outside],
                    capture_started_unix_ns=0,
                    run_root=owned,
                )
            inside = owned / "trace.nsys-rep"
            inside.write_bytes(b"report")
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "before capture"
            ):
                validate_fresh_artifacts(
                    [inside],
                    capture_started_unix_ns=0,
                    run_root=owned,
                    preexisting_paths=(inside,),
                )

    def test_symlink_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.nsys-rep"
            target.write_bytes(b"report")
            linked = root / "linked.nsys-rep"
            linked.symlink_to(target)
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "owned regular file"
            ):
                validate_fresh_artifacts(
                    [linked],
                    capture_started_unix_ns=0,
                    run_root=root,
                )

    def test_nsys_requires_cuda_kernel_and_os_runtime_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.write_bytes(b"nsys")
            stats = {
                name: {
                    "returncode": 0,
                    "stdout": (
                        "Time (%),Total Time (ns),Num Calls,Name\n"
                        "100.0,123,2,operation\n"
                    ),
                    "stderr": "",
                }
                for name in (
                    "cuda_api_sum",
                    "cuda_gpu_kern_sum",
                    "osrt_sum",
                    "nvtx_sum",
                )
            }
            result = validate_nsys_report(
                report,
                capture_started_unix_ns=0,
                run_root=root,
                preexisting_paths=(),
                stats=stats,
                capture_boundary=capture_boundary(),
            )
            self.assertTrue(
                result["reports"]["cuda_gpu_kern_sum"]["available"]
            )
            stats["cuda_gpu_kern_sum"]["stdout"] = "SKIPPED: no kernels\n"
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "cuda_gpu_kern_sum"
            ):
                validate_nsys_report(
                    report,
                    capture_started_unix_ns=0,
                    run_root=root,
                    preexisting_paths=(),
                    stats=stats,
                    capture_boundary=capture_boundary(),
                )

    def test_nsys_header_without_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.write_bytes(b"nsys")
            header = "Time (%),Total Time (ns),Num Calls,Name\n"
            stats = {
                name: {
                    "returncode": 0,
                    "stdout": header,
                    "stderr": "",
                }
                for name in (
                    "cuda_api_sum",
                    "cuda_gpu_kern_sum",
                    "osrt_sum",
                )
            }
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "cuda_api_sum"
            ):
                validate_nsys_report(
                    report,
                    capture_started_unix_ns=0,
                    run_root=root,
                    preexisting_paths=(),
                    stats=stats,
                    capture_boundary=capture_boundary(),
                )

    def test_nsys_kernel_instances_column_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.write_bytes(b"nsys")
            stats = {
                "cuda_api_sum": {
                    "returncode": 0,
                    "stdout": "Num Calls,Name\n2,cudaLaunchKernel\n",
                    "stderr": "",
                },
                "cuda_gpu_kern_sum": {
                    "returncode": 0,
                    "stdout": "Instances,Name\n1,vector_kernel\n",
                    "stderr": "",
                },
                "osrt_sum": {
                    "returncode": 0,
                    "stdout": "Num Calls,Name\n3,poll\n",
                    "stderr": "",
                },
            }
            result = validate_nsys_report(
                report,
                capture_started_unix_ns=0,
                run_root=root,
                preexisting_paths=(),
                stats=stats,
                capture_boundary=capture_boundary(),
            )
            self.assertEqual(
                result["reports"]["cuda_gpu_kern_sum"][
                    "data_row_count"
                ],
                1,
            )

    def test_nsys_nvtx_range_column_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.write_bytes(b"nsys")
            stats = {
                "cuda_api_sum": {
                    "returncode": 0,
                    "stdout": "Num Calls,Name\n2,cudaLaunchKernel\n",
                    "stderr": "",
                },
                "cuda_gpu_kern_sum": {
                    "returncode": 0,
                    "stdout": "Instances,Name\n1,vector_kernel\n",
                    "stderr": "",
                },
                "osrt_sum": {
                    "returncode": 0,
                    "stdout": "Num Calls,Name\n3,poll\n",
                    "stderr": "",
                },
                "nvtx_sum": {
                    "returncode": 0,
                    "stdout": "Instances,Range\n2,execute_context\n",
                    "stderr": "",
                },
            }
            result = validate_nsys_report(
                report,
                capture_started_unix_ns=0,
                run_root=root,
                preexisting_paths=(),
                stats=stats,
                capture_boundary=capture_boundary(),
            )
            self.assertTrue(result["reports"]["nvtx_sum"]["available"])
            self.assertEqual(
                result["reports"]["nvtx_sum"]["data_row_count"],
                1,
            )

    def test_rbln_report_requires_host_and_device_timing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "profile.pb"
            report.write_bytes(b"protobuf")
            result = validate_rbln_reports(
                [report],
                capture_started_unix_ns=0,
                run_root=root,
                preexisting_paths=(),
                strings_results={
                    str(report): {
                        "returncode": 0,
                        "stdout": (
                            "Host\nNeural Engine Clusters\nTask DMA\n"
                        ),
                    }
                },
                capture_boundary=capture_boundary(),
            )
            self.assertTrue(result["host_timing_present"])
            self.assertTrue(result["device_timing_present"])
            self.assertEqual(result["format"], "perfetto_trace_protobuf")
            self.assertEqual(
                result["structural_parse"],
                "deferred_to_official_trace_processor",
            )
            self.assertNotIn(
                "Neural Engine Clusters",
                result["reader_evidence"][0]["stdout_sha256"],
            )
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "host/device"
            ):
                validate_rbln_reports(
                    [report],
                    capture_started_unix_ns=0,
                    run_root=root,
                    preexisting_paths=(),
                    strings_results={
                        str(report): {
                            "returncode": 0,
                            "stdout": "Host only\n",
                        }
                    },
                    capture_boundary=capture_boundary(),
                )


class DetailedTelemetryTests(unittest.TestCase):
    def test_per_sample_streams_are_nonempty_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / "gpu"
            npu = root / "npu"
            samples = [
                {"monotonic_ns": 100, "load_average": [1.0]},
                {"monotonic_ns": 600, "load_average": [2.0]},
            ]
            result = persist_per_sample_streams(
                gpu_root=gpu,
                npu_root=npu,
                gpu_metrics=(
                    metric("gpu", "resource.gpu.utilization", 10.0),
                ),
                npu_metrics=(
                    metric("npu", "resource.npu.utilization", 20.0),
                ),
                system_metrics=(
                    metric("gpu", "resource.cpu.utilization", None),
                ),
                collector_samples=samples,
            )
            self.assertEqual(result["collector_sample_count"], 2)
            self.assertEqual(result["actual_interval_ns"]["average"], 500)
            for path in result["paths"].values():
                self.assertGreater(Path(path).stat().st_size, 0)
            self.assertEqual(
                result["aggregates"]["system"][
                    "resource.cpu.utilization"
                ]["unavailable_count"],
                1,
            )

    def test_empty_stream_fails_instead_of_recording_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "gpu.*empty"
            ):
                persist_per_sample_streams(
                    gpu_root=Path(directory) / "gpu",
                    npu_root=Path(directory) / "npu",
                    gpu_metrics=(),
                    npu_metrics=(
                        metric("npu", "resource.npu.utilization", 1.0),
                    ),
                    system_metrics=(
                        metric("gpu", "resource.cpu.utilization", 1.0),
                    ),
                    collector_samples=({"monotonic_ns": 1},),
                )


class DetailedAlignmentAndStatsTests(unittest.TestCase):
    def test_profiler_clock_domain_uses_v1_ns_and_preserves_native_unit(self):
        result = build_profiler_clock_domain(
            run_id="run",
            clock_domain_id="gpu:torch-chrome-trace",
            host_id="host",
            clock_type=ClockType.EXTERNAL,
            profile_kind="gpu_torch",
            native_timestamp_unit="chrome_trace_microseconds",
            alignment_status="partial",
        )
        self.assertEqual(result.unit, "ns")
        self.assertEqual(
            result.attributes["hybrid.native_timestamp_unit"],
            "chrome_trace_microseconds",
        )
        self.assertTrue(result.attributes["hybrid.raw_timestamp_preserved"])

    def test_nsys_target_must_be_owned_descendant_group_leader(self):
        self.assertEqual(
            validate_owned_wrapper_child_leader(
                wrapper_pid=10,
                target_pid=30,
                target_pgid=30,
                parent_by_pid={30: 20, 20: 10},
            ),
            30,
        )
        with self.assertRaisesRegex(
            DetailedProfileValidationError, "not an owned"
        ):
            validate_owned_wrapper_child_leader(
                wrapper_pid=10,
                target_pid=30,
                target_pgid=30,
                parent_by_pid={30: 20, 20: 1},
            )
        with self.assertRaisesRegex(
            DetailedProfileValidationError, "lead its own"
        ):
            validate_owned_wrapper_child_leader(
                wrapper_pid=10,
                target_pid=30,
                target_pgid=20,
                parent_by_pid={30: 20, 20: 10},
            )

    def test_unproven_profiler_clock_stays_partial(self):
        result = build_profiler_alignment(
            profiler_type="torch",
            native_clock_domain="torch-chrome-trace",
            native_timestamp_unit="us",
            canonical_clock_domain="host-monotonic",
            anchors=alignment_anchors(),
            native_capture_start=10,
            native_capture_end=20,
        )
        self.assertEqual(result["alignment_status"], "partial")
        self.assertIsNone(result["offset_ns"])
        self.assertIsNone(result["uncertainty_ns"])
        self.assertTrue(result["unaligned_profiler_events"])

    def test_groundless_same_clock_alignment_is_rejected(self):
        with self.assertRaisesRegex(
            DetailedProfileValidationError, "different.*clocks"
        ):
            build_profiler_alignment(
                profiler_type="nsys",
                native_clock_domain="nsys-native",
                native_timestamp_unit="ns",
                canonical_clock_domain="host-monotonic",
                anchors=alignment_anchors(),
                native_capture_start=1,
                native_capture_end=2,
                same_clock_evidence={
                    "method": "clock_descriptor_identity",
                    "clock_domain_id": "host-monotonic",
                },
            )

    def test_missing_start_stop_alignment_anchors_are_rejected(self):
        with self.assertRaisesRegex(
            DetailedProfileValidationError, "start and stop"
        ):
            build_profiler_alignment(
                profiler_type="torch",
                native_clock_domain="torch-native",
                native_timestamp_unit="us",
                canonical_clock_domain="host-monotonic",
                anchors=(),
                native_capture_start=1,
                native_capture_end=2,
            )

    def test_duplicate_alignment_anchor_is_rejected(self):
        start, stop = alignment_anchors()
        with self.assertRaisesRegex(
            DetailedProfileValidationError, "exactly one"
        ):
            build_profiler_alignment(
                profiler_type="torch",
                native_clock_domain="torch-native",
                native_timestamp_unit="us",
                canonical_clock_domain="host-monotonic",
                anchors=(start, dict(start), stop),
                native_capture_start=1,
                native_capture_end=2,
            )

    def test_valid_torch_trace_requires_boundary_and_known_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json"
            path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "Forward",
                                "cat": "cpu_op",
                                "ph": "X",
                                "ts": 1,
                                "dur": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "boundary evidence"
            ):
                validate_torch_traces(
                    [path],
                    target="npu",
                    capture_started_unix_ns=0,
                )
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "unsupported.*target"
            ):
                validate_torch_traces(
                    [path],
                    target="other",  # type: ignore[arg-type]
                    capture_started_unix_ns=0,
                    capture_boundary=capture_boundary(),
                )

    def test_proxy_stats_require_matching_records_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "runtime-markers-12.jsonl"
            marker.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            stats = marker.with_suffix(".stats.json")
            stats.write_text(
                json.dumps(
                    {
                        "records": 2,
                        "bytes": marker.stat().st_size,
                        "average_write_ns": 10,
                        "max_write_ns": 20,
                        "dropped": 0,
                        "duplicates": 0,
                    }
                ),
                encoding="utf-8",
            )
            result = validate_proxy_marker_stats(root)
            self.assertEqual(result["coverage"], "complete")
            self.assertEqual(result["records"], 2)
            stats.unlink()
            with self.assertRaisesRegex(
                DetailedProfileValidationError, "missing"
            ):
                validate_proxy_marker_stats(root)

    def test_overhead_handles_zero_and_unavailable(self):
        result = compare_overhead(
            {
                "bool": True,
                "e2e_ns": 100,
                "infinite": float("inf"),
                "missing": None,
                "not_a_number": float("nan"),
                "zero": 0,
            },
            {
                "bool": 1,
                "e2e_ns": 125,
                "infinite": 10,
                "missing": 10,
                "not_a_number": 10,
                "zero": 5,
            },
        )
        self.assertEqual(result["e2e_ns"]["absolute_delta"], 25)
        self.assertEqual(result["e2e_ns"]["relative_delta"], 0.25)
        self.assertIsNone(result["bool"]["absolute_delta"])
        self.assertIsNone(result["infinite"]["absolute_delta"])
        self.assertIsNone(result["missing"]["absolute_delta"])
        self.assertIsNone(result["not_a_number"]["absolute_delta"])
        self.assertIsNone(result["zero"]["relative_delta"])


if __name__ == "__main__":
    unittest.main()
