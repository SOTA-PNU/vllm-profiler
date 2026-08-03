"""Fake end-to-end hybrid bundle and CLI tests."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.hybrid import (
    AlignmentMethod,
    HybridBundleMerger,
    HybridMergeConfig,
    build_hybrid_plan,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceType,
    MetricSample,
    RunPaths,
    RunStatus,
    read_json,
    read_jsonl,
    validate_record,
    write_jsonl,
)

from tests.hybrid_fixtures import (
    GPU_MARKERS,
    NPU_MARKERS,
    build_source_bundle,
    event,
)


class HybridCase:
    def __init__(
        self,
        directory,
        *,
        same_host=True,
        offset_ns=0,
        gpu_host=None,
        npu_host=None,
        gpu_clock=None,
        npu_clock=None,
        gpu_markers=GPU_MARKERS,
        npu_markers=NPU_MARKERS,
        npu_times=None,
        include_artifact=False,
    ):
        root = Path(directory)
        host_gpu = gpu_host or ("host-0" if same_host else "gpu-host")
        host_npu = npu_host or ("host-0" if same_host else "npu-host")
        clock_gpu = gpu_clock or ("mono" if same_host else "gpu-mono")
        clock_npu = npu_clock or ("mono" if same_host else "npu-mono")
        gpu_times = tuple(
            1_000_000 + index * 100_000 for index in range(len(gpu_markers))
        )
        canonical_npu = tuple(
            1_900_000 + index * 100_000 for index in range(len(npu_markers))
        )
        npu_times = npu_times or tuple(value + offset_ns for value in canonical_npu)
        self.gpu = build_source_bundle(
            root / "gpu-source",
            device_type=DeviceType.GPU,
            host_id=host_gpu,
            clock_domain_id=clock_gpu,
            markers=gpu_markers,
            timestamps=gpu_times,
            include_artifact=include_artifact,
        )
        self.npu = build_source_bundle(
            root / "npu-source",
            device_type=DeviceType.NPU,
            host_id=host_npu,
            clock_domain_id=clock_npu,
            markers=npu_markers,
            timestamps=npu_times,
            include_artifact=include_artifact,
        )
        self.output_root = root / "hybrid-runs"
        self.output = self.output_root / "hybrid"


def merge_config(case, **overrides):
    same = read_json(case.gpu / "manifest.json").hosts[0].host_id == read_json(
        case.npu / "manifest.json"
    ).hosts[0].host_id
    values = {
        "run_root": case.output_root,
        "run_id": "hybrid",
        "gpu_run": case.gpu,
        "npu_run": case.npu,
        "alignment_method": (
            AlignmentMethod.SAME_CLOCK_DOMAIN if same else AlignmentMethod.FAKE
        ),
    }
    values.update(overrides)
    return HybridMergeConfig(**values)


class HybridConfigTests(unittest.TestCase):
    def test_relative_run_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_root"):
            HybridMergeConfig(
                run_root=Path("runs"),
                run_id="hybrid",
                gpu_run=Path("/tmp/gpu"),
                npu_run=Path("/tmp/npu"),
            )

    def test_same_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            HybridMergeConfig(
                run_root=Path("/tmp/runs"),
                run_id="hybrid",
                gpu_run=Path("/tmp/source"),
                npu_run=Path("/tmp/source"),
            )

    def test_negative_uncertainty_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            HybridMergeConfig(
                run_root=Path("/tmp/runs"),
                run_id="hybrid",
                gpu_run=Path("/tmp/gpu"),
                npu_run=Path("/tmp/npu"),
                max_uncertainty_ns=-1,
            )

    def test_invalid_probe_limits_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimum_probe_samples"):
            HybridMergeConfig(
                run_root=Path("/tmp/runs"),
                run_id="hybrid",
                gpu_run=Path("/tmp/gpu"),
                npu_run=Path("/tmp/npu"),
                probe_count=3,
                minimum_probe_samples=4,
            )

    def test_plan_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            config = merge_config(case)
            plan = build_hybrid_plan(config)
            self.assertFalse(config.paths.root.exists())
            self.assertFalse(plan["executes"])
            self.assertFalse(plan["creates_perfetto_trace"])


class HybridBundleTests(unittest.TestCase):
    def test_same_host_merge_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.SUCCEEDED)
            self.assertEqual(result.joined_request_count, 1)

    def test_same_host_different_clock_is_not_same_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory, npu_clock="npu-mono")
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertTrue(any("identical" in reason for reason in result.reasons))

    def test_same_clock_name_different_host_is_not_same_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(
                directory,
                same_host=False,
                gpu_clock="mono",
                npu_clock="mono",
            )
            result = HybridBundleMerger(
                merge_config(
                    case,
                    alignment_method=AlignmentMethod.SAME_CLOCK_DOMAIN,
                )
            ).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertTrue(any("identical" in reason for reason in result.reasons))

    def test_cross_host_plus_50ms_aligns(self):
        with tempfile.TemporaryDirectory() as directory:
            offset = 50_000_000
            case = HybridCase(directory, same_host=False, offset_ns=offset)
            result = HybridBundleMerger(
                merge_config(case, fake_offset_ns=offset)
            ).merge()
            self.assertIs(result.status, RunStatus.SUCCEEDED)
            events = read_jsonl(case.output / "events/events.jsonl")
            decode = next(row for row in events if row.event_name == "decode_loop_start")
            self.assertEqual(decode.timestamp_ns, 1_900_000)

    def test_cross_host_jitter_stays_within_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            offset = 50_000_000
            case = HybridCase(directory, same_host=False, offset_ns=offset)
            result = HybridBundleMerger(
                merge_config(
                    case,
                    fake_offset_ns=offset,
                    fake_jitter_ns=20_000,
                )
            ).merge()
            self.assertIs(result.status, RunStatus.SUCCEEDED)
            self.assertLessEqual(result.uncertainty_ns, 1_000_000)

    def test_canonical_events_preserve_original_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            HybridBundleMerger(merge_config(case)).merge()
            events = read_jsonl(case.output / "events/events.jsonl")
            self.assertTrue(
                all(
                    "hybrid.original_timestamp_ns" in row.attributes
                    for row in events
                )
            )
            self.assertTrue(
                all(row.clock_domain_id == "hybrid-canonical" for row in events)
            )

    def test_resource_metric_preserves_device(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            HybridBundleMerger(merge_config(case)).merge()
            metrics = read_jsonl(case.output / "metrics/metrics.jsonl")
            resource = [
                row
                for row in metrics
                if isinstance(row, MetricSample)
                and row.metric_name.startswith("resource.")
            ]
            self.assertEqual(
                {row.device_type for row in resource},
                {DeviceType.GPU, DeviceType.NPU},
            )

    def test_all_phase_metrics_are_calculated(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            HybridBundleMerger(merge_config(case)).merge()
            metrics = read_jsonl(case.output / "metrics/metrics.jsonl")
            latency = {
                row.metric_name: row
                for row in metrics
                if isinstance(row, MetricSample)
                and row.metric_name.startswith("latency.")
                and row.request_id == "request-1"
            }
            for name in (
                "latency.e2e",
                "latency.prefill",
                "latency.kv_export",
                "latency.kv_transform",
                "latency.kv_transfer",
                "latency.decode",
                "latency.sampling",
            ):
                self.assertIs(latency[name].availability, Availability.AVAILABLE)

    def test_missing_marker_is_partial_and_metric_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            markers = tuple(
                name for name in GPU_MARKERS if name != "kv_transform_start"
            )
            case = HybridCase(directory, gpu_markers=markers)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.PARTIAL)
            metrics = read_jsonl(case.output / "metrics/metrics.jsonl")
            transform_metric = next(
                row
                for row in metrics
                if row.metric_name == "latency.kv_transform"
            )
            self.assertIs(
                transform_metric.availability,
                Availability.NOT_AVAILABLE,
            )

    def test_zero_duration_is_available_not_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(
                directory,
                npu_times=(
                    1_900_000,
                    2_000_000,
                    2_100_000,
                    2_300_000,
                    2_300_000,
                    2_400_000,
                    2_500_000,
                ),
            )
            HybridBundleMerger(merge_config(case)).merge()
            metrics = read_jsonl(case.output / "metrics/metrics.jsonl")
            sampling = next(row for row in metrics if row.metric_name == "latency.sampling")
            self.assertIs(sampling.availability, Availability.AVAILABLE)
            self.assertEqual(sampling.value, 0)

    def test_high_uncertainty_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            offset = 50_000_000
            case = HybridCase(directory, same_host=False, offset_ns=offset)
            result = HybridBundleMerger(
                merge_config(
                    case,
                    fake_offset_ns=offset,
                    fake_delay_ns=2_000_000,
                    max_uncertainty_ns=100_000,
                )
            ).merge()
            self.assertIs(result.status, RunStatus.PARTIAL)
            self.assertGreater(result.uncertainty_ns, 100_000)

    def test_ambiguous_request_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            gpu_paths = RunPaths(case.gpu.parent, case.gpu.name)
            npu_paths = RunPaths(case.npu.parent, case.npu.name)
            gpu_events = [
                replace(
                    row,
                    request_id="gpu-request",
                    attributes={**row.attributes, "hybrid.transfer_id": "shared"},
                )
                for row in read_jsonl(gpu_paths.events)
            ]
            npu_source = read_jsonl(npu_paths.events)
            npu_events = [
                replace(
                    row,
                    event_id=f"{candidate}:{row.event_id}",
                    request_id=candidate,
                    timestamp_ns=row.timestamp_ns + offset,
                    attributes={**row.attributes, "hybrid.transfer_id": "shared"},
                )
                for candidate, offset in (("npu-1", 0), ("npu-2", 10_000))
                for row in npu_source
            ]
            write_jsonl(gpu_paths.events, gpu_events, overwrite=True)
            write_jsonl(npu_paths.events, npu_events, overwrite=True)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.PARTIAL)
            summary = json.loads(
                (case.output / "summary/hybrid_summary.json").read_text()
            )
            self.assertTrue(
                any(item["status"] == "ambiguous" for item in summary["joins"])
            )

    def test_ordering_violation_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            times = tuple(1_700_000 + index * 100_000 for index in range(len(NPU_MARKERS)))
            case = HybridCase(directory, npu_times=times)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.PARTIAL)
            self.assertTrue(any("ordering" in reason for reason in result.reasons))

    def test_uncertainty_overlap_is_not_definite(self):
        with tempfile.TemporaryDirectory() as directory:
            offset = 50_000_000
            times = tuple(
                value + offset
                for value in (
                    1_799_950,
                    2_000_000,
                    2_100_000,
                    2_200_000,
                    2_300_000,
                    2_400_000,
                    2_500_000,
                )
            )
            case = HybridCase(
                directory,
                same_host=False,
                offset_ns=offset,
                npu_times=times,
            )
            result = HybridBundleMerger(
                merge_config(
                    case,
                    fake_offset_ns=offset,
                    fake_delay_ns=100,
                    max_uncertainty_ns=1_000,
                )
            ).merge()
            self.assertIs(result.status, RunStatus.PARTIAL)
            self.assertFalse(
                any("definite marker" in reason for reason in result.reasons)
            )

    def test_source_descriptors_preserve_manifest_hash_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory, include_artifact=True)
            HybridBundleMerger(merge_config(case)).merge()
            descriptor = json.loads(
                (case.output / "sources/gpu-source.json").read_text()
            )
            self.assertEqual(len(descriptor["source_manifest_sha256"]), 64)
            self.assertEqual(descriptor["source_role"], "gpu")
            self.assertEqual(
                descriptor["source_artifacts"][0]["relative_path"],
                "raw/client/source.log",
            )

    def test_output_records_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            HybridBundleMerger(merge_config(case)).merge()
            validate_record(read_json(case.output / "manifest.json"))
            for relative in (
                "clocks/clock_domains.jsonl",
                "clocks/sync_points.jsonl",
                "clocks/transforms.jsonl",
                "events/events.jsonl",
                "metrics/metrics.jsonl",
                "artifacts/artifacts.jsonl",
            ):
                for record in read_jsonl(case.output / relative):
                    validate_record(record)

    def test_existing_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            HybridBundleMerger(merge_config(case)).merge()
            with self.assertRaises(FileExistsError):
                HybridBundleMerger(merge_config(case)).merge()

    def test_corrupt_source_stream_returns_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            (case.npu / "events/events.jsonl").write_text("{bad json}\n")
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            summary = json.loads(
                (case.output / "summary/hybrid_summary.json").read_text()
            )
            self.assertEqual(summary["status"], "failed")

    def test_corrupt_source_artifact_returns_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory, include_artifact=True)
            (case.gpu / "raw/client/source.log").write_text("changed\n")
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)

    def test_unknown_source_host_returns_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            paths = RunPaths(case.gpu.parent, case.gpu.name)
            events = [
                replace(row, host_id="unknown-host")
                for row in read_jsonl(paths.events)
            ]
            write_jsonl(paths.events, events, overwrite=True)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertIn("unknown host", result.reasons[0])

    def test_non_fake_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            manifest = read_json(case.gpu / "manifest.json")
            manifest.attributes["hybrid.fake_source"] = False
            from perfetto_hetero_profiler.schema import write_json
            write_json(case.gpu / "manifest.json", manifest, overwrite=True)
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertFalse(case.output.exists())

    def test_source_device_identity_collision_returns_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            manifest = read_json(case.npu / "manifest.json")
            manifest.devices[0] = replace(
                manifest.devices[0],
                device_id="gpu-0",
            )
            from perfetto_hetero_profiler.schema import write_json
            write_json(case.npu / "manifest.json", manifest, overwrite=True)
            paths = RunPaths(case.npu.parent, case.npu.name)
            write_jsonl(
                paths.events,
                [
                    replace(row, device_id="gpu-0")
                    for row in read_jsonl(paths.events)
                ],
                overwrite=True,
            )
            write_jsonl(
                paths.metrics,
                [
                    replace(row, device_id="gpu-0")
                    for row in read_jsonl(paths.metrics)
                ],
                overwrite=True,
            )
            result = HybridBundleMerger(merge_config(case)).merge()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertIn("identity collision", result.reasons[0])


class HybridCliTests(unittest.TestCase):
    def test_help(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                main(["merge", "hybrid", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--max-uncertainty-ns", stdout.getvalue())

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runs"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "merge",
                        "hybrid",
                        "--run-root",
                        str(output),
                        "--run-id",
                        "dry",
                        "--gpu-run",
                        str(root / "gpu"),
                        "--npu-run",
                        str(root / "npu"),
                        "--alignment-method",
                        "fake",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(stdout.getvalue())["executes"])

    def test_fake_merge_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "merge",
                        "hybrid",
                        "--run-root",
                        str(case.output_root),
                        "--run-id",
                        "hybrid",
                        "--gpu-run",
                        str(case.gpu),
                        "--npu-run",
                        str(case.npu),
                        "--alignment-method",
                        "same-clock-domain",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("succeeded:", stdout.getvalue())
            manifest = read_json(case.output / "manifest.json")
            self.assertIs(manifest.status, RunStatus.SUCCEEDED)

    def test_missing_gpu_run_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            code = main(
                [
                    "merge",
                    "hybrid",
                    "--run-root",
                    str(case.output_root),
                    "--run-id",
                    "hybrid",
                    "--gpu-run",
                    str(Path(directory) / "missing-gpu"),
                    "--npu-run",
                    str(case.npu),
                ]
            )
            self.assertEqual(code, 1)

    def test_missing_npu_run_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            code = main(
                [
                    "merge",
                    "hybrid",
                    "--run-root",
                    str(case.output_root),
                    "--run-id",
                    "hybrid",
                    "--gpu-run",
                    str(case.gpu),
                    "--npu-run",
                    str(Path(directory) / "missing-npu"),
                ]
            )
            self.assertEqual(code, 1)

    def test_non_fake_cli_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            case = HybridCase(directory)
            manifest = read_json(case.npu / "manifest.json")
            manifest.attributes["hybrid.fake_source"] = False
            from perfetto_hetero_profiler.schema import write_json
            write_json(case.npu / "manifest.json", manifest, overwrite=True)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "merge",
                        "hybrid",
                        "--run-root",
                        str(case.output_root),
                        "--run-id",
                        "hybrid",
                        "--gpu-run",
                        str(case.gpu),
                        "--npu-run",
                        str(case.npu),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("accepts fake source bundles only", stdout.getvalue())
            self.assertFalse(case.output.exists())

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "merge",
                    "hybrid",
                    "--run-root",
                    "/tmp/runs",
                    "--run-id",
                    "bad",
                    "--gpu-run",
                    "/tmp/gpu",
                    "--npu-run",
                    "/tmp/npu",
                    "--alignment-method",
                    "invalid",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_threshold_returns_two(self):
        code = main(
            [
                "merge",
                "hybrid",
                "--run-root",
                "/tmp/runs",
                "--run-id",
                "bad",
                "--gpu-run",
                "/tmp/gpu",
                "--npu-run",
                "/tmp/npu",
                "--max-uncertainty-ns",
                "-1",
                "--dry-run",
            ]
        )
        self.assertEqual(code, 2)

    def test_existing_version_cli_regression(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["version"]), 0)
        self.assertIn("hetero-profiler", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
