"""NPU telemetry, planning, CLI, and fake CPU-only smoke tests."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.collectors.base import CollectorError, CollectorState
from perfetto_hetero_profiler.collectors.npu import (
    NpuDeviceInfo,
    NpuRunCollector,
    NpuRunConfig,
    NpuTelemetryCollector,
    RblnSmiClient,
    build_npu_run_plan,
    build_rbln_profile_plan,
)
from perfetto_hetero_profiler.schema import (
    Availability,
    DeviceType,
    ProfileMode,
    RunMode,
    RunStatus,
    read_json,
    read_jsonl,
)


FIXTURES = Path(__file__).parent / "fixtures" / "rbln_smi"
ONE_DEVICE = (FIXTURES / "one_device.json").read_text(encoding="utf-8")
UNSUPPORTED = (FIXTURES / "unsupported_fields.json").read_text(encoding="utf-8")


def fake_client(raw=ONE_DEVICE, *, return_code=0, stderr=""):
    def runner(argv, **kwargs):
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "3.0.0\n", "")
        return subprocess.CompletedProcess(argv, return_code, raw, stderr)

    return RblnSmiClient(runner=runner)


def run_cli(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = main(argv)
        except SystemExit as error:
            code = int(error.code)
    return code, stdout.getvalue(), stderr.getvalue()


class NpuTelemetryTests(unittest.TestCase):
    def collector(self, raw=ONE_DEVICE):
        ticks = iter((1_000_000_000, 1_100_000_000, 1_200_000_000))
        return NpuTelemetryCollector(
            run_id="run",
            host_id="host-a",
            clock_domain_id="clock-a",
            sample_interval_ms=100,
            client=fake_client(raw),
            known_npu_indices=(0,),
            monotonic_ns=lambda: next(ticks),
        )

    def test_lifecycle_reuses_base_collector(self):
        collector = self.collector()
        self.assertIs(collector.state, CollectorState.CREATED)
        collector.prepare()
        collector.start()
        collector.sample()
        collector.stop()
        self.assertIs(collector.state, CollectorState.STOPPED)

    def test_sample_before_start_rejected(self):
        with self.assertRaises(CollectorError):
            self.collector().sample()

    def test_duplicate_start_rejected(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        with self.assertRaises(CollectorError):
            collector.start()

    def test_stop_is_idempotent(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        collector.stop()
        collector.stop()
        self.assertIs(collector.state, CollectorState.STOPPED)

    def test_three_official_npu_metrics(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        metrics = collector.sample()
        self.assertEqual(
            {metric.metric_name for metric in metrics},
            {
                "resource.npu.utilization",
                "resource.npu.memory_used",
                "resource.npu.power",
            },
        )

    def test_metric_identity_fields(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        metric = collector.sample()[0]
        self.assertEqual(metric.device_type, DeviceType.NPU)
        self.assertEqual(metric.device_id, "npu-0")
        self.assertEqual(metric.host_id, "host-a")
        self.assertEqual(metric.clock_domain_id, "clock-a")

    def test_actual_zero_stays_available(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        metrics = collector.sample()
        self.assertTrue(
            all(
                metric.availability is Availability.AVAILABLE
                for metric in metrics
                if metric.value == 0
            )
        )

    def test_command_error_produces_error_metrics(self):
        collector = NpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=100,
            client=fake_client("", return_code=2, stderr="failed"),
            known_npu_indices=(0, 1),
            monotonic_ns=lambda: 1,
        )
        collector.prepare()
        collector.start()
        metrics = collector.sample()
        self.assertEqual(len(metrics), 6)
        self.assertTrue(all(item.availability is Availability.ERROR for item in metrics))
        self.assertTrue(all("failed" in item.reason for item in metrics))

    def test_structurally_unsupported_metrics_emit_once(self):
        collector = self.collector(UNSUPPORTED)
        collector.prepare()
        collector.start()
        first = collector.sample()
        second = collector.sample()
        self.assertEqual(len(first), 3)
        self.assertEqual(
            [metric.metric_name for metric in second],
            ["resource.npu.utilization"],
        )

    def test_unsupported_state_is_independent_per_device(self):
        document = json.loads((FIXTURES / "two_devices.json").read_text())
        for device in document["devices"]:
            device.pop("memory")
            device.pop("card_power")
        ticks = iter((1_000_000_000, 1_100_000_000))
        collector = NpuTelemetryCollector(
            run_id="run",
            host_id="host-a",
            clock_domain_id="clock-a",
            sample_interval_ms=100,
            client=fake_client(json.dumps(document)),
            known_npu_indices=(0, 1),
            monotonic_ns=lambda: next(ticks),
        )
        collector.prepare()
        collector.start()
        first = collector.sample()
        second = collector.sample()
        unsupported = [
            metric
            for metric in first
            if metric.availability is Availability.NOT_AVAILABLE
        ]
        self.assertEqual(
            {(metric.device_id, metric.metric_name) for metric in unsupported},
            {
                ("npu-0", "resource.npu.memory_used"),
                ("npu-0", "resource.npu.power"),
                ("npu-1", "resource.npu.memory_used"),
                ("npu-1", "resource.npu.power"),
            },
        )
        self.assertEqual(
            {metric.device_id for metric in second},
            {"npu-0", "npu-1"},
        )

    def test_unsupported_state_resets_for_new_collector_instance(self):
        first = self.collector(UNSUPPORTED)
        second = self.collector(UNSUPPORTED)
        for collector in (first, second):
            collector.prepare()
            collector.start()
            self.assertEqual(len(collector.sample()), 3)

    def test_available_sample_resets_structural_missing_suppression(self):
        samples = iter((UNSUPPORTED, ONE_DEVICE, UNSUPPORTED))

        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, next(samples), "")

        ticks = iter((1_000_000_000, 1_100_000_000, 1_200_000_000))
        collector = NpuTelemetryCollector(
            run_id="run",
            host_id="host-a",
            clock_domain_id="clock-a",
            sample_interval_ms=100,
            client=RblnSmiClient(runner=runner),
            monotonic_ns=lambda: next(ticks),
        )
        collector.prepare()
        collector.start()
        collector.sample()
        collector.sample()
        third = collector.sample()
        self.assertIn(
            "resource.npu.power",
            {metric.metric_name for metric in third},
        )

    def test_temperature_never_becomes_normalized_metric(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        names = {metric.metric_name for metric in collector.sample()}
        self.assertNotIn("resource.npu.temperature", names)
        self.assertFalse(any("temperature" in name for name in names))

    def test_last_raw_output_is_retained(self):
        collector = self.collector()
        collector.prepare()
        collector.start()
        collector.sample()
        self.assertEqual(collector.last_raw_output, ONE_DEVICE)

    def test_unexpected_client_failure_marks_collector_failed(self):
        class BrokenClient:
            def query(self):
                raise ValueError("unexpected parser defect")

        collector = NpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=100,
            client=BrokenClient(),
            monotonic_ns=lambda: 1,
        )
        collector.prepare()
        collector.start()
        with self.assertRaisesRegex(ValueError, "parser defect"):
            collector.sample()
        self.assertIs(collector.state, CollectorState.FAILED)


class NpuPlanningTests(unittest.TestCase):
    def config(self, root, mode=ProfileMode.MONITOR):
        return NpuRunConfig(
            run_root=root,
            run_id="npu-run",
            profile_mode=mode,
            sample_interval_ms=100,
            command=(sys.executable, "-c", "print('ok')"),
        )

    def test_minimum_interval_enforced(self):
        with self.assertRaisesRegex(ValueError, ">= 100"):
            NpuRunConfig(
                run_root="/tmp/runs",
                run_id="run",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=99,
                command=("true",),
            )

    def test_negative_device_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            NpuRunConfig(
                run_root="/tmp/runs",
                run_id="run",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=("true",),
                device_ids=(-1,),
            )

    def test_monitor_plan_disables_profiler(self):
        plan = build_npu_run_plan(self.config("/tmp/runs"))
        self.assertIs(plan["rbln_profiler_enabled"], False)
        self.assertNotIn("detailed_profile", plan)

    def test_dry_plan_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            plan = build_npu_run_plan(self.config(root))
            self.assertFalse(root.exists())
            self.assertIs(plan["executes"], False)

    def test_detailed_plan_is_pure_and_unverified(self):
        plan = build_npu_run_plan(
            self.config("/tmp/runs", ProfileMode.DETAILED_PROFILE)
        )
        profile = plan["detailed_profile"]
        self.assertIs(profile["execution_verified"], False)
        self.assertIs(profile["activate_profiler"], True)

    def test_profile_plan_names_capture_and_start_stop_apis(self):
        plan = build_rbln_profile_plan()
        self.assertEqual(plan.capture_reports_api, "rebel.capture_reports()")
        self.assertIn("profiler_start", plan.start_api)
        self.assertIn("profiler_done", plan.stop_api)

    def test_profile_plan_does_not_expose_private_vendor_module(self):
        plan = build_rbln_profile_plan()
        self.assertNotIn("core_ori", repr(plan))
        self.assertNotIn("profiler_backend", repr(plan))

    def test_profile_output_is_relative(self):
        self.assertEqual(
            build_rbln_profile_plan().output_directory,
            "raw/npu/rbln-profiler",
        )

    def test_absolute_profile_output_rejected(self):
        with self.assertRaises(ValueError):
            build_rbln_profile_plan("/tmp/report")

    def test_detailed_execution_is_blocked_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            config = self.config(root, ProfileMode.DETAILED_PROFILE)
            with self.assertRaises(NotImplementedError):
                NpuRunCollector(config, npu_client=fake_client()).run()
            self.assertFalse(root.exists())


class NpuFakeSmokeTests(unittest.TestCase):
    def config(self, root, run_id="fake-smoke", command=None):
        return NpuRunConfig(
            run_root=root,
            run_id=run_id,
            profile_mode=ProfileMode.MONITOR,
            sample_interval_ms=100,
            command=command
            or (
                sys.executable,
                "-c",
                "import time; print('fake-npu-ok'); time.sleep(0.12)",
            ),
            npu_devices=(
                NpuDeviceInfo(
                    0, "RBLN-CA22", "normal", 16877879296, "3.0.0"
                ),
            ),
        )

    def test_fake_cpu_only_smoke_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            result = NpuRunCollector(config, npu_client=fake_client()).run()
            self.assertIs(result.status, RunStatus.SUCCEEDED)
            self.assertEqual(result.return_code, 0)

    def test_fake_smoke_bundle_validates_and_has_npu_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            NpuRunCollector(config, npu_client=fake_client()).run()
            manifest = read_json(config.paths.manifest)
            self.assertIs(manifest.mode, RunMode.NPU_ONLY)
            self.assertIs(manifest.status, RunStatus.SUCCEEDED)
            self.assertEqual(manifest.devices[0].device_type, DeviceType.NPU)
            self.assertEqual(manifest.software[1].version, "3.0.0")

    def test_fake_smoke_has_clock_events_metrics_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            result = NpuRunCollector(config, npu_client=fake_client()).run()
            self.assertEqual(len(read_jsonl(config.paths.clock_domains)), 1)
            events = read_jsonl(config.paths.events)
            metrics = read_jsonl(config.paths.metrics)
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertEqual(
                [event.event_name for event in events],
                [
                    "collector.run_start",
                    "collector.child_process_start",
                    "collector.child_process_end",
                ],
            )
            self.assertTrue(
                {
                    "resource.npu.utilization",
                    "resource.npu.memory_used",
                    "resource.npu.power",
                    "resource.cpu.utilization",
                    "resource.system.memory_used",
                    "resource.cpu.memory_used",
                }.issubset({metric.metric_name for metric in metrics})
            )
            self.assertEqual(
                {artifact.relative_path for artifact in artifacts},
                {
                    "raw/client/stdout.log",
                    "raw/client/stderr.log",
                    "raw/npu/rbln-smi-last.json",
                },
            )
            self.assertEqual(result.event_count, 3)

    def test_stdout_and_stderr_are_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            NpuRunCollector(config, npu_client=fake_client()).run()
            self.assertEqual(
                (config.paths.root / "raw/client/stdout.log").read_text().strip(),
                "fake-npu-ok",
            )
            self.assertEqual(
                (config.paths.root / "raw/client/stderr.log").read_text(), ""
            )

    def test_raw_rbln_smi_json_is_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            NpuRunCollector(config, npu_client=fake_client()).run()
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertIn(
                "raw/npu/rbln-smi-last.json",
                {artifact.relative_path for artifact in artifacts},
            )

    def test_artifact_size_and_sha256_match_files(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            NpuRunCollector(config, npu_client=fake_client()).run()
            for artifact in read_jsonl(config.paths.artifacts):
                path = config.paths.root / artifact.relative_path
                self.assertEqual(artifact.size_bytes, path.stat().st_size)
                self.assertEqual(
                    artifact.sha256,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_sensitive_raw_fields_do_not_enter_normalized_records(self):
        document = json.loads(ONE_DEVICE)
        document["devices"][0].update(
            {
                "sid": "fixture-sensitive-sid",
                "uuid": "fixture-sensitive-uuid",
                "board_info": "fixture-sensitive-board",
                "pci": {"bus_id": "fixture-sensitive-pci"},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            config = NpuRunConfig(
                run_root=Path(directory),
                run_id="sensitive-normalization",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "print('ok')"),
            )
            NpuRunCollector(
                config, npu_client=fake_client(json.dumps(document))
            ).run()
            normalized = (
                config.paths.manifest.read_text()
                + config.paths.metrics.read_text()
                + config.paths.events.read_text()
                + config.paths.artifacts.read_text()
            )
            self.assertNotIn("fixture-sensitive-", normalized)
            self.assertIn(
                "fixture-sensitive-",
                (config.paths.root / "raw/npu/rbln-smi-last.json").read_text(),
            )

    def test_child_failure_marks_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(
                Path(directory),
                command=(sys.executable, "-c", "raise SystemExit(4)"),
            )
            result = NpuRunCollector(config, npu_client=fake_client()).run()
            self.assertEqual((result.status, result.return_code), (RunStatus.FAILED, 4))

    def test_child_timeout_marks_failed_and_records_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            config = NpuRunConfig(
                run_root=Path(directory),
                run_id="timeout",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "import time; time.sleep(5)"),
                timeout_sec=0.02,
                npu_devices=(NpuDeviceInfo(0, "RBLN-CA22", "normal"),),
            )
            result = NpuRunCollector(config, npu_client=fake_client()).run()
            events = read_jsonl(config.paths.events)
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertTrue(events[2].attributes["vendor.timed_out"])

    def test_telemetry_failure_marks_partial_and_writes_error_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            result = NpuRunCollector(
                config,
                npu_client=fake_client("", return_code=2, stderr="no telemetry"),
            ).run()
            self.assertIs(result.status, RunStatus.PARTIAL)
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertIn(
                "raw/system/collector-errors.json",
                {artifact.relative_path for artifact in artifacts},
            )

    def test_existing_run_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            config.paths.create()
            config.paths.manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                NpuRunCollector(config, npu_client=fake_client()).run()


class NpuCliTests(unittest.TestCase):
    def base(self, root):
        return [
            "collect",
            "npu",
            "--run-root",
            str(root),
            "--run-id",
            "dry",
            "--dry-run",
            "--command",
            "python3",
            "-c",
            "print('hello')",
        ]

    def test_collect_npu_help(self):
        code, stdout, _ = run_cli(["collect", "npu", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("--device-id", stdout)

    def test_monitor_dry_run(self):
        code, stdout, stderr = run_cli(self.base("/tmp/runs"))
        plan = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(plan["mode"], "npu_only")
        self.assertEqual(plan["profile_mode"], "monitor")

    def test_detailed_profile_dry_run(self):
        argv = self.base("/tmp/runs")
        argv[argv.index("--dry-run"):argv.index("--dry-run")] = [
            "--profile-mode",
            "detailed-profile",
        ]
        code, stdout, _ = run_cli(argv)
        self.assertEqual(code, 0)
        self.assertIn("detailed_profile", json.loads(stdout))

    def test_dry_run_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            code, _, _ = run_cli(self.base(root))
            self.assertEqual(code, 0)
            self.assertFalse(root.exists())

    def test_invalid_interval_returns_two(self):
        argv = self.base("/tmp/runs")
        argv[argv.index("--dry-run"):argv.index("--dry-run")] = [
            "--sample-interval-ms",
            "99",
        ]
        code, _, stderr = run_cli(argv)
        self.assertEqual(code, 2)
        self.assertIn(">= 100", stderr)

    def test_missing_command_returns_two(self):
        code, _, _ = run_cli(
            ["collect", "npu", "--run-root", "/tmp/runs", "--run-id", "dry"]
        )
        self.assertEqual(code, 2)

    def test_invalid_device_returns_two(self):
        argv = self.base("/tmp/runs")
        argv[argv.index("--dry-run"):argv.index("--dry-run")] = [
            "--device-id",
            "-1",
        ]
        code, _, stderr = run_cli(argv)
        self.assertEqual(code, 2)
        self.assertIn("non-negative", stderr)

    def test_selected_devices_appear_in_plan(self):
        argv = self.base("/tmp/runs")
        argv[argv.index("--dry-run"):argv.index("--dry-run")] = [
            "--device-id",
            "0",
            "--device-id",
            "1",
        ]
        code, stdout, _ = run_cli(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["device_ids"], [0, 1])

    def test_existing_gpu_cli_regression(self):
        code, stdout, _ = run_cli(["collect", "gpu", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("--sample-interval-ms", stdout)


if __name__ == "__main__":
    unittest.main()
