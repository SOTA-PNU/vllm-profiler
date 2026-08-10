"""GPU run planning, CLI, and CPU-only orchestration tests."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.collectors.gpu import (
    GpuDeviceInfo,
    GpuRunCollector,
    GpuRunConfig,
    NvidiaSmiClient,
    build_detailed_profile_plan,
    build_gpu_run_plan,
    build_nsys_argv,
)
from perfetto_hetero_profiler.schema import (
    ProfileMode,
    RunMode,
    RunStatus,
    read_json,
    read_jsonl,
)


GPU_ROW = "0, Fake GPU, 0, 0, 1000, N/A\n"


def fake_gpu_client():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, GPU_ROW, "")

    return NvidiaSmiClient(runner=runner)


class PlanningTests(unittest.TestCase):
    def config(self, root):
        return GpuRunConfig(
            run_root=root,
            run_id="gpu-run",
            profile_mode=ProfileMode.MONITOR,
            sample_interval_ms=100,
            command=(sys.executable, "-c", "print('ok')"),
        )

    def test_minimum_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, ">= 100"):
                GpuRunConfig(
                    run_root=Path(directory),
                    run_id="run",
                    profile_mode=ProfileMode.MONITOR,
                    sample_interval_ms=99,
                    command=("true",),
                )

    def test_dry_run_plan_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            plan = build_gpu_run_plan(self.config(root))
            self.assertFalse(root.exists())
            self.assertIs(plan["executes"], False)
            self.assertIn("metrics", plan["outputs"])

    def test_nsys_plan_is_argv(self):
        argv = build_nsys_argv(("python3", "work.py"))
        self.assertEqual(argv[:2], ("nsys", "profile"))
        self.assertEqual(argv[-2:], ("python3", "work.py"))

    def test_nsys_absolute_output_rejected(self):
        with self.assertRaises(ValueError):
            build_nsys_argv(("true",), "/tmp/report")

    def test_detailed_plan_contains_vllm_endpoints(self):
        plan = build_detailed_profile_plan(("true",))
        self.assertEqual(plan.torch.start_endpoint, "/start_profile")
        self.assertEqual(plan.torch.stop_endpoint, "/stop_profile")

    def test_detailed_execution_rejected_without_creating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            config = GpuRunConfig(
                run_root=root,
                run_id="run",
                profile_mode=ProfileMode.DETAILED_PROFILE,
                sample_interval_ms=100,
                command=("true",),
            )
            with self.assertRaises(NotImplementedError):
                GpuRunCollector(config).run()
            self.assertFalse(root.exists())


class GpuRunIntegrationTests(unittest.TestCase):
    def test_cpu_only_child_writes_schema_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GpuRunConfig(
                run_root=Path(directory),
                run_id="smoke",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "print('hello')"),
                gpu_devices=(GpuDeviceInfo(0, "Fake GPU", 1000 * 1024 * 1024),),
            )
            result = GpuRunCollector(config, gpu_client=fake_gpu_client()).run()
            manifest = read_json(config.paths.manifest)
            events = read_jsonl(config.paths.events)
            metrics = read_jsonl(config.paths.metrics)
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertIn(result.status, {RunStatus.SUCCEEDED, RunStatus.PARTIAL})
            self.assertIs(manifest.mode, RunMode.GPU_ONLY)
            self.assertEqual(len(events), 3)
            self.assertEqual(
                [event.event_name for event in events],
                [
                    "collector.run_start",
                    "collector.child_process_start",
                    "collector.child_process_end",
                ],
            )
            self.assertGreaterEqual(len(metrics), 5)
            self.assertGreaterEqual(len(artifacts), 3)
            self.assertEqual(
                (config.paths.root / "raw/client/stdout.log").read_text().strip(),
                "hello",
            )

    def test_child_failure_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GpuRunConfig(
                run_root=Path(directory),
                run_id="failure",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "raise SystemExit(3)"),
                gpu_devices=(GpuDeviceInfo(0, "Fake GPU"),),
            )
            result = GpuRunCollector(config, gpu_client=fake_gpu_client()).run()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertEqual(result.return_code, 3)

    def test_child_start_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GpuRunConfig(
                run_root=Path(directory),
                run_id="missing-command",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=("command-that-does-not-exist-gpu-collector",),
                gpu_devices=(GpuDeviceInfo(0, "Fake GPU"),),
            )
            result = GpuRunCollector(config, gpu_client=fake_gpu_client()).run()
            manifest = read_json(config.paths.manifest)
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertEqual(result.return_code, 127)
            self.assertTrue(manifest.attributes["vendor.collector_errors"])
            self.assertIn(
                "raw/system/collector-errors.json",
                {artifact.relative_path for artifact in artifacts},
            )

    def test_orchestrator_timeout_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GpuRunConfig(
                run_root=Path(directory),
                run_id="timeout",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "import time; time.sleep(5)"),
                timeout_sec=0.02,
                gpu_devices=(GpuDeviceInfo(0, "Fake GPU"),),
            )
            result = GpuRunCollector(config, gpu_client=fake_gpu_client()).run()
            events = read_jsonl(config.paths.events)
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertTrue(events[-1].attributes["vendor.timed_out"])

    def test_artifact_paths_are_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GpuRunConfig(
                run_root=Path(directory),
                run_id="paths",
                profile_mode=ProfileMode.MONITOR,
                sample_interval_ms=100,
                command=(sys.executable, "-c", "pass"),
                gpu_devices=(GpuDeviceInfo(0, "Fake GPU"),),
            )
            GpuRunCollector(config, gpu_client=fake_gpu_client()).run()
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertTrue(
                all(not Path(record.relative_path).is_absolute() for record in artifacts)
            )


class GpuCliTests(unittest.TestCase):
    def run_cli(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main(argv)
            except SystemExit as error:
                code = int(error.code)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_collect_gpu_help(self):
        code, stdout, _ = self.run_cli(["collect", "gpu", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("--sample-interval-ms", stdout)

    def test_dry_run_exit_zero_and_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            code, stdout, stderr = self.run_cli(
                [
                    "collect", "gpu", "--run-root", str(root), "--run-id", "dry",
                    "--dry-run", "--command", "python3", "-c", "print('hello')",
                ]
            )
            plan = json.loads(stdout)
            self.assertEqual((code, stderr), (0, ""))
            self.assertFalse(root.exists())
            self.assertIs(plan["executes"], False)

    def test_invalid_interval_exit_two(self):
        code, _, stderr = self.run_cli(
            [
                "collect", "gpu", "--run-root", "/tmp/runs", "--run-id", "dry",
                "--sample-interval-ms", "99", "--dry-run", "--command", "true",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn(">= 100", stderr)

    def test_missing_command_exit_two(self):
        code, _, _ = self.run_cli(
            ["collect", "gpu", "--run-root", "/tmp/runs", "--run-id", "dry"]
        )
        self.assertEqual(code, 2)

    def test_existing_version_command_regression(self):
        code, stdout, stderr = self.run_cli(["version"])
        self.assertEqual((code, stdout.strip(), stderr), (0, "hetero-profiler 0.1.0", ""))


if __name__ == "__main__":
    unittest.main()
