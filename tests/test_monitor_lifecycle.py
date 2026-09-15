"""Failure injection for the monitor run lifecycle shared by GPU and NPU."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perfetto_hetero_profiler.collectors import monitor
from perfetto_hetero_profiler.collectors.base import BaseCollector
from perfetto_hetero_profiler.collectors.gpu import (
    GpuRunCollector,
    GpuRunConfig,
    NvmlClient,
)
from perfetto_hetero_profiler.collectors.npu import (
    NpuRunCollector,
    NpuRunConfig,
    RblnSmiClient,
)
from perfetto_hetero_profiler.schema import (
    ProfileMode,
    RunStatus,
    read_json,
    read_jsonl,
)
from tests.support.gpu_fakes import FakeBinding, FakeDriverNotLoaded
from tests.support.gpu_fakes import client as fake_gpu_client
from tests.support.paths import RBLN_SMI_FIXTURES

CHILD = (sys.executable, "-c", "print('lifecycle')")
ONE_DEVICE = (RBLN_SMI_FIXTURES / "one_device.json").read_text(encoding="utf-8")


def fake_npu_client(raw=ONE_DEVICE, *, return_code=0, stderr=""):
    def runner(argv, **kwargs):
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "3.0.0\n", "")
        return subprocess.CompletedProcess(argv, return_code, raw, stderr)

    return RblnSmiClient(runner=runner)


class FailingCollector(BaseCollector):
    """Device telemetry that fails at one chosen lifecycle step."""

    def __init__(self, *, failing_step: str) -> None:
        super().__init__()
        self.failing_step = failing_step
        self.stop_calls = 0
        self.last_raw_snapshot: str | None = None
        self.last_raw_output: str | None = None

    def _fail_if_selected(self, step: str) -> None:
        if self.failing_step == step:
            raise RuntimeError(f"injected {step} failure")

    def _prepare(self) -> None:
        self._fail_if_selected("prepare")

    def _start(self) -> None:
        self._fail_if_selected("start")

    def _sample(self) -> list[object]:
        self._fail_if_selected("sample")
        return []

    def _stop(self) -> None:
        self.stop_calls += 1
        self._fail_if_selected("stop")


class GpuLifecycleFailureTests(unittest.TestCase):
    """Every GPU path also exercises the shared lifecycle template."""

    def config(self, root: Path, **overrides) -> GpuRunConfig:
        return GpuRunConfig(
            run_root=root,
            run_id="gpu-lifecycle",
            profile_mode=ProfileMode.MONITOR,
            sample_interval_ms=100,
            command=overrides.pop("command", CHILD),
            **overrides,
        )

    def collector(self, config: GpuRunConfig, **overrides) -> GpuRunCollector:
        return GpuRunCollector(
            config,
            gpu_client=overrides.pop("gpu_client", fake_gpu_client()),
            sleep=lambda _seconds: None,
            **overrides,
        )

    def test_discovery_failure_is_recorded_without_losing_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            client = NvmlClient(
                binding=FakeBinding(count_error=FakeDriverNotLoaded("no driver"))
            )
            result = self.collector(config, gpu_client=client).run()

            self.assertIs(result.status, RunStatus.PARTIAL)
            self.assertEqual(result.return_code, 0)
            manifest = read_json(config.paths.manifest)
            errors = manifest.attributes["vendor.collector_errors"]
            self.assertTrue(errors)
            self.assertIn("driver", errors[0].lower())
            device = manifest.devices[0]
            self.assertEqual(device.model, "unknown")
            self.assertEqual(device.status, "unknown")
            errors_path = config.paths.root / monitor.COLLECTOR_ERRORS_RELATIVE_PATH
            self.assertTrue(errors_path.is_file())

    def test_initial_manifest_write_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            with mock.patch.object(
                monitor,
                "publish_run_manifest",
                side_effect=OSError("initial manifest is unwritable"),
            ):
                with self.assertRaisesRegex(OSError, "initial manifest"):
                    self.collector(config).run()
            self.assertFalse(config.paths.manifest.exists())

    def test_final_manifest_write_failure_propagates_after_streams_land(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            calls: list[bool] = []

            def publish(path, manifest, *, initial=False):
                calls.append(initial)
                if initial:
                    return None
                raise OSError("final manifest is unwritable")

            with mock.patch.object(monitor, "publish_run_manifest", publish):
                with self.assertRaisesRegex(OSError, "final manifest"):
                    self.collector(config).run()

            self.assertEqual(calls, [True, False])
            # The normalized streams are published before the final manifest,
            # so the failure never costs already-collected evidence.
            self.assertTrue(config.paths.events.is_file())
            self.assertTrue(config.paths.metrics.is_file())
            self.assertTrue(config.paths.artifacts.is_file())

    def test_child_start_failure_is_reported_as_a_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(
                Path(directory) / "runs",
                command=(str(Path(directory) / "missing-binary"),),
            )
            result = self.collector(config).run()

            self.assertIs(result.status, RunStatus.FAILED)
            self.assertEqual(result.return_code, 127)
            manifest = read_json(config.paths.manifest)
            errors = manifest.attributes["vendor.collector_errors"]
            self.assertTrue(
                any(error.startswith("run orchestration:") for error in errors)
            )

    def test_raw_artifact_write_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            with mock.patch.object(
                Path,
                "write_text",
                side_effect=OSError("raw artifact is unwritable"),
            ):
                with self.assertRaisesRegex(OSError, "raw artifact"):
                    self.collector(config).run()

    def test_nvml_shutdown_failure_never_blocks_the_remaining_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            binding = FakeBinding(shutdown_error=FakeDriverNotLoaded("late unload"))
            result = self.collector(
                config, gpu_client=NvmlClient(binding=binding)
            ).run()

            self.assertIs(result.status, RunStatus.PARTIAL)
            manifest = read_json(config.paths.manifest)
            errors = manifest.attributes["vendor.collector_errors"]
            self.assertTrue(
                any("GpuTelemetryCollector stop" in error for error in errors)
            )
            self.assertEqual(binding.shutdown_calls, 1)
            # The system collector is stopped after the device collector; its
            # samples must still be published despite the device stop failure.
            metric_names = {metric.metric_name for metric in read_jsonl(config.paths.metrics)}
            self.assertTrue(
                any(name.startswith("resource.system.") for name in metric_names)
            )


class InjectedTelemetryFailureTests(unittest.TestCase):
    """Telemetry start/sample/stop failures keep the run and its cleanup."""

    def run_with_failing_telemetry(self, directory: str, step: str):
        config = GpuRunConfig(
            run_root=Path(directory) / "runs",
            run_id="telemetry-failure",
            profile_mode=ProfileMode.MONITOR,
            sample_interval_ms=100,
            command=CHILD,
        )
        telemetry = FailingCollector(failing_step=step)
        collector = GpuRunCollector(
            config, gpu_client=fake_gpu_client(), sleep=lambda _seconds: None
        )
        with mock.patch.object(
            GpuRunCollector, "_device_telemetry", return_value=telemetry
        ):
            return config, telemetry, collector.run()

    def assert_error_recorded(self, config, result, status, fragment):
        self.assertIs(result.status, status)
        errors = read_json(config.paths.manifest).attributes[
            "vendor.collector_errors"
        ]
        self.assertTrue(
            any(fragment in error for error in errors),
            msg=f"{fragment!r} missing from {errors!r}",
        )

    def test_telemetry_start_failure_is_recorded_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            config, telemetry, result = self.run_with_failing_telemetry(
                directory, "start"
            )
            # A start failure aborts before the child runs, so the run is
            # failed rather than partial, and the collector is still stopped.
            self.assert_error_recorded(
                config,
                result,
                RunStatus.FAILED,
                "run orchestration: injected start failure",
            )
            self.assertEqual(result.return_code, 127)
            self.assertEqual(telemetry.stop_calls, 1)

    def test_telemetry_sample_failure_is_recorded_per_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            config, telemetry, result = self.run_with_failing_telemetry(
                directory, "sample"
            )
            self.assert_error_recorded(
                config,
                result,
                RunStatus.PARTIAL,
                "FailingCollector: injected sample failure",
            )
            self.assertEqual(telemetry.stop_calls, 1)

    def test_telemetry_stop_failure_does_not_block_later_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, telemetry, result = self.run_with_failing_telemetry(
                directory, "stop"
            )
            self.assert_error_recorded(
                config,
                result,
                RunStatus.PARTIAL,
                "FailingCollector stop: injected stop failure",
            )
            self.assertEqual(telemetry.stop_calls, 1)
            # Cleanup runs in reverse registration order and each stop is
            # guarded, so the earlier failure still leaves system metrics.
            metric_names = {
                metric.metric_name for metric in read_jsonl(config.paths.metrics)
            }
            self.assertTrue(
                any(name.startswith("resource.system.") for name in metric_names)
            )


class NpuLifecycleFailureTests(unittest.TestCase):
    """The NPU collector keeps its own error semantics on the shared template."""

    def config(self, root: Path, **overrides) -> NpuRunConfig:
        return NpuRunConfig(
            run_root=root,
            run_id="npu-lifecycle",
            profile_mode=ProfileMode.MONITOR,
            sample_interval_ms=100,
            command=overrides.pop("command", CHILD),
            **overrides,
        )

    def test_discovery_failure_records_rbln_errors_and_error_events(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            result = NpuRunCollector(
                config,
                npu_client=fake_npu_client(return_code=1, stderr="rbln-smi exploded"),
                sleep=lambda _seconds: None,
            ).run()

            self.assertIs(result.status, RunStatus.PARTIAL)
            manifest = read_json(config.paths.manifest)
            errors = manifest.attributes["vendor.collector_errors"]
            self.assertTrue(errors)
            self.assertEqual(manifest.devices[0].model, "unknown")
            # Unlike the GPU collector, NPU runs also publish collector errors
            # as events; that device-specific behaviour must survive.
            events = read_jsonl(config.paths.events)
            error_events = [
                event for event in events if event.event_name == "collector.error"
            ]
            self.assertEqual(len(error_events), len(errors))
            self.assertEqual(result.event_count, 3 + len(errors))

    def test_child_start_failure_is_reported_as_a_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(
                Path(directory) / "runs",
                command=(str(Path(directory) / "missing-binary"),),
            )
            result = NpuRunCollector(
                config, npu_client=fake_npu_client(), sleep=lambda _seconds: None
            ).run()

            self.assertIs(result.status, RunStatus.FAILED)
            self.assertEqual(result.return_code, 127)

    def test_final_manifest_write_failure_propagates_after_streams_land(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")

            def publish(path, manifest, *, initial=False):
                if initial:
                    return None
                raise OSError("final manifest is unwritable")

            with mock.patch.object(monitor, "publish_run_manifest", publish):
                with self.assertRaisesRegex(OSError, "final manifest"):
                    NpuRunCollector(
                        config,
                        npu_client=fake_npu_client(),
                        sleep=lambda _seconds: None,
                    ).run()

            self.assertTrue(config.paths.events.is_file())
            self.assertTrue(config.paths.artifacts.is_file())


if __name__ == "__main__":
    unittest.main()
