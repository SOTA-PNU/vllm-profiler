"""Configuration, dry-run, and CLI tests for GPU vLLM collection."""

import contextlib
import gzip
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.gpu import vllm_collection as collection_module
from perfetto_hetero_profiler.gpu.vllm_collection import (
    GpuVllmCollectionConfig,
    GpuVllmCollectionRunner,
    build_vllm_collection_plan,
)
from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation
from perfetto_hetero_profiler.collectors.gpu import NvmlClient, NvmlError
from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    read_json,
    read_jsonl,
    RunStatus,
)


class CollectionConfigTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "run_root": Path("/runs"),
            "run_id": "gpu-monitor-test",
            "model": Path("/models/qwen"),
            "profile_mode": "monitor",
            "host": "127.0.0.1",
            "port": 18080,
            "startup_timeout_sec": 180,
            "request_timeout_sec": 60,
            "shutdown_timeout_sec": 60,
            "sample_interval_ms": 500,
            "gpu_memory_utilization": 0.25,
            "max_model_len": 2048,
            "warmup_requests": 1,
            "measured_requests": 2,
            "max_output_tokens": 8,
            "vllm_bin": Path("/venv/bin/vllm"),
        }
        values.update(changes)
        return GpuVllmCollectionConfig(**values)

    def test_plan_is_nonexecuting(self) -> None:
        plan = build_vllm_collection_plan(self.config())
        self.assertFalse(plan["executes"])
        self.assertFalse(plan["workload"]["stores_prompt_or_generated_text"])

    def test_plan_does_not_create_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(
                run_root=Path(directory), run_id="does-not-exist"
            )
            build_vllm_collection_plan(config)
            self.assertFalse(config.paths.root.exists())

    def test_torch_plan_uses_separate_directory(self) -> None:
        config = self.config(profile_mode="torch", run_id="torch-run")
        argv = build_vllm_collection_plan(config)["server_argv"]
        self.assertIn(
            "--profiler-config.torch_profiler_dir=/runs/torch-run/raw/gpu/torch",
            argv,
        )

    def test_nsys_plan_uses_run_local_output(self) -> None:
        config = self.config(profile_mode="nsys", run_id="nsys-run")
        argv = build_vllm_collection_plan(config)["server_argv"]
        index = argv.index("--output")
        self.assertEqual(argv[index + 1], "/runs/nsys-run/raw/gpu/nsys/vllm-smoke")

    def test_invalid_interval_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "500 or 1000"):
            self.config(sample_interval_ms=250)

    def test_measured_request_limit_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "measured_requests"):
            self.config(measured_requests=3)

    def test_relative_run_root_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "run_root"):
            self.config(run_root=Path("runs"))


class CollectionCliTests(unittest.TestCase):
    def test_help_lists_profiler_modes(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(SystemExit, "0"):
                main(["collect", "gpu-vllm", "--help"])
        self.assertIn("monitor,torch,nsys", output.getvalue())

    def test_dry_run_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "collect",
                        "gpu-vllm",
                        "--run-root",
                        str(run_root),
                        "--run-id",
                        "dry-run",
                        "--model",
                        "/models/qwen",
                        "--vllm-bin",
                        "/venv/bin/vllm",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(run_root.exists())
            self.assertFalse(json.loads(output.getvalue())["executes"])


def FakeGpuClient():
    from tests.test_gpu_telemetry import FakeBinding

    return NvmlClient(binding=FakeBinding())


class FakeServer:
    def __init__(self, _config, stdout, stderr):
        self.stdout = stdout
        self.stderr = stderr
        self.process = None
        self.started_monotonic_ns = None
        self.base_url = "http://127.0.0.1:18080"

    def start(self):
        self.stdout.parent.mkdir(parents=True, exist_ok=True)
        self.stdout.write_text("server stdout\n", encoding="utf-8")
        self.stderr.write_text("", encoding="utf-8")
        self.process = SimpleNamespace(pid=os.getpid())
        self.started_monotonic_ns = 100

    def wait_ready(self, _timeout):
        return 200

    def stop(self, _timeout):
        return 0


class FakeCompletionClient:
    def __init__(self, _base_url, timeout_sec):
        self.timeout_sec = timeout_sec
        self.close_calls = 0

    def close(self):
        self.close_calls += 1

    def complete(self, *, request_id, **_kwargs):
        offset = 1000 if "warmup" in request_id else 2000
        return CompletionObservation(
            request_id=request_id,
            received_ns=offset,
            token_timestamps_ns=(offset + 10, offset + 20),
            done_ns=offset + 30,
            input_tokens=4,
            output_tokens=2,
            total_tokens=6,
            http_status=200,
        )


class CollectionFakeIntegrationTests(unittest.TestCase):
    def test_discovery_failure_shuts_down_nvml_before_server_start(self) -> None:
        from tests.test_gpu_telemetry import FakeBinding

        with tempfile.TemporaryDirectory() as directory:
            binding = FakeBinding(devices=[])
            config = CollectionConfigTests().config(
                run_root=Path(directory), run_id="no-gpu", measured_requests=1
            )
            runner = GpuVllmCollectionRunner(
                config,
                gpu_client=NvmlClient(binding=binding),
                server_factory=FakeServer,
                client_factory=FakeCompletionClient,
            )
            with self.assertRaisesRegex(NvmlError, "no GPUs"):
                runner.run()
            self.assertEqual(binding.shutdown_calls, 1)

    def test_monitor_writes_valid_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="fake-monitor",
                warmup_requests=1,
                measured_requests=1,
            )
            result = GpuVllmCollectionRunner(
                config,
                gpu_client=FakeGpuClient(),
                server_factory=FakeServer,
                client_factory=FakeCompletionClient,
            ).run()
            self.assertEqual(result.status, RunStatus.SUCCEEDED)
            manifest = read_json(config.paths.manifest)
            self.assertEqual(manifest.status, RunStatus.SUCCEEDED)
            self.assertTrue(read_jsonl(config.paths.events))
            self.assertTrue(read_jsonl(config.paths.metrics))
            artifacts = read_jsonl(config.paths.artifacts)
            self.assertGreaterEqual(len(artifacts), 4)
            nvml = next(
                artifact for artifact in artifacts if artifact.artifact_id == "nvml-last"
            )
            self.assertEqual(nvml.relative_path, "raw/gpu/nvml-last.json")
            self.assertEqual((nvml.format, nvml.producer), ("json", "nvml"))
            summary = json.loads(
                (config.paths.root / "summary/smoke.json").read_text()
            )
            self.assertEqual(summary["final"]["status"], "succeeded")
            request_summary = (
                config.paths.root / "raw/client/requests.json"
            ).read_text()
            self.assertNotIn("Explain a computer cache", request_summary)

    def test_torch_trace_discovery_excludes_auxiliary_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="fake-torch",
                profile_mode="torch",
                measured_requests=1,
            )
            torch_root = config.paths.root / "raw/gpu/torch"
            torch_root.mkdir(parents=True)
            trace = torch_root / "rank0.pt.trace.json.gz"
            trace.write_bytes(b"trace")
            (torch_root / "profiler_out_0.txt").write_text("table")
            files = GpuVllmCollectionRunner(
                config, gpu_client=FakeGpuClient()
            )._profile_files(config.paths.root)
            self.assertEqual(files, [trace])

    def test_torch_plain_and_gzip_traces_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="valid-traces",
                profile_mode="torch",
                measured_requests=1,
            )
            torch_root = config.paths.root / "raw/gpu/torch"
            torch_root.mkdir(parents=True)
            plain = torch_root / "api.pt.trace.json"
            plain.write_text('{"traceEvents":[]}', encoding="utf-8")
            compressed = torch_root / "worker.pt.trace.json.gz"
            with gzip.open(compressed, "wt", encoding="utf-8") as stream:
                json.dump({"traceEvents": []}, stream)
            errors = []
            result = GpuVllmCollectionRunner(
                config, gpu_client=FakeGpuClient()
            )._validate_profile_files([plain, compressed], errors)
            self.assertEqual(result["valid_files"], 2)
            self.assertEqual(errors, [])

    def test_empty_torch_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="empty-trace",
                profile_mode="torch",
                measured_requests=1,
            )
            trace = config.paths.root / "raw/gpu/torch/empty.pt.trace.json"
            trace.parent.mkdir(parents=True)
            trace.touch()
            errors = []
            result = GpuVllmCollectionRunner(
                config, gpu_client=FakeGpuClient()
            )._validate_profile_files([trace], errors)
            self.assertEqual(result["valid_files"], 0)
            self.assertIn("empty profiler artifact", errors[0])

    def test_corrupt_plain_and_gzip_traces_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="corrupt-traces",
                profile_mode="torch",
                measured_requests=1,
            )
            torch_root = config.paths.root / "raw/gpu/torch"
            torch_root.mkdir(parents=True)
            plain = torch_root / "bad.pt.trace.json"
            plain.write_text("{not-json", encoding="utf-8")
            compressed = torch_root / "bad.pt.trace.json.gz"
            compressed.write_bytes(b"\x1f\x8btruncated")
            errors = []
            result = GpuVllmCollectionRunner(
                config, gpu_client=FakeGpuClient()
            )._validate_profile_files([plain, compressed], errors)
            self.assertEqual(result["valid_files"], 0)
            self.assertEqual(len(errors), 2)

    def test_torch_auxiliary_is_raw_log_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = CollectionConfigTests().config(
                run_root=Path(directory),
                run_id="artifact-kinds",
                profile_mode="torch",
                measured_requests=1,
            )
            root = config.paths.root
            torch_root = root / "raw/gpu/torch"
            torch_root.mkdir(parents=True)
            trace = torch_root / "rank0.pt.trace.json"
            trace.write_text('{"traceEvents":[]}', encoding="utf-8")
            auxiliary = torch_root / "profiler_out_0.txt"
            auxiliary.write_text("table", encoding="utf-8")
            stdout = root / "raw/gpu/stdout.log"
            stderr = root / "raw/gpu/stderr.log"
            requests = root / "raw/client/requests.json"
            requests.parent.mkdir(parents=True)
            for path in (stdout, stderr, requests):
                path.write_text("", encoding="utf-8")
            artifacts = GpuVllmCollectionRunner(
                config, gpu_client=FakeGpuClient()
            )._artifacts(root, stdout, stderr, requests, [trace], None)
            by_path = {item.relative_path: item for item in artifacts}
            self.assertEqual(len(by_path), len(artifacts))
            self.assertEqual(
                by_path["raw/gpu/torch/rank0.pt.trace.json"].artifact_kind,
                ArtifactKind.TORCH_TRACE,
            )
            aux = by_path["raw/gpu/torch/profiler_out_0.txt"]
            self.assertEqual(aux.artifact_kind, ArtifactKind.RAW_LOG)
            self.assertEqual(aux.format, "txt")
            self.assertEqual(aux.producer, "gpu-vllm-smoke")


class CollectionFailureInjectionTests(unittest.TestCase):
    def run_case(
        self,
        directory,
        *,
        mode="monitor",
        failures=(),
        torch_artifact="valid",
        telemetry_errors=(),
    ):
        failures = set(failures)
        calls = []
        config = CollectionConfigTests().config(
            run_root=Path(directory), run_id="injected", profile_mode=mode,
            warmup_requests=1, measured_requests=1,
        )

        class Gpu:
            def __init__(self):
                self.shutdown_calls = 0

            def query(self):
                calls.append("gpu.query")
                if "discovery" in failures:
                    raise NvmlError("injected discovery failure")
                return SimpleNamespace(rows=(SimpleNamespace(
                    device_id="gpu-0", name="Fake GPU", index=0,
                    memory_total_bytes=SimpleNamespace(value=1024),
                ),))

            def shutdown(self):
                calls.append("gpu.shutdown")
                self.shutdown_calls += 1
                if "nvml_shutdown" in failures:
                    raise RuntimeError("injected NVML shutdown failure")

        class Telemetry:
            instances = []

            def __init__(self, _config, _pid_provider, _gpu):
                self.metrics = []
                self.errors = list(telemetry_errors)
                self.gpu = SimpleNamespace(last_raw_snapshot=None)
                self.stop_calls = 0
                self.__class__.instances.append(self)

            def start(self):
                calls.append("telemetry.start")

            def stop(self):
                calls.append("telemetry.stop")
                self.stop_calls += 1
                if "telemetry_stop" in failures:
                    raise RuntimeError("injected telemetry stop failure")

        class Server:
            instances = []

            def __init__(self, server_config, stdout, stderr):
                self.config = server_config
                self.stdout = stdout
                self.stderr = stderr
                self.process = None
                self.started_monotonic_ns = None
                self.base_url = "http://127.0.0.1:18080"
                self.stop_calls = 0
                self.alive = False
                self.__class__.instances.append(self)

            def start(self):
                calls.append("server.start")
                self.stdout.parent.mkdir(parents=True, exist_ok=True)
                self.stdout.write_text("stdout\n", encoding="utf-8")
                self.stderr.write_text("stderr\n", encoding="utf-8")
                if "server_start" in failures:
                    raise RuntimeError("injected server start failure")
                self.process = SimpleNamespace(pid=os.getpid())
                self.started_monotonic_ns = 100
                self.alive = True

            def wait_ready(self, _timeout):
                calls.append("server.wait_ready")
                if "readiness" in failures:
                    raise TimeoutError("injected readiness failure")
                return 200

            def stop(self, _timeout):
                calls.append("server.stop")
                self.stop_calls += 1
                self.alive = False
                if self.config.nsys_output is not None:
                    report = self.config.nsys_output.with_suffix(".nsys-rep")
                    report.parent.mkdir(parents=True, exist_ok=True)
                    report.write_bytes(b"nsys")
                if "server_stop" in failures:
                    raise RuntimeError("injected server stop failure")
                return 0

        class Client:
            instances = []

            def __init__(self, _base_url, timeout_sec):
                calls.append("client.create")
                self.timeout_sec = timeout_sec
                self.close_calls = 0
                self.__class__.instances.append(self)

            def complete(self, *, request_id, **_kwargs):
                kind = "warmup" if "warmup" in request_id else "measured"
                calls.append(f"client.{kind}")
                if kind in failures:
                    raise RuntimeError(f"injected {kind} failure")
                offset = 1000 if kind == "warmup" else 2000
                return CompletionObservation(
                    request_id=request_id, received_ns=offset,
                    token_timestamps_ns=(offset + 10, offset + 20),
                    done_ns=offset + 30, input_tokens=4, output_tokens=2,
                    total_tokens=6, http_status=200,
                )

            def close(self):
                calls.append("client.close")
                self.close_calls += 1
                if "client_close" in failures:
                    raise RuntimeError("injected client close failure")

        profile_stop_calls = []

        def post_empty(_base_url, endpoint, _timeout):
            calls.append(endpoint)
            if endpoint == "/start_profile" and "torch_start" in failures:
                raise RuntimeError("injected torch start failure")
            if endpoint == "/stop_profile":
                profile_stop_calls.append(endpoint)
                if "torch_stop" in failures:
                    raise RuntimeError("injected torch stop failure")
                trace = config.paths.root / "raw/gpu/torch/trace.pt.trace.json"
                if torch_artifact == "missing":
                    return
                trace.parent.mkdir(parents=True, exist_ok=True)
                if torch_artifact == "empty":
                    trace.touch()
                elif torch_artifact == "corrupt":
                    trace.write_text("{bad", encoding="utf-8")
                else:
                    trace.write_text('{"traceEvents":[]}\n', encoding="utf-8")

        gpu = Gpu()
        self.last_injection_gpu = gpu
        self.last_injection_calls = calls
        clock = iter(range(1_700_000_000_000_000_000, 1_700_000_000_000_001_000))
        runner = GpuVllmCollectionRunner(
            config, gpu_client=gpu, server_factory=Server,
            client_factory=Client, unix_time_ns=clock.__next__,
        )
        with patch.object(collection_module, "_TelemetryThread", Telemetry), \
                patch.object(collection_module, "post_empty", post_empty):
            result = runner.run()
        return SimpleNamespace(
            config=config, result=result, calls=calls, gpu=gpu,
            telemetry=Telemetry.instances[-1], server=Server.instances[-1],
            client=Client.instances[-1] if Client.instances else None,
            profile_stop_calls=profile_stop_calls,
        )

    def assert_closed(self, case):
        self.assertEqual(case.telemetry.stop_calls, 1)
        self.assertEqual(case.gpu.shutdown_calls, 1)
        self.assertEqual(case.server.stop_calls, 1)
        self.assertFalse(case.server.alive)
        if case.client is not None:
            self.assertEqual(case.client.close_calls, 1)
            self.assertLess(case.calls.index("client.close"),
                            case.calls.index("telemetry.stop"))
        if case.profile_stop_calls:
            self.assertLess(case.calls.index("/stop_profile"),
                            case.calls.index("client.close"))
        self.assertLess(case.calls.index("telemetry.stop"),
                        case.calls.index("gpu.shutdown"))
        self.assertLess(case.calls.index("gpu.shutdown"),
                        case.calls.index("server.stop"))
        for path in (
            case.config.paths.manifest, case.config.paths.events,
            case.config.paths.metrics, case.config.paths.artifacts,
            case.config.paths.root / "summary/smoke.json",
        ):
            self.assertTrue(path.is_file(), path)
        self.assertFalse(list(case.config.paths.root.rglob("*.tmp")))

    def test_monitor_torch_and_nsys_success_cleanup_once(self):
        for mode in ("monitor", "torch", "nsys"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                case = self.run_case(directory, mode=mode)
                self.assertEqual(case.result.status, RunStatus.SUCCEEDED)
                self.assertEqual(read_json(case.config.paths.manifest).status,
                                 RunStatus.SUCCEEDED)
                self.assert_closed(case)
                self.assertLess(case.calls.index("client.close"),
                                case.calls.index("telemetry.stop"))
                self.assertLess(case.calls.index("telemetry.stop"),
                                case.calls.index("gpu.shutdown"))
                self.assertLess(case.calls.index("gpu.shutdown"),
                                case.calls.index("server.stop"))
                if mode == "torch":
                    self.assertEqual(case.profile_stop_calls, ["/stop_profile"])

    def test_execution_failures_preserve_failed_bundle_and_cleanup(self):
        cases = (
            ("server_start", "monitor"), ("readiness", "monitor"),
            ("warmup", "monitor"), ("torch_start", "torch"),
            ("measured", "torch"), ("torch_stop", "torch"),
        )
        for failure, mode in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                case = self.run_case(directory, mode=mode, failures={failure})
                self.assertEqual(case.result.status, RunStatus.FAILED)
                manifest = read_json(case.config.paths.manifest)
                self.assertEqual(manifest.status, RunStatus.FAILED)
                summary = json.loads((case.config.paths.root / "summary/smoke.json").read_text())
                self.assertEqual(summary["final"]["errors"], list(case.result.errors))
                self.assert_closed(case)
                if failure == "measured":
                    self.assertEqual(case.profile_stop_calls, ["/stop_profile"])
                if failure == "torch_stop":
                    self.assertEqual(case.profile_stop_calls, ["/stop_profile"])

    def test_cleanup_failures_do_not_block_later_cleanup(self):
        expected = {
            "client_close": "client cleanup: injected client close failure",
            "telemetry_stop": "telemetry cleanup: injected telemetry stop failure",
            "nvml_shutdown": "NVML cleanup: injected NVML shutdown failure",
            "server_stop": "server cleanup: injected server stop failure",
        }
        for failure, message in expected.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                case = self.run_case(directory, failures={failure})
                self.assertIn(message, case.result.errors)
                self.assertEqual(case.result.status, RunStatus.FAILED)
                self.assert_closed(case)

    def test_telemetry_sample_error_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.run_case(
                directory, telemetry_errors=("GpuTelemetryCollector: sample failed",)
            )
            self.assertEqual(
                case.result.errors, ("GpuTelemetryCollector: sample failed",)
            )
            self.assertEqual(case.result.status, RunStatus.FAILED)
            self.assert_closed(case)

    def test_missing_empty_and_corrupt_torch_artifacts_fail(self):
        expected = {
            "missing": "torch profiler produced no report",
            "empty": "empty profiler artifact",
            "corrupt": "invalid torch trace",
        }
        for artifact, message in expected.items():
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                case = self.run_case(directory, mode="torch", torch_artifact=artifact)
                self.assertEqual(case.result.status, RunStatus.FAILED)
                self.assertTrue(any(message in error for error in case.result.errors))
                self.assert_closed(case)

    def test_multiple_errors_remain_ordered_and_cleanup_completes(self):
        failures = {
            "measured", "torch_stop", "client_close", "telemetry_stop",
            "nvml_shutdown", "server_stop",
        }
        with tempfile.TemporaryDirectory() as directory:
            case = self.run_case(directory, mode="torch", failures=failures)
            self.assertEqual(
                case.result.errors[:5],
                (
                    "RuntimeError: injected measured failure",
                    "stop_profile cleanup: injected torch stop failure",
                    "client cleanup: injected client close failure",
                    "telemetry cleanup: injected telemetry stop failure",
                    "NVML cleanup: injected NVML shutdown failure",
                ),
            )
            self.assertEqual(
                case.result.errors[5], "server cleanup: injected server stop failure"
            )
            self.assertEqual(case.profile_stop_calls, ["/stop_profile"])
            self.assert_closed(case)

    def test_discovery_failure_only_shuts_down_nvml(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NvmlError, "injected discovery failure"):
                self.run_case(directory, failures={"discovery"})
            self.assertEqual(self.last_injection_gpu.shutdown_calls, 1)
            self.assertEqual(self.last_injection_calls, ["gpu.query", "gpu.shutdown"])


if __name__ == "__main__":
    unittest.main()
