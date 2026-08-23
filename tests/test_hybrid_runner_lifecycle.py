"""CPU-only failure and interruption tests for hybrid process ownership."""

import json
from dataclasses import dataclass
from pathlib import Path
import threading
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

from perfetto_hetero_profiler.hybrid.runner import (
    HybridRunner,
    HybridRunnerError,
    _Layout,
    _TelemetryWorker,
    _profile_call,
    _shutdown_integrity,
    _wait_http,
    _wait_runtime_marker_completion,
)
from perfetto_hetero_profiler.hybrid import (
    AlignmentMethod,
    HybridBundleMerger,
    HybridMergeConfig,
)
from perfetto_hetero_profiler.hybrid.runner_config import load_hybrid_runner_config
from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation
from perfetto_hetero_profiler.perfetto.loader import load_hybrid_run
from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    DeviceType,
    RunPaths,
    RunStatus,
    read_json,
    read_jsonl,
)

from tests.hybrid_fixtures import GPU_MARKERS, NPU_MARKERS, build_source_bundle
from tests.test_hybrid_runner_config import document


class _Telemetry:
    instances = []

    def __init__(self, *_args):
        self.started = False
        self.stopped = False
        self.errors = []
        self.gpu_metrics = []
        self.npu_metrics = []
        self.system_metrics = []
        self.gpu = mock.Mock(last_raw_output=None)
        self.npu = mock.Mock(last_raw_output=None)
        self.boundary_roles = []
        self.request_window = None
        self.actions = []
        self.instances.append(self)

    def start(self):
        self.started = True
        self.actions.append("start")

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.actions.append("stop")

    def capture_boundary(self, role):
        self.boundary_roles.append(role)
        self.actions.append(f"capture:{role}")
        return {}

    def set_request_window(self, start_ns, end_ns):
        self.request_window = (start_ns, end_ns)
        self.actions.append("request-window")

    def lifecycle(self):
        return {
            "requested_interval_ms": 500,
            "request_start_ns": None,
            "request_end_ns": None,
            "boundaries": {},
            "streams": {"gpu": {}, "npu": {}, "system": {}},
            "errors": [],
        }


class _Result:
    return_code = 0
    terminated = True
    killed = False


class _Process:
    instances = []
    fail_name = None

    def __init__(self, _spec, stdout, _stderr):
        self.spec = _spec
        self.name = stdout.name.split(".")[0]
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self):
        if self.name == self.fail_name:
            raise RuntimeError(f"{self.name} start failed")
        self.started = True

    def poll(self):
        return None

    def stop_leader_first(self, **_kwargs):
        self.stopped = True
        return _Result()


class _ClosingClient:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def complete(self, *, request_id, **_kwargs):
        return CompletionObservation(
            request_id=request_id,
            received_ns=10,
            token_timestamps_ns=(20,),
            done_ns=30,
            input_tokens=5,
            output_tokens=1,
            total_tokens=6,
            http_status=200,
        )

    def close(self):
        self.closed = True
        raise RuntimeError("close failed")


class _SuccessfulClient(_ClosingClient):
    def close(self):
        self.closed = True


class _WrittenTelemetry:
    gpu_metrics = []
    npu_metrics = []
    system_metrics = []
    gpu = mock.Mock(last_raw_output=None)
    npu = mock.Mock(last_raw_output=None)

    @staticmethod
    def lifecycle():
        return {
            "requested_interval_ms": 500,
            "request_start_ns": None,
            "request_end_ns": None,
            "boundaries": {},
            "streams": {"gpu": {}, "npu": {}, "system": {}},
            "errors": [],
        }


@dataclass
class _WorkerMetric:
    timestamp_ns: int
    interval_ns: int | None
    attributes: dict


class _BlockingCollector:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def sample(self):
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test collector release timed out")
        return [_WorkerMetric(100 + self.calls, 10, {})]


class _SequencedCollector:
    def __init__(self):
        self.calls = 0
        self.started = [threading.Event() for _ in range(3)]
        self.release = [threading.Event() for _ in range(3)]

    def sample(self):
        call = self.calls
        self.calls += 1
        if call < len(self.started):
            self.started[call].set()
            if not self.release[call].wait(timeout=2):
                raise TimeoutError("test collector release timed out")
        return [_WorkerMetric(100 + self.calls, 10, {})]


def _config(root: Path):
    value = document(root)
    for directory in (
        root / "model", root / "cache", root / "prefill/bin", root / "decode/bin"
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        root / "prefill/bin/vllm", root / "decode/bin/vllm", root / "python",
        root / "trace_processor_shell", root / "nsys",
    ):
        path.write_text("executable", encoding="utf-8")
    (root / "cache/model-0.rbln").write_bytes(b"zero")
    (root / "cache/model-2.rbln").write_bytes(b"two")
    path = root / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return load_hybrid_runner_config(path)


class TelemetryWorkerTests(unittest.TestCase):
    def test_boundary_result_survives_later_background_sample(self) -> None:
        collector = _SequencedCollector()
        target = []
        worker = _TelemetryWorker(
            name="device",
            collector=collector,
            target=target,
            interval_sec=0,
            errors=[],
        )
        worker.start()
        self.assertTrue(collector.started[0].wait(timeout=1))
        ticket = worker.request("baseline")
        collector.release[0].set()
        self.assertTrue(collector.started[1].wait(timeout=1))
        collector.release[1].set()
        self.assertTrue(collector.started[2].wait(timeout=1))

        sample = worker.wait(ticket, timeout_sec=1)

        collector.release[2].set()
        worker.stop(timeout_sec=1)
        self.assertEqual(sample["role"], "baseline")
        self.assertEqual(
            [item.attributes["telemetry.sample_role"] for item in target[:2]],
            ["baseline", "background"],
        )

    def test_final_boundary_joins_inflight_poll_without_duplicate_query(self) -> None:
        collector = _BlockingCollector()
        target = []
        errors = []
        worker = _TelemetryWorker(
            name="device",
            collector=collector,
            target=target,
            interval_sec=60,
            errors=errors,
        )
        worker.start()
        self.assertTrue(collector.started.wait(timeout=1))
        ticket = worker.request("final")
        collector.release.set()
        sample = worker.wait(ticket, timeout_sec=1)
        worker.stop(timeout_sec=1)

        self.assertEqual(collector.calls, 1)
        self.assertEqual(sample["role"], "final")
        self.assertEqual(target[0].attributes["telemetry.sample_role"], "final")
        self.assertEqual(errors, [])

    def test_final_boundary_is_the_terminal_sample(self) -> None:
        collector = _SequencedCollector()
        worker = _TelemetryWorker(
            name="device",
            collector=collector,
            target=[],
            interval_sec=0,
            errors=[],
        )
        worker.start()
        self.assertTrue(collector.started[0].wait(timeout=1))
        ticket = worker.request("final")
        collector.release[0].set()

        sample = worker.wait(ticket, timeout_sec=1)

        self.assertEqual(sample["role"], "final")
        self.assertFalse(collector.started[1].wait(timeout=0.05))
        with self.assertRaisesRegex(HybridRunnerError, "not running"):
            worker.request("final")
        worker.stop(timeout_sec=1)
        self.assertEqual(collector.calls, 1)

    def test_sampler_exception_is_terminal_and_stop_is_idempotent(self) -> None:
        class BrokenCollector:
            calls = 0

            def sample(self):
                self.calls += 1
                raise RuntimeError("sample failed")

        collector = BrokenCollector()
        errors = []
        worker = _TelemetryWorker(
            name="broken",
            collector=collector,
            target=[],
            interval_sec=0.001,
            errors=errors,
        )
        worker.start()
        worker.thread.join(timeout=1)
        worker.stop(timeout_sec=1)
        worker.stop(timeout_sec=1)

        self.assertEqual(collector.calls, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("sample failed", errors[0])


class ShutdownIntegrityTests(unittest.TestCase):
    def test_known_nixl_segfault_is_diagnostic_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = _Layout(Path(directory), "run")
            raw = layout.coordinator / "raw"
            raw.mkdir(parents=True)
            (raw / "decode.stderr.log").write_text(
                "Segfault encountered in rtnl_tc_unregister\n",
                encoding="utf-8",
            )
            result = _shutdown_integrity(layout, {})

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["reason"], "native_sigsegv_rtnl_tc_unregister")
        self.assertTrue(result["demo_only"])


class HybridRunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        _Telemetry.instances.clear()
        _Process.instances.clear()
        _Process.fail_name = None

    def test_partial_startup_failure_cleans_only_started_children(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _Process.fail_name = "prefill"
            runner = HybridRunner(
                _config(root), run_root=root / "runs", run_id="partial",
                profile_mode="monitor", process_factory=_Process,
            )
            with mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._Telemetry", _Telemetry
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._port_available",
                return_value=True,
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._wait_http",
                return_value=1,
            ), mock.patch.object(
                runner, "_profile_metadata", return_value=None
            ), mock.patch.object(runner, "_write_sources", return_value=None):
                result = runner.run()
            by_name = {item.name: item for item in _Process.instances}
            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertTrue(by_name["decode"].stopped)
            self.assertFalse(by_name["prefill"].stopped)
            self.assertFalse(by_name["proxy"].started)
            self.assertTrue(_Telemetry.instances[0].stopped)

    def test_server_pythonpath_uses_configured_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            runner = HybridRunner(
                config, run_root=root / "runs", run_id="pythonpath",
                profile_mode="monitor", process_factory=_Process,
            )
            runner._processes(
                {
                    "prefill": ("prefill",),
                    "decode": ("decode",),
                    "proxy": ("proxy",),
                }
            )
            by_name = {item.name: item for item in _Process.instances}
            self.assertEqual(
                by_name["prefill"].spec.env_overrides["PYTHONPATH"],
                str(config.prefill.pythonpath),
            )
            self.assertEqual(
                by_name["decode"].spec.env_overrides["PYTHONPATH"],
                str(config.decode.pythonpath),
            )
            self.assertIn("PYTHONPATH", by_name["decode"].spec.env_allowlist)

    def test_keyboard_interrupt_runs_reverse_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = HybridRunner(
                _config(root), run_root=root / "runs", run_id="interrupt",
                profile_mode="monitor", process_factory=_Process,
            )
            with mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._Telemetry", _Telemetry
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._port_available",
                return_value=True,
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._wait_http",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run()
            by_name = {item.name: item for item in _Process.instances}
            self.assertTrue(by_name["decode"].stopped)
            self.assertFalse(by_name["prefill"].started)
            self.assertTrue(_Telemetry.instances[0].stopped)

    def test_marker_completion_requires_proxy_and_decode_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "event_name": "decode_loop_end",
                    "correlation_id": "request-1",
                },
                {
                    "event_name": "response_done",
                    "correlation_id": "request-1",
                },
            ]
            (root / "markers.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            _wait_runtime_marker_completion(root, {"request-1"}, 0.1)

    def test_marker_completion_timeout_reports_missing_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "markers.jsonl").write_text(
                json.dumps(
                    {
                        "event_name": "response_done",
                        "correlation_id": "request-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                HybridRunnerError, "decode_loop_end"
            ):
                _wait_runtime_marker_completion(root, {"request-1"}, 0.01)

    def test_readiness_timeout_is_bounded(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch(
            "perfetto_hetero_profiler.hybrid.runner.time.monotonic",
            side_effect=(0.0, 2.0),
        ), mock.patch(
            "perfetto_hetero_profiler.hybrid.runner.urlopen",
            side_effect=URLError("not ready"),
        ), mock.patch(
            "perfetto_hetero_profiler.hybrid.runner.time.sleep"
        ):
            with self.assertRaisesRegex(TimeoutError, "readiness timed out"):
                _wait_http("http://127.0.0.1:1", "/v1/models", process, 1.0)

    def test_readiness_detects_early_process_exit(self) -> None:
        process = mock.Mock()
        process.poll.return_value = 17
        with self.assertRaisesRegex(HybridRunnerError, "code 17"):
            _wait_http("http://127.0.0.1:1", "/v1/models", process, 1.0)

    def test_profiler_start_failure_is_classified_at_api_boundary(self) -> None:
        with mock.patch(
            "perfetto_hetero_profiler.hybrid.runner.urlopen",
            side_effect=URLError("refused"),
        ):
            with self.assertRaisesRegex(HybridRunnerError, "/start_profile"):
                _profile_call("http://127.0.0.1:1", "/start_profile", 1.0)

    def test_client_close_failure_does_not_skip_profiler_or_source_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = HybridRunner(
                _config(root), run_root=root / "runs", run_id="close-failure",
                profile_mode="gpu-torch", process_factory=_Process,
                client_factory=_ClosingClient,
            )
            start = {
                "before_monotonic_ns": 1,
                "after_monotonic_ns": 2,
                "before_unix_ns": 101,
                "after_unix_ns": 102,
                "http_status": 200,
                "response_body": "started",
            }
            stop = {
                "before_monotonic_ns": 40,
                "after_monotonic_ns": 50,
                "before_unix_ns": 140,
                "after_unix_ns": 150,
                "http_status": 200,
                "response_body": "stopped",
            }
            with mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._Telemetry", _Telemetry
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._port_available",
                return_value=True,
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._wait_http",
                return_value=1,
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._wait_runtime_marker_completion"
            ), mock.patch(
                "perfetto_hetero_profiler.hybrid.runner._profile_call",
                side_effect=(start, stop),
            ) as profile_call, mock.patch.object(
                runner, "_profile_metadata", return_value=None
            ), mock.patch.object(
                runner, "_write_sources"
            ) as write_sources:
                result = runner.run()
            self.assertIs(result.status, RunStatus.FAILED)
            self.assertTrue(any("client close" in error for error in result.errors))
            self.assertEqual(profile_call.call_count, 2)
            write_sources.assert_called_once()
            telemetry = _Telemetry.instances[0]
            self.assertEqual(telemetry.boundary_roles, ["baseline", "final"])
            self.assertEqual(telemetry.request_window[1], 30)
            self.assertLess(
                telemetry.actions.index("request-window"),
                telemetry.actions.index("capture:final"),
            )
            self.assertLess(
                telemetry.actions.index("capture:final"),
                telemetry.actions.index("stop"),
            )

    def test_source_clock_metadata_is_schema_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runner = HybridRunner(
                _config(root), run_root=runs, run_id="source-write",
                profile_mode="monitor", process_factory=_Process,
            )
            RunPaths(runs, "source-write-gpu").create()
            RunPaths(runs, "source-write-npu").create()
            marker_root = runs / "source-write-coordinator/raw/runtime_markers"
            marker_root.mkdir(parents=True)
            (marker_root / "proxy-markers.jsonl").write_text(
                "", encoding="utf-8"
            )
            observation = CompletionObservation(
                request_id="request-1", received_ns=10,
                token_timestamps_ns=(20,), done_ns=30,
                input_tokens=5, output_tokens=1, total_tokens=6,
                http_status=200,
            )
            with mock.patch.object(
                runner, "_marker_events", return_value=([], [])
            ):
                runner._write_sources(
                    warmups=[], observations=[observation],
                    telemetry=_WrittenTelemetry(), profile=None, errors=[],
                )
            for role in ("gpu", "npu"):
                clock = read_jsonl(
                    runs / f"source-write-{role}/clocks/clock_domains.jsonl"
                )[0]
                self.assertEqual(
                    clock.attributes["clock.source"], "time.monotonic_ns"
                )

    def test_marker_failure_preserves_request_and_telemetry_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runner = HybridRunner(
                _config(root), run_root=runs, run_id="marker-failure",
                profile_mode="monitor", process_factory=_Process,
            )
            RunPaths(runs, "marker-failure-gpu").create()
            RunPaths(runs, "marker-failure-npu").create()
            marker_root = runs / "marker-failure-coordinator/raw/runtime_markers"
            marker_root.mkdir(parents=True)
            (marker_root / "proxy-markers.jsonl").write_text(
                "", encoding="utf-8"
            )
            observation = CompletionObservation(
                request_id="request-1", received_ns=10,
                token_timestamps_ns=(20,), done_ns=30,
                input_tokens=5, output_tokens=1, total_tokens=6,
                http_status=200,
            )

            with mock.patch.object(
                runner,
                "_marker_events",
                side_effect=HybridRunnerError("marker validation failed"),
            ), self.assertRaisesRegex(HybridRunnerError, "marker validation failed"):
                runner._write_sources(
                    warmups=[], observations=[observation],
                    telemetry=_WrittenTelemetry(), profile=None, errors=[],
                )

            self.assertTrue(
                (runs / "marker-failure-coordinator/requests.json").is_file()
            )
            self.assertTrue(
                (runs / "marker-failure-coordinator/telemetry_lifecycle.json").is_file()
            )
            for role in ("gpu", "npu"):
                bundle = runs / f"marker-failure-{role}"
                self.assertTrue((bundle / "metrics/metrics.jsonl").is_file())
                self.assertTrue(
                    (bundle / "summary/telemetry_lifecycle.json").is_file()
                )
                manifest = read_json(bundle / "manifest.json")
                self.assertIs(manifest.status, RunStatus.FAILED)

    def test_torch_artifact_uses_native_converter_format_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runner = HybridRunner(
                _config(root / "assets"), run_root=runs, run_id="torch-format",
                profile_mode="gpu-torch", process_factory=_Process,
            )
            RunPaths(runs, "torch-format-gpu").create()
            for relative in (
                "raw/client/measured_requests.jsonl",
                "events/events.jsonl",
                "metrics/metrics.jsonl",
            ):
                (runs / "torch-format-gpu" / relative).write_text(
                    "", encoding="utf-8"
                )
            trace = (
                runs / "torch-format-gpu/raw/gpu/torch/"
                "rank0.pt.trace.json.gz"
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_bytes(b"gzip-trace")
            (runs / "torch-format-gpu/clocks/profiler_alignment.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (runs / "torch-format-gpu/summary/detailed_profile.json").write_text(
                "{}\n", encoding="utf-8"
            )
            profile = {
                "root": runs / "torch-format-gpu",
                "alignment": {
                    "native_clock_domain": "gpu:torch-chrome-trace",
                },
                "detail": {
                    "files": [{"path": trace.relative_to(trace.parents[3]).as_posix()}],
                },
            }

            artifacts = runner._artifacts(profile["root"], "gpu", profile)
            torch_artifact = next(
                item
                for item in artifacts
                if item.artifact_kind is ArtifactKind.TORCH_TRACE
            )

            self.assertEqual(torch_artifact.format, "chrome_trace_json_gzip")
            self.assertEqual(
                torch_artifact.clock_domain_id, "gpu:torch-chrome-trace"
            )

    def test_runner_closeout_publishes_loader_compatible_detached_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            gpu = build_source_bundle(
                runs / "closeout-gpu", device_type=DeviceType.GPU,
                host_id="localhost", clock_domain_id="host-monotonic",
                markers=GPU_MARKERS,
                timestamps=tuple(
                    1_000_000 + index * 100_000
                    for index in range(len(GPU_MARKERS))
                ),
            )
            npu = build_source_bundle(
                runs / "closeout-npu", device_type=DeviceType.NPU,
                host_id="localhost", clock_domain_id="host-monotonic",
                markers=NPU_MARKERS,
                timestamps=tuple(
                    1_900_000 + index * 100_000
                    for index in range(len(NPU_MARKERS))
                ),
            )
            merged = HybridBundleMerger(
                HybridMergeConfig(
                    run_root=runs, run_id="closeout", gpu_run=gpu,
                    npu_run=npu,
                    alignment_method=AlignmentMethod.SAME_CLOCK_DOMAIN,
                    coordinator_host_id="localhost",
                    canonical_clock_domain_id="hybrid-canonical",
                )
            ).merge()
            self.assertIs(merged.status, RunStatus.SUCCEEDED)
            coordinator = runs / "closeout-coordinator"
            coordinator.mkdir()
            (coordinator / "result.json").write_text(
                '{"status":"succeeded"}\n', encoding="utf-8"
            )
            runner = HybridRunner(
                _config(root / "assets"), run_root=runs, run_id="closeout",
                profile_mode="monitor", process_factory=_Process,
            )
            runner._create_closeout()
            loaded = load_hybrid_run(runs / "closeout")
            self.assertEqual(loaded.manifest.run_id, "closeout")
            self.assertGreater(loaded.closeout_artifact_count, 0)

    def test_failure_classification_ignores_profiler_in_absolute_path(self) -> None:
        from perfetto_hetero_profiler.hybrid.runner import classify_failure

        message = (
            "postprocess PerfettoInputError: "
            "/home/user/perfetto-hetero-profiler/run is invalid"
        )
        self.assertEqual(classify_failure(message), "postprocess")


if __name__ == "__main__":
    unittest.main()
