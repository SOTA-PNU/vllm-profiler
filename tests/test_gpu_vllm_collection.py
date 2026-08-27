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

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.gpu.vllm_collection import (
    GpuVllmCollectionConfig,
    GpuVllmCollectionRunner,
    build_vllm_collection_plan,
)
from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation
from perfetto_hetero_profiler.collectors.gpu.nvidia_smi import (
    NvidiaSmiQueryResult,
    parse_nvidia_smi_csv,
)
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


class FakeGpuClient:
    def query(self):
        raw = "0, Fake GPU, 0, 0, 1024, 10\n"
        return NvidiaSmiQueryResult(
            rows=parse_nvidia_smi_csv(raw),
            raw_output=raw,
        )


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


if __name__ == "__main__":
    unittest.main()
