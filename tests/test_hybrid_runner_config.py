"""CPU-only contract tests for the reusable hybrid runner."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.hybrid.runner import build_hybrid_run_plan
from perfetto_hetero_profiler.hybrid.runner_config import (
    HybridRunnerConfigError,
    load_hybrid_runner_config,
    validate_hybrid_invocation,
)


def document(root: Path) -> dict:
    return {
        "schema_version": "1.0",
        "model": {
            "path": str(root / "model"),
            "served_name": "example-model",
            "rbln_cache_path": str(root / "cache"),
        },
        "prefill": {
            "executable": str(root / "prefill/bin/vllm"),
            "working_directory": str(root / "prefill"),
            "host": "127.0.0.1",
            "http_port": 18100,
            "nixl_port": 18559,
            "extra_args": [],
        },
        "decode": {
            "executable": str(root / "decode/bin/vllm"),
            "working_directory": str(root / "decode"),
            "host": "127.0.0.1",
            "http_port": 18200,
            "nixl_port": 18659,
            "extra_args": [],
        },
        "proxy": {
            "python": str(root / "python"),
            "entry_point": "perfetto_hetero_profiler.hybrid.proxy",
            "host": "127.0.0.1",
            "http_port": 18192,
        },
        "workload": {
            "prompt": "Explain cache briefly.",
            "warmup_requests": 1,
            "measured_requests": 2,
            "max_output_tokens": 8,
            "temperature": 0,
            "streaming": True,
        },
        "runtime": {
            "max_model_len": 512,
            "block_size": 512,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.2,
            "gpu_indices": [0],
            "npu_indices": [0],
        },
        "connectors": {
            "prefill": {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_producer",
                "kv_buffer_device": "cuda",
                "kv_load_failure_policy": "fail",
            },
            "decode": {
                "kv_connector": "RblnNixlConnector",
                "kv_role": "kv_consumer",
                "kv_buffer_device": "cpu",
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": {
                    "remote_nixl_memory_type": "VRAM",
                    "rbln_external_kv_format": "host_visible_hnd_to_runtime_private",
                    "rbln_external_kv_source_dtype": "bfloat16",
                },
            },
        },
        "profilers": {
            "gpu_torch_subdir": "raw/gpu/torch",
            "gpu_nsys_basename": "raw/gpu/nsys/gpu-prefill",
            "npu_torch_subdir": "raw/npu/torch",
            "npu_rbln_subdir": "raw/npu/rbln-profiler",
        },
        "telemetry": {"sample_interval_ms": 500},
        "timeouts": {"startup_sec": 300, "request_sec": 60, "shutdown_sec": 60},
        "tools": {
            "trace_processor": str(root / "trace_processor_shell"),
            "nsys": str(root / "nsys"),
        },
        "offline": True,
    }


class HybridRunnerConfigTests(unittest.TestCase):
    def load(self, root: Path, value: dict | None = None):
        path = root / "config.json"
        path.write_text(json.dumps(value or document(root)), encoding="utf-8")
        return load_hybrid_runner_config(path)

    def test_valid_config_and_plan_are_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            plan = build_hybrid_run_plan(
                config, run_root=runs, run_id="example", profile_mode="monitor"
            )
            self.assertFalse(plan["executes"])
            self.assertFalse(runs.exists())
            self.assertEqual(plan["outputs"]["hybrid"], str(runs / "example"))

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["unexpected"] = True
            with self.assertRaisesRegex(HybridRunnerConfigError, "unknown config"):
                self.load(root, value)

    def test_relative_config_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(HybridRunnerConfigError, "--config"):
            load_hybrid_runner_config(Path("config.json"))

    def test_prompt_and_prompt_file_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["workload"]["prompt_file"] = str(root / "prompt.txt")
            with self.assertRaisesRegex(HybridRunnerConfigError, "exactly one"):
                self.load(root, value)

    def test_online_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["offline"] = False
            with self.assertRaisesRegex(HybridRunnerConfigError, "offline"):
                self.load(root, value)

    def test_duplicate_and_colliding_ports_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["decode"]["http_port"] = value["prefill"]["http_port"]
            with self.assertRaisesRegex(HybridRunnerConfigError, "ports must differ"):
                self.load(root, value)

    def test_existing_output_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            target = runs / "same-gpu"
            target.mkdir(parents=True)
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_hybrid_invocation(
                    config, run_root=runs, run_id="same", profile_mode="monitor"
                )
            self.assertEqual(marker.read_text(), "keep")

    def test_symlink_run_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(HybridRunnerConfigError, "symlink"):
                validate_hybrid_invocation(
                    config, run_root=linked, run_id="run", profile_mode="monitor"
                )

    def test_cli_dry_run_uses_overrides_and_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(document(root)), encoding="utf-8")
            runs = root / "runs"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "collect", "hybrid", "--config", str(config_path),
                        "--run-root", str(runs), "--run-id", "dry-run",
                        "--profile-mode", "gpu-torch", "--warmup-requests", "0",
                        "--measured-requests", "1", "--max-output-tokens", "4",
                        "--prompt", "override", "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["profile_mode"], "gpu-torch")
            self.assertEqual(value["workload"]["warmup_requests"], 0)
            self.assertFalse(runs.exists())

    def test_all_modes_enable_only_one_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            for mode in ("monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"):
                plan = build_hybrid_run_plan(
                    config, run_root=root / "runs", run_id=mode, profile_mode=mode
                )
                prefill = plan["commands"]["prefill"]
                decode = plan["commands"]["decode"]
                torch_count = sum("profiler=torch" in arg for arg in [*prefill, *decode])
                rbln_count = int(mode == "npu-rbln")
                nsys_count = int(prefill[0] == str(config.nsys_executable))
                self.assertLessEqual(torch_count + rbln_count + nsys_count, 1)


if __name__ == "__main__":
    unittest.main()
