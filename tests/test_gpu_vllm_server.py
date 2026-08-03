"""Unit tests for vLLM server planning and safety constraints."""

import os
from pathlib import Path
import unittest

from perfetto_hetero_profiler.gpu.vllm_server import (
    VllmServerConfig,
    build_server_argv,
    server_environment,
)


class VllmServerPlanTests(unittest.TestCase):
    def config(self, **changes) -> VllmServerConfig:
        values = {
            "model": Path("/models/qwen"),
            "host": "127.0.0.1",
            "port": 18080,
            "gpu_memory_utilization": 0.25,
            "max_model_len": 2048,
            "vllm_bin": Path("/venv/bin/vllm"),
        }
        values.update(changes)
        return VllmServerConfig(**values)

    def test_vllm_bin_argv(self) -> None:
        argv = build_server_argv(self.config())
        self.assertEqual(argv[:3], ("/venv/bin/vllm", "serve", "/models/qwen"))
        self.assertIn("--enforce-eager", argv)
        self.assertIn("--no-async-scheduling", argv)

    def test_python_module_argv(self) -> None:
        config = self.config(
            vllm_bin=None, server_python=Path("/venv/bin/python")
        )
        argv = build_server_argv(config)
        self.assertEqual(
            argv[:4],
            (
                "/venv/bin/python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
            ),
        )

    def test_torch_nested_cli_syntax(self) -> None:
        config = self.config(torch_profiler_dir=Path("/runs/a/raw/gpu/torch"))
        argv = build_server_argv(config)
        self.assertIn("--profiler-config.profiler=torch", argv)
        self.assertIn(
            "--profiler-config.torch_profiler_dir=/runs/a/raw/gpu/torch", argv
        )

    def test_nsys_wraps_server(self) -> None:
        argv = build_server_argv(
            self.config(nsys_output=Path("/runs/a/raw/gpu/nsys/report"))
        )
        self.assertEqual(argv[:3], ("nsys", "profile", "--trace=cuda,nvtx,osrt"))
        self.assertIn("--sample=none", argv)
        self.assertIn("--cpuctxsw=none", argv)
        self.assertIn("--force-overwrite=false", argv)

    def test_torch_and_nsys_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot"):
            self.config(
                torch_profiler_dir=Path("/runs/a/torch"),
                nsys_output=Path("/runs/a/nsys/report"),
            )

    def test_exactly_one_executable(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.config(vllm_bin=None)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.config(server_python=Path("/python"))

    def test_rejects_non_loopback_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            self.config(host="0.0.0.0")

    def test_rejects_high_memory_utilization(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.50"):
            self.config(gpu_memory_utilization=0.51)

    def test_rejects_large_model_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "2048"):
            self.config(max_model_len=2049)

    def test_rejects_relative_profile_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.config(torch_profiler_dir=Path("relative"))
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.config(nsys_output=Path("relative"))

    def test_offline_environment(self) -> None:
        environment = server_environment(self.config())
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")

    def test_online_environment_does_not_force_offline(self) -> None:
        previous_hf = os.environ.pop("HF_HUB_OFFLINE", None)
        previous_transformers = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            environment = server_environment(self.config(offline=False))
        finally:
            if previous_hf is not None:
                os.environ["HF_HUB_OFFLINE"] = previous_hf
            if previous_transformers is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers
        self.assertNotIn("HF_HUB_OFFLINE", environment)
        self.assertNotIn("TRANSFORMERS_OFFLINE", environment)


if __name__ == "__main__":
    unittest.main()
