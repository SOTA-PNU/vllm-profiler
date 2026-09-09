"""Shared hybrid runner configuration fixture."""

from __future__ import annotations

from pathlib import Path


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
            "pythonpath": str(root / "prefill"),
            "host": "127.0.0.1",
            "http_port": 18100,
            "nixl_port": 18559,
            "extra_args": [],
        },
        "decode": {
            "executable": str(root / "decode/bin/vllm"),
            "working_directory": str(root / "decode"),
            "pythonpath": str(root / "decode"),
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
