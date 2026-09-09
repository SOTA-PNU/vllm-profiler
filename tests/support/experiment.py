"""Shared experiment configuration fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hybrid_document(root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model": {"path": str(root / "model"), "served_name": "Qwen3-0.6B", "rbln_cache_path": str(root / "cache")},
        "prefill": {"executable": str(root / "prefill"), "working_directory": str(root), "host": "127.0.0.1", "http_port": 18100, "nixl_port": 18559, "extra_args": []},
        "decode": {"executable": str(root / "decode"), "working_directory": str(root), "host": "127.0.0.1", "http_port": 18200, "nixl_port": 18659, "extra_args": []},
        "proxy": {"python": str(root / "python"), "entry_point": "perfetto_hetero_profiler.hybrid.proxy", "host": "127.0.0.1", "http_port": 18192},
        "workload": {"prompt": "Capital of South Korea is", "warmup_requests": 2, "measured_requests": 10, "max_output_tokens": 8, "temperature": 0, "streaming": True},
        "runtime": {"max_model_len": 512, "block_size": 512, "max_num_seqs": 1, "gpu_memory_utilization": 0.2, "gpu_indices": [0], "npu_indices": [0]},
        "connectors": {"prefill": {"kv_role": "kv_producer"}, "decode": {"kv_role": "kv_consumer"}},
        "profilers": {"gpu_torch_subdir": "raw/gpu/torch", "gpu_nsys_basename": "raw/gpu/nsys/gpu", "npu_torch_subdir": "raw/npu/torch", "npu_rbln_subdir": "raw/npu/rbln"},
        "telemetry": {"sample_interval_ms": 500},
        "timeouts": {"startup_sec": 300, "request_sec": 60, "shutdown_sec": 60},
        "tools": {"trace_processor": str(root / "trace_processor"), "nsys": str(root / "nsys")},
        "offline": True,
    }


def write_config(root: Path, *, mutate_hybrid=None):
    hybrid = hybrid_document(root)
    if mutate_hybrid:
        mutate_hybrid(hybrid)
    hybrid_path = root / "hybrid.json"
    hybrid_path.write_text(json.dumps(hybrid), encoding="utf-8")
    digest = hashlib.sha256(hybrid_path.read_bytes()).hexdigest()
    config_path = root / "experiment.json"
    config_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": "test-repeatability",
        "hybrid_config": {"path": str(hybrid_path), "sha256": digest},
        "schedule": {"seed": 20260807, "max_hardware_attempts": 42},
    }), encoding="utf-8")
    return config_path
