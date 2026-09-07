"""Strict configuration for the reusable GPU-prefill/NPU-decode runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
import json
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from ..schema.jsonschema_runtime import (
    JsonSchemaFailure,
    compile_schema,
    validate_schema_document,
)
from ..schema.validation import validate_run_id
from .layout import HybridRunLayout


HybridProfileMode = Literal[
    "monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"
]
PROFILE_MODES = frozenset(
    {"monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"}
)
HYBRID_RUNNER_CONFIG_SCHEMA_NAME = "hybrid_runner_config.schema.json"


class HybridRunnerConfigError(ValueError):
    """The runner configuration is malformed or unsafe."""


@lru_cache(maxsize=1)
def _config_validator():
    resource = resources.files(__package__)
    for component in ("json", "v1", HYBRID_RUNNER_CONFIG_SCHEMA_NAME):
        resource = resource.joinpath(component)
    return compile_schema(json.loads(resource.read_text(encoding="utf-8")))


def _validate_structure(value: object) -> None:
    try:
        validate_schema_document(value, _config_validator(), root_path="config")
    except JsonSchemaFailure as error:
        message = (
            "unknown config field"
            if error.message == "unknown field"
            else error.message
        )
        raise HybridRunnerConfigError(f"{error.field_path}: {message}") from error


def _reject_nonfinite_constant(_: str) -> None:
    raise ValueError("non-finite numeric constants are not valid JSON")


def _absolute_path(value: str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise HybridRunnerConfigError(f"{field} must be absolute")
    return path


def _relative_output(value: str, field: str) -> Path:
    windows = PureWindowsPath(value)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HybridRunnerConfigError(f"{field} must be a safe relative path")
    return Path(*path.parts)


def _bounded_int(value: int, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not minimum <= value <= maximum:
        raise HybridRunnerConfigError(
            f"{field} must be an integer from {minimum} through {maximum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ServerConfig:
    executable: Path
    working_directory: Path
    pythonpath: Path
    host: str
    http_port: int
    nixl_port: int
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    prompt: str | None
    prompt_file: Path | None
    warmup_requests: int
    measured_requests: int
    max_output_tokens: int
    temperature: float
    streaming: bool

    def prompt_text(self) -> str:
        if self.prompt_file is not None:
            text = self.prompt_file.read_text(encoding="utf-8")
            if not text.strip():
                raise HybridRunnerConfigError("workload.prompt_file is empty")
            return text
        assert self.prompt is not None
        return self.prompt


@dataclass(frozen=True, slots=True)
class ProfilerOutputConfig:
    gpu_torch_subdir: Path
    gpu_nsys_basename: Path
    npu_torch_subdir: Path
    npu_rbln_subdir: Path


@dataclass(frozen=True, slots=True)
class HybridRunnerConfig:
    config_path: Path
    model_path: Path
    served_model_name: str
    rbln_cache_path: Path
    prefill: ServerConfig
    decode: ServerConfig
    proxy_python: Path
    proxy_entry_point: str
    proxy_host: str
    proxy_port: int
    workload: WorkloadConfig
    prefill_connector: dict[str, Any]
    decode_connector: dict[str, Any]
    profiler_outputs: ProfilerOutputConfig
    max_model_len: int
    block_size: int
    max_num_seqs: int
    gpu_memory_utilization: float
    gpu_indices: tuple[int, ...]
    npu_indices: tuple[int, ...]
    sample_interval_ms: int
    startup_timeout_sec: float
    request_timeout_sec: float
    shutdown_timeout_sec: float
    trace_processor_path: Path | None
    nsys_executable: Path
    offline: bool

    def __post_init__(self) -> None:
        ports = (self.prefill.http_port, self.decode.http_port, self.proxy_port)
        if len(set(ports)) != len(ports):
            raise HybridRunnerConfigError("prefill, decode, and proxy ports must differ")
        nixl_ports = (self.prefill.nixl_port, self.decode.nixl_port)
        if len(set(nixl_ports)) != 2 or set(ports) & set(nixl_ports):
            raise HybridRunnerConfigError("HTTP and NIXL ports must all be unique")
        if not self.offline:
            raise HybridRunnerConfigError("hybrid execution requires offline=true")

    def with_overrides(
        self,
        *,
        prompt: str | None = None,
        prompt_file: Path | None = None,
        warmup_requests: int | None = None,
        measured_requests: int | None = None,
        max_output_tokens: int | None = None,
    ) -> "HybridRunnerConfig":
        if prompt is not None and prompt_file is not None:
            raise HybridRunnerConfigError("--prompt and --prompt-file are exclusive")
        workload = self.workload
        if prompt is not None or prompt_file is not None:
            if prompt_file is not None and not prompt_file.is_absolute():
                raise HybridRunnerConfigError("--prompt-file must be absolute")
            workload = replace(workload, prompt=prompt, prompt_file=prompt_file)
        if warmup_requests is not None:
            workload = replace(
                workload,
                warmup_requests=_bounded_int(
                    warmup_requests, "--warmup-requests", 0, 1000
                ),
            )
        if measured_requests is not None:
            workload = replace(
                workload,
                measured_requests=_bounded_int(
                    measured_requests, "--measured-requests", 1, 1000
                ),
            )
        if max_output_tokens is not None:
            workload = replace(
                workload,
                max_output_tokens=_bounded_int(
                    max_output_tokens, "--max-output-tokens", 1, 16
                ),
            )
        return replace(self, workload=workload)


def _server(value: dict[str, Any], field: str) -> ServerConfig:
    extra = value.get("extra_args", [])
    controlled = (
        "--host", "--port", "--block-size", "--max-model-len",
        "--max-num-seqs", "--served-model-name", "--kv-transfer-config",
        "--profiler-config", "--gpu-memory-utilization",
    )
    conflict = next(
        (item for item in extra if item.startswith(controlled)), None
    )
    if conflict is not None:
        raise HybridRunnerConfigError(
            f"{field}.extra_args cannot override runner-controlled option: {conflict}"
        )
    host = value["host"]
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise HybridRunnerConfigError(f"{field}.host must be loopback")
    working_directory = _absolute_path(
        value["working_directory"], f"{field}.working_directory"
    )
    pythonpath = _absolute_path(
        value.get("pythonpath", str(working_directory)),
        f"{field}.pythonpath",
    )
    return ServerConfig(
        executable=_absolute_path(value["executable"], f"{field}.executable"),
        working_directory=working_directory,
        pythonpath=pythonpath,
        host=host,
        http_port=value["http_port"],
        nixl_port=value["nixl_port"],
        extra_args=tuple(extra),
    )


def load_hybrid_runner_config(path: Path) -> HybridRunnerConfig:
    """Load a strict versioned JSON document without changing the environment."""

    path = Path(path)
    if not path.is_absolute():
        raise HybridRunnerConfigError("--config must be an absolute path")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise HybridRunnerConfigError(f"cannot read config: {error}") from error
    _validate_structure(document)
    assert isinstance(document, dict)
    root = document
    model = root["model"]
    proxy = root["proxy"]
    workload = root["workload"]
    assert isinstance(model, dict)
    assert isinstance(proxy, dict)
    assert isinstance(workload, dict)
    prompt = workload.get("prompt")
    prompt_file = workload.get("prompt_file")
    prompt_value = prompt if prompt is not None else None
    prompt_path = (
        _absolute_path(prompt_file, "workload.prompt_file")
        if prompt_file is not None
        else None
    )
    runtime = root["runtime"]
    telemetry = root["telemetry"]
    timeouts = root["timeouts"]
    tool_config = root["tools"]
    connectors = root["connectors"]
    profilers = root["profilers"]
    assert all(
        isinstance(value, dict)
        for value in (runtime, telemetry, timeouts, tool_config, connectors, profilers)
    )
    prefill_connector = connectors["prefill"]
    decode_connector = connectors["decode"]
    assert isinstance(prefill_connector, dict)
    assert isinstance(decode_connector, dict)
    if prefill_connector.get("kv_role") != "kv_producer":
        raise HybridRunnerConfigError(
            "connectors.prefill.kv_role must be kv_producer"
        )
    if decode_connector.get("kv_role") != "kv_consumer":
        raise HybridRunnerConfigError(
            "connectors.decode.kv_role must be kv_consumer"
        )
    trace_processor = tool_config.get("trace_processor")
    proxy_host = proxy["host"]
    if proxy_host not in {"127.0.0.1", "localhost", "::1"}:
        raise HybridRunnerConfigError("proxy.host must be loopback")
    proxy_entry_point = proxy["entry_point"]
    return HybridRunnerConfig(
        config_path=path,
        model_path=_absolute_path(model["path"], "model.path"),
        served_model_name=model["served_name"],
        rbln_cache_path=_absolute_path(
            model["rbln_cache_path"], "model.rbln_cache_path"
        ),
        prefill=_server(root["prefill"], "prefill"),
        decode=_server(root["decode"], "decode"),
        proxy_python=_absolute_path(proxy["python"], "proxy.python"),
        proxy_entry_point=proxy_entry_point,
        proxy_host=proxy_host,
        proxy_port=proxy["http_port"],
        workload=WorkloadConfig(
            prompt=prompt_value,
            prompt_file=prompt_path,
            warmup_requests=workload["warmup_requests"],
            measured_requests=workload["measured_requests"],
            max_output_tokens=workload["max_output_tokens"],
            temperature=workload["temperature"],
            streaming=workload["streaming"],
        ),
        prefill_connector=prefill_connector,
        decode_connector=decode_connector,
        profiler_outputs=ProfilerOutputConfig(
            gpu_torch_subdir=_relative_output(
                profilers["gpu_torch_subdir"], "profilers.gpu_torch_subdir"
            ),
            gpu_nsys_basename=_relative_output(
                profilers["gpu_nsys_basename"], "profilers.gpu_nsys_basename"
            ),
            npu_torch_subdir=_relative_output(
                profilers["npu_torch_subdir"], "profilers.npu_torch_subdir"
            ),
            npu_rbln_subdir=_relative_output(
                profilers["npu_rbln_subdir"], "profilers.npu_rbln_subdir"
            ),
        ),
        max_model_len=runtime["max_model_len"],
        block_size=runtime["block_size"],
        max_num_seqs=runtime["max_num_seqs"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        gpu_indices=tuple(runtime["gpu_indices"]),
        npu_indices=tuple(runtime["npu_indices"]),
        sample_interval_ms=telemetry["sample_interval_ms"],
        startup_timeout_sec=timeouts["startup_sec"],
        request_timeout_sec=timeouts["request_sec"],
        shutdown_timeout_sec=timeouts["shutdown_sec"],
        trace_processor_path=(
            _absolute_path(trace_processor, "tools.trace_processor")
            if trace_processor is not None
            else None
        ),
        nsys_executable=_absolute_path(tool_config["nsys"], "tools.nsys"),
        offline=root["offline"],
    )


def validate_hybrid_invocation(
    config: HybridRunnerConfig,
    *,
    run_root: Path,
    run_id: str,
    profile_mode: str,
) -> None:
    """Validate paths and an output identity before any directory is created."""

    validate_run_id(run_id)
    if profile_mode not in PROFILE_MODES:
        raise HybridRunnerConfigError(f"unsupported profile mode: {profile_mode}")
    run_root = Path(run_root)
    if not run_root.is_absolute():
        raise HybridRunnerConfigError("--run-root must be an absolute path")
    current = Path(run_root.anchor)
    for part in run_root.parts[1:]:
        current /= part
        if current.is_symlink():
            raise HybridRunnerConfigError(
                f"--run-root must not traverse a symlink: {current}"
            )
    layout = HybridRunLayout(run_root, run_id)
    existing = next(
        (
            path
            for path in dict.fromkeys((layout.bundle, *layout.legacy_roots))
            if os.path.lexists(path)
        ),
        None,
    )
    if existing is not None:
        raise FileExistsError(f"run output already exists: {existing}")
    if config.workload.max_output_tokens >= config.max_model_len:
        raise HybridRunnerConfigError(
            "max_output_tokens must leave room below max_model_len"
        )
