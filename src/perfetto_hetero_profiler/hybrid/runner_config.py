"""Strict configuration for the reusable GPU-prefill/NPU-decode runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Literal

from ..schema.validation import validate_run_id


HybridProfileMode = Literal[
    "monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"
]
PROFILE_MODES = frozenset(
    {"monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"}
)


class HybridRunnerConfigError(ValueError):
    """The runner configuration is malformed or unsafe."""


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HybridRunnerConfigError(f"{field} must be an object")
    return dict(value)


def _keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise HybridRunnerConfigError(f"unknown {field} field: {unknown[0]}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HybridRunnerConfigError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise HybridRunnerConfigError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise HybridRunnerConfigError(
            f"{field} must be in [{minimum}, {maximum}]"
        )
    return value


def _number(value: object, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridRunnerConfigError(f"{field} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise HybridRunnerConfigError(
            f"{field} must be in [{minimum:g}, {maximum:g}]"
        )
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise HybridRunnerConfigError(f"{field} must be a boolean")
    return value


def _absolute_path(value: object, field: str, *, no_symlink: bool = False) -> Path:
    path = Path(_string(value, field))
    if not path.is_absolute():
        raise HybridRunnerConfigError(f"{field} must be an absolute path")
    if no_symlink and path.exists() and path.is_symlink():
        raise HybridRunnerConfigError(f"{field} must not be a symlink")
    return path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    executable: Path
    working_directory: Path
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
            raise HybridRunnerConfigError("Phase 7A requires offline=true")

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
                warmup_requests=_integer(
                    warmup_requests, "--warmup-requests", 0, 1000
                ),
            )
        if measured_requests is not None:
            workload = replace(
                workload,
                measured_requests=_integer(
                    measured_requests, "--measured-requests", 1, 1000
                ),
            )
        if max_output_tokens is not None:
            workload = replace(
                workload,
                max_output_tokens=_integer(
                    max_output_tokens, "--max-output-tokens", 1, 16
                ),
            )
        return replace(self, workload=workload)


def _server(document: object, field: str) -> ServerConfig:
    value = _object(document, field)
    _keys(
        value,
        {"executable", "working_directory", "host", "http_port", "nixl_port", "extra_args"},
        field,
    )
    extra = value.get("extra_args", [])
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        raise HybridRunnerConfigError(f"{field}.extra_args must be a string array")
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
    host = _string(value.get("host"), f"{field}.host")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise HybridRunnerConfigError(f"{field}.host must be loopback")
    return ServerConfig(
        executable=_absolute_path(value.get("executable"), f"{field}.executable"),
        working_directory=_absolute_path(
            value.get("working_directory"), f"{field}.working_directory"
        ),
        host=host,
        http_port=_integer(value.get("http_port"), f"{field}.http_port", 1, 65535),
        nixl_port=_integer(value.get("nixl_port"), f"{field}.nixl_port", 1, 65535),
        extra_args=tuple(extra),
    )


def _relative_output(value: object, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise HybridRunnerConfigError(
            f"{field} must be a safe relative output path"
        )
    return path


def load_hybrid_runner_config(path: Path) -> HybridRunnerConfig:
    """Load a strict versioned JSON document without changing the environment."""

    path = Path(path)
    if not path.is_absolute():
        raise HybridRunnerConfigError("--config must be an absolute path")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HybridRunnerConfigError(f"cannot read config: {error}") from error
    root = _object(document, "config")
    _keys(
        root,
        {
            "schema_version", "model", "prefill", "decode", "proxy", "workload",
            "runtime", "connectors", "profilers", "telemetry", "timeouts",
            "tools", "offline",
        },
        "config",
    )
    if root.get("schema_version") != "1.0":
        raise HybridRunnerConfigError("schema_version must be '1.0'")

    model = _object(root.get("model"), "model")
    _keys(model, {"path", "served_name", "rbln_cache_path"}, "model")
    proxy = _object(root.get("proxy"), "proxy")
    _keys(proxy, {"python", "entry_point", "host", "http_port"}, "proxy")
    workload = _object(root.get("workload"), "workload")
    _keys(
        workload,
        {"prompt", "prompt_file", "warmup_requests", "measured_requests", "max_output_tokens", "temperature", "streaming"},
        "workload",
    )
    prompt = workload.get("prompt")
    prompt_file = workload.get("prompt_file")
    if (prompt is None) == (prompt_file is None):
        raise HybridRunnerConfigError(
            "workload requires exactly one of prompt or prompt_file"
        )
    prompt_value = _string(prompt, "workload.prompt") if prompt is not None else None
    prompt_path = (
        _absolute_path(prompt_file, "workload.prompt_file")
        if prompt_file is not None
        else None
    )
    runtime = _object(root.get("runtime"), "runtime")
    _keys(
        runtime,
        {"max_model_len", "block_size", "max_num_seqs", "gpu_memory_utilization", "gpu_indices", "npu_indices"},
        "runtime",
    )
    telemetry = _object(root.get("telemetry"), "telemetry")
    _keys(telemetry, {"sample_interval_ms"}, "telemetry")
    timeouts = _object(root.get("timeouts"), "timeouts")
    _keys(timeouts, {"startup_sec", "request_sec", "shutdown_sec"}, "timeouts")
    tools = _object(root.get("tools"), "tools")
    _keys(tools, {"trace_processor", "nsys"}, "tools")
    connectors = _object(root.get("connectors"), "connectors")
    _keys(connectors, {"prefill", "decode"}, "connectors")
    prefill_connector = _object(connectors.get("prefill"), "connectors.prefill")
    decode_connector = _object(connectors.get("decode"), "connectors.decode")
    if prefill_connector.get("kv_role") != "kv_producer":
        raise HybridRunnerConfigError(
            "connectors.prefill.kv_role must be kv_producer"
        )
    if decode_connector.get("kv_role") != "kv_consumer":
        raise HybridRunnerConfigError(
            "connectors.decode.kv_role must be kv_consumer"
        )
    profilers = _object(root.get("profilers"), "profilers")
    _keys(
        profilers,
        {"gpu_torch_subdir", "gpu_nsys_basename", "npu_torch_subdir", "npu_rbln_subdir"},
        "profilers",
    )

    def indices(name: str) -> tuple[int, ...]:
        values = runtime.get(name)
        if not isinstance(values, list) or not values:
            raise HybridRunnerConfigError(f"runtime.{name} must be a non-empty array")
        parsed = tuple(_integer(value, f"runtime.{name}", 0, 1024) for value in values)
        if len(parsed) != len(set(parsed)):
            raise HybridRunnerConfigError(f"runtime.{name} contains duplicates")
        return parsed

    trace_processor = tools.get("trace_processor")
    proxy_host = _string(proxy.get("host"), "proxy.host")
    if proxy_host not in {"127.0.0.1", "localhost", "::1"}:
        raise HybridRunnerConfigError("proxy.host must be loopback")
    proxy_entry_point = _string(
        proxy.get("entry_point"), "proxy.entry_point"
    )
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", proxy_entry_point) is None:
        raise HybridRunnerConfigError(
            "proxy.entry_point must be a Python module name"
        )
    return HybridRunnerConfig(
        config_path=path,
        model_path=_absolute_path(model.get("path"), "model.path"),
        served_model_name=_string(model.get("served_name"), "model.served_name"),
        rbln_cache_path=_absolute_path(
            model.get("rbln_cache_path"), "model.rbln_cache_path"
        ),
        prefill=_server(root.get("prefill"), "prefill"),
        decode=_server(root.get("decode"), "decode"),
        proxy_python=_absolute_path(proxy.get("python"), "proxy.python"),
        proxy_entry_point=proxy_entry_point,
        proxy_host=proxy_host,
        proxy_port=_integer(proxy.get("http_port"), "proxy.http_port", 1, 65535),
        workload=WorkloadConfig(
            prompt=prompt_value,
            prompt_file=prompt_path,
            warmup_requests=_integer(workload.get("warmup_requests"), "workload.warmup_requests", 0, 1000),
            measured_requests=_integer(workload.get("measured_requests"), "workload.measured_requests", 1, 1000),
            max_output_tokens=_integer(workload.get("max_output_tokens"), "workload.max_output_tokens", 1, 16),
            temperature=_number(workload.get("temperature"), "workload.temperature", 0, 2),
            streaming=_boolean(workload.get("streaming"), "workload.streaming"),
        ),
        prefill_connector=prefill_connector,
        decode_connector=decode_connector,
        profiler_outputs=ProfilerOutputConfig(
            gpu_torch_subdir=_relative_output(
                profilers.get("gpu_torch_subdir"), "profilers.gpu_torch_subdir"
            ),
            gpu_nsys_basename=_relative_output(
                profilers.get("gpu_nsys_basename"), "profilers.gpu_nsys_basename"
            ),
            npu_torch_subdir=_relative_output(
                profilers.get("npu_torch_subdir"), "profilers.npu_torch_subdir"
            ),
            npu_rbln_subdir=_relative_output(
                profilers.get("npu_rbln_subdir"), "profilers.npu_rbln_subdir"
            ),
        ),
        max_model_len=_integer(runtime.get("max_model_len"), "runtime.max_model_len", 1, 131072),
        block_size=_integer(runtime.get("block_size"), "runtime.block_size", 1, 65536),
        max_num_seqs=_integer(runtime.get("max_num_seqs"), "runtime.max_num_seqs", 1, 4096),
        gpu_memory_utilization=_number(runtime.get("gpu_memory_utilization"), "runtime.gpu_memory_utilization", 0.01, 1.0),
        gpu_indices=indices("gpu_indices"),
        npu_indices=indices("npu_indices"),
        sample_interval_ms=_integer(telemetry.get("sample_interval_ms"), "telemetry.sample_interval_ms", 50, 60000),
        startup_timeout_sec=_number(timeouts.get("startup_sec"), "timeouts.startup_sec", 1, 3600),
        request_timeout_sec=_number(timeouts.get("request_sec"), "timeouts.request_sec", 1, 3600),
        shutdown_timeout_sec=_number(timeouts.get("shutdown_sec"), "timeouts.shutdown_sec", 1, 600),
        trace_processor_path=(
            _absolute_path(trace_processor, "tools.trace_processor")
            if trace_processor is not None
            else None
        ),
        nsys_executable=_absolute_path(tools.get("nsys"), "tools.nsys"),
        offline=_boolean(root.get("offline"), "offline"),
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
        if current.exists() and current.is_symlink():
            raise HybridRunnerConfigError(
                f"--run-root must not traverse a symlink: {current}"
            )
    targets = tuple(
        run_root / suffix
        for suffix in (
            run_id,
            f"{run_id}-gpu",
            f"{run_id}-npu",
            f"{run_id}-coordinator",
            f"{run_id}-perfetto",
            f"{run_id}-perfetto-request-focused",
            f"{run_id}-overview",
            f"{run_id}-closeout-recovery",
            f"{run_id}-publication",
        )
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"run output already exists: {existing[0]}")
    if config.workload.max_output_tokens >= config.max_model_len:
        raise HybridRunnerConfigError(
            "max_output_tokens must leave room below max_model_len"
        )
