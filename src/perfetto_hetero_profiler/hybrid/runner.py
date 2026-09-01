"""Reusable GPU-prefill/NPU-decode collection runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..collectors.command import CommandSpec, mask_command
from ..collectors.gpu import (
    NVML_DISTRIBUTION,
    NVML_DISTRIBUTION_VERSION,
    GpuTelemetryCollector,
    NvmlClient,
)
from ..collectors.npu import NpuTelemetryCollector
from ..collectors.npu.rbln_smi import RblnSmiClient
from ..collectors.process import ManagedProcess
from ..collectors.system import ProcTelemetryCollector
from ..gpu.openai_client import CompletionObservation, OpenAICompletionClient
from ..gpu.workload import measured_window_metrics, observation_metrics
from ..schema import (
    ArtifactKind,
    ArtifactReference,
    ClockDomain,
    ClockType,
    DeviceDescriptor,
    DeviceType,
    HostDescriptor,
    ModelDescriptor,
    ProfileMode,
    RunManifest,
    RunMode,
    RunPaths,
    RunStatus,
    SoftwareDescriptor,
    WorkloadDescriptor,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    create_detached_recovery,
)
from ..support.files import fingerprint_tree, sha256_file
from ..support.json_io import write_jsonl_exclusive, write_pretty_json
from ..support.network import port_available
from ..collectors.telemetry import CollectorGroup, SampleTicket, TelemetryWorker
from .bundle import HybridBundleMerger
from .config import AlignmentMethod, HybridMergeConfig
from .detailed_profile import (
    build_profiler_alignment,
    build_profiler_clock_domain,
    validate_nsys_report,
    validate_rbln_reports,
    validate_torch_traces,
)
from .join import validate_marker_order
from .runner_config import HybridProfileMode, HybridRunnerConfig
from .runtime_markers import ingest_runtime_marker_files


HOST_ID = "localhost"
CLOCK_DOMAIN_ID = "host-monotonic"
_ENV_ALLOWLIST = (
    "PATH", "PYTHONPATH", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "VLLM_KV_CACHE_LAYOUT", "UCX_NET_DEVICES", "VLLM_NIXL_SIDE_CHANNEL_PORT",
    "VLLM_RBLN_USE_VLLM_MODEL", "VLLM_RBLN_COMPILE_MODEL", "RBLN_DEVICES",
    "VLLM_CACHE_ROOT", "VLLM_RBLN_RUNTIME_MARKER_DIR",
    "VLLM_RBLN_RUNTIME_MARKER_HOST_ID",
    "VLLM_RBLN_RUNTIME_MARKER_CLOCK_DOMAIN_ID",
    "VLLM_RBLN_NIXL_READ_DIAGNOSTIC", "RBLN_PROFILER",
    "VLLM_RBLN_DEVICE_PROFILER_DIR",
)


class HybridRunnerError(RuntimeError):
    """Stable user-facing runner failure."""


def classify_failure(message: str) -> str:
    """Map a safe error message to one stable operational failure class."""
    lowered = message.lower()
    if "postprocess" in lowered or "perfetto" in lowered or "overview" in lowered:
        return "postprocess"
    if "required executable" in lowered or "required directory" in lowered or "port" in lowered:
        return "preflight"
    if "readiness" in lowered or "before readiness" in lowered:
        return "readiness"
    if (
        "/start_profile" in lowered
        or "/stop_profile" in lowered
        or "profiler cleanup" in lowered
    ):
        return "profiler_control"
    if "marker" in lowered or "correlation" in lowered or "join" in lowered:
        return "marker_validation"
    if "cache" in lowered or "compile" in lowered:
        return "model_cache_reuse"
    if "artifact" in lowered or "fingerprint" in lowered or "deterministic" in lowered:
        return "artifact_validation"
    if "cleanup" in lowered or "sigkill" in lowered or "shutdown integrity" in lowered:
        return "cleanup"
    if "request" in lowered or "completion" in lowered:
        return "workload"
    return "runtime"


def _shutdown_integrity(
    layout: "_Layout", shutdown: dict[str, Any]
) -> dict[str, Any]:
    logs: dict[str, str] = {}
    for name in ("prefill", "decode", "proxy"):
        path = layout.coordinator / f"raw/{name}.stderr.log"
        logs[name] = (
            path.read_text(encoding="utf-8", errors="replace")
            if path.is_file()
            else ""
        )
    combined = "\n".join(logs.values())
    known_nixl = (
        "Segfault encountered" in logs["decode"]
        and "rtnl_tc_unregister" in logs["decode"]
    )
    signatures = {
        "segfault_encountered": "Segfault encountered" in combined,
        "rtnl_tc_unregister": "rtnl_tc_unregister" in combined,
        "sigsegv": "SIGSEGV" in combined,
        "native_abort": "Aborted" in combined or "SIGABRT" in combined,
        "python_fatal_error": "Fatal Python error" in combined,
    }
    fatal = known_nixl or any(signatures.values())
    if known_nixl:
        status = "invalid"
        reason = "native_sigsegv_rtnl_tc_unregister"
    elif fatal:
        status = "invalid"
        reason = "unexpected_native_shutdown_failure"
    else:
        status = "valid"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "demo_only": known_nixl,
        "known_nixl_shutdown_signature": known_nixl,
        "signatures": signatures,
        "process_results": shutdown,
    }


@dataclass(frozen=True, slots=True)
class HybridRunResult:
    status: RunStatus
    run_directory: Path
    gpu_run_directory: Path
    npu_run_directory: Path
    coordinator_directory: Path
    perfetto_directory: Path | None
    request_focused_perfetto_directory: Path | None
    overview_directory: Path | None
    recovery_directory: Path | None
    publication_directory: Path
    warmup_count: int
    measured_count: int
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Layout:
    run_root: Path
    run_id: str

    @property
    def hybrid(self) -> Path:
        return self.run_root / self.run_id

    @property
    def gpu(self) -> Path:
        return self.run_root / f"{self.run_id}-gpu"

    @property
    def npu(self) -> Path:
        return self.run_root / f"{self.run_id}-npu"

    @property
    def coordinator(self) -> Path:
        return self.run_root / f"{self.run_id}-coordinator"

    @property
    def perfetto(self) -> Path:
        return self.run_root / f"{self.run_id}-perfetto"

    @property
    def overview(self) -> Path:
        return self.run_root / f"{self.run_id}-overview"

    @property
    def request_perfetto(self) -> Path:
        return self.run_root / f"{self.run_id}-perfetto-request-focused"

    @property
    def recovery(self) -> Path:
        return self.run_root / f"{self.run_id}-closeout-recovery"

    @property
    def publication(self) -> Path:
        return self.run_root / f"{self.run_id}-publication"


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _plain_json(path: Path, value: object) -> None:
    write_pretty_json(path, value)


def _plain_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_exclusive(path, rows)


def _cache_fingerprint(root: Path) -> list[dict[str, object]]:
    return fingerprint_tree(root, pattern="*.rbln", include_mtime=True)


def _tree_fingerprint(root: Path) -> list[dict[str, object]]:
    return fingerprint_tree(root)


def _port_available(host: str, port: int) -> bool:
    return port_available(host, port)

def _wait_http(
    base_url: str,
    endpoint: str,
    process: ManagedProcess,
    timeout_sec: float,
) -> int:
    deadline = time.monotonic() + timeout_sec
    last_error = "not contacted"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise HybridRunnerError(
                f"process exited before readiness with code {code}"
            )
        try:
            with urlopen(f"{base_url}{endpoint}", timeout=1.0) as response:
                if response.status == 200:
                    return time.monotonic_ns()
                last_error = f"HTTP {response.status}"
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"readiness timed out for {base_url}{endpoint}: {last_error}")


def _profile_call(base_url: str, endpoint: str, timeout_sec: float) -> dict[str, Any]:
    before_mono = time.monotonic_ns()
    before_unix = time.time_ns()
    request = Request(
        f"{base_url}{endpoint}",
        data=b"",
        method="POST",
        headers={"Content-Length": "0"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except (HTTPError, URLError, TimeoutError) as error:
        raise HybridRunnerError(f"{endpoint} failed: {error}") from error
    after_mono = time.monotonic_ns()
    after_unix = time.time_ns()
    if status != 200:
        raise HybridRunnerError(f"{endpoint} returned HTTP {status}")
    return {
        "endpoint": endpoint,
        "before_monotonic_ns": before_mono,
        "after_monotonic_ns": after_mono,
        "before_unix_ns": before_unix,
        "after_unix_ns": after_unix,
        "http_status": status,
        "response_body": body,
    }


def _wait_runtime_marker_completion(
    marker_directory: Path,
    request_ids: set[str],
    timeout_sec: float,
) -> None:
    """Wait until proxy and decode completion markers are durably observable."""
    deadline = time.monotonic() + timeout_sec
    observed: dict[str, set[str]] = {request_id: set() for request_id in request_ids}
    while time.monotonic() < deadline:
        for path in sorted(marker_directory.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                correlation = row.get("correlation_id", row.get("request_id"))
                name = row.get("event_name")
                if correlation in observed and isinstance(name, str):
                    observed[correlation].add(name)
        if all(
            {"decode_loop_end", "response_done"}.issubset(names)
            for names in observed.values()
        ):
            return
        time.sleep(0.05)
    missing = {
        request_id: sorted({"decode_loop_end", "response_done"} - names)
        for request_id, names in observed.items()
        if not {"decode_loop_end", "response_done"}.issubset(names)
    }
    raise HybridRunnerError(
        f"runtime marker completion timed out: {missing}"
    )


_SampleTicket = SampleTicket


class _TelemetryWorker(TelemetryWorker):
    """Runner-bound compatibility name for the shared polling worker."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, error_type=HybridRunnerError)


class _Telemetry:
    def __init__(self, config: HybridRunnerConfig, layout: _Layout) -> None:
        self.config = config
        self.errors: list[str] = []
        self.gpu_metrics = []
        self.npu_metrics = []
        self.system_metrics = []
        self.workers_started = False
        self.stopped = False
        self.boundaries: dict[str, dict[str, dict[str, Any]]] = {}
        self.request_start_ns: int | None = None
        self.request_end_ns: int | None = None
        self.gpu = GpuTelemetryCollector(
            run_id=f"{layout.run_id}-gpu",
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            sample_interval_ms=config.sample_interval_ms,
            client=NvmlClient(),
            known_gpu_indices=config.gpu_indices,
        )
        self.npu = NpuTelemetryCollector(
            run_id=f"{layout.run_id}-npu",
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            sample_interval_ms=config.sample_interval_ms,
            client=RblnSmiClient(device_ids=config.npu_indices),
            known_npu_indices=config.npu_indices,
        )
        self.system = ProcTelemetryCollector(
            run_id=f"{layout.run_id}-gpu",
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
        )
        self.collector_group = CollectorGroup(
            (self.gpu, self.npu, self.system), errors=self.errors
        )
        interval_sec = config.sample_interval_ms / 1000
        self.workers = {
            "gpu": _TelemetryWorker(
                name="gpu", collector=self.gpu, target=self.gpu_metrics,
                interval_sec=interval_sec, errors=self.errors,
            ),
            "npu": _TelemetryWorker(
                name="npu", collector=self.npu, target=self.npu_metrics,
                interval_sec=interval_sec, errors=self.errors,
            ),
            "system": _TelemetryWorker(
                name="system", collector=self.system, target=self.system_metrics,
                interval_sec=interval_sec, errors=self.errors,
            ),
        }

    def start(self) -> None:
        self.collector_group.start()
        for worker in self.workers.values():
            worker.start()
        self.workers_started = True

    def capture_boundary(self, role: str) -> dict[str, dict[str, Any]]:
        if not self.workers_started or self.stopped:
            raise HybridRunnerError("telemetry boundary requested while stopped")
        tickets = {
            name: worker.request(role) for name, worker in self.workers.items()
        }
        timeout_sec = max(10.0, self.config.sample_interval_ms / 100.0)
        samples = {
            name: self.workers[name].wait(ticket, timeout_sec)
            for name, ticket in tickets.items()
        }
        empty = [name for name, sample in samples.items() if not sample["sample_count"]]
        if empty:
            raise HybridRunnerError(
                f"{role} telemetry sample was empty: {', '.join(empty)}"
            )
        self.boundaries[role] = samples
        return samples

    def set_request_window(self, start_ns: int, end_ns: int) -> None:
        if end_ns < start_ns:
            raise HybridRunnerError("telemetry request window is reversed")
        self.request_start_ns = start_ns
        self.request_end_ns = end_ns

    def stop(self) -> None:
        if self.stopped:
            return
        timeout_sec = max(10.0, self.config.sample_interval_ms / 100.0)
        if self.workers_started:
            for worker in self.workers.values():
                worker.stop(timeout_sec)
        self.collector_group.close()
        self.workers_started = False
        self.stopped = True

    def lifecycle(self) -> dict[str, Any]:
        streams = {}
        for name, metrics in (
            ("gpu", self.gpu_metrics),
            ("npu", self.npu_metrics),
            ("system", self.system_metrics),
        ):
            by_timestamp: dict[int, object] = {}
            roles: dict[str, set[int]] = {}
            for metric in metrics:
                by_timestamp.setdefault(metric.timestamp_ns, metric)
                role = metric.attributes.get("telemetry.sample_role")
                if isinstance(role, str):
                    roles.setdefault(role, set()).add(metric.timestamp_ns)
            ordered = list(by_timestamp.values())
            timestamps = [metric.timestamp_ns for metric in ordered]
            actual_intervals = [
                metric.interval_ns
                for index, metric in enumerate(ordered)
                if index > 0 and isinstance(metric.interval_ns, int)
            ]
            interval_consistent = all(
                metric.interval_ns == metric.timestamp_ns - ordered[index - 1].timestamp_ns
                for index, metric in enumerate(ordered)
                if index > 0
            )
            background_during_request = 0
            if self.request_start_ns is not None and self.request_end_ns is not None:
                background_during_request = sum(
                    self.request_start_ns <= timestamp <= self.request_end_ns
                    for timestamp in roles.get("background", set())
                )
            streams[name] = {
                "metric_record_count": len(metrics),
                "sample_batch_count": len(ordered),
                "first_timestamp_ns": timestamps[0] if timestamps else None,
                "last_timestamp_ns": timestamps[-1] if timestamps else None,
                "timestamps_strictly_increasing": all(
                    current > previous
                    for previous, current in zip(timestamps, timestamps[1:])
                ),
                "interval_matches_timestamp_delta": interval_consistent,
                "actual_interval_ns": {
                    "count": len(actual_intervals),
                    "min": min(actual_intervals) if actual_intervals else None,
                    "max": max(actual_intervals) if actual_intervals else None,
                    "mean": (
                        sum(actual_intervals) / len(actual_intervals)
                        if actual_intervals else None
                    ),
                },
                "role_sample_count": {
                    role: len(values) for role, values in sorted(roles.items())
                },
                "background_samples_during_request": background_during_request,
            }
        return {
            "requested_interval_ms": self.config.sample_interval_ms,
            "request_start_ns": self.request_start_ns,
            "request_end_ns": self.request_end_ns,
            "boundaries": self.boundaries,
            "streams": streams,
            "errors": list(self.errors),
        }


def build_hybrid_run_plan(
    config: HybridRunnerConfig,
    *,
    run_root: Path,
    run_id: str,
    profile_mode: HybridProfileMode,
) -> dict[str, object]:
    layout = _Layout(Path(run_root), run_id)
    commands = _commands(config, layout, profile_mode)
    return {
        "executes": False,
        "run_id": run_id,
        "profile_mode": profile_mode,
        "offline": config.offline,
        "commands": {name: mask_command(list(argv)) for name, argv in commands.items()},
        "workload": {
            "warmup_requests": config.workload.warmup_requests,
            "measured_requests": config.workload.measured_requests,
            "max_output_tokens": config.workload.max_output_tokens,
            "temperature": config.workload.temperature,
            "streaming": config.workload.streaming,
            "stores_prompt_or_response": False,
        },
        "outputs": {
            "hybrid": str(layout.hybrid),
            "gpu_source": str(layout.gpu),
            "npu_source": str(layout.npu),
            "coordinator": str(layout.coordinator),
            "perfetto": str(layout.perfetto),
            "request_focused_perfetto": str(layout.request_perfetto),
            "external_html_overview": str(layout.overview),
            "closeout_recovery": str(layout.recovery),
            "publication": str(layout.publication),
        },
        "creates_output": False,
    }


def _base_vllm_argv(
    executable: Path,
    model: Path,
    host: str,
    port: int,
    served_name: str,
    config: HybridRunnerConfig,
    *,
    gpu: bool,
    extra_args: tuple[str, ...],
) -> list[str]:
    argv = [
        str(executable), "serve", str(model), "--host", host, "--port", str(port),
        "--block-size", str(config.block_size), "--tensor-parallel-size", "1",
        "--served-model-name", served_name, "--max-model-len",
        str(config.max_model_len), "--max-num-seqs", str(config.max_num_seqs),
    ]
    if gpu:
        argv.extend(
            [
                "--enforce-eager", "--gpu-memory-utilization",
                str(config.gpu_memory_utilization),
                "--kv-transfer-config",
                json.dumps(config.prefill_connector, separators=(",", ":")),
            ]
        )
    else:
        argv.extend(
            [
                "--kv-transfer-config",
                json.dumps(config.decode_connector, separators=(",", ":")),
            ]
        )
    argv.extend(extra_args)
    return argv


def _commands(
    config: HybridRunnerConfig, layout: _Layout, profile_mode: HybridProfileMode
) -> dict[str, tuple[str, ...]]:
    prefill = _base_vllm_argv(
        config.prefill.executable, config.model_path, config.prefill.host,
        config.prefill.http_port, config.served_model_name, config, gpu=True,
        extra_args=config.prefill.extra_args,
    )
    decode = _base_vllm_argv(
        config.decode.executable, config.model_path, config.decode.host,
        config.decode.http_port, config.served_model_name, config, gpu=False,
        extra_args=config.decode.extra_args,
    )
    if profile_mode == "gpu-torch":
        prefill.extend(
            [
                "--profiler-config.profiler=torch",
                f"--profiler-config.torch_profiler_dir={layout.gpu / config.profiler_outputs.gpu_torch_subdir}",
                "--profiler-config.torch_profiler_use_gzip=true",
            ]
        )
    elif profile_mode == "gpu-nsys":
        prefill.extend(["--profiler-config.profiler=cuda"])
        prefill = [
            str(config.nsys_executable), "profile", "--trace=cuda,nvtx,osrt",
            "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop", "--force-overwrite=false", "--output",
            str(layout.gpu / config.profiler_outputs.gpu_nsys_basename), *prefill,
        ]
    elif profile_mode == "npu-torch":
        decode.extend(
            [
                "--profiler-config.profiler=torch",
                f"--profiler-config.torch_profiler_dir={layout.npu / config.profiler_outputs.npu_torch_subdir}",
                "--profiler-config.torch_profiler_use_gzip=true",
            ]
        )
    elif profile_mode == "npu-rbln":
        decode.extend(["--profiler-config.profiler=cuda"])
    proxy = (
        str(config.proxy_python), "-m", config.proxy_entry_point,
        "--host", config.proxy_host, "--port", str(config.proxy_port),
        "--prefill-host", config.prefill.host, "--prefill-port",
        str(config.prefill.http_port), "--decode-host", config.decode.host,
        "--decode-port", str(config.decode.http_port), "--timeout-sec",
        str(config.request_timeout_sec), "--marker-file",
        str(layout.coordinator / "raw/runtime_markers/proxy-markers.jsonl"),
        "--host-id", HOST_ID, "--clock-domain-id", CLOCK_DOMAIN_ID,
    )
    return {"prefill": tuple(prefill), "decode": tuple(decode), "proxy": proxy}


class HybridRunner:
    def __init__(
        self,
        config: HybridRunnerConfig,
        *,
        run_root: Path,
        run_id: str,
        profile_mode: HybridProfileMode,
        enable_telemetry: bool = True,
        process_factory: Callable[..., ManagedProcess] = ManagedProcess,
        client_factory: Callable[..., OpenAICompletionClient] = OpenAICompletionClient,
    ) -> None:
        self.config = config
        self.layout = _Layout(Path(run_root), run_id)
        self.profile_mode = profile_mode
        self.enable_telemetry = enable_telemetry
        self.process_factory = process_factory
        self.client_factory = client_factory
        self._runtime_marker_capability = False

    def run(self) -> HybridRunResult:
        config, layout = self.config, self.layout
        self._preflight()
        RunPaths(layout.run_root, f"{layout.run_id}-gpu").create()
        RunPaths(layout.run_root, f"{layout.run_id}-npu").create()
        layout.coordinator.mkdir(parents=True)
        layout.publication.mkdir(parents=True)
        for path in (
            layout.gpu / config.profiler_outputs.gpu_torch_subdir,
            layout.gpu / config.profiler_outputs.gpu_nsys_basename.parent,
            layout.npu / config.profiler_outputs.npu_torch_subdir,
            layout.npu / config.profiler_outputs.npu_rbln_subdir,
            layout.coordinator / "raw/runtime_markers",
        ):
            path.mkdir(parents=True, exist_ok=True)
        commands = _commands(config, layout, self.profile_mode)
        _plain_json(layout.coordinator / "execution_plan.json", build_hybrid_run_plan(
            config, run_root=layout.run_root, run_id=layout.run_id,
            profile_mode=self.profile_mode,
        ))
        cache_before = _cache_fingerprint(config.rbln_cache_path)
        _plain_json(layout.coordinator / "cache_before.json", cache_before)

        processes = self._processes(commands)
        telemetry = _Telemetry(config, layout)
        observations: list[CompletionObservation] = []
        warmups: list[CompletionObservation] = []
        errors: list[str] = []
        boundary: dict[str, Any] | None = None
        capture_started_unix_ns: int | None = None
        request_start_ns: int | None = None
        request_end_ns: int | None = None
        profiler_active = False
        client: OpenAICompletionClient | None = None
        started: list[str] = []
        shutdown: dict[str, Any] = {}
        owned_processes: dict[str, dict[str, Any]] = {}
        try:
            processes["decode"].start()
            started.append("decode")
            self._record_started_process(owned_processes, "decode", processes["decode"])
            if self.enable_telemetry:
                telemetry.start()
            _wait_http(
                f"http://{config.decode.host}:{config.decode.http_port}",
                "/v1/models", processes["decode"], config.startup_timeout_sec,
            )
            self._compile_gate()
            processes["prefill"].start()
            started.append("prefill")
            self._record_started_process(owned_processes, "prefill", processes["prefill"])
            _wait_http(
                f"http://{config.prefill.host}:{config.prefill.http_port}",
                "/v1/models", processes["prefill"], config.startup_timeout_sec,
            )
            processes["proxy"].start()
            started.append("proxy")
            self._record_started_process(owned_processes, "proxy", processes["proxy"])
            _wait_http(
                f"http://{config.proxy_host}:{config.proxy_port}",
                "/healthcheck", processes["proxy"], config.startup_timeout_sec,
            )
            client = self.client_factory(
                f"http://{config.proxy_host}:{config.proxy_port}",
                timeout_sec=config.request_timeout_sec,
            )
            prompt = config.workload.prompt_text()
            for index in range(config.workload.warmup_requests):
                warmups.append(self._request(client, prompt, f"warmup-{index:03d}"))

            if self.enable_telemetry:
                telemetry.capture_boundary("baseline")

            target_url = self._profile_url()
            if target_url is not None:
                capture_started_unix_ns = time.time_ns()
                start = _profile_call(target_url, "/start_profile", config.request_timeout_sec)
                profiler_active = True
            else:
                start = None
            request_start_ns = time.monotonic_ns()
            for index in range(config.workload.measured_requests):
                observations.append(self._request(client, prompt, f"measured-{index:03d}"))
            response_end_ns = max(item.done_ns for item in observations)
            if self.enable_telemetry:
                telemetry.set_request_window(request_start_ns, response_end_ns)
                try:
                    telemetry.capture_boundary("final")
                except Exception as error:
                    errors.append(f"final telemetry: {type(error).__name__}: {error}")
                telemetry.stop()
            _wait_runtime_marker_completion(
                layout.coordinator / "raw/runtime_markers",
                {item.request_id for item in observations},
                config.request_timeout_sec,
            )
            request_end_ns = time.monotonic_ns()
            if target_url is not None:
                stop = _profile_call(target_url, "/stop_profile", config.request_timeout_sec)
                profiler_active = False
                assert start is not None
                boundary = self._boundary(start, stop, request_start_ns, request_end_ns)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            if profiler_active:
                try:
                    target_url = self._profile_url()
                    assert target_url is not None
                    _profile_call(target_url, "/stop_profile", config.request_timeout_sec)
                except Exception as error:
                    errors.append(f"profiler cleanup: {error}")
            if client is not None:
                try:
                    client.close()
                except Exception as error:
                    errors.append(f"client close: {error}")
            telemetry.stop()
            errors.extend(telemetry.errors)
            for name in reversed(started):
                try:
                    leader_signal = (
                        signal.SIGINT
                        if name == "prefill" and self.profile_mode == "gpu-nsys"
                        else signal.SIGTERM
                    )
                    result = processes[name].stop_leader_first(
                        leader_signal=leader_signal
                    )
                    shutdown[name] = {
                        "return_code": result.return_code,
                        "terminated": result.terminated,
                        "killed": result.killed,
                    }
                    if result.killed:
                        errors.append(f"{name} required SIGKILL")
                    owned_processes.setdefault(name, {})["cleanup"] = shutdown[name]
                except Exception as error:
                    errors.append(f"{name} cleanup: {error}")
                    owned_processes.setdefault(name, {})["cleanup_error"] = str(error)
            _plain_json(
                layout.coordinator / "owned_processes.json",
                {
                    "processes": owned_processes,
                    "start_order": started,
                    "cleanup_order": list(reversed(started)),
                },
            )

        cache_after = _cache_fingerprint(config.rbln_cache_path)
        _plain_json(layout.coordinator / "cache_after.json", cache_after)
        if cache_before != cache_after:
            errors.append("persistent RBLN model cache fingerprint changed")
        _plain_json(layout.coordinator / "cleanup.json", shutdown)
        shutdown_integrity = _shutdown_integrity(layout, shutdown)
        _plain_json(
            layout.coordinator / "shutdown_integrity.json", shutdown_integrity
        )
        shutdown_error = (
            "shutdown integrity invalid: " + str(shutdown_integrity["reason"])
            if shutdown_integrity["status"] == "invalid"
            else None
        )
        try:
            self._compile_gate()
            profile = self._profile_metadata(
                boundary=boundary,
                capture_started_unix_ns=capture_started_unix_ns,
            )
            self._write_sources(
                warmups=warmups,
                observations=observations,
                telemetry=telemetry,
                profile=profile,
                errors=errors,
            )
            source_before = {
                "gpu": _tree_fingerprint(layout.gpu),
                "npu": _tree_fingerprint(layout.npu),
            }
            if not errors and len(observations) == config.workload.measured_requests:
                merge = HybridBundleMerger(
                    HybridMergeConfig(
                        run_root=layout.run_root,
                        run_id=layout.run_id,
                        gpu_run=layout.gpu,
                        npu_run=layout.npu,
                        alignment_method=AlignmentMethod.SAME_CLOCK_DOMAIN,
                        coordinator_host_id=HOST_ID,
                        canonical_clock_domain_id="hybrid-canonical",
                        allow_non_fake_sources=True,
                    )
                ).merge()
                if merge.status is not RunStatus.SUCCEEDED:
                    errors.append(
                        "hybrid merge did not succeed: " + "; ".join(merge.reasons)
                    )
                else:
                    source_after = {
                        "gpu": _tree_fingerprint(layout.gpu),
                        "npu": _tree_fingerprint(layout.npu),
                    }
                    if source_before != source_after:
                        raise HybridRunnerError(
                            "normalized source fingerprint changed during derivation"
                        )
                    _plain_json(
                        layout.coordinator / "source_fingerprint.json",
                        {
                            "unchanged": source_before == source_after,
                            "gpu": source_before["gpu"],
                            "npu": source_before["npu"],
                        },
                    )
                    _plain_json(
                        layout.coordinator / "result.json",
                        {
                            "run_id": layout.run_id,
                            "profile_mode": self.profile_mode,
                            "status": "failed" if shutdown_error else "succeeded",
                            "inference_status": "succeeded",
                            "shutdown_integrity": shutdown_integrity["status"],
                            "shutdown_reason": shutdown_integrity["reason"],
                            "demo_only": shutdown_integrity["demo_only"],
                            "stage": (
                                "diagnostic_collection_closeout"
                                if shutdown_error
                                else "immutable_collection_closeout"
                            ),
                            "hardware_rerun": True,
                            "warmup_completed": len(warmups),
                            "measured_completed": len(observations),
                            "errors": [shutdown_error] if shutdown_error else [],
                        },
                    )
                    self._create_closeout()
                    self._derive_products()
        except Exception as error:
            errors.append(f"postprocess {type(error).__name__}: {error}")

        if shutdown_error is not None:
            errors.append(shutdown_error)

        status = RunStatus.SUCCEEDED if not errors else RunStatus.FAILED
        result_payload = {
                "run_id": layout.run_id,
                "profile_mode": self.profile_mode,
                "status": status.value,
                "warmup_completed": len(warmups),
                "measured_completed": len(observations),
                "errors": errors,
                "failures": [
                    {"failure_class": classify_failure(error), "message": error}
                    for error in errors
                ],
                "outputs": {
                    "hybrid": str(layout.hybrid), "gpu": str(layout.gpu),
                    "npu": str(layout.npu), "perfetto": str(layout.perfetto),
                    "request_focused_perfetto": str(layout.request_perfetto),
                    "external_html_overview": str(layout.overview),
                    "closeout_recovery": str(layout.recovery),
                    "publication": str(layout.publication),
                },
            }
        if not (layout.coordinator / "result.json").exists():
            _plain_json(layout.coordinator / "result.json", result_payload)
        _plain_json(layout.publication / "result.json", result_payload)
        return HybridRunResult(
            status=status,
            run_directory=layout.hybrid,
            gpu_run_directory=layout.gpu,
            npu_run_directory=layout.npu,
            coordinator_directory=layout.coordinator,
            perfetto_directory=layout.perfetto if layout.perfetto.exists() else None,
            request_focused_perfetto_directory=(
                layout.request_perfetto
                if layout.request_perfetto.exists()
                else None
            ),
            overview_directory=layout.overview if layout.overview.exists() else None,
            recovery_directory=layout.recovery if layout.recovery.exists() else None,
            publication_directory=layout.publication,
            warmup_count=len(warmups),
            measured_count=len(observations),
            errors=tuple(errors),
        )

    @staticmethod
    def _record_started_process(
        records: dict[str, dict[str, Any]],
        name: str,
        managed: ManagedProcess,
    ) -> None:
        process = getattr(managed, "process", None)
        records[name] = {
            "pid": getattr(process, "pid", None),
            "process_group_id": getattr(managed, "process_group_id", None),
            "owned": True,
        }

    def _preflight(self) -> None:
        config = self.config
        required_files = [
            config.prefill.executable, config.decode.executable, config.proxy_python
        ]
        if self.profile_mode == "gpu-nsys":
            required_files.append(config.nsys_executable)
        if config.trace_processor_path is not None:
            required_files.append(config.trace_processor_path)
        missing = [path for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"required executable is missing: {missing[0]}")
        required_dirs = [
            config.model_path, config.rbln_cache_path,
            config.prefill.working_directory, config.decode.working_directory,
            config.prefill.pythonpath, config.decode.pythonpath,
        ]
        missing_dirs = [path for path in required_dirs if not path.is_dir()]
        if missing_dirs:
            raise FileNotFoundError(f"required directory is missing: {missing_dirs[0]}")
        ports = (
            (config.prefill.host, config.prefill.http_port),
            (config.decode.host, config.decode.http_port),
            (config.proxy_host, config.proxy_port),
            (config.prefill.host, config.prefill.nixl_port),
            (config.decode.host, config.decode.nixl_port),
        )
        busy = [f"{host}:{port}" for host, port in ports if not _port_available(host, port)]
        if busy:
            raise HybridRunnerError(f"configured port is already occupied: {busy[0]}")
        if len(_cache_fingerprint(config.rbln_cache_path)) < 2:
            raise HybridRunnerError("RBLN cache has fewer than two model artifacts")

    def _processes(self, commands: dict[str, tuple[str, ...]]) -> dict[str, ManagedProcess]:
        config, layout = self.config, self.layout
        common = {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "UCX_NET_DEVICES": "all",
        }
        prefill_env = {
            **common,
            "PYTHONPATH": str(config.prefill.pythonpath),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.gpu_indices)),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(config.prefill.nixl_port),
        }
        marker_dir = layout.coordinator / "raw/runtime_markers"
        decode_env = {
            **common,
            "PYTHONPATH": str(config.decode.pythonpath),
            "RBLN_DEVICES": ",".join(map(str, config.npu_indices)),
            "VLLM_RBLN_USE_VLLM_MODEL": "1",
            "VLLM_RBLN_COMPILE_MODEL": "1",
            "VLLM_CACHE_ROOT": str(config.rbln_cache_path),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(config.decode.nixl_port),
            "VLLM_RBLN_RUNTIME_MARKER_DIR": str(marker_dir),
            "VLLM_RBLN_RUNTIME_MARKER_HOST_ID": HOST_ID,
            "VLLM_RBLN_RUNTIME_MARKER_CLOCK_DOMAIN_ID": CLOCK_DOMAIN_ID,
            "VLLM_RBLN_NIXL_READ_DIAGNOSTIC": "1",
        }
        if self.profile_mode == "npu-rbln":
            decode_env.update(
                {
                    "RBLN_PROFILER": "1",
                    "VLLM_RBLN_DEVICE_PROFILER_DIR": str(
                        layout.npu / config.profiler_outputs.npu_rbln_subdir
                    ),
                }
            )
        proxy_pythonpath = str(Path(__file__).resolve().parents[2])
        proxy_env = {"PYTHONPATH": proxy_pythonpath}
        specs = {
            "prefill": CommandSpec(
                argv=commands["prefill"], cwd=config.prefill.working_directory,
                env_overrides=prefill_env,
                env_allowlist=_ENV_ALLOWLIST,
                terminate_grace_sec=config.shutdown_timeout_sec,
            ),
            "decode": CommandSpec(
                argv=commands["decode"], cwd=config.decode.working_directory,
                env_overrides=decode_env,
                env_allowlist=_ENV_ALLOWLIST,
                terminate_grace_sec=config.shutdown_timeout_sec,
            ),
            "proxy": CommandSpec(
                argv=commands["proxy"], cwd=Path(__file__).resolve().parents[3],
                env_overrides=proxy_env, env_allowlist=_ENV_ALLOWLIST,
                terminate_grace_sec=min(5.0, config.shutdown_timeout_sec),
            ),
        }
        return {
            name: self.process_factory(
                spec,
                layout.coordinator / f"raw/{name}.stdout.log",
                layout.coordinator / f"raw/{name}.stderr.log",
            )
            for name, spec in specs.items()
        }

    def _request(
        self, client: OpenAICompletionClient, prompt: str, suffix: str
    ) -> CompletionObservation:
        return client.complete(
            model=self.config.served_model_name,
            request_id=f"{self.layout.run_id}-{suffix}",
            prompt=prompt,
            max_output_tokens=self.config.workload.max_output_tokens,
            temperature=self.config.workload.temperature,
            stream=self.config.workload.streaming,
        )

    def _profile_url(self) -> str | None:
        if self.profile_mode.startswith("gpu-"):
            return f"http://{self.config.prefill.host}:{self.config.prefill.http_port}"
        if self.profile_mode.startswith("npu-"):
            return f"http://{self.config.decode.host}:{self.config.decode.http_port}"
        return None

    @staticmethod
    def _boundary(
        start: dict[str, Any], stop: dict[str, Any], request_start: int, request_end: int
    ) -> dict[str, Any]:
        return {
            "start_before_monotonic_ns": start["before_monotonic_ns"],
            "start_after_monotonic_ns": start["after_monotonic_ns"],
            "request_start_monotonic_ns": request_start,
            "request_end_monotonic_ns": request_end,
            "stop_before_monotonic_ns": stop["before_monotonic_ns"],
            "stop_after_monotonic_ns": stop["after_monotonic_ns"],
            "start_http_status": start["http_status"],
            "stop_http_status": stop["http_status"],
            "api": {"start": start, "stop": stop},
        }

    def _compile_gate(self) -> None:
        path = self.layout.coordinator / "raw/decode.stderr.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        forbidden = [label for label in ("Compile(#0)", "Compile(#2)") if label in text]
        sampler = text.count("Compile(#1)")
        result = {
            "prefill_or_decode_model_compile": forbidden,
            "sampler_compile_count": sampler,
            "model_reuse_accepted": not forbidden,
        }
        _plain_json(self.layout.coordinator / "compile_gate.json", result)
        if forbidden:
            raise HybridRunnerError(
                "persistent model graph recompile detected: " + ", ".join(forbidden)
            )

    def _profile_metadata(
        self,
        *,
        boundary: dict[str, Any] | None,
        capture_started_unix_ns: int | None,
    ) -> dict[str, Any] | None:
        if self.profile_mode == "monitor":
            return None
        if boundary is None or capture_started_unix_ns is None:
            raise HybridRunnerError("detailed profiler boundary is unavailable")
        mode = self.profile_mode
        root = self.layout.gpu if mode.startswith("gpu-") else self.layout.npu
        if mode in {"gpu-torch", "npu-torch"}:
            folder = root / (
                self.config.profiler_outputs.gpu_torch_subdir
                if mode == "gpu-torch"
                else self.config.profiler_outputs.npu_torch_subdir
            )
            paths = sorted(
                path for path in folder.rglob("*")
                if path.is_file() and str(path).endswith((".pt.trace.json", ".pt.trace.json.gz"))
            )
            detail = validate_torch_traces(
                paths,
                target="gpu" if mode == "gpu-torch" else "npu",
                capture_started_unix_ns=capture_started_unix_ns,
                run_root=root,
                forbidden_text=(self.config.workload.prompt_text(),),
                capture_boundary=boundary,
            )
            native_clock = "gpu:torch-chrome-trace" if mode == "gpu-torch" else "npu:torch-chrome-trace"
            native_unit = "chrome_trace_microseconds"
            kind = "gpu_torch" if mode == "gpu-torch" else "npu_vllm"
        elif mode == "gpu-nsys":
            reports = sorted(
                (root / self.config.profiler_outputs.gpu_nsys_basename.parent).glob(
                    "*.nsys-rep"
                )
            )
            if len(reports) != 1:
                raise HybridRunnerError("Nsight produced != 1 .nsys-rep")
            sqlite_path = reports[0].with_suffix(".sqlite")
            exported = subprocess.run(
                [
                    str(self.config.nsys_executable), "export", "--type", "sqlite",
                    "--force-overwrite=false", "--output", str(sqlite_path),
                    str(reports[0]),
                ],
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            if (
                exported.returncode != 0
                or not sqlite_path.is_file()
                or sqlite_path.stat().st_size <= 0
            ):
                raise HybridRunnerError(
                    "official nsys SQLite export failed: "
                    + exported.stderr[-1000:]
                )
            stats = self._nsys_stats(reports[0])
            detail = validate_nsys_report(
                reports[0], capture_started_unix_ns=capture_started_unix_ns,
                run_root=root, preexisting_paths=(), stats=stats,
                capture_boundary=boundary,
            )
            native_clock, native_unit, kind = (
                "gpu:nsight-systems-native", "nsight-report-native", "gpu_nsys"
            )
            detail["sqlite_export"] = {
                "path": sqlite_path.relative_to(root).as_posix(),
                "size_bytes": sqlite_path.stat().st_size,
                "sha256": _sha256(sqlite_path),
                "official_exporter": "nsys export --type sqlite",
            }
        else:
            paths = sorted(
                (root / self.config.profiler_outputs.npu_rbln_subdir).glob("*.pb")
            )
            strings_results = {}
            for path in paths:
                result = subprocess.run(
                    ["strings", str(path)], text=True, capture_output=True,
                    timeout=30, check=False,
                )
                strings_results[str(path)] = {
                    "returncode": result.returncode, "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            detail = validate_rbln_reports(
                paths, capture_started_unix_ns=capture_started_unix_ns,
                run_root=root, preexisting_paths=(),
                strings_results=strings_results, capture_boundary=boundary,
            )
            detail["report_count"] = len(paths)
            native_clock, native_unit, kind = (
                "npu:rbln-profiler-native", "rbln_report_native", "npu_rbln"
            )
        detail.update({"kind": kind, "enabled": True, "api": boundary["api"]})
        anchors = [
            {
                "kind": "profiler_start_api",
                "before_monotonic_ns": boundary["start_before_monotonic_ns"],
                "after_monotonic_ns": boundary["start_after_monotonic_ns"],
                "before_unix_ns": boundary["api"]["start"]["before_unix_ns"],
                "after_unix_ns": boundary["api"]["start"]["after_unix_ns"],
                "http_status": 200,
                "request_start_monotonic_ns": boundary["request_start_monotonic_ns"],
            },
            {
                "kind": "profiler_stop_api",
                "before_monotonic_ns": boundary["stop_before_monotonic_ns"],
                "after_monotonic_ns": boundary["stop_after_monotonic_ns"],
                "before_unix_ns": boundary["api"]["stop"]["before_unix_ns"],
                "after_unix_ns": boundary["api"]["stop"]["after_unix_ns"],
                "http_status": 200,
                "request_end_monotonic_ns": boundary["request_end_monotonic_ns"],
            },
        ]
        alignment = build_profiler_alignment(
            profiler_type=kind,
            native_clock_domain=native_clock,
            native_timestamp_unit=native_unit,
            canonical_clock_domain=CLOCK_DOMAIN_ID,
            anchors=anchors,
            native_capture_start=detail.get("native_timestamp_min"),
            native_capture_end=detail.get("native_timestamp_max"),
        )
        return {"root": root, "detail": detail, "alignment": alignment}

    def _nsys_stats(self, report: Path) -> dict[str, dict[str, object]]:
        results = {}
        for name in ("cuda_api_sum", "cuda_gpu_kern_sum", "osrt_sum", "nvtx_sum"):
            result = subprocess.run(
                [str(self.config.nsys_executable), "stats", "--report", name,
                 "--format", "csv", "--output", "-", str(report)],
                text=True, capture_output=True, timeout=120, check=False,
            )
            results[name] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        _plain_json(self.layout.gpu / "summary/nsys_stats.json", results)
        return results

    def _marker_events(self, observations: list[CompletionObservation]) -> tuple[list, list]:
        root = self.layout.coordinator / "raw/runtime_markers"
        proxy_paths = [root / "proxy-markers.jsonl"]
        npu_paths = sorted(path for path in root.glob("runtime-markers-*.jsonl"))
        measured_ids = {item.request_id for item in observations}
        gpu_events = ingest_runtime_marker_files(
            proxy_paths, run_id=f"{self.layout.run_id}-gpu",
            expected_host_id=HOST_ID, expected_clock_domain_id=CLOCK_DOMAIN_ID,
            process_devices={"proxy": (DeviceType.GPU, "gpu-0")},
        )
        npu_events = ingest_runtime_marker_files(
            npu_paths, run_id=f"{self.layout.run_id}-npu",
            expected_host_id=HOST_ID, expected_clock_domain_id=CLOCK_DOMAIN_ID,
            process_devices={
                "npu_engine": (DeviceType.NPU, "npu-0"),
                "npu_model_runner": (DeviceType.NPU, "npu-0"),
            },
        )
        gpu_filtered = [event for event in gpu_events if event.request_id in measured_ids]
        npu_filtered = [
            event for event in npu_events
            if event.attributes.get("hybrid.correlation_id") in measured_ids
        ]
        self._runtime_marker_capability = any(
            event.attributes.get("hybrid.marker_version") == "1.1.0"
            for event in (*gpu_filtered, *npu_filtered)
        )
        for request_id in measured_ids:
            validation = validate_marker_order(
                [
                    *[event for event in gpu_filtered if event.request_id == request_id],
                    *[
                        event for event in npu_filtered
                        if event.attributes.get("hybrid.correlation_id") == request_id
                    ],
                ]
            )
            if validation.status != "valid":
                raise HybridRunnerError(
                    f"canonical marker validation failed for {request_id}: "
                    f"missing={validation.missing_markers}, "
                    f"duplicates={validation.duplicate_markers}, "
                    f"pairing={validation.pairing_issues}"
                )
        return gpu_filtered, npu_filtered

    def _write_sources(
        self,
        *,
        warmups: list[CompletionObservation],
        observations: list[CompletionObservation],
        telemetry: _Telemetry,
        profile: dict[str, Any] | None,
        errors: list[str],
    ) -> None:
        marker_error: Exception | None = None
        try:
            gpu_events, npu_events = self._marker_events(observations)
        except Exception as error:
            # Request and telemetry evidence must survive a later marker or
            # normalization failure. Invalid markers remain available as raw
            # artifacts, but are not emitted as normalized events.
            marker_error = error
            gpu_events, npu_events = [], []
        gpu_metrics = [*telemetry.gpu_metrics, *telemetry.system_metrics]
        for observation in observations:
            gpu_metrics.extend(observation_metrics(f"{self.layout.run_id}-gpu", observation))
        gpu_metrics.extend(measured_window_metrics(f"{self.layout.run_id}-gpu", observations))
        source_data = {
            "gpu": (self.layout.gpu, gpu_events, gpu_metrics),
            "npu": (self.layout.npu, npu_events, telemetry.npu_metrics),
        }
        telemetry_lifecycle = telemetry.lifecycle()
        _plain_json(
            self.layout.coordinator / "telemetry_lifecycle.json",
            telemetry_lifecycle,
        )
        _plain_json(
            self.layout.gpu / "summary/telemetry_lifecycle.json",
            {
                "requested_interval_ms": telemetry_lifecycle["requested_interval_ms"],
                "request_start_ns": telemetry_lifecycle["request_start_ns"],
                "request_end_ns": telemetry_lifecycle["request_end_ns"],
                "boundaries": {
                    role: {
                        name: sample
                        for name, sample in samples.items()
                        if name in {"gpu", "system"}
                    }
                    for role, samples in telemetry_lifecycle["boundaries"].items()
                },
                "streams": {
                    name: telemetry_lifecycle["streams"][name]
                    for name in ("gpu", "system")
                },
                "errors": telemetry_lifecycle["errors"],
            },
        )
        _plain_json(
            self.layout.npu / "summary/telemetry_lifecycle.json",
            {
                "requested_interval_ms": telemetry_lifecycle["requested_interval_ms"],
                "request_start_ns": telemetry_lifecycle["request_start_ns"],
                "request_end_ns": telemetry_lifecycle["request_end_ns"],
                "boundaries": {
                    role: {"npu": samples["npu"]}
                    for role, samples in telemetry_lifecycle["boundaries"].items()
                },
                "streams": {"npu": telemetry_lifecycle["streams"]["npu"]},
                "errors": telemetry_lifecycle["errors"],
            },
        )
        measured_rows = [
            {
                "request_id": item.request_id,
                "http_status": item.http_status,
                "request_start_ns": item.received_ns,
                "http_response_start_ns": item.response_started_ns,
                "valid_token_timestamps_ns": list(item.token_timestamps_ns),
                "stream_end_ns": item.done_ns,
                "status": "succeeded",
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
                "e2e_ns": item.e2e_ns,
                "ttft_ns": item.ttft_ns,
                "tpot_ns": item.tpot_ns,
            }
            for item in observations
        ]
        _plain_jsonl(self.layout.gpu / "raw/client/measured_requests.jsonl", measured_rows)
        marker_root = self.layout.coordinator / "raw/runtime_markers"
        gpu_marker_root = self.layout.gpu / "raw/runtime"
        npu_marker_root = self.layout.npu / "raw/runtime"
        gpu_marker_root.mkdir(parents=True, exist_ok=True)
        npu_marker_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            marker_root / "proxy-markers.jsonl",
            gpu_marker_root / "proxy-markers.jsonl",
        )
        for path in sorted(marker_root.glob("runtime-markers-*")):
            if path.is_file():
                shutil.copyfile(path, npu_marker_root / path.name)
        if telemetry.gpu.last_raw_snapshot is not None:
            (self.layout.gpu / "raw/gpu/nvml-last.json").write_text(
                telemetry.gpu.last_raw_snapshot, encoding="utf-8"
            )
        if telemetry.npu.last_raw_output is not None:
            (self.layout.npu / "raw/npu/rbln-smi-last.json").write_text(
                telemetry.npu.last_raw_output, encoding="utf-8"
            )
        for role, (root, events, metrics) in source_data.items():
            run_id = f"{self.layout.run_id}-{role}"
            write_jsonl(root / "events/events.jsonl", events)
            write_jsonl(root / "metrics/metrics.jsonl", metrics)
            write_jsonl(
                root / "clocks/clock_domains.jsonl",
                [
                    ClockDomain(
                        run_id=run_id, clock_domain_id=CLOCK_DOMAIN_ID,
                        host_id=HOST_ID, clock_type=ClockType.MONOTONIC, unit="ns",
                        monotonic=True, adjustable=False,
                        attributes={"clock.source": "time.monotonic_ns"},
                    )
                ],
            )
        if profile is not None:
            root = profile["root"]
            _plain_json(root / "summary/detailed_profile.json", profile["detail"])
            _plain_json(root / "clocks/profiler_alignment.json", profile["alignment"])
            clocks = list(read_jsonl(root / "clocks/clock_domains.jsonl"))
            clocks.append(
                build_profiler_clock_domain(
                    run_id=root.name,
                    clock_domain_id=profile["alignment"]["native_clock_domain"],
                    host_id=HOST_ID,
                    clock_type=ClockType.EXTERNAL,
                    profile_kind=profile["alignment"]["profiler_type"],
                    native_timestamp_unit=profile["alignment"]["native_timestamp_unit"],
                    alignment_status="partial",
                )
            )
            write_jsonl(root / "clocks/clock_domains.jsonl", clocks, overwrite=True)
        for role, (root, _events, _metrics) in source_data.items():
            artifacts = self._artifacts(root, role, profile)
            write_jsonl(root / "artifacts/artifacts.jsonl", artifacts)
            manifest = self._manifest(
                role=role,
                status=(
                    RunStatus.FAILED
                    if errors or marker_error is not None
                    else RunStatus.SUCCEEDED
                ),
                detailed=profile is not None and profile["root"] == root,
            )
            write_json(root / "manifest.json", manifest)
        _plain_json(
            self.layout.coordinator / "requests.json",
            {
                "warmup_request_ids": [item.request_id for item in warmups],
                "measured": measured_rows,
                "stores_prompt_or_generated_text": False,
                "clock": "CLOCK_MONOTONIC_NS",
                "method_id": "independent_streaming_client_v1",
            },
        )
        if marker_error is not None:
            raise marker_error

    def _artifacts(
        self, root: Path, role: str, profile: dict[str, Any] | None
    ) -> list[ArtifactReference]:
        run_id = root.name
        files: list[tuple[Path, ArtifactKind, str, str | None]] = []
        coordinator = self.layout.coordinator
        server_name = "prefill" if role == "gpu" else "decode"
        for suffix in ("stdout", "stderr"):
            source = coordinator / f"raw/{server_name}.{suffix}.log"
            if source.is_file():
                files.append((source, ArtifactKind.RAW_LOG, "text", None))
        if role == "gpu":
            files.append((root / "raw/client/measured_requests.jsonl", ArtifactKind.RAW_LOG, "jsonl", None))
        files.extend(
            [
                (root / "events/events.jsonl", ArtifactKind.EVENT_STREAM, "jsonl", None),
                (root / "metrics/metrics.jsonl", ArtifactKind.METRIC_STREAM, "jsonl", None),
            ]
        )
        for path in sorted((root / "raw/runtime").glob("*")):
            if path.is_file():
                files.append((path, ArtifactKind.RAW_LOG, "jsonl" if path.suffix == ".jsonl" else "json", None))
        telemetry_path = (
            root / "raw/gpu/nvml-last.json"
            if role == "gpu"
            else root / "raw/npu/rbln-smi-last.json"
        )
        if telemetry_path.is_file():
            files.append(
                (
                    telemetry_path,
                    ArtifactKind.TELEMETRY,
                    "json",
                    None,
                )
            )
        lifecycle_path = root / "summary/telemetry_lifecycle.json"
        if lifecycle_path.is_file():
            files.append((lifecycle_path, ArtifactKind.OTHER, "json", None))
        if profile is not None and profile["root"] == root:
            alignment = root / "clocks/profiler_alignment.json"
            detail = root / "summary/detailed_profile.json"
            files.extend(
                [
                    (alignment, ArtifactKind.OTHER, "json", None),
                    (detail, ArtifactKind.OTHER, "json", None),
                ]
            )
            native_clock = profile["alignment"]["native_clock_domain"]
            for record in profile["detail"]["files"]:
                path = root / record["path"]
                if self.profile_mode in {"gpu-torch", "npu-torch"}:
                    kind = ArtifactKind.TORCH_TRACE
                    format_name = (
                        "chrome_trace_json_gzip"
                        if path.name.endswith(".json.gz")
                        else "chrome_trace_json"
                    )
                elif self.profile_mode == "gpu-nsys":
                    kind, format_name = ArtifactKind.NSYS_REPORT, "nsys-rep"
                else:
                    kind, format_name = ArtifactKind.RBLN_REPORT, "vendor-rbln-pb"
                files.append((path, kind, format_name, native_clock))
            if self.profile_mode == "gpu-nsys":
                for path in sorted(
                    (root / self.config.profiler_outputs.gpu_nsys_basename.parent).glob(
                        "*.sqlite"
                    )
                ):
                    files.append((path, ArtifactKind.OTHER, "sqlite", None))
        artifacts = []
        for index, (path, kind, format_name, clock) in enumerate(files):
            if path.is_relative_to(root):
                relative = path.relative_to(root).as_posix()
            else:
                # Logs remain owned by the coordinator; source bundles require local refs.
                target = root / "raw/server" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                path, relative = target, target.relative_to(root).as_posix()
            is_nvml = role == "gpu" and path.name == "nvml-last.json"
            artifacts.append(
                ArtifactReference(
                    run_id=run_id,
                    artifact_id=(
                        "nvml-last" if is_nvml else f"{role}-artifact-{index:03d}"
                    ),
                    artifact_kind=kind, relative_path=relative,
                    format=format_name,
                    producer=("nvml" if is_nvml else "hetero-profiler-hybrid-runner"),
                    created_at_unix_ns=path.stat().st_mtime_ns,
                    size_bytes=path.stat().st_size, sha256=_sha256(path),
                    host_id=HOST_ID, clock_domain_id=clock,
                    attributes={"hybrid.profile_mode": self.profile_mode},
                )
            )
        return artifacts

    def _manifest(self, *, role: str, status: RunStatus, detailed: bool) -> RunManifest:
        config = self.config
        device_type = DeviceType.GPU if role == "gpu" else DeviceType.NPU
        indices = config.gpu_indices if role == "gpu" else config.npu_indices
        return RunManifest(
            run_id=f"{self.layout.run_id}-{role}",
            mode=RunMode.GPU_ONLY if role == "gpu" else RunMode.NPU_ONLY,
            profile_mode=ProfileMode.DETAILED_PROFILE if detailed else ProfileMode.MONITOR,
            status=status,
            created_at_unix_ns=time.time_ns(),
            models=[ModelDescriptor(
                role="prefill" if role == "gpu" else "decode",
                model_id=str(config.model_path), revision=None,
                tokenizer_id=None, dtype="bfloat16",
            )],
            workload=WorkloadDescriptor(
                request_count=config.workload.measured_requests, concurrency=1,
                request_rate_per_s=None, input_tokens=None,
                output_tokens=config.workload.max_output_tokens,
                max_model_len=config.max_model_len,
                warmup_requests=config.workload.warmup_requests,
            ),
            hosts=[HostDescriptor(
                host_id=HOST_ID, role=role,
                hostname=platform.node() or "localhost",
                operating_system=platform.system() or "unknown",
                architecture=platform.machine() or "unknown",
            )],
            software=[
                SoftwareDescriptor(
                    name="vllm-rbln" if role == "npu" else "vllm",
                    version=None,
                    role=f"{role}-server",
                    path=str(
                        config.decode.executable
                        if role == "npu"
                        else config.prefill.executable
                    ),
                ),
                *(
                    [
                        SoftwareDescriptor(
                            name=NVML_DISTRIBUTION,
                            version=NVML_DISTRIBUTION_VERSION,
                            role="gpu-telemetry",
                            path=None,
                        )
                    ]
                    if role == "gpu"
                    else []
                ),
            ],
            devices=[DeviceDescriptor(
                host_id=HOST_ID, device_type=device_type,
                device_id=f"{role}-{index}",
                vendor="NVIDIA" if role == "gpu" else "Rebellions",
                model="discovered-by-telemetry", status="available",
                memory_total_bytes=None,
                attributes={
                    "nvml.gpu_index" if role == "gpu" else "device.index": index
                },
            ) for index in indices],
            configuration={
                "profile_mode": self.profile_mode,
                "max_model_len": config.max_model_len,
                "block_size": config.block_size,
                "offline": config.offline,
            },
            attributes={
                "hybrid.source_role": role,
                "hybrid.real_source": True,
                "hybrid.runner": "collect hybrid",
                **(
                    {
                        "hybrid.runtime_marker_version": "1.1.0",
                        "hybrid.runtime_marker_capabilities": [
                            "transfer_wait_observability_v1"
                        ],
                    }
                    if self._runtime_marker_capability
                    else {}
                ),
            },
        )

    def _derive_products(self) -> None:
        from ..overview.generator import OverviewGenerationConfig, generate_overview
        from ..perfetto.converter import PerfettoConversionConfig, convert_perfetto

        include_details = self.profile_mode != "monitor"
        conversion = convert_perfetto(
            PerfettoConversionConfig(
                run_directory=self.layout.hybrid,
                output_directory=self.layout.perfetto,
                trace_processor_path=self.config.trace_processor_path,
                include_native_details=include_details,
                request_focused=False,
            )
        )
        _plain_json(self.layout.publication / "perfetto_result.json", conversion)
        request_conversion = convert_perfetto(
            PerfettoConversionConfig(
                run_directory=self.layout.hybrid,
                output_directory=self.layout.request_perfetto,
                trace_processor_path=self.config.trace_processor_path,
                include_native_details=include_details,
                request_focused=True,
            )
        )
        _plain_json(
            self.layout.publication / "request_focused_perfetto_result.json",
            request_conversion,
        )
        overview = generate_overview(
            OverviewGenerationConfig(
                run_directory=self.layout.hybrid,
                perfetto_directory=self.layout.perfetto,
                output_directory=self.layout.overview,
                trace_processor_path=self.config.trace_processor_path,
            )
        )
        _plain_json(self.layout.publication / "overview_result.json", overview)
        with tempfile.TemporaryDirectory(prefix="runner-determinism-") as directory:
            temporary = Path(directory)
            second_perfetto = temporary / "perfetto"
            convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=self.layout.hybrid,
                    output_directory=second_perfetto,
                    trace_processor_path=self.config.trace_processor_path,
                    include_native_details=include_details,
                    request_focused=False,
                )
            )
            first_traces = {
                path.name: _sha256(path)
                for path in sorted(self.layout.perfetto.glob("*.pftrace"))
            }
            second_traces = {
                path.name: _sha256(path)
                for path in sorted(second_perfetto.glob("*.pftrace"))
            }
            if first_traces != second_traces:
                raise HybridRunnerError(
                    "repeated Perfetto conversion was not byte-for-byte deterministic"
                )
            second_request_perfetto = temporary / "perfetto-request-focused"
            convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=self.layout.hybrid,
                    output_directory=second_request_perfetto,
                    trace_processor_path=self.config.trace_processor_path,
                    include_native_details=include_details,
                    request_focused=True,
                )
            )
            first_request_traces = {
                path.name: _sha256(path)
                for path in sorted(self.layout.request_perfetto.glob("*.pftrace"))
            }
            second_request_traces = {
                path.name: _sha256(path)
                for path in sorted(second_request_perfetto.glob("*.pftrace"))
            }
            if first_request_traces != second_request_traces:
                raise HybridRunnerError(
                    "repeated request-focused Perfetto conversion was not "
                    "byte-for-byte deterministic"
                )
            second_overview = temporary / "overview"
            generate_overview(
                OverviewGenerationConfig(
                    run_directory=self.layout.hybrid,
                    perfetto_directory=self.layout.perfetto,
                    output_directory=second_overview,
                    trace_processor_path=self.config.trace_processor_path,
                )
            )
            overview_hashes = {
                name: _sha256(self.layout.overview / name)
                for name in ("overview.json", "overview.html")
            }
            repeated_overview_hashes = {
                name: _sha256(second_overview / name)
                for name in ("overview.json", "overview.html")
            }
            if overview_hashes != repeated_overview_hashes:
                raise HybridRunnerError(
                    "repeated Overview generation was not byte-for-byte deterministic"
                )
            _plain_json(
                self.layout.publication / "determinism.json",
                {
                    "perfetto_byte_identical": True,
                    "perfetto_sha256": first_traces,
                    "request_focused_perfetto_byte_identical": True,
                    "request_focused_perfetto_sha256": first_request_traces,
                    "overview_byte_identical": True,
                    "overview_sha256": overview_hashes,
                    "temporary_repeat_preserved": False,
                },
            )

    def _create_closeout(self) -> None:
        create_detached_recovery(
            self.layout.recovery,
            {
                "coordinator": self.layout.coordinator,
                "gpu": self.layout.gpu,
                "hybrid": self.layout.hybrid,
                "npu": self.layout.npu,
            },
            {
                "schema_version": "1.0.0",
                "record_type": "closeout_recovery_result",
                "source_run_id": self.layout.run_id,
                "success": True,
                "hardware_rerun": False,
                "postprocess_only": True,
            },
            required_artifacts=(
                ("coordinator", "result.json"),
                ("gpu", "manifest.json"),
                ("hybrid", "artifacts/artifacts.jsonl"),
                ("hybrid", "clocks/clock_domains.jsonl"),
                ("hybrid", "clocks/transforms.jsonl"),
                ("hybrid", "events/events.jsonl"),
                ("hybrid", "manifest.json"),
                ("hybrid", "metrics/metrics.jsonl"),
                ("npu", "manifest.json"),
            ),
        )
