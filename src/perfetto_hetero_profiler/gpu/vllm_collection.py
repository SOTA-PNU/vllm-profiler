"""End-to-end local vLLM profiling collection for monitor, torch, and nsys."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time
from typing import Callable, Literal
import uuid

from ..artifact_compatibility import (
    LEGACY_GPU_COLLECTION_PRODUCER,
    LEGACY_GPU_COLLECTION_SUMMARY,
    LEGACY_GPU_NSYS_OUTPUT,
)
from ..collectors.gpu import (
    NVML_DISTRIBUTION,
    NVML_DISTRIBUTION_VERSION,
    GpuTelemetryCollector,
    NvmlClient,
)
from ..collectors.system import ProcTelemetryCollector
from ..schema import (
    Availability,
    ArtifactKind,
    ArtifactReference,
    ClockDomain,
    ClockType,
    DeviceDescriptor,
    DeviceType,
    HostDescriptor,
    MetricSample,
    ModelDescriptor,
    ProfileMode,
    RunManifest,
    RunMode,
    RunPaths,
    RunStatus,
    SoftwareDescriptor,
    WorkloadDescriptor,
    write_json,
    write_jsonl,
)
from .openai_client import CompletionObservation, OpenAICompletionClient
from .vllm_server import (
    ManagedVllmServer,
    VllmServerConfig,
    build_server_argv,
    post_empty,
)
from .workload import (
    CLOCK_DOMAIN_ID,
    HOST_ID,
    measured_window_metrics,
    observation_events,
    observation_metrics,
)


ProfileKind = Literal["monitor", "torch", "nsys"]
PROMPT = "Explain a computer cache in one short sentence."


@dataclass(frozen=True)
class GpuVllmCollectionConfig:
    run_root: Path
    run_id: str
    model: Path
    profile_mode: ProfileKind
    host: str
    port: int
    startup_timeout_sec: float
    request_timeout_sec: float
    shutdown_timeout_sec: float
    sample_interval_ms: int
    gpu_memory_utilization: float
    max_model_len: int
    warmup_requests: int
    measured_requests: int
    max_output_tokens: int
    server_python: Path | None = None
    vllm_bin: Path | None = None
    offline: bool = True

    def __post_init__(self) -> None:
        if self.profile_mode not in {"monitor", "torch", "nsys"}:
            raise ValueError("profile_mode must be monitor, torch, or nsys")
        if not self.run_root.is_absolute():
            raise ValueError("run_root must be absolute")
        if not self.model.is_absolute():
            raise ValueError("model must be absolute")
        if self.sample_interval_ms not in {500, 1000}:
            raise ValueError("sample_interval_ms must be 500 or 1000")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be >= 0")
        if not 1 <= self.measured_requests <= 2:
            raise ValueError("measured_requests must be in [1, 2]")
        if not 1 <= self.max_output_tokens <= 16:
            raise ValueError("max_output_tokens must be in [1, 16]")
        for name in (
            "startup_timeout_sec",
            "request_timeout_sec",
            "shutdown_timeout_sec",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        # Reuse server-level validation.
        self.server_config

    @property
    def paths(self) -> RunPaths:
        return RunPaths(self.run_root, self.run_id)

    @property
    def server_config(self) -> VllmServerConfig:
        root = self.paths.root
        torch_dir = (
            root / "raw/gpu/torch" if self.profile_mode == "torch" else None
        )
        nsys_output = (
            root / LEGACY_GPU_NSYS_OUTPUT
            if self.profile_mode == "nsys"
            else None
        )
        return VllmServerConfig(
            model=self.model,
            host=self.host,
            port=self.port,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            server_python=self.server_python,
            vllm_bin=self.vllm_bin,
            torch_profiler_dir=torch_dir,
            nsys_output=nsys_output,
            offline=self.offline,
        )


@dataclass(frozen=True)
class GpuVllmCollectionResult:
    run_directory: Path
    status: RunStatus
    startup_ns: int | None
    warmup_count: int
    measured_count: int
    event_count: int
    metric_count: int
    artifact_count: int
    errors: tuple[str, ...]


def build_vllm_collection_plan(
    config: GpuVllmCollectionConfig,
) -> dict[str, object]:
    """Return a side-effect-free, secret-free execution plan."""
    paths = config.paths
    return {
        "executes": False,
        "run_id": config.run_id,
        "run_directory": str(paths.root),
        "profile_mode": config.profile_mode,
        "offline": config.offline,
        "server_argv": list(build_server_argv(config.server_config)),
        "workload": {
            "warmup_requests": config.warmup_requests,
            "measured_requests": config.measured_requests,
            "max_output_tokens": config.max_output_tokens,
            "stream": True,
            "stores_prompt_or_generated_text": False,
        },
        "outputs": {
            "manifest": str(paths.manifest),
            "events": str(paths.events),
            "metrics": str(paths.metrics),
            "artifacts": str(paths.artifacts),
            "summary": str(paths.root / LEGACY_GPU_COLLECTION_SUMMARY),
        },
    }


class _TelemetryThread:
    def __init__(
        self,
        config: GpuVllmCollectionConfig,
        pid_provider: Callable[[], int | None],
        gpu_client: NvmlClient,
    ) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.metrics: list[MetricSample] = []
        self.errors: list[str] = []
        self.gpu = GpuTelemetryCollector(
            run_id=config.run_id,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            sample_interval_ms=config.sample_interval_ms,
            client=gpu_client,
            known_gpu_indices=(0,),
        )
        self.system = ProcTelemetryCollector(
            run_id=config.run_id,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            pid_provider=pid_provider,
        )
        self.thread = threading.Thread(
            target=self._run,
            name=f"telemetry-{config.run_id}",
            daemon=True,
        )

    def start(self) -> None:
        for collector in (self.gpu, self.system):
            collector.prepare()
            collector.start()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(2.0, self.config.sample_interval_ms / 500))
        for collector in (self.system, self.gpu):
            try:
                collector.stop()
            except Exception as error:  # pragma: no cover - defensive cleanup
                self.errors.append(f"{type(collector).__name__} stop: {error}")

    def _run(self) -> None:
        interval_sec = self.config.sample_interval_ms / 1000
        while not self.stop_event.is_set():
            for collector in (self.gpu, self.system):
                try:
                    self.metrics.extend(collector.sample())
                except Exception as error:
                    self.errors.append(f"{type(collector).__name__}: {error}")
            self.stop_event.wait(interval_sec)


class GpuVllmCollectionRunner:
    def __init__(
        self,
        config: GpuVllmCollectionConfig,
        *,
        gpu_client: NvmlClient | None = None,
        server_factory: Callable[..., ManagedVllmServer] = ManagedVllmServer,
        client_factory: Callable[..., OpenAICompletionClient] = OpenAICompletionClient,
        unix_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config
        self.gpu_client = gpu_client or NvmlClient()
        self.server_factory = server_factory
        self.client_factory = client_factory
        self.unix_time_ns = unix_time_ns

    def run(self) -> GpuVllmCollectionResult:
        config = self.config
        paths = config.paths
        paths.create()
        (paths.root / "raw/gpu/torch").mkdir(parents=True, exist_ok=True)
        (paths.root / "raw/gpu/nsys").mkdir(parents=True, exist_ok=True)
        server_stdout = paths.root / "raw/gpu/vllm-server.stdout.log"
        server_stderr = paths.root / "raw/gpu/vllm-server.stderr.log"
        request_summary_path = paths.root / "raw/client/requests.json"
        summary_path = paths.root / LEGACY_GPU_COLLECTION_SUMMARY

        try:
            devices = self.gpu_client.query()
            write_json(
                paths.manifest, self._manifest(devices, RunStatus.RUNNING, ())
            )
            write_jsonl(
                paths.clock_domains,
                [
                    ClockDomain(
                        run_id=config.run_id,
                        clock_domain_id=CLOCK_DOMAIN_ID,
                        host_id=HOST_ID,
                        clock_type=ClockType.MONOTONIC,
                        unit="ns",
                        monotonic=True,
                        adjustable=False,
                        attributes={"vendor.clock_source": "time.monotonic_ns"},
                    )
                ],
            )
        except BaseException:
            self.gpu_client.shutdown()
            raise

        server = self.server_factory(
            config.server_config, server_stdout, server_stderr
        )
        telemetry = _TelemetryThread(
            config,
            lambda: server.process.pid if server.process is not None else None,
            self.gpu_client,
        )
        warmups: list[CompletionObservation] = []
        measured: list[CompletionObservation] = []
        errors: list[str] = []
        startup_ns: int | None = None
        server_return_code: int | None = None
        profiler_started = False
        try:
            server.start()
            telemetry.start()
            ready_ns = server.wait_ready(config.startup_timeout_sec)
            assert server.started_monotonic_ns is not None
            startup_ns = ready_ns - server.started_monotonic_ns
            client = self.client_factory(
                server.base_url, timeout_sec=config.request_timeout_sec
            )
            model_name = str(config.model)
            for index in range(config.warmup_requests):
                warmups.append(
                    client.complete(
                        model=model_name,
                        request_id=f"{config.run_id}-warmup-{index}",
                        prompt=PROMPT,
                        max_output_tokens=config.max_output_tokens,
                    )
                )
            if config.profile_mode == "torch":
                post_empty(
                    server.base_url, "/start_profile", config.request_timeout_sec
                )
                profiler_started = True
            for index in range(config.measured_requests):
                measured.append(
                    client.complete(
                        model=model_name,
                        request_id=f"{config.run_id}-measured-{index}",
                        prompt=PROMPT,
                        max_output_tokens=config.max_output_tokens,
                    )
                )
            if config.profile_mode == "torch":
                post_empty(
                    server.base_url, "/stop_profile", config.request_timeout_sec
                )
                profiler_started = False
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            if profiler_started:
                try:
                    post_empty(
                        server.base_url, "/stop_profile", config.request_timeout_sec
                    )
                except Exception as error:
                    errors.append(f"stop_profile cleanup: {error}")
            telemetry.stop()
            try:
                self.gpu_client.shutdown()
            except Exception as error:
                errors.append(f"NVML cleanup: {error}")
            try:
                server_return_code = server.stop(config.shutdown_timeout_sec)
            except Exception as error:
                errors.append(f"server cleanup: {error}")

        errors.extend(telemetry.errors)
        errors.extend(
            f"{metric.metric_name}: {metric.reason}"
            for metric in telemetry.metrics
            if metric.availability is Availability.ERROR
        )
        events = [
            event for observation in measured for event in observation_events(config.run_id, observation)
        ]
        metrics = list(telemetry.metrics)
        for observation in measured:
            metrics.extend(observation_metrics(config.run_id, observation))
        metrics.extend(measured_window_metrics(config.run_id, measured))
        write_jsonl(paths.events, events)
        write_jsonl(paths.metrics, metrics)
        request_summary = {
            "run_id": config.run_id,
            "warmup_request_ids": [item.request_id for item in warmups],
            "measured": [
                {
                    "request_id": item.request_id,
                    "http_status": item.http_status,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "total_tokens": item.total_tokens,
                    "token_timestamp_count": len(item.token_timestamps_ns),
                }
                for item in measured
            ],
            "stores_prompt_or_generated_text": False,
        }
        _write_plain_json(request_summary_path, request_summary)
        nvml_raw_path = paths.root / "raw/gpu/nvml-last.json"
        if telemetry.gpu.last_raw_snapshot is not None:
            nvml_raw_path.write_text(
                telemetry.gpu.last_raw_snapshot, encoding="utf-8"
            )

        profile_files = self._profile_files(paths.root)
        profile_validation = self._validate_profile_files(profile_files, errors)
        artifacts = self._artifacts(
            paths.root,
            server_stdout,
            server_stderr,
            request_summary_path,
            profile_files,
            nvml_raw_path if nvml_raw_path.exists() else None,
        )
        write_jsonl(paths.artifacts, artifacts)
        succeeded = (
            not errors
            and len(warmups) == config.warmup_requests
            and len(measured) == config.measured_requests
            and (config.profile_mode == "monitor" or bool(profile_files))
        )
        status = RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED
        final_manifest = self._manifest(devices, status, tuple(errors))
        temporary_manifest = paths.manifest.with_name(
            f".{paths.manifest.name}.{uuid.uuid4().hex}.tmp"
        )
        write_json(temporary_manifest, final_manifest)
        os.replace(temporary_manifest, paths.manifest)
        summary = self._summary(
            status=status,
            startup_ns=startup_ns,
            warmups=warmups,
            measured=measured,
            telemetry=telemetry.metrics,
            profile_files=profile_files,
            profile_validation=profile_validation,
            server_return_code=server_return_code,
            errors=errors,
        )
        _write_plain_json(summary_path, summary)
        return GpuVllmCollectionResult(
            run_directory=paths.root,
            status=status,
            startup_ns=startup_ns,
            warmup_count=len(warmups),
            measured_count=len(measured),
            event_count=len(events),
            metric_count=len(metrics),
            artifact_count=len(artifacts),
            errors=tuple(errors),
        )

    def _manifest(
        self, discovery: object, status: RunStatus, errors: tuple[str, ...]
    ) -> RunManifest:
        config = self.config
        rows = discovery.rows  # type: ignore[attr-defined]
        devices = [
            DeviceDescriptor(
                host_id=HOST_ID,
                device_type=DeviceType.GPU,
                device_id=row.device_id,
                vendor="NVIDIA",
                model=row.name,
                status="available",
                memory_total_bytes=(
                    int(row.memory_total_bytes.value)
                    if row.memory_total_bytes.value is not None
                    else None
                ),
                attributes={"nvml.gpu_index": row.index},
            )
            for row in rows
        ]
        return RunManifest(
            run_id=config.run_id,
            mode=RunMode.GPU_ONLY,
            profile_mode=(
                ProfileMode.MONITOR
                if config.profile_mode == "monitor"
                else ProfileMode.DETAILED_PROFILE
            ),
            status=status,
            created_at_unix_ns=self.unix_time_ns(),
            models=[
                ModelDescriptor(
                    role="served",
                    model_id=str(config.model),
                    revision=None,
                    tokenizer_id=None,
                    dtype=None,
                )
            ],
            workload=WorkloadDescriptor(
                request_count=config.measured_requests,
                concurrency=1,
                request_rate_per_s=None,
                input_tokens=None,
                output_tokens=config.max_output_tokens,
                max_model_len=config.max_model_len,
                warmup_requests=config.warmup_requests,
            ),
            hosts=[
                HostDescriptor(
                    host_id=HOST_ID,
                    role="gpu",
                    hostname=platform.node() or HOST_ID,
                    operating_system=platform.system() or "unknown",
                    architecture=platform.machine() or "unknown",
                )
            ],
            software=[
                SoftwareDescriptor(
                    name="vllm",
                    version="0.18.0",
                    role="inference-server",
                    path=str(config.vllm_bin or config.server_python),
                ),
                SoftwareDescriptor(
                    name=NVML_DISTRIBUTION,
                    version=NVML_DISTRIBUTION_VERSION,
                    role="gpu-telemetry",
                    path=None,
                ),
            ],
            devices=devices,
            configuration={
                "profile_mode": config.profile_mode,
                "host": config.host,
                "port": config.port,
                "sample_interval_ms": config.sample_interval_ms,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_model_len": config.max_model_len,
                "warmup_requests": config.warmup_requests,
                "measured_requests": config.measured_requests,
                "max_output_tokens": config.max_output_tokens,
                "offline": config.offline,
                "server_argv": list(build_server_argv(config.server_config)),
            },
            attributes={
                "vendor.collector": LEGACY_GPU_COLLECTION_PRODUCER,
                "vendor.collector_errors": list(errors),
            },
        )

    def _profile_files(self, run_root: Path) -> list[Path]:
        if self.config.profile_mode == "torch":
            root = run_root / "raw/gpu/torch"
            return sorted(
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path.name.endswith(".pt.trace.json")
                    or path.name.endswith(".pt.trace.json.gz")
                )
            )
        if self.config.profile_mode == "nsys":
            root = run_root / "raw/gpu/nsys"
            return sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".nsys-rep", ".qdrep"}
            )
        return []

    def _validate_profile_files(
        self, files: list[Path], errors: list[str]
    ) -> dict[str, object]:
        if self.config.profile_mode == "monitor":
            return {"required": False, "valid_files": 0}
        if not files:
            errors.append(f"{self.config.profile_mode} profiler produced no report")
            return {"required": True, "valid_files": 0}
        valid = 0
        for path in files:
            if path.stat().st_size <= 0:
                errors.append(f"empty profiler artifact: {path.name}")
                continue
            if self.config.profile_mode == "torch":
                try:
                    opener = gzip.open if path.suffix == ".gz" else open
                    with opener(path, "rt", encoding="utf-8") as stream:
                        document = json.load(stream)
                    if not isinstance(document, dict):
                        raise ValueError("trace root must be an object")
                except (OSError, EOFError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"invalid torch trace {path.name}: {error}")
                    continue
            valid += 1
        return {"required": True, "valid_files": valid}

    def _artifacts(
        self,
        root: Path,
        stdout: Path,
        stderr: Path,
        requests: Path,
        profile_files: list[Path],
        nvml_raw: Path | None,
    ) -> list[ArtifactReference]:
        items = [
            self._artifact(root, stdout, "vllm-stdout", ArtifactKind.RAW_LOG, "text"),
            self._artifact(root, stderr, "vllm-stderr", ArtifactKind.RAW_LOG, "text"),
            self._artifact(
                root, requests, "client-requests", ArtifactKind.RAW_LOG, "json"
            ),
        ]
        if nvml_raw is not None:
            items.append(
                self._artifact(
                    root,
                    nvml_raw,
                    "nvml-last",
                    ArtifactKind.TELEMETRY,
                    "json",
                    producer="nvml",
                )
            )
        for index, path in enumerate(profile_files):
            kind = (
                ArtifactKind.TORCH_TRACE
                if self.config.profile_mode == "torch"
                else ArtifactKind.NSYS_REPORT
            )
            items.append(
                self._artifact(
                    root,
                    path,
                    f"{self.config.profile_mode}-profile-{index}",
                    kind,
                    "json.gz" if path.suffix == ".gz" else path.suffix.lstrip("."),
                )
            )
        if self.config.profile_mode == "torch":
            trace_set = set(profile_files)
            auxiliary_files = sorted(
                path
                for path in (root / "raw/gpu/torch").rglob("*")
                if path.is_file() and path not in trace_set
            )
            for index, path in enumerate(auxiliary_files):
                items.append(
                    self._artifact(
                        root,
                        path,
                        f"torch-auxiliary-{index}",
                        ArtifactKind.RAW_LOG,
                        path.suffix.lstrip(".") or "binary",
                    )
                )
        return items

    def _artifact(
        self,
        root: Path,
        path: Path,
        artifact_id: str,
        kind: ArtifactKind,
        format_name: str,
        *,
        producer: str = LEGACY_GPU_COLLECTION_PRODUCER,
    ) -> ArtifactReference:
        return ArtifactReference(
            run_id=self.config.run_id,
            artifact_id=artifact_id,
            artifact_kind=kind,
            relative_path=path.relative_to(root).as_posix(),
            format=format_name or "binary",
            producer=producer,
            created_at_unix_ns=self.unix_time_ns(),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            host_id=HOST_ID,
            attributes={"vendor.profile_mode": self.config.profile_mode},
        )

    def _summary(
        self,
        *,
        status: RunStatus,
        startup_ns: int | None,
        warmups: list[CompletionObservation],
        measured: list[CompletionObservation],
        telemetry: list[MetricSample],
        profile_files: list[Path],
        profile_validation: dict[str, object],
        server_return_code: int | None,
        errors: list[str],
    ) -> dict[str, object]:
        def values(name: str) -> list[float]:
            return [
                float(item.value)
                for item in telemetry
                if item.metric_name == name and item.value is not None
            ]

        latencies = [item.e2e_ns for item in measured]
        ttfts = [item.ttft_ns for item in measured if item.ttft_ns is not None]
        tpots = [item.tpot_ns for item in measured if item.tpot_ns is not None]
        gpu_util = values("resource.gpu.utilization")
        gpu_memory = values("resource.gpu.memory_used")
        gpu_power = values("resource.gpu.power")
        return {
            "run_id": self.config.run_id,
            "profile_mode": self.config.profile_mode,
            "startup": {
                "ready": startup_ns is not None,
                "duration_ns": startup_ns,
            },
            "requests": {
                "warmup_expected": self.config.warmup_requests,
                "warmup_completed": len(warmups),
                "measured_expected": self.config.measured_requests,
                "measured_completed": len(measured),
                "http_statuses": [item.http_status for item in measured],
            },
            "latency": {
                "e2e_ns": latencies,
                "ttft_ns": ttfts,
                "tpot_ns": tpots,
            },
            "tokens": {
                "input": sum(item.input_tokens for item in measured),
                "output": sum(item.output_tokens for item in measured),
                "total": sum(item.total_tokens for item in measured),
            },
            "telemetry": {
                "sample_record_count": len(telemetry),
                "gpu_utilization_percent_avg": _average(gpu_util),
                "gpu_memory_used_bytes_max": max(gpu_memory, default=None),
                "gpu_power_w_avg": _average(gpu_power),
            },
            "artifact": {
                "profile_files": [
                    {
                        "relative_path": path.relative_to(
                            self.config.paths.root
                        ).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in profile_files
                ],
                "validation": profile_validation,
            },
            "final": {
                "status": status.value,
                "server_return_code": server_return_code,
                "errors": errors,
            },
        }


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_plain_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
