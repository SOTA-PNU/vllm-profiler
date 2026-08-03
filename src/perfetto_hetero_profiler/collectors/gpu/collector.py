"""GPU-only run orchestration for low-overhead monitor collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Callable

from ...schema import (
    ArtifactKind,
    ArtifactReference,
    ClockDomain,
    ClockType,
    DeviceDescriptor,
    DeviceType,
    EventRecord,
    EventType,
    HostDescriptor,
    ModelDescriptor,
    Phase,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
    WorkloadDescriptor,
    write_json,
    write_jsonl,
)
from ..command import mask_command
from ..process import ManagedProcess
from ..run import run_monitored_process
from ..system import ProcTelemetryCollector
from .config import GpuDeviceInfo, GpuRunConfig
from .nvidia_smi import NvidiaSmiClient, NvidiaSmiCommandError
from .profiling import build_detailed_profile_plan
from .telemetry import GpuTelemetryCollector


HOST_CLOCK_DOMAIN = "host-monotonic"


@dataclass(frozen=True)
class GpuRunResult:
    status: RunStatus
    return_code: int
    event_count: int
    metric_count: int
    artifact_count: int
    run_directory: Path


def build_gpu_run_plan(config: GpuRunConfig) -> dict[str, object]:
    paths = config.paths
    plan: dict[str, object] = {
        "mode": RunMode.GPU_ONLY.value,
        "profile_mode": config.profile_mode.value,
        "run_id": config.run_id,
        "run_directory": str(paths.root),
        "sample_interval_ms": config.sample_interval_ms,
        "command": config.command_spec.safe_plan(),
        "outputs": {
            "manifest": str(paths.manifest),
            "clock_domains": str(paths.clock_domains),
            "events": str(paths.events),
            "metrics": str(paths.metrics),
            "artifacts": str(paths.artifacts),
            "stdout": str(paths.root / "raw/client/stdout.log"),
            "stderr": str(paths.root / "raw/client/stderr.log"),
        },
        "executes": False,
    }
    if config.profile_mode is ProfileMode.DETAILED_PROFILE:
        detailed = build_detailed_profile_plan(config.command)
        plan["detailed_profile"] = {
            "nsys_argv": list(detailed.nsys_argv),
            "torch": asdict(detailed.torch),
            "warning": detailed.simultaneous_warning,
        }
    return plan


class GpuRunCollector:
    def __init__(
        self,
        config: GpuRunConfig,
        *,
        gpu_client: NvidiaSmiClient | None = None,
        proc_root: Path = Path("/proc"),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_time_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.gpu_client = gpu_client or NvidiaSmiClient()
        self.proc_root = Path(proc_root)
        self.monotonic_ns = monotonic_ns
        self.unix_time_ns = unix_time_ns
        self.sleep = sleep

    def run(self) -> GpuRunResult:
        if self.config.profile_mode is ProfileMode.DETAILED_PROFILE:
            raise NotImplementedError(
                "detailed-profile execution is deferred to Phase 2B; use --dry-run"
            )
        paths = self.config.paths
        paths.create()
        devices, discovery_error = self._discover_devices()
        manifest = self._manifest(devices, RunStatus.RUNNING, ())
        self._replace_manifest(paths.manifest, manifest, initial=True)
        clock = ClockDomain(
            run_id=self.config.run_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            host_id=self.config.host_alias,
            clock_type=ClockType.MONOTONIC,
            unit="ns",
            monotonic=True,
            adjustable=False,
            attributes={"vendor.clock_source": "time.monotonic_ns"},
        )
        write_jsonl(paths.clock_domains, [clock])

        stdout_path = paths.root / "raw/client/stdout.log"
        stderr_path = paths.root / "raw/client/stderr.log"
        process = ManagedProcess(
            self.config.command_spec,
            stdout_path,
            stderr_path,
            monotonic_ns=self.monotonic_ns,
        )
        gpu = GpuTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.config.host_alias,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            sample_interval_ms=self.config.sample_interval_ms,
            client=self.gpu_client,
            known_gpu_indices=tuple(device.index for device in devices),
            monotonic_ns=self.monotonic_ns,
        )
        system = ProcTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.config.host_alias,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            pid_provider=lambda: (
                process.process.pid if process.process is not None else None
            ),
            proc_root=self.proc_root,
            monotonic_ns=self.monotonic_ns,
        )

        run_start_ns = self.monotonic_ns()
        errors = [discovery_error] if discovery_error else []
        monitored = run_monitored_process(
            process,
            (gpu, system),
            sample_interval_ms=self.config.sample_interval_ms,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            monotonic_ns=self.monotonic_ns,
            sleep=self.sleep,
        )
        errors.extend(monitored.errors)
        events: list[EventRecord] = [
            self._event("run-start", "collector.run_start", run_start_ns),
            self._event(
                "child-start",
                "collector.child_process_start",
                monitored.command.started_monotonic_ns,
                process_id=monitored.process_id,
            ),
            self._event(
                "child-end",
                "collector.child_process_end",
                monitored.command.ended_monotonic_ns,
                process_id=monitored.process_id,
                attributes={
                    "vendor.return_code": monitored.command.return_code,
                    "vendor.timed_out": monitored.command.timed_out,
                },
            ),
        ]
        errors.extend(
            f"{metric.metric_name}: {metric.reason}"
            for metric in monitored.metrics
            if metric.availability.value == "error"
        )
        artifacts = self._write_raw_artifacts(
            stdout_path, stderr_path, gpu.last_raw_output, tuple(errors)
        )
        write_jsonl(paths.events, events)
        write_jsonl(paths.metrics, monitored.metrics)
        write_jsonl(paths.artifacts, artifacts)

        if monitored.command.return_code != 0 or monitored.command.timed_out:
            status = RunStatus.FAILED
        elif errors:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.SUCCEEDED
        final_manifest = self._manifest(devices, status, tuple(errors))
        self._replace_manifest(paths.manifest, final_manifest)
        return GpuRunResult(
            status=status,
            return_code=monitored.command.return_code,
            event_count=len(events),
            metric_count=len(monitored.metrics),
            artifact_count=len(artifacts),
            run_directory=paths.root,
        )

    def _discover_devices(self) -> tuple[tuple[GpuDeviceInfo, ...], str | None]:
        if self.config.gpu_devices:
            return self.config.gpu_devices, None
        try:
            result = self.gpu_client.query()
            devices = tuple(
                GpuDeviceInfo(
                    index=row.index,
                    name=row.name,
                    memory_total_bytes=(
                        int(row.memory_total_bytes.value)
                        if row.memory_total_bytes.value is not None
                        else None
                    ),
                )
                for row in result.rows
            )
            return devices, None
        except NvidiaSmiCommandError as error:
            return (GpuDeviceInfo(index=0, name="unknown"),), str(error)

    def _manifest(
        self,
        devices: tuple[GpuDeviceInfo, ...],
        status: RunStatus,
        errors: tuple[str, ...],
    ) -> RunManifest:
        return RunManifest(
            run_id=self.config.run_id,
            mode=RunMode.GPU_ONLY,
            profile_mode=self.config.profile_mode,
            status=status,
            created_at_unix_ns=self.unix_time_ns(),
            models=[
                ModelDescriptor(
                    role="served",
                    model_id=self.config.model_id,
                    revision=None,
                    tokenizer_id=None,
                    dtype=None,
                )
            ],
            workload=WorkloadDescriptor(
                request_count=None,
                concurrency=None,
                request_rate_per_s=None,
                input_tokens=None,
                output_tokens=None,
                max_model_len=None,
                warmup_requests=None,
            ),
            hosts=[
                HostDescriptor(
                    host_id=self.config.host_alias,
                    role="gpu",
                    hostname=self.config.host_alias,
                    operating_system=platform.system() or "unknown",
                    architecture=platform.machine() or "unknown",
                )
            ],
            software=[
                SoftwareDescriptor(
                    name="python",
                    version=platform.python_version(),
                    role="child-runtime",
                    path=sys.executable,
                ),
                SoftwareDescriptor(
                    name="nvidia-smi",
                    version=None,
                    role="gpu-telemetry",
                    path=shutil.which("nvidia-smi"),
                ),
            ],
            devices=[
                DeviceDescriptor(
                    host_id=self.config.host_alias,
                    device_type=DeviceType.GPU,
                    device_id=device.device_id,
                    vendor="NVIDIA",
                    model=device.name,
                    status="available" if device.name != "unknown" else "unknown",
                    memory_total_bytes=device.memory_total_bytes,
                    attributes={"nvidia_smi.gpu_index": device.index},
                )
                for device in devices
            ],
            configuration={
                "sample_interval_ms": self.config.sample_interval_ms,
                "command": mask_command(self.config.command),
                "cwd": str(self.config.cwd) if self.config.cwd else None,
                "timeout_sec": self.config.timeout_sec,
            },
            attributes={
                "vendor.collector": "gpu-monitor",
                "vendor.collector_errors": list(errors),
            },
        )

    def _event(
        self,
        event_id: str,
        event_name: str,
        timestamp_ns: int,
        *,
        process_id: int | None = None,
        attributes: dict[str, object] | None = None,
    ) -> EventRecord:
        return EventRecord(
            run_id=self.config.run_id,
            event_id=event_id,
            event_name=event_name,
            event_type=EventType.INSTANT,
            phase=Phase.SYSTEM,
            host_id=self.config.host_alias,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            timestamp_ns=timestamp_ns,
            process_id=process_id,
            attributes=attributes or {},
        )

    def _write_raw_artifacts(
        self,
        stdout_path: Path,
        stderr_path: Path,
        raw_gpu: str | None,
        errors: tuple[str, ...],
    ) -> list[ArtifactReference]:
        paths = self.config.paths
        artifacts = [
            self._artifact("child-stdout", "raw/client/stdout.log", stdout_path),
            self._artifact("child-stderr", "raw/client/stderr.log", stderr_path),
        ]
        if raw_gpu is not None:
            raw_path = paths.root / "raw/gpu/nvidia-smi-last.csv"
            raw_path.write_text(raw_gpu, encoding="utf-8")
            artifacts.append(
                self._artifact(
                    "nvidia-smi-last", "raw/gpu/nvidia-smi-last.csv", raw_path
                )
            )
        if errors:
            error_path = paths.root / "raw/system/collector-errors.json"
            error_path.write_text(
                json.dumps({"errors": list(errors)}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts.append(
                self._artifact(
                    "collector-errors",
                    "raw/system/collector-errors.json",
                    error_path,
                    format_name="json",
                )
            )
        return artifacts

    def _artifact(
        self,
        artifact_id: str,
        relative_path: str,
        actual_path: Path,
        *,
        format_name: str = "text",
    ) -> ArtifactReference:
        return ArtifactReference(
            run_id=self.config.run_id,
            artifact_id=artifact_id,
            artifact_kind=ArtifactKind.RAW_LOG,
            relative_path=relative_path,
            format=format_name,
            producer="gpu-monitor",
            created_at_unix_ns=self.unix_time_ns(),
            size_bytes=actual_path.stat().st_size,
            attributes={},
        )

    @staticmethod
    def _replace_manifest(
        path: Path, manifest: RunManifest, *, initial: bool = False
    ) -> None:
        if initial:
            write_json(path, manifest)
            return
        temporary = path.with_name(f".{path.name}.tmp")
        write_json(temporary, manifest)
        os.replace(temporary, path)


def format_plan_json(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
