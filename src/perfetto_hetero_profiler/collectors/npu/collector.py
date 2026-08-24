"""NPU-only run orchestration for low-overhead monitor collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
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
from .config import NpuDeviceInfo, NpuRunConfig
from .profiling import build_rbln_profile_plan
from .rbln_smi import RblnSmiClient, RblnSmiCommandError
from .telemetry import NpuTelemetryCollector


HOST_CLOCK_DOMAIN = "host-monotonic"
_PACKAGE_NAMES = ("rebel-compiler", "optimum-rbln", "vllm-rbln")


@dataclass(frozen=True)
class NpuRunResult:
    status: RunStatus
    return_code: int
    event_count: int
    metric_count: int
    artifact_count: int
    run_directory: Path


def build_npu_run_plan(config: NpuRunConfig) -> dict[str, object]:
    paths = config.paths
    plan: dict[str, object] = {
        "mode": RunMode.NPU_ONLY.value,
        "profile_mode": config.profile_mode.value,
        "run_id": config.run_id,
        "run_directory": str(paths.root),
        "sample_interval_ms": config.sample_interval_ms,
        "device_ids": list(config.device_ids),
        "command": config.command_spec.safe_plan(),
        "rbln_profiler_enabled": False,
        "outputs": {
            "manifest": str(paths.manifest),
            "clock_domains": str(paths.clock_domains),
            "events": str(paths.events),
            "metrics": str(paths.metrics),
            "artifacts": str(paths.artifacts),
            "stdout": str(paths.root / "raw/client/stdout.log"),
            "stderr": str(paths.root / "raw/client/stderr.log"),
            "rbln_smi": str(paths.root / "raw/npu/rbln-smi-last.json"),
        },
        "executes": False,
    }
    if config.profile_mode is ProfileMode.DETAILED_PROFILE:
        plan["rbln_profiler_enabled"] = True
        plan["detailed_profile"] = asdict(build_rbln_profile_plan())
    return plan


class NpuRunCollector:
    def __init__(
        self,
        config: NpuRunConfig,
        *,
        npu_client: RblnSmiClient | None = None,
        proc_root: Path = Path("/proc"),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_time_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.npu_client = npu_client or RblnSmiClient(device_ids=config.device_ids)
        self.proc_root = Path(proc_root)
        self.monotonic_ns = monotonic_ns
        self.unix_time_ns = unix_time_ns
        self.sleep = sleep
        self._rbln_smi_version: str | None = None
        self._kmd_version: str | None = None

    def run(self) -> NpuRunResult:
        if (
            self.config.profile_mode is ProfileMode.DETAILED_PROFILE
            and not self.config.allow_detailed_execution
        ):
            raise NotImplementedError(
                "RBLN detailed-profile execution requires the runtime collector; use --dry-run"
            )
        paths = self.config.paths
        paths.create()
        version_error = self._read_rbln_smi_version()
        devices, discovery_error = self._discover_devices()
        initial_errors = tuple(
            error for error in (version_error, discovery_error) if error is not None
        )
        self._replace_manifest(
            paths.manifest,
            self._manifest(devices, RunStatus.RUNNING, initial_errors),
            initial=True,
        )
        write_jsonl(
            paths.clock_domains,
            [
                ClockDomain(
                    run_id=self.config.run_id,
                    clock_domain_id=HOST_CLOCK_DOMAIN,
                    host_id=self.config.host_id,
                    clock_type=ClockType.MONOTONIC,
                    unit="ns",
                    monotonic=True,
                    adjustable=False,
                    attributes={"vendor.clock_source": "time.monotonic_ns"},
                )
            ],
        )

        stdout_path = paths.root / "raw/client/stdout.log"
        stderr_path = paths.root / "raw/client/stderr.log"
        process = ManagedProcess(
            self.config.command_spec,
            stdout_path,
            stderr_path,
            monotonic_ns=self.monotonic_ns,
        )
        npu = NpuTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.config.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            sample_interval_ms=self.config.sample_interval_ms,
            client=self.npu_client,
            known_npu_indices=tuple(device.index for device in devices),
            monotonic_ns=self.monotonic_ns,
        )
        system = ProcTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.config.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            pid_provider=lambda: (
                process.process.pid if process.process is not None else None
            ),
            proc_root=self.proc_root,
            monotonic_ns=self.monotonic_ns,
        )
        run_start_ns = self.monotonic_ns()
        monitored = run_monitored_process(
            process,
            (npu, system),
            sample_interval_ms=self.config.sample_interval_ms,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            monotonic_ns=self.monotonic_ns,
            sleep=self.sleep,
        )
        errors = [*initial_errors, *monitored.errors]
        errors.extend(
            f"{metric.metric_name}: {metric.reason}"
            for metric in monitored.metrics
            if metric.availability.value == "error"
        )
        events = [
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
                    "vendor.terminated": monitored.command.terminated,
                    "vendor.killed": monitored.command.killed,
                },
            ),
        ]
        for index, error in enumerate(errors):
            events.append(
                self._event(
                    f"collector-error-{index}",
                    "collector.error",
                    monitored.command.ended_monotonic_ns,
                    attributes={"vendor.error": error},
                )
            )
        artifacts = self._write_raw_artifacts(
            stdout_path, stderr_path, npu.last_raw_output, tuple(errors)
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
        self._replace_manifest(
            paths.manifest, self._manifest(devices, status, tuple(errors))
        )
        return NpuRunResult(
            status=status,
            return_code=monitored.command.return_code,
            event_count=len(events),
            metric_count=len(monitored.metrics),
            artifact_count=len(artifacts),
            run_directory=paths.root,
        )

    def _read_rbln_smi_version(self) -> str | None:
        try:
            self._rbln_smi_version = self.npu_client.version()
        except RblnSmiCommandError as error:
            return f"rbln-smi version: {error}"
        return None

    def _discover_devices(self) -> tuple[tuple[NpuDeviceInfo, ...], str | None]:
        if self.config.npu_devices:
            return self.config.npu_devices, None
        try:
            result = self.npu_client.query()
            self._kmd_version = result.kmd_version
            rows = result.rows
            if self.config.device_ids:
                selected = set(self.config.device_ids)
                rows = tuple(row for row in rows if row.index in selected)
                found = {row.index for row in rows}
                missing = selected - found
                if missing:
                    return (
                        tuple(self._unknown_device(index) for index in self.config.device_ids),
                        f"rbln-smi did not report requested NPU device {min(missing)}",
                    )
            return (
                tuple(
                    NpuDeviceInfo(
                        index=row.index,
                        name=row.name,
                        status=row.status,
                        memory_total_bytes=(
                            int(row.memory_total_bytes.value)
                            if row.memory_total_bytes.value is not None
                            else None
                        ),
                        firmware_version=row.firmware_version,
                    )
                    for row in rows
                ),
                None,
            )
        except RblnSmiCommandError as error:
            indices = self.config.device_ids or (0,)
            return tuple(self._unknown_device(index) for index in indices), str(error)

    @staticmethod
    def _unknown_device(index: int) -> NpuDeviceInfo:
        return NpuDeviceInfo(index=index, name="unknown", status="unknown")

    def _manifest(
        self,
        devices: tuple[NpuDeviceInfo, ...],
        status: RunStatus,
        errors: tuple[str, ...],
    ) -> RunManifest:
        return RunManifest(
            run_id=self.config.run_id,
            mode=RunMode.NPU_ONLY,
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
                    host_id=self.config.host_id,
                    role="npu",
                    hostname=self.config.host_id,
                    operating_system=platform.system() or "unknown",
                    architecture=platform.machine() or "unknown",
                )
            ],
            software=self._software_descriptors(),
            devices=[
                DeviceDescriptor(
                    host_id=self.config.host_id,
                    device_type=DeviceType.NPU,
                    device_id=device.device_id,
                    vendor="Rebellions",
                    model=device.name,
                    status=device.status,
                    memory_total_bytes=device.memory_total_bytes,
                    attributes={
                        "rbln_smi.npu_index": device.index,
                        "rbln_smi.firmware_version": device.firmware_version,
                    },
                )
                for device in devices
            ],
            configuration={
                "sample_interval_ms": self.config.sample_interval_ms,
                "command": mask_command(self.config.command),
                "cwd": str(self.config.cwd) if self.config.cwd else None,
                "timeout_sec": self.config.timeout_sec,
                "device_ids": list(self.config.device_ids),
                "device_selection_scope": "rbln-smi telemetry only",
                "rbln_profiler_enabled": (
                    self.config.profile_mode is ProfileMode.DETAILED_PROFILE
                ),
            },
            attributes={
                "vendor.collector": "npu-monitor",
                "vendor.collector_errors": list(errors),
                "rbln_smi.kmd_version": self._kmd_version,
                "rbln_smi.unsupported_metric_policy": "first-sample-once-per-device",
            },
        )

    def _software_descriptors(self) -> list[SoftwareDescriptor]:
        descriptors = [
            SoftwareDescriptor(
                name="python",
                version=platform.python_version(),
                role="child-runtime",
                path=sys.executable,
            ),
            SoftwareDescriptor(
                name="rbln-smi",
                version=self._rbln_smi_version,
                role="npu-telemetry",
                path=shutil.which("rbln-smi"),
            ),
        ]
        for package_name in _PACKAGE_NAMES:
            try:
                version = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                version = None
            descriptors.append(
                SoftwareDescriptor(
                    name=package_name,
                    version=version,
                    role="npu-runtime",
                    path=None,
                )
            )
        return descriptors

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
            host_id=self.config.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            timestamp_ns=timestamp_ns,
            process_id=process_id,
            attributes=attributes or {},
        )

    def _write_raw_artifacts(
        self,
        stdout_path: Path,
        stderr_path: Path,
        raw_npu: str | None,
        errors: tuple[str, ...],
    ) -> list[ArtifactReference]:
        artifacts = [
            self._artifact("child-stdout", "raw/client/stdout.log", stdout_path),
            self._artifact("child-stderr", "raw/client/stderr.log", stderr_path),
        ]
        if raw_npu is not None:
            raw_path = self.config.paths.root / "raw/npu/rbln-smi-last.json"
            raw_path.write_text(raw_npu, encoding="utf-8")
            artifacts.append(
                self._artifact(
                    "rbln-smi-last",
                    "raw/npu/rbln-smi-last.json",
                    raw_path,
                    format_name="json",
                )
            )
        if errors:
            error_path = self.config.paths.root / "raw/system/collector-errors.json"
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
            producer="npu-monitor",
            created_at_unix_ns=self.unix_time_ns(),
            size_bytes=actual_path.stat().st_size,
            sha256=self._sha256(actual_path),
            attributes={},
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
