"""Shared monitor run lifecycle for the GPU-only and NPU-only collectors.

Both device collectors publish the same run shape: create the run directory,
discover devices, publish the initial manifest, record the host clock domain,
run the child under device and system telemetry, emit the run/child events,
write the raw artifacts, decide the status and republish the manifest.  Only
the device-specific parts differ, and those stay in the subclass hooks below.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Generic, TypeVar

from ..schema import (
    ArtifactKind,
    ArtifactReference,
    ClockDomain,
    ClockType,
    DeviceDescriptor,
    EventRecord,
    EventType,
    HostDescriptor,
    ModelDescriptor,
    Phase,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
    WorkloadDescriptor,
    write_jsonl,
)
from ..schema.manifests import publish_run_manifest
from .base import BaseCollector
from .command import mask_command
from .process import CommandResult, ManagedProcess
from .run import run_monitored_process
from .system import SystemTelemetryCollector

HOST_CLOCK_DOMAIN = "host-monotonic"
STDOUT_RELATIVE_PATH = "raw/client/stdout.log"
STDERR_RELATIVE_PATH = "raw/client/stderr.log"
COLLECTOR_ERRORS_RELATIVE_PATH = "raw/system/collector-errors.json"


@dataclass(frozen=True)
class MonitorRunResult:
    """Outcome of one monitor run, shared by the GPU and NPU result types."""

    status: RunStatus
    return_code: int
    event_count: int
    metric_count: int
    artifact_count: int
    run_directory: Path


def write_json_document(path: Path, payload: dict[str, object]) -> None:
    """Write one human-readable JSON document, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


ResultT = TypeVar("ResultT", bound=MonitorRunResult)


class MonitorRunCollector(Generic[ResultT]):
    """Template for one device monitor run; subclasses supply device hooks."""

    #: ``True`` when the device collector records collector errors as events.
    emits_collector_error_events: ClassVar[bool] = False
    #: Producer recorded on every artifact this collector publishes.
    artifact_producer: ClassVar[str] = "monitor"
    #: Run-relative path of the last raw device telemetry snapshot.
    raw_device_relative_path: ClassVar[str]
    #: Artifact identity recorded for that snapshot.
    raw_device_artifact_id: ClassVar[str]
    raw_device_producer: ClassVar[str | None] = None
    raw_device_kind: ClassVar[ArtifactKind] = ArtifactKind.RAW_LOG
    #: Device-specific frozen result type returned by :meth:`run`.
    result_type: type[ResultT]

    def __init__(
        self,
        config,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_time_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.monotonic_ns = monotonic_ns
        self.unix_time_ns = unix_time_ns
        self.sleep = sleep

    def run(self) -> ResultT:
        self._reject_unsupported_profile_mode()
        paths = self.config.paths
        paths.create()
        devices, initial_errors = self._discover_devices()
        publish_run_manifest(
            paths.manifest,
            self._manifest(devices, RunStatus.RUNNING, initial_errors),
            initial=True,
        )
        write_jsonl(paths.clock_domains, [self._clock_domain()])

        stdout_path = paths.root / STDOUT_RELATIVE_PATH
        stderr_path = paths.root / STDERR_RELATIVE_PATH
        process = ManagedProcess(
            self.config.command_spec,
            stdout_path,
            stderr_path,
            monotonic_ns=self.monotonic_ns,
        )
        device_telemetry = self._device_telemetry(devices)
        system = SystemTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            pid_provider=lambda: (
                process.process.pid if process.process is not None else None
            ),
            monotonic_ns=self.monotonic_ns,
        )

        run_start_ns = self.monotonic_ns()
        monitored = run_monitored_process(
            process,
            (device_telemetry, system),
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
                attributes=self._child_end_attributes(monitored.command),
            ),
        ]
        if self.emits_collector_error_events:
            events.extend(
                self._event(
                    f"collector-error-{index}",
                    "collector.error",
                    monitored.command.ended_monotonic_ns,
                    attributes={"vendor.error": error},
                )
                for index, error in enumerate(errors)
            )
        artifacts = self._write_raw_artifacts(
            stdout_path,
            stderr_path,
            self._raw_device_snapshot(device_telemetry),
            tuple(errors),
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
        publish_run_manifest(
            paths.manifest, self._manifest(devices, status, tuple(errors))
        )
        return self.result_type(
            status=status,
            return_code=monitored.command.return_code,
            event_count=len(events),
            metric_count=len(monitored.metrics),
            artifact_count=len(artifacts),
            run_directory=paths.root,
        )

    def _clock_domain(self) -> ClockDomain:
        return ClockDomain(
            run_id=self.config.run_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            host_id=self.host_id,
            clock_type=ClockType.MONOTONIC,
            unit="ns",
            monotonic=True,
            adjustable=False,
            attributes={"vendor.clock_source": "time.monotonic_ns"},
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
            host_id=self.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            timestamp_ns=timestamp_ns,
            process_id=process_id,
            attributes=attributes or {},
        )

    def _write_raw_artifacts(
        self,
        stdout_path: Path,
        stderr_path: Path,
        raw_device_snapshot: str | None,
        errors: tuple[str, ...],
    ) -> list[ArtifactReference]:
        artifacts = [
            self._artifact("child-stdout", STDOUT_RELATIVE_PATH, stdout_path),
            self._artifact("child-stderr", STDERR_RELATIVE_PATH, stderr_path),
        ]
        if raw_device_snapshot is not None:
            raw_path = self.config.paths.root / self.raw_device_relative_path
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_device_snapshot, encoding="utf-8")
            artifacts.append(
                self._artifact(
                    self.raw_device_artifact_id,
                    self.raw_device_relative_path,
                    raw_path,
                    format_name="json",
                    producer=self.raw_device_producer,
                    kind=self.raw_device_kind,
                )
            )
        if errors:
            error_path = self.config.paths.root / COLLECTOR_ERRORS_RELATIVE_PATH
            write_json_document(error_path, {"errors": list(errors)})
            artifacts.append(
                self._artifact(
                    "collector-errors",
                    COLLECTOR_ERRORS_RELATIVE_PATH,
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
        producer: str | None = None,
        kind: ArtifactKind = ArtifactKind.RAW_LOG,
    ) -> ArtifactReference:
        return ArtifactReference(
            run_id=self.config.run_id,
            artifact_id=artifact_id,
            artifact_kind=kind,
            relative_path=relative_path,
            format=format_name,
            producer=self.artifact_producer if producer is None else producer,
            created_at_unix_ns=self.unix_time_ns(),
            size_bytes=actual_path.stat().st_size,
            sha256=self._artifact_sha256(actual_path),
            attributes={},
        )

    def _artifact_sha256(self, actual_path: Path) -> str | None:
        """Digest recorded on published artifacts; GPU records none."""

        return None

    def _child_end_attributes(self, command: CommandResult) -> dict[str, object]:
        return {
            "vendor.return_code": command.return_code,
            "vendor.timed_out": command.timed_out,
        }

    def _manifest_record(
        self,
        *,
        mode: RunMode,
        host_role: str,
        status: RunStatus,
        software: list[SoftwareDescriptor],
        devices: list[DeviceDescriptor],
        configuration: dict[str, object],
        attributes: dict[str, object],
    ) -> RunManifest:
        """Assemble one run manifest around the identical device-free fields."""

        return RunManifest(
            run_id=self.config.run_id,
            mode=mode,
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
                    host_id=self.host_id,
                    role=host_role,
                    hostname=self.host_id,
                    operating_system=platform.system() or "unknown",
                    architecture=platform.machine() or "unknown",
                )
            ],
            software=software,
            devices=devices,
            configuration=configuration,
            attributes=attributes,
        )

    def _base_configuration(self) -> dict[str, object]:
        """Configuration keys both device collectors record identically."""

        return {
            "sample_interval_ms": self.config.sample_interval_ms,
            "command": mask_command(self.config.command),
            "cwd": str(self.config.cwd) if self.config.cwd else None,
            "timeout_sec": self.config.timeout_sec,
        }

    def _python_runtime_descriptor(self) -> SoftwareDescriptor:
        return SoftwareDescriptor(
            name="python",
            version=platform.python_version(),
            role="child-runtime",
            path=sys.executable,
        )

    # --- device-specific hooks; every subclass supplies all of these ------

    @property
    def host_id(self) -> str:
        raise NotImplementedError

    def _reject_unsupported_profile_mode(self) -> None:
        raise NotImplementedError

    def _discover_devices(self) -> tuple[tuple[object, ...], tuple[str, ...]]:
        raise NotImplementedError

    def _device_telemetry(self, devices: tuple[object, ...]) -> BaseCollector:
        raise NotImplementedError

    def _raw_device_snapshot(self, device_telemetry: BaseCollector) -> str | None:
        raise NotImplementedError

    def _manifest(
        self,
        devices: tuple[object, ...],
        status: RunStatus,
        errors: tuple[str, ...],
    ) -> RunManifest:
        raise NotImplementedError


def monitor_run_plan(config, *, mode: RunMode) -> dict[str, object]:
    """Dry-run plan fields both device run planners publish identically."""

    paths = config.paths
    return {
        "mode": mode.value,
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
            "stdout": str(paths.root / STDOUT_RELATIVE_PATH),
            "stderr": str(paths.root / STDERR_RELATIVE_PATH),
        },
        "executes": False,
    }
