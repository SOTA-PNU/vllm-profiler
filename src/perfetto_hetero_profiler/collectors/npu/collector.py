"""NPU-only run orchestration for low-overhead monitor collection."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable

from ...runtime_metadata import topology_metadata
from ...schema import (
    DeviceDescriptor,
    DeviceType,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
)
from ...support.files import sha256_file
from ..monitor import (
    HOST_CLOCK_DOMAIN,
    MonitorRunCollector,
    MonitorRunResult,
    monitor_run_plan,
)
from ..process import CommandResult
from .config import NpuDeviceInfo, NpuRunConfig
from .profiling import build_rbln_profile_plan
from .rbln_smi import RblnSmiClient, RblnSmiCommandError
from .telemetry import NpuTelemetryCollector

_PACKAGE_NAMES = ("rebel-compiler", "optimum-rbln", "vllm-rbln")
RAW_RBLN_SMI_RELATIVE_PATH = "raw/npu/rbln-smi-last.json"


@dataclass(frozen=True)
class NpuRunResult(MonitorRunResult):
    """Outcome of one NPU-only monitor run."""


def build_npu_run_plan(config: NpuRunConfig) -> dict[str, object]:
    plan = monitor_run_plan(config, mode=RunMode.NPU_ONLY)
    plan["device_ids"] = list(config.device_ids)
    plan["rbln_profiler_enabled"] = False
    plan["outputs"]["rbln_smi"] = str(
        config.paths.root / RAW_RBLN_SMI_RELATIVE_PATH
    )
    if config.profile_mode is ProfileMode.DETAILED_PROFILE:
        plan["rbln_profiler_enabled"] = True
        plan["detailed_profile"] = asdict(build_rbln_profile_plan())
    return plan


class NpuRunCollector(MonitorRunCollector[NpuRunResult]):
    artifact_producer = "npu-monitor"
    emits_collector_error_events = True
    raw_device_relative_path = RAW_RBLN_SMI_RELATIVE_PATH
    raw_device_artifact_id = "rbln-smi-last"
    result_type = NpuRunResult

    def __init__(
        self,
        config: NpuRunConfig,
        *,
        npu_client: RblnSmiClient | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_time_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            config,
            monotonic_ns=monotonic_ns,
            unix_time_ns=unix_time_ns,
            sleep=sleep,
        )
        self.npu_client = npu_client or RblnSmiClient(device_ids=config.device_ids)
        self._rbln_smi_version: str | None = None
        self._kmd_version: str | None = None

    @property
    def host_id(self) -> str:
        return self.config.host_id

    def _reject_unsupported_profile_mode(self) -> None:
        if (
            self.config.profile_mode is ProfileMode.DETAILED_PROFILE
            and not self.config.allow_detailed_execution
        ):
            raise NotImplementedError(
                "RBLN detailed-profile execution requires the runtime collector; use --dry-run"
            )

    def _child_end_attributes(self, command: CommandResult) -> dict[str, object]:
        return {
            **super()._child_end_attributes(command),
            "vendor.terminated": command.terminated,
            "vendor.killed": command.killed,
        }

    def _artifact_sha256(self, actual_path: Path) -> str | None:
        return sha256_file(actual_path)

    def _read_rbln_smi_version(self) -> str | None:
        try:
            self._rbln_smi_version = self.npu_client.version()
        except RblnSmiCommandError as error:
            return f"rbln-smi version: {error}"
        return None

    def _discover_devices(self) -> tuple[tuple[NpuDeviceInfo, ...], tuple[str, ...]]:
        version_error = self._read_rbln_smi_version()
        devices, discovery_error = self._query_devices()
        return devices, tuple(
            error for error in (version_error, discovery_error) if error is not None
        )

    def _query_devices(self) -> tuple[tuple[NpuDeviceInfo, ...], str | None]:
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

    def _device_telemetry(
        self, devices: tuple[NpuDeviceInfo, ...]
    ) -> NpuTelemetryCollector:
        return NpuTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            sample_interval_ms=self.config.sample_interval_ms,
            client=self.npu_client,
            known_npu_indices=tuple(device.index for device in devices),
            monotonic_ns=self.monotonic_ns,
        )

    def _raw_device_snapshot(
        self, device_telemetry: NpuTelemetryCollector
    ) -> str | None:
        return device_telemetry.last_raw_output

    def _manifest(
        self,
        devices: tuple[NpuDeviceInfo, ...],
        status: RunStatus,
        errors: tuple[str, ...],
    ) -> RunManifest:
        return self._manifest_record(
            mode=RunMode.NPU_ONLY,
            host_role="npu",
            status=status,
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
                **self._base_configuration(),
                "device_ids": list(self.config.device_ids),
                "device_selection_scope": "rbln-smi telemetry only",
                "rbln_profiler_enabled": (
                    self.config.profile_mode is ProfileMode.DETAILED_PROFILE
                ),
                "runtime_metadata": {
                    "topology": topology_metadata(RunMode.NPU_ONLY),
                },
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
            self._python_runtime_descriptor(),
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
                    role="collector-environment",
                    path=None,
                )
            )
        return descriptors


def format_plan_json(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
