"""GPU-only run orchestration for low-overhead monitor collection."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

from ...runtime_metadata import topology_metadata
from ...schema import (
    ArtifactKind,
    DeviceDescriptor,
    DeviceType,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
)
from ..monitor import (
    HOST_CLOCK_DOMAIN,
    MonitorRunCollector,
    MonitorRunResult,
    monitor_run_plan,
)
from .config import GpuDeviceInfo, GpuRunConfig
from .nvml import (
    NVML_DISTRIBUTION,
    NVML_DISTRIBUTION_VERSION,
    NvmlClient,
    NvmlError,
)
from .profiling import build_detailed_profile_plan
from .telemetry import GpuTelemetryCollector

RAW_NVML_RELATIVE_PATH = "raw/gpu/nvml-last.json"


@dataclass(frozen=True)
class GpuRunResult(MonitorRunResult):
    """Outcome of one GPU-only monitor run."""


def build_gpu_run_plan(config: GpuRunConfig) -> dict[str, object]:
    plan = monitor_run_plan(config, mode=RunMode.GPU_ONLY)
    if config.profile_mode is ProfileMode.DETAILED_PROFILE:
        detailed = build_detailed_profile_plan(config.command)
        plan["detailed_profile"] = {
            "nsys_argv": list(detailed.nsys_argv),
            "torch": asdict(detailed.torch),
            "warning": detailed.simultaneous_warning,
        }
    return plan


class GpuRunCollector(MonitorRunCollector[GpuRunResult]):
    artifact_producer = "gpu-monitor"
    raw_device_relative_path = RAW_NVML_RELATIVE_PATH
    raw_device_artifact_id = "nvml-last"
    raw_device_producer = "nvml"
    raw_device_kind = ArtifactKind.TELEMETRY
    result_type = GpuRunResult

    def __init__(
        self,
        config: GpuRunConfig,
        *,
        gpu_client: NvmlClient | None = None,
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
        self.gpu_client = gpu_client or NvmlClient()

    @property
    def host_id(self) -> str:
        return self.config.host_alias

    def _reject_unsupported_profile_mode(self) -> None:
        if self.config.profile_mode is ProfileMode.DETAILED_PROFILE:
            raise NotImplementedError(
                "detailed-profile execution requires the runtime collector; use --dry-run"
            )

    def _discover_devices(self) -> tuple[tuple[GpuDeviceInfo, ...], tuple[str, ...]]:
        if self.config.gpu_devices:
            return self.config.gpu_devices, ()
        try:
            result = self.gpu_client.query()
        except NvmlError as error:
            return (GpuDeviceInfo(index=0, name="unknown"),), (str(error),)
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
        return devices, ()

    def _device_telemetry(
        self, devices: tuple[GpuDeviceInfo, ...]
    ) -> GpuTelemetryCollector:
        return GpuTelemetryCollector(
            run_id=self.config.run_id,
            host_id=self.host_id,
            clock_domain_id=HOST_CLOCK_DOMAIN,
            sample_interval_ms=self.config.sample_interval_ms,
            client=self.gpu_client,
            known_gpu_indices=tuple(device.index for device in devices),
            monotonic_ns=self.monotonic_ns,
        )

    def _raw_device_snapshot(
        self, device_telemetry: GpuTelemetryCollector
    ) -> str | None:
        return device_telemetry.last_raw_snapshot

    def _manifest(
        self,
        devices: tuple[GpuDeviceInfo, ...],
        status: RunStatus,
        errors: tuple[str, ...],
    ) -> RunManifest:
        return self._manifest_record(
            mode=RunMode.GPU_ONLY,
            host_role="gpu",
            status=status,
            software=[
                self._python_runtime_descriptor(),
                SoftwareDescriptor(
                    name=NVML_DISTRIBUTION,
                    version=NVML_DISTRIBUTION_VERSION,
                    role="gpu-telemetry",
                    path=None,
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
                    attributes={"nvml.gpu_index": device.index},
                )
                for device in devices
            ],
            configuration={
                **self._base_configuration(),
                "runtime_metadata": {
                    "topology": topology_metadata(RunMode.GPU_ONLY),
                },
            },
            attributes={
                "vendor.collector": "gpu-monitor",
                "vendor.collector_errors": list(errors),
            },
        )


def format_plan_json(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
