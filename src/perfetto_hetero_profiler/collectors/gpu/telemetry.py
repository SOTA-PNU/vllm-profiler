"""Convert NVML snapshots into schema v1 GPU resource metrics."""

from __future__ import annotations

import time
from typing import Callable

from ...schema import Availability, DeviceType, MetricSample
from ..telemetry import DeviceTelemetryCollector
from .nvml import NvmlClient, NvmlError, NvmlValue, nvml_error_snapshot

_METRICS = (
    ("resource.gpu.utilization", "percent", "utilization_percent"),
    ("resource.gpu.memory_used", "bytes", "memory_used_bytes"),
    ("resource.gpu.power", "W", "power_watts"),
)


class GpuTelemetryCollector(DeviceTelemetryCollector):
    def __init__(
        self,
        *,
        run_id: str,
        host_id: str,
        clock_domain_id: str,
        sample_interval_ms: int,
        client: NvmlClient | None = None,
        known_gpu_indices: tuple[int, ...] = (0,),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(
            run_id=run_id,
            host_id=host_id,
            clock_domain_id=clock_domain_id,
            sample_interval_ms=sample_interval_ms,
            device_type=DeviceType.GPU,
            device_prefix="gpu",
            provenance_namespace="nvml",
            monotonic_ns=monotonic_ns,
        )
        self.client = client or NvmlClient()
        self.known_gpu_indices = known_gpu_indices
        self.last_raw_snapshot: str | None = None
        self.discovered_rows = ()

    def _prepare(self) -> None:
        # Keep lifecycle ownership explicit, but defer a capability failure to
        # sample() so known devices receive schema-valid error evidence.
        try:
            self.client.initialize()
        except NvmlError:
            pass

    def _stop(self) -> None:
        self.client.shutdown()

    def _sample(self) -> list[MetricSample]:
        try:
            result = self.client.query()
        except NvmlError as error:
            timestamp_ns = self.monotonic_ns()
            interval_ns = self._interval(timestamp_ns)
            reason = str(error)
            self.last_raw_snapshot = nvml_error_snapshot(
                self.known_gpu_indices, reason
            )
            return [
                self._device_metric(
                    index=index,
                    name=name,
                    unit=unit,
                    parsed=NvmlValue(None, Availability.ERROR, reason),
                    timestamp_ns=timestamp_ns,
                    interval_ns=interval_ns,
                )
                for index in self.known_gpu_indices
                for name, unit, _ in _METRICS
            ]
        timestamp_ns = self.monotonic_ns()
        interval_ns = self._interval(timestamp_ns)
        self.last_raw_snapshot = result.raw_snapshot
        self.discovered_rows = result.rows
        records: list[MetricSample] = []
        for row in result.rows:
            records.extend(
                self._device_metric(
                    index=row.index,
                    name=name,
                    unit=unit,
                    parsed=getattr(row, field),
                    timestamp_ns=timestamp_ns,
                    interval_ns=interval_ns,
                )
                for name, unit, field in _METRICS
            )
        return records
