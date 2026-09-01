"""Convert NVML snapshots into schema v1 GPU resource metrics."""

from __future__ import annotations

import time
from typing import Callable

from ..base import BaseCollector
from ...schema import (
    Availability,
    DeviceType,
    MetricKind,
    MetricSample,
    MetricScope,
    ValueOrigin,
)
from .nvml import NvmlClient, NvmlError, NvmlValue, nvml_error_snapshot


_METRICS = (
    ("resource.gpu.utilization", "percent", "utilization_percent"),
    ("resource.gpu.memory_used", "bytes", "memory_used_bytes"),
    ("resource.gpu.power", "W", "power_watts"),
)


class GpuTelemetryCollector(BaseCollector):
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
        super().__init__()
        self.run_id = run_id
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.sample_interval_ms = sample_interval_ms
        self.client = client or NvmlClient()
        self.known_gpu_indices = known_gpu_indices
        self.monotonic_ns = monotonic_ns
        self.last_raw_snapshot: str | None = None
        self.discovered_rows = ()
        self._previous_timestamp_ns: int | None = None

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
                self._metric(
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
                self._metric(
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

    def _interval(self, timestamp_ns: int) -> int:
        interval_ns = (
            self.sample_interval_ms * 1_000_000
            if self._previous_timestamp_ns is None
            else timestamp_ns - self._previous_timestamp_ns
        )
        self._previous_timestamp_ns = timestamp_ns
        return interval_ns

    def _metric(
        self,
        *,
        index: int,
        name: str,
        unit: str,
        parsed: NvmlValue,
        timestamp_ns: int,
        interval_ns: int,
    ) -> MetricSample:
        return MetricSample(
            run_id=self.run_id,
            metric_name=name,
            metric_kind=MetricKind.GAUGE,
            scope=MetricScope.DEVICE,
            host_id=self.host_id,
            clock_domain_id=self.clock_domain_id,
            timestamp_ns=timestamp_ns,
            availability=parsed.availability,
            origin=ValueOrigin.MEASURED,
            unit=unit,
            value=parsed.value,
            device_type=DeviceType.GPU,
            device_id=f"gpu-{index}",
            interval_ns=interval_ns,
            reason=parsed.reason,
            dimensions={},
            attributes={
                "nvml.gpu_index": index,
                "nvml.query_field": name,
            },
        )
