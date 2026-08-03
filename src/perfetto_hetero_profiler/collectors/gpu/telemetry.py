"""Convert nvidia-smi samples into schema v1 MetricSample records."""

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
from .nvidia_smi import NvidiaSmiClient, NvidiaSmiCommandError, ParsedValue


class GpuTelemetryCollector(BaseCollector):
    def __init__(
        self,
        *,
        run_id: str,
        host_id: str,
        clock_domain_id: str,
        sample_interval_ms: int,
        client: NvidiaSmiClient | None = None,
        known_gpu_indices: tuple[int, ...] = (0,),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.sample_interval_ms = sample_interval_ms
        self.client = client or NvidiaSmiClient()
        self.known_gpu_indices = known_gpu_indices
        self.monotonic_ns = monotonic_ns
        self.last_raw_output: str | None = None
        self.discovered_rows = ()
        self._previous_timestamp_ns: int | None = None

    def _sample(self) -> list[MetricSample]:
        timestamp_ns = self.monotonic_ns()
        interval_ns = (
            self.sample_interval_ms * 1_000_000
            if self._previous_timestamp_ns is None
            else timestamp_ns - self._previous_timestamp_ns
        )
        self._previous_timestamp_ns = timestamp_ns
        try:
            result = self.client.query()
        except NvidiaSmiCommandError as error:
            return [
                self._metric(
                    index=index,
                    name=name,
                    unit=unit,
                    parsed=ParsedValue(None, Availability.ERROR, str(error)),
                    timestamp_ns=timestamp_ns,
                    interval_ns=interval_ns,
                )
                for index in self.known_gpu_indices
                for name, unit in (
                    ("resource.gpu.utilization", "percent"),
                    ("resource.gpu.memory_used", "bytes"),
                    ("resource.gpu.power", "W"),
                )
            ]
        self.last_raw_output = result.raw_output
        self.discovered_rows = result.rows
        records: list[MetricSample] = []
        for row in result.rows:
            records.extend(
                (
                    self._metric(
                        index=row.index,
                        name="resource.gpu.utilization",
                        unit="percent",
                        parsed=row.utilization_percent,
                        timestamp_ns=timestamp_ns,
                        interval_ns=interval_ns,
                    ),
                    self._metric(
                        index=row.index,
                        name="resource.gpu.memory_used",
                        unit="bytes",
                        parsed=row.memory_used_bytes,
                        timestamp_ns=timestamp_ns,
                        interval_ns=interval_ns,
                    ),
                    self._metric(
                        index=row.index,
                        name="resource.gpu.power",
                        unit="W",
                        parsed=row.power_watts,
                        timestamp_ns=timestamp_ns,
                        interval_ns=interval_ns,
                    ),
                )
            )
        return records

    def _metric(
        self,
        *,
        index: int,
        name: str,
        unit: str,
        parsed: ParsedValue,
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
                "nvidia_smi.gpu_index": index,
                "nvidia_smi.query_field": name,
            },
        )
