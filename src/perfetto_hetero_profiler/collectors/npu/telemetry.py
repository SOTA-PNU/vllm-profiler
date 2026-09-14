"""Convert ``rbln-smi`` samples into schema v1 metric records."""

from __future__ import annotations

import time
from typing import Callable

from ...schema import Availability, DeviceType, MetricSample
from ..telemetry import DeviceTelemetryCollector
from .rbln_smi import ParsedValue, RblnSmiClient, RblnSmiCommandError


class NpuTelemetryCollector(DeviceTelemetryCollector):
    def __init__(
        self,
        *,
        run_id: str,
        host_id: str,
        clock_domain_id: str,
        sample_interval_ms: int,
        client: RblnSmiClient | None = None,
        known_npu_indices: tuple[int, ...] = (0,),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(
            run_id=run_id,
            host_id=host_id,
            clock_domain_id=clock_domain_id,
            sample_interval_ms=sample_interval_ms,
            device_type=DeviceType.NPU,
            device_prefix="npu",
            provenance_namespace="rbln_smi",
            monotonic_ns=monotonic_ns,
        )
        self.client = client or RblnSmiClient()
        self.known_npu_indices = known_npu_indices
        self.last_raw_output: str | None = None
        self.discovered_rows = ()
        self._reported_unsupported: set[tuple[str, str, str]] = set()

    def _sample(self) -> list[MetricSample]:
        try:
            result = self.client.query()
        except RblnSmiCommandError as error:
            timestamp_ns = self.monotonic_ns()
            interval_ns = self._interval(timestamp_ns)
            return [
                self._device_metric(
                    index=index,
                    name=name,
                    unit=unit,
                    parsed=ParsedValue(None, Availability.ERROR, str(error)),
                    timestamp_ns=timestamp_ns,
                    interval_ns=interval_ns,
                )
                for index in self.known_npu_indices
                for name, unit in self._metric_definitions()
            ]
        timestamp_ns = self.monotonic_ns()
        interval_ns = self._interval(timestamp_ns)
        self.last_raw_output = result.raw_output
        self.discovered_rows = result.rows
        records: list[MetricSample] = []
        for row in result.rows:
            values = (
                ("resource.npu.utilization", "percent", row.utilization_percent),
                ("resource.npu.memory_used", "bytes", row.memory_used_bytes),
                ("resource.npu.power", "W", row.power_watts),
            )
            for name, unit, parsed in values:
                marker = (self.host_id, row.device_id, name)
                if not parsed.structurally_unsupported:
                    self._reported_unsupported.discard(marker)
                if parsed.structurally_unsupported and marker in self._reported_unsupported:
                    continue
                records.append(
                    self._device_metric(
                        index=row.index,
                        name=name,
                        unit=unit,
                        parsed=parsed,
                        timestamp_ns=timestamp_ns,
                        interval_ns=interval_ns,
                    )
                )
                if parsed.structurally_unsupported:
                    self._reported_unsupported.add(marker)
        return records

    @staticmethod
    def _metric_definitions() -> tuple[tuple[str, str], ...]:
        return (
            ("resource.npu.utilization", "percent"),
            ("resource.npu.memory_used", "bytes"),
            ("resource.npu.power", "W"),
        )
