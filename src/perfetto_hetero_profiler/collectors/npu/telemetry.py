"""Convert ``rbln-smi`` samples into schema v1 metric records."""

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
from .rbln_smi import ParsedValue, RblnSmiClient, RblnSmiCommandError


class NpuTelemetryCollector(BaseCollector):
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
        super().__init__()
        self.run_id = run_id
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.sample_interval_ms = sample_interval_ms
        self.client = client or RblnSmiClient()
        self.known_npu_indices = known_npu_indices
        self.monotonic_ns = monotonic_ns
        self.last_raw_output: str | None = None
        self.discovered_rows = ()
        self._previous_timestamp_ns: int | None = None
        self._reported_unsupported: set[tuple[str, str, str]] = set()

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
        except RblnSmiCommandError as error:
            return [
                self._metric(
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
                    self._metric(
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
            device_type=DeviceType.NPU,
            device_id=f"npu-{index}",
            interval_ns=interval_ns,
            reason=parsed.reason,
            dimensions={},
            attributes={
                "rbln_smi.npu_index": index,
                "rbln_smi.query_field": name,
            },
        )
