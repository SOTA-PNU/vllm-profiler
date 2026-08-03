"""Dependency-free Linux /proc monitor telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from .base import BaseCollector
from ..schema import (
    Availability,
    MetricKind,
    MetricSample,
    MetricScope,
    ValueOrigin,
)


@dataclass(frozen=True)
class CpuTimes:
    total: int
    idle: int


def parse_proc_stat(text: str) -> CpuTimes:
    line = next((item for item in text.splitlines() if item.startswith("cpu ")), None)
    if line is None:
        raise ValueError("aggregate cpu line is missing")
    fields = line.split()[1:]
    if len(fields) < 4:
        raise ValueError("aggregate cpu line has too few fields")
    try:
        values = [int(field) for field in fields]
    except ValueError as error:
        raise ValueError("aggregate cpu counters must be integers") from error
    if any(value < 0 for value in values):
        raise ValueError("aggregate cpu counters must be non-negative")
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return CpuTimes(total=sum(values), idle=idle)


def parse_meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        if not fields:
            continue
        try:
            number = int(fields[0])
        except ValueError as error:
            raise ValueError(f"{key} must be an integer") from error
        multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
        values[key] = number * multiplier
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("MemTotal and MemAvailable are required")
    if values["MemAvailable"] > values["MemTotal"]:
        raise ValueError("MemAvailable cannot exceed MemTotal")
    return values["MemTotal"], values["MemAvailable"]


def parse_process_rss(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) < 2:
                break
            try:
                value = int(fields[1])
            except ValueError as error:
                raise ValueError("VmRSS must be an integer") from error
            if value < 0:
                raise ValueError("VmRSS must be non-negative")
            return value * 1024
    raise ValueError("VmRSS is missing")


class ProcTelemetryCollector(BaseCollector):
    """Collect host CPU, host memory, and optional child RSS metrics."""

    def __init__(
        self,
        *,
        run_id: str,
        host_id: str,
        clock_domain_id: str,
        pid_provider: Callable[[], int | None] | None = None,
        proc_root: Path = Path("/proc"),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.pid_provider = pid_provider or (lambda: None)
        self.proc_root = Path(proc_root)
        self.monotonic_ns = monotonic_ns
        self._previous_cpu: CpuTimes | None = None
        self._previous_timestamp_ns: int | None = None

    def _sample(self) -> list[MetricSample]:
        timestamp_ns = self.monotonic_ns()
        interval_ns = (
            None
            if self._previous_timestamp_ns is None
            else timestamp_ns - self._previous_timestamp_ns
        )
        records = [
            self._cpu_metric(timestamp_ns, interval_ns),
            self._memory_metric(timestamp_ns, interval_ns),
        ]
        pid = self.pid_provider()
        if pid is not None:
            records.append(self._process_memory_metric(pid, timestamp_ns, interval_ns))
        self._previous_timestamp_ns = timestamp_ns
        return records

    def _cpu_metric(self, timestamp_ns: int, interval_ns: int | None) -> MetricSample:
        current = parse_proc_stat(
            (self.proc_root / "stat").read_text(encoding="utf-8")
        )
        if self._previous_cpu is None:
            availability = Availability.NOT_AVAILABLE
            value = None
            reason = "first sample establishes /proc/stat baseline"
            attributes = {"procfs.baseline": True}
        else:
            total_delta = current.total - self._previous_cpu.total
            idle_delta = current.idle - self._previous_cpu.idle
            if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
                availability = Availability.ERROR
                value = None
                reason = "invalid /proc/stat counter delta"
            else:
                availability = Availability.AVAILABLE
                value = 100.0 * (total_delta - idle_delta) / total_delta
                reason = None
            attributes = {
                "procfs.total_delta": total_delta,
                "procfs.idle_delta": idle_delta,
            }
        self._previous_cpu = current
        return self._metric(
            name="resource.cpu.utilization",
            kind=MetricKind.GAUGE,
            scope=MetricScope.HOST,
            unit="percent",
            value=value,
            availability=availability,
            reason=reason,
            timestamp_ns=timestamp_ns,
            interval_ns=interval_ns,
            attributes=attributes,
        )

    def _memory_metric(
        self, timestamp_ns: int, interval_ns: int | None
    ) -> MetricSample:
        try:
            total, available = parse_meminfo(
                (self.proc_root / "meminfo").read_text(encoding="utf-8")
            )
            value = total - available
            availability = Availability.AVAILABLE
            reason = None
            attributes = {
                "procfs.mem_total_bytes": total,
                "procfs.mem_available_bytes": available,
            }
        except (OSError, ValueError) as error:
            value = None
            availability = Availability.ERROR
            reason = str(error)
            attributes = {"procfs.source": "meminfo"}
        return self._metric(
            name="resource.system.memory_used",
            kind=MetricKind.GAUGE,
            scope=MetricScope.HOST,
            unit="bytes",
            value=value,
            availability=availability,
            reason=reason,
            timestamp_ns=timestamp_ns,
            interval_ns=interval_ns,
            attributes=attributes,
        )

    def _process_memory_metric(
        self, pid: int, timestamp_ns: int, interval_ns: int | None
    ) -> MetricSample:
        try:
            value = parse_process_rss(
                (self.proc_root / str(pid) / "status").read_text(encoding="utf-8")
            )
            availability = Availability.AVAILABLE
            reason = None
        except FileNotFoundError:
            value = None
            availability = Availability.NOT_AVAILABLE
            reason = "child process has exited"
        except (OSError, ValueError) as error:
            value = None
            availability = Availability.ERROR
            reason = str(error)
        return self._metric(
            name="resource.cpu.memory_used",
            kind=MetricKind.GAUGE,
            scope=MetricScope.PROCESS,
            unit="bytes",
            value=value,
            availability=availability,
            reason=reason,
            timestamp_ns=timestamp_ns,
            interval_ns=interval_ns,
            attributes={"procfs.pid": pid},
            dimensions={"process_id": str(pid)},
        )

    def _metric(
        self,
        *,
        name: str,
        kind: MetricKind,
        scope: MetricScope,
        unit: str,
        value: int | float | None,
        availability: Availability,
        reason: str | None,
        timestamp_ns: int,
        interval_ns: int | None,
        attributes: dict[str, object],
        dimensions: dict[str, object] | None = None,
    ) -> MetricSample:
        return MetricSample(
            run_id=self.run_id,
            metric_name=name,
            metric_kind=kind,
            scope=scope,
            host_id=self.host_id,
            clock_domain_id=self.clock_domain_id,
            timestamp_ns=timestamp_ns,
            availability=availability,
            origin=ValueOrigin.MEASURED,
            unit=unit,
            value=value,
            interval_ns=interval_ns,
            reason=reason,
            dimensions=dimensions or {},
            attributes=attributes,
        )
