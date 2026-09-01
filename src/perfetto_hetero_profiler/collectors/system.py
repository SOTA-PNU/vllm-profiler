"""psutil-backed host and process resource telemetry."""

from __future__ import annotations

from dataclasses import replace
import math
from numbers import Real
import time
from typing import Any, Callable

import psutil

from .base import BaseCollector
from ..schema import Availability, MetricKind, MetricSample, MetricScope, ValueOrigin


class _MetricValueError(ValueError):
    """A sanitized invalid measurement or counter delta."""


def _number(value: object, field: str, *, integer: bool = False) -> int | float:
    valid_type = isinstance(value, int) if integer else isinstance(value, Real)
    if isinstance(value, bool) or not valid_type:
        raise _MetricValueError(f"psutil {field} returned invalid data")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise _MetricValueError(f"psutil {field} returned invalid data")
    return int(value) if integer else result


def _cpu_snapshot(times: object) -> tuple[tuple[float, ...], float, float]:
    fields = getattr(times, "_fields", ())
    if not fields or "idle" not in fields:
        raise _MetricValueError("psutil CPU times returned invalid data")
    names = tuple(name for name in fields if name not in {"guest", "guest_nice"})
    values = tuple(float(_number(getattr(times, name), f"CPU {name}")) for name in names)
    idle = float(_number(getattr(times, "idle"), "CPU idle"))
    if "iowait" in fields:
        idle += float(_number(getattr(times, "iowait"), "CPU iowait"))
    return values, sum(values), idle


class SystemTelemetryCollector(BaseCollector):
    """Collect host CPU, host memory, and optional child RSS metrics."""

    def __init__(
        self,
        *,
        run_id: str,
        host_id: str,
        clock_domain_id: str,
        pid_provider: Callable[[], int | None] | None = None,
        psutil_module: Any = psutil,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.pid_provider = pid_provider or (lambda: None)
        self.psutil = psutil_module
        self.monotonic_ns = monotonic_ns
        self._previous_cpu: tuple[tuple[float, ...], float, float] | None = None
        self._previous_timestamp_ns: int | None = None

    def _sample(self) -> list[MetricSample]:
        records = [self._cpu_metric(), self._memory_metric()]
        pid = self.pid_provider()
        if pid is not None:
            records.append(self._process_memory_metric(pid))
        timestamp_ns = self.monotonic_ns()
        interval_ns = (
            None
            if self._previous_timestamp_ns is None
            else timestamp_ns - self._previous_timestamp_ns
        )
        self._previous_timestamp_ns = timestamp_ns
        return [
            replace(record, timestamp_ns=timestamp_ns, interval_ns=interval_ns)
            for record in records
        ]

    def _cpu_metric(self) -> MetricSample:
        try:
            current = _cpu_snapshot(self.psutil.cpu_times())
            previous = self._previous_cpu
            self._previous_cpu = current
            if previous is None:
                availability, value = Availability.NOT_AVAILABLE, None
                reason = "first sample establishes psutil CPU baseline"
                attributes = {"psutil.baseline": True}
            else:
                deltas = tuple(
                    now - old for now, old in zip(current[0], previous[0])
                )
                total_delta = current[1] - previous[1]
                idle_delta = current[2] - previous[2]
                if (
                    len(current[0]) != len(previous[0])
                    or any(delta < 0 for delta in deltas)
                    or total_delta <= 0
                    or idle_delta < 0
                    or idle_delta > total_delta
                ):
                    raise _MetricValueError("invalid psutil CPU counter delta")
                availability = Availability.AVAILABLE
                value = 100.0 * (total_delta - idle_delta) / total_delta
                reason = None
                attributes = {
                    "psutil.total_delta": total_delta,
                    "psutil.idle_delta": idle_delta,
                }
        except Exception as error:
            availability, value = Availability.ERROR, None
            reason = str(error) if isinstance(error, _MetricValueError) else "psutil CPU query failed"
            attributes = {"psutil.source": "cpu_times"}
        return self._metric(
            "resource.cpu.utilization", MetricScope.HOST, "percent",
            value, availability, reason, attributes=attributes,
        )

    def _memory_metric(self) -> MetricSample:
        try:
            memory = self.psutil.virtual_memory()
            total = _number(memory.total, "total memory", integer=True)
            available = _number(memory.available, "available memory", integer=True)
            if available > total:
                raise _MetricValueError("psutil available memory exceeds total memory")
            value, availability, reason = total - available, Availability.AVAILABLE, None
            attributes = {
                "psutil.mem_total_bytes": total,
                "psutil.mem_available_bytes": available,
            }
        except Exception as error:
            value, availability = None, Availability.ERROR
            reason = str(error) if isinstance(error, _MetricValueError) else "psutil virtual memory query failed"
            attributes = {"psutil.source": "virtual_memory"}
        return self._metric(
            "resource.system.memory_used", MetricScope.HOST, "bytes",
            value, availability, reason, attributes=attributes,
        )

    def _process_memory_metric(self, pid: int) -> MetricSample:
        try:
            value = _number(
                self.psutil.Process(pid).memory_info().rss,
                "process RSS",
                integer=True,
            )
            availability, reason = Availability.AVAILABLE, None
        except self.psutil.ZombieProcess:
            value, availability, reason = None, Availability.NOT_AVAILABLE, "child process is a zombie"
        except self.psutil.NoSuchProcess:
            value, availability, reason = None, Availability.NOT_AVAILABLE, "child process has exited"
        except self.psutil.AccessDenied:
            value, availability, reason = None, Availability.ERROR, "process RSS access is denied"
        except Exception as error:
            value, availability = None, Availability.ERROR
            reason = str(error) if isinstance(error, _MetricValueError) else "psutil process RSS query failed"
        return self._metric(
            "resource.cpu.memory_used", MetricScope.PROCESS, "bytes",
            value, availability, reason,
            attributes={"psutil.pid": pid}, dimensions={"process_id": str(pid)},
        )

    def _metric(
        self,
        name: str,
        scope: MetricScope,
        unit: str,
        value: int | float | None,
        availability: Availability,
        reason: str | None,
        *,
        attributes: dict[str, object],
        dimensions: dict[str, object] | None = None,
    ) -> MetricSample:
        return MetricSample(
            run_id=self.run_id, metric_name=name, metric_kind=MetricKind.GAUGE,
            scope=scope, host_id=self.host_id, clock_domain_id=self.clock_domain_id,
            timestamp_ns=0, availability=availability, origin=ValueOrigin.MEASURED,
            unit=unit, value=value, interval_ns=None, reason=reason,
            dimensions=dimensions or {}, attributes=attributes,
        )
