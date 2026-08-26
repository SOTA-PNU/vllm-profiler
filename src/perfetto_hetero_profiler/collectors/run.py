"""Shared monitor loop for an explicitly owned child process."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from .base import BaseCollector
from .process import CommandResult, ManagedProcess


@dataclass(frozen=True)
class MonitorProcessResult:
    command: CommandResult
    metrics: tuple[Any, ...]
    errors: tuple[str, ...]
    process_id: int | None


def run_monitored_process(
    process: ManagedProcess,
    collectors: Iterable[BaseCollector],
    *,
    sample_interval_ms: int,
    stdout_path: Path,
    stderr_path: Path,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> MonitorProcessResult:
    """Run a child and sample collectors, always stopping prepared collectors."""
    active_collectors = tuple(collectors)
    metrics: list[Any] = []
    errors: list[str] = []
    result: CommandResult | None = None
    child_start_ns: int | None = None

    def stop_collector(collector: BaseCollector) -> None:
        try:
            collector.stop()
        except Exception as error:
            errors.append(f"{type(collector).__name__} stop: {error}")

    with ExitStack() as cleanup:
        for collector in active_collectors:
            cleanup.callback(stop_collector, collector)
        try:
            for collector in active_collectors:
                collector.prepare()
                collector.start()
            child_start_ns = monotonic_ns()
            process.start()
            deadline_ns = (
                None
                if process.spec.timeout_sec is None
                else process.started_monotonic_ns
                + int(process.spec.timeout_sec * 1_000_000_000)
            )
            first_sample = True
            while True:
                if not first_sample:
                    if process.poll() is not None:
                        break
                    if deadline_ns is not None and monotonic_ns() >= deadline_ns:
                        result = process.stop(timed_out=True)
                        break
                for collector in active_collectors:
                    try:
                        metrics.extend(collector.sample())
                    except Exception as error:
                        errors.append(f"{type(collector).__name__}: {error}")
                first_sample = False
                if process.poll() is not None:
                    break
                if deadline_ns is not None and monotonic_ns() >= deadline_ns:
                    result = process.stop(timed_out=True)
                    break
                sleep(sample_interval_ms / 1000)
            if result is None:
                result = process.wait()
        except Exception as error:
            errors.append(f"run orchestration: {error}")
            if process.process is not None:
                result = process.stop()
            else:
                ended_ns = monotonic_ns()
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.touch(exist_ok=True)
                stderr_path.touch(exist_ok=True)
                result = CommandResult(
                    return_code=127,
                    started_monotonic_ns=child_start_ns or ended_ns,
                    ended_monotonic_ns=ended_ns,
                    timed_out=False,
                    terminated=False,
                    killed=False,
                )

    assert result is not None
    return MonitorProcessResult(
        command=result,
        metrics=tuple(metrics),
        errors=tuple(errors),
        process_id=process.process.pid if process.process is not None else None,
    )
