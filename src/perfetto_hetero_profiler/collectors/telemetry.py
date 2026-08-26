"""Shared polling and boundary-sample lifecycle for telemetry adapters."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Protocol


class SamplingAdapter(Protocol):
    """Device-specific collector surface consumed by the polling worker."""

    def sample(self) -> Sequence[Any]: ...


class ManagedCollector(Protocol):
    """Collector lifecycle required by a telemetry bundle."""

    def prepare(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def finalize(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class SampleTicket:
    role: str
    generation: int
    requested_ns: int


class TelemetryWorker:
    """Serialize one adapter's baseline, background, and final samples."""

    def __init__(
        self,
        *,
        name: str,
        collector: SamplingAdapter,
        target: MutableSequence[Any],
        interval_sec: float,
        errors: list[str],
        error_type: type[RuntimeError] = RuntimeError,
    ) -> None:
        self.name = name
        self.collector = collector
        self.target = target
        self.interval_sec = interval_sec
        self.errors = errors
        self.error_type = error_type
        self.condition = threading.Condition()
        self.stopping = False
        self.inflight = False
        self.failed = False
        self.sampling_complete = False
        self.generation = 0
        self.pending_ticket: SampleTicket | None = None
        self.active_boundary_ticket: SampleTicket | None = None
        self.completed_boundaries: dict[SampleTicket, dict[str, Any]] = {}
        self.last_sample: dict[str, Any] | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"hybrid-telemetry-{name}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def request(self, role: str) -> SampleTicket:
        if role not in {"baseline", "final"}:
            raise ValueError(f"unsupported telemetry boundary role: {role}")
        requested_ns = time.monotonic_ns()
        with self.condition:
            if self.stopping or self.failed or self.sampling_complete:
                raise self.error_type(f"{self.name} telemetry is not running")
            if self.pending_ticket is not None or self.active_boundary_ticket is not None:
                raise self.error_type(
                    f"{self.name} telemetry already has a boundary request"
                )
            ticket = SampleTicket(role, self.generation, requested_ns)
            self.pending_ticket = ticket
            self.condition.notify_all()
            return ticket

    def wait(self, ticket: SampleTicket, timeout_sec: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        with self.condition:
            while True:
                sample = self.completed_boundaries.pop(ticket, None)
                if sample is not None:
                    if sample.get("error") is not None:
                        raise self.error_type(
                            f"{self.name} telemetry boundary failed: {sample['error']}"
                        )
                    return dict(sample)
                if self.failed:
                    detail = (
                        self.last_sample.get("error")
                        if self.last_sample is not None
                        else None
                    )
                    raise self.error_type(
                        f"{self.name} telemetry boundary failed"
                        + (f": {detail}" if detail else "")
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self.name} telemetry {ticket.role} sample timed out"
                    )
                self.condition.wait(remaining)

    def stop(self, timeout_sec: float) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=timeout_sec)
        if self.thread.is_alive():
            self.errors.append(f"{self.name} telemetry thread did not stop")

    def _run(self) -> None:
        next_deadline = time.monotonic()
        while True:
            with self.condition:
                while not self.stopping and self.pending_ticket is None:
                    remaining = next_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(remaining)
                if self.stopping:
                    return
                started_ticket = self.pending_ticket
                self.pending_ticket = None
                self.active_boundary_ticket = started_ticket
                self.inflight = True

            query_started_ns = time.monotonic_ns()
            try:
                records = list(self.collector.sample())
                error: str | None = None
            except Exception as caught:
                records = []
                error = f"{type(caught).__name__}: {caught}"
            query_completed_ns = time.monotonic_ns()

            with self.condition:
                # A boundary requested during a poll consumes that poll instead
                # of triggering a duplicate device query.
                boundary_ticket = self.pending_ticket or self.active_boundary_ticket
                self.pending_ticket = None
                self.active_boundary_ticket = None
                role = boundary_ticket.role if boundary_ticket is not None else "background"
                requested_ns = (
                    boundary_ticket.requested_ns if boundary_ticket is not None else None
                )
                sequence = self.generation + 1
                tagged = [
                    replace(
                        record,
                        attributes={
                            **record.attributes,
                            "telemetry.sample_role": role,
                            "telemetry.sample_sequence": sequence,
                            "telemetry.query_started_ns": query_started_ns,
                            "telemetry.query_completed_ns": query_completed_ns,
                        },
                    )
                    for record in records
                ]
                self.target.extend(tagged)
                timestamps = sorted({record.timestamp_ns for record in tagged})
                self.generation = sequence
                self.last_sample = {
                    "role": role,
                    "sequence": sequence,
                    "requested_ns": requested_ns,
                    "query_started_ns": query_started_ns,
                    "query_completed_ns": query_completed_ns,
                    "sample_count": len(tagged),
                    "sample_timestamps_ns": timestamps,
                    "error": error,
                }
                if boundary_ticket is not None:
                    self.completed_boundaries[boundary_ticket] = self.last_sample
                    if boundary_ticket.role == "final":
                        self.sampling_complete = True
                self.inflight = False
                if error is not None:
                    self.failed = True
                    self.errors.append(f"{self.name} telemetry: {error}")
                self.condition.notify_all()
            if error is not None or self.sampling_complete:
                return
            next_deadline = time.monotonic() + self.interval_sec


class CollectorGroup:
    """Start collectors in order and finalize them in reverse order."""

    def __init__(
        self, collectors: Sequence[ManagedCollector], *, errors: list[str]
    ) -> None:
        self.collectors = tuple(collectors)
        self.errors = errors
        self.started: list[ManagedCollector] = []
        self.closed = False

    def start(self) -> None:
        for collector in self.collectors:
            collector.prepare()
            collector.start()
            self.started.append(collector)

    def close(self) -> None:
        if self.closed:
            return
        for collector in reversed(self.started):
            try:
                collector.stop()
                collector.finalize()
            except Exception as error:
                self.errors.append(f"{type(collector).__name__} finalize: {error}")
        self.started.clear()
        self.closed = True

    def __enter__(self) -> "CollectorGroup":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
