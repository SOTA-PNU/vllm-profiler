"""NTP-style four-timestamp clock estimation without a network dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import time


class ClockSyncError(RuntimeError):
    """Clock probing did not produce a usable estimate."""


@dataclass(frozen=True)
class ClockProbeSample:
    t0_ns: int
    t1_ns: int
    t2_ns: int
    t3_ns: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.t0_ns, self.t1_ns, self.t2_ns, self.t3_ns)):
            raise ValueError("clock probe timestamps must be non-negative")
        if self.t3_ns < self.t0_ns:
            raise ValueError("coordinator receive precedes send")
        if self.t2_ns < self.t1_ns:
            raise ValueError("target send precedes receive")
        if self.round_trip_ns < 0:
            raise ValueError("remote processing time exceeds total elapsed time")

    @property
    def round_trip_ns(self) -> int:
        return (self.t3_ns - self.t0_ns) - (self.t2_ns - self.t1_ns)

    @property
    def offset_ns(self) -> int:
        return ((self.t1_ns - self.t0_ns) + (self.t2_ns - self.t3_ns)) // 2


@dataclass(frozen=True)
class ClockEstimate:
    offset_ns: int
    uncertainty_ns: int
    round_trip_ns: int
    sample_count: int
    selected_index: int
    median_offset_ns: int
    method: str
    samples: tuple[ClockProbeSample, ...]


class ClockProbeTransport(Protocol):
    def probe(self) -> ClockProbeSample:
        """Return one complete four-timestamp observation."""


class LocalClockProbeTransport:
    """Probe one process-local monotonic clock without sockets or a daemon."""

    def __init__(self, monotonic_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._monotonic_ns = monotonic_ns

    def probe(self) -> ClockProbeSample:
        t0_ns = self._monotonic_ns()
        t1_ns = self._monotonic_ns()
        t2_ns = self._monotonic_ns()
        t3_ns = self._monotonic_ns()
        return ClockProbeSample(t0_ns, t1_ns, t2_ns, t3_ns)


class FakeClockProbeTransport:
    """Deterministic remote clock with configurable delay and failure cases."""

    def __init__(
        self,
        *,
        offset_ns: int = 0,
        delay_ns: int = 100_000,
        jitter_ns: int = 0,
        asymmetry_ns: int = 0,
        processing_ns: int = 1_000,
        start_ns: int = 1_000_000_000,
        timeout_indices: tuple[int, ...] = (),
        malformed_indices: tuple[int, ...] = (),
    ) -> None:
        if delay_ns < 0 or jitter_ns < 0 or processing_ns < 0:
            raise ValueError("fake probe delays must be non-negative")
        self.offset_ns = offset_ns
        self.delay_ns = delay_ns
        self.jitter_ns = jitter_ns
        self.asymmetry_ns = asymmetry_ns
        self.processing_ns = processing_ns
        self.start_ns = start_ns
        self.timeout_indices = frozenset(timeout_indices)
        self.malformed_indices = frozenset(malformed_indices)
        self.calls = 0

    def probe(self) -> ClockProbeSample:
        index = self.calls
        self.calls += 1
        if index in self.timeout_indices:
            raise TimeoutError("fake clock probe timeout")
        if index in self.malformed_indices:
            raise ValueError("fake malformed clock response")
        jitter = ((index % 3) - 1) * self.jitter_ns
        outbound_ns = max(0, self.delay_ns + self.asymmetry_ns + jitter)
        inbound_ns = max(0, self.delay_ns - self.asymmetry_ns - jitter)
        t0_ns = self.start_ns + index * 10_000_000
        t1_ns = t0_ns + outbound_ns + self.offset_ns
        t2_ns = t1_ns + self.processing_ns
        t3_ns = t0_ns + outbound_ns + self.processing_ns + inbound_ns
        return ClockProbeSample(t0_ns, t1_ns, t2_ns, t3_ns)


def same_clock_estimate() -> ClockEstimate:
    """An explicitly verified identical domain needs no probe or correction."""
    return ClockEstimate(
        offset_ns=0,
        uncertainty_ns=0,
        round_trip_ns=0,
        sample_count=0,
        selected_index=0,
        median_offset_ns=0,
        method="same_clock_domain",
        samples=(),
    )


def collect_probe_samples(
    transport: ClockProbeTransport,
    *,
    count: int = 7,
    minimum_samples: int = 5,
) -> tuple[ClockProbeSample, ...]:
    if count < 1 or not 1 <= minimum_samples <= count:
        raise ValueError("invalid clock probe sample limits")
    samples: list[ClockProbeSample] = []
    errors: list[str] = []
    for _ in range(count):
        try:
            sample = transport.probe()
            if not isinstance(sample, ClockProbeSample):
                raise ValueError("transport returned a malformed response")
            samples.append(sample)
        except (TimeoutError, ValueError) as error:
            errors.append(str(error))
    if len(samples) < minimum_samples:
        suffix = f": {'; '.join(errors)}" if errors else ""
        raise ClockSyncError(
            f"insufficient clock probe samples ({len(samples)}/{minimum_samples})"
            f"{suffix}"
        )
    return tuple(samples)


def _integer_median(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def estimate_clock(
    samples: tuple[ClockProbeSample, ...],
    *,
    minimum_samples: int = 5,
) -> ClockEstimate:
    if len(samples) < minimum_samples:
        raise ClockSyncError(
            f"insufficient clock probe samples ({len(samples)}/{minimum_samples})"
        )
    offsets = [sample.offset_ns for sample in samples]
    median_offset = _integer_median(offsets)
    deviations = [abs(value - median_offset) for value in offsets]
    median_deviation = _integer_median(deviations)
    threshold = max(1, median_deviation * 3)
    candidates = [
        (index, sample)
        for index, sample in enumerate(samples)
        if abs(sample.offset_ns - median_offset) <= threshold
    ]
    if len(candidates) < minimum_samples:
        candidates = list(enumerate(samples))
    selected_index, selected = min(
        candidates, key=lambda item: (item[1].round_trip_ns, item[0])
    )
    uncertainty = max(
        selected.round_trip_ns // 2,
        abs(selected.offset_ns - median_offset),
    )
    return ClockEstimate(
        offset_ns=selected.offset_ns,
        uncertainty_ns=uncertainty,
        round_trip_ns=selected.round_trip_ns,
        sample_count=len(samples),
        selected_index=selected_index,
        median_offset_ns=median_offset,
        method="ntp_style_four_timestamp",
        samples=samples,
    )


def probe_clock(
    transport: ClockProbeTransport,
    *,
    count: int = 7,
    minimum_samples: int = 5,
) -> ClockEstimate:
    return estimate_clock(
        collect_probe_samples(
            transport, count=count, minimum_samples=minimum_samples
        ),
        minimum_samples=minimum_samples,
    )
