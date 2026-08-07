"""Pure Phase 7 accuracy tolerance and exact-reconciliation checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TypeAlias


Number: TypeAlias = int | float


class AccuracyError(ValueError):
    """An accuracy observation or policy is invalid."""


class ClientLatencyMetric(str, Enum):
    """Client latency metrics with fixed Phase 7 tolerances."""

    E2E = "latency.e2e"
    TTFT = "latency.ttft"
    TPOT = "latency.tpot"


@dataclass(frozen=True, slots=True)
class AccuracyCheck:
    """One deterministic reference/observed reconciliation."""

    metric_name: str
    method_id: str
    canonical_unit: str
    reference: Number
    observed: Number
    absolute_error: Number
    tolerance: Number
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "method_id": self.method_id,
            "canonical_unit": self.canonical_unit,
            "reference": self.reference,
            "observed": self.observed,
            "absolute_error": self.absolute_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def _finite_nonnegative(value: object, *, field: str) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AccuracyError(f"{field} must be a non-boolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise AccuracyError(f"{field} is outside the finite numeric range") from error
    if not finite:
        raise AccuracyError(f"{field} must be finite")
    if value < 0:
        raise AccuracyError(f"{field} must be non-negative")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AccuracyError(f"{field} must be a non-negative integer")
    return value


def _metric(value: ClientLatencyMetric | str) -> ClientLatencyMetric:
    aliases = {
        "e2e": ClientLatencyMetric.E2E,
        "ttft": ClientLatencyMetric.TTFT,
        "tpot": ClientLatencyMetric.TPOT,
    }
    if isinstance(value, str) and value in aliases:
        return aliases[value]
    try:
        return ClientLatencyMetric(value)
    except (TypeError, ValueError) as error:
        raise AccuracyError(f"unsupported client latency metric: {value!r}") from error


def client_latency_accuracy(
    metric_name: ClientLatencyMetric | str,
    *,
    reference_ns: Number,
    observed_ns: Number,
) -> AccuracyCheck:
    """Apply the fixed Phase 7 independent-client latency tolerance.

    E2E and TTFT use ``max(2 ms, 2% of reference)``.  TPOT uses
    ``max(1 ms, 5% of reference)``.  Equality at the tolerance boundary
    passes; the policy must not be widened after observing results.
    """

    metric = _metric(metric_name)
    reference = _finite_nonnegative(reference_ns, field="reference_ns")
    observed = _finite_nonnegative(observed_ns, field="observed_ns")
    if metric is ClientLatencyMetric.TPOT:
        floor_ns = 1_000_000
        relative_fraction = 0.05
        method = "client_tpot_max_1ms_5pct_v1"
    else:
        floor_ns = 2_000_000
        relative_fraction = 0.02
        method = "client_e2e_ttft_max_2ms_2pct_v1"
    tolerance = max(floor_ns, abs(reference) * relative_fraction)
    error = abs(observed - reference)
    try:
        finite_result = math.isfinite(tolerance) and math.isfinite(error)
    except OverflowError as exc:
        raise AccuracyError(
            "accuracy calculation is outside the finite numeric range"
        ) from exc
    if not finite_result:
        raise AccuracyError("accuracy calculation produced a non-finite value")
    return AccuracyCheck(
        metric_name=metric.value,
        method_id=method,
        canonical_unit="ns",
        reference=reference,
        observed=observed,
        absolute_error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
    )


def exact_integer_accuracy(
    metric_name: str,
    *,
    reference: int,
    observed: int,
    canonical_unit: str = "count",
    method_id: str = "exact_integer_equality_v1",
) -> AccuracyCheck:
    """Require exact non-negative integer equality.

    This contract covers request/token counts and same-clock marker-derived
    timestamps or durations.  Separate wrapper functions provide stable
    method IDs for report provenance.
    """

    if not isinstance(metric_name, str) or not metric_name:
        raise AccuracyError("metric_name must be a non-empty string")
    if not isinstance(canonical_unit, str) or not canonical_unit:
        raise AccuracyError("canonical_unit must be a non-empty string")
    if not isinstance(method_id, str) or not method_id:
        raise AccuracyError("method_id must be a non-empty string")
    expected = _nonnegative_integer(reference, field="reference")
    actual = _nonnegative_integer(observed, field="observed")
    error = abs(actual - expected)
    return AccuracyCheck(
        metric_name=metric_name,
        method_id=method_id,
        canonical_unit=canonical_unit,
        reference=expected,
        observed=actual,
        absolute_error=error,
        tolerance=0,
        passed=error == 0,
    )


def exact_count_accuracy(
    metric_name: str,
    *,
    reference: int,
    observed: int,
) -> AccuracyCheck:
    """Require exact request, token, or success-count equality."""

    return exact_integer_accuracy(
        metric_name,
        reference=reference,
        observed=observed,
        canonical_unit="count",
        method_id="exact_count_equality_v1",
    )


def exact_marker_accuracy(
    metric_name: str,
    *,
    reference_ns: int,
    observed_ns: int,
) -> AccuracyCheck:
    """Require exact same-clock marker timestamp/duration equality."""

    return exact_integer_accuracy(
        metric_name,
        reference=reference_ns,
        observed=observed_ns,
        canonical_unit="ns",
        method_id="exact_same_clock_integer_ns_v1",
    )


__all__ = [
    "AccuracyCheck",
    "AccuracyError",
    "ClientLatencyMetric",
    "Number",
    "client_latency_accuracy",
    "exact_count_accuracy",
    "exact_integer_accuracy",
    "exact_marker_accuracy",
]
