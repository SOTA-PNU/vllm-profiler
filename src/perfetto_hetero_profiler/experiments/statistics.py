"""Deterministic descriptive statistics and paired profiler overhead.

The functions in this module are deliberately pure.  They do not inspect
hardware, files, clocks, or process state.  Callers must select valid trials
before passing observations here; this module then enforces exact round
pairing and finite, non-boolean numeric inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import TypeAlias


Number: TypeAlias = int | float


class StatisticsError(ValueError):
    """A statistic or paired comparison is not well-defined."""


class OverheadDirection(str, Enum):
    """Sign convention for a paired relative overhead value."""

    INCREASE = "increase"
    THROUGHPUT_DEGRADATION = "throughput_degradation"


def _finite_number(value: object, *, field: str) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{field} must be a non-boolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise StatisticsError(f"{field} is outside the finite numeric range") from error
    if not finite:
        raise StatisticsError(f"{field} must be finite")
    return value


def _finite_result(value: Number, *, field: str) -> Number:
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise StatisticsError(
            f"{field} is outside the finite numeric range"
        ) from error
    if not finite:
        raise StatisticsError(f"{field} overflowed to a non-finite value")
    return value


def _expected_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StatisticsError("expected_pair_count must be a positive integer")
    return value


def percentile_r7(
    values: Sequence[Number],
    probability: Number,
) -> float:
    """Return the Hyndman-Fan type-7 percentile for finite values."""

    probability_value = _finite_number(probability, field="probability")
    if probability_value < 0 or probability_value > 1:
        raise StatisticsError("probability must be in [0, 1]")
    ordered = sorted(
        _finite_number(value, field=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if not ordered:
        raise StatisticsError("percentile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability_value
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    result = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    return float(_finite_result(result, field="percentile"))


def _median(ordered: Sequence[Number]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(
        _finite_result(
            (ordered[middle - 1] + ordered[middle]) / 2,
            field="median",
        )
    )


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Descriptive statistics for one non-empty observation vector."""

    sample_count: int
    mean: float
    median: float
    minimum: Number
    maximum: Number
    sample_standard_deviation: float | None
    sample_standard_deviation_unavailable_reason: str | None
    coefficient_of_variation: float | None
    coefficient_of_variation_unavailable_reason: str | None
    median_absolute_deviation: float
    p50: float
    p95: float

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sample_standard_deviation": self.sample_standard_deviation,
            "sample_standard_deviation_unavailable_reason": (
                self.sample_standard_deviation_unavailable_reason
            ),
            "coefficient_of_variation": self.coefficient_of_variation,
            "coefficient_of_variation_unavailable_reason": (
                self.coefficient_of_variation_unavailable_reason
            ),
            "median_absolute_deviation": self.median_absolute_deviation,
            "p50": self.p50,
            "p95": self.p95,
        }


def summarize_distribution(values: Sequence[Number]) -> DistributionSummary:
    """Calculate the profiler experiment descriptive-statistics contract.

    Standard deviation is the sample standard deviation with denominator
    ``n - 1``.  MAD is the unscaled median absolute deviation.  CV is
    ``sample_standard_deviation / abs(mean)`` and is unavailable for a zero
    mean or when sample standard deviation itself is unavailable.
    """

    ordered = sorted(
        _finite_number(value, field=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if not ordered:
        raise StatisticsError("at least one observation is required")

    count = len(ordered)
    mean = _finite_result(math.fsum(ordered) / count, field="mean")
    median = _median(ordered)
    minimum = ordered[0]
    maximum = ordered[-1]
    deviations = [abs(value - median) for value in ordered]
    mad = _median(sorted(deviations))

    if count < 2:
        sample_stddev = None
        sample_stddev_reason = (
            "sample standard deviation requires at least two observations"
        )
        cv = None
        cv_reason = "coefficient of variation requires sample standard deviation"
    else:
        variance = math.fsum((value - mean) ** 2 for value in ordered) / (
            count - 1
        )
        variance = _finite_result(variance, field="sample variance")
        sample_stddev = float(
            _finite_result(
                math.sqrt(max(variance, 0.0)),
                field="sample standard deviation",
            )
        )
        sample_stddev_reason = None
        if mean == 0:
            cv = None
            cv_reason = "coefficient of variation is unavailable for a zero mean"
        else:
            cv = float(
                _finite_result(
                    sample_stddev / abs(mean),
                    field="coefficient of variation",
                )
            )
            cv_reason = None

    return DistributionSummary(
        sample_count=count,
        mean=float(mean),
        median=median,
        minimum=minimum,
        maximum=maximum,
        sample_standard_deviation=sample_stddev,
        sample_standard_deviation_unavailable_reason=sample_stddev_reason,
        coefficient_of_variation=cv,
        coefficient_of_variation_unavailable_reason=cv_reason,
        median_absolute_deviation=mad,
        p50=percentile_r7(ordered, 0.50),
        p95=percentile_r7(ordered, 0.95),
    )


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One condition/reference comparison in the same experiment round."""

    round_index: int
    reference: Number
    observed: Number
    absolute_delta: Number
    overhead_ratio: float | None
    overhead_ratio_unavailable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "reference": self.reference,
            "observed": self.observed,
            "absolute_delta": self.absolute_delta,
            "overhead_ratio": self.overhead_ratio,
            "overhead_ratio_unavailable_reason": (
                self.overhead_ratio_unavailable_reason
            ),
        }


@dataclass(frozen=True, slots=True)
class PairedOverheadSummary:
    """Exact round pairs plus their descriptive delta distributions."""

    direction: OverheadDirection
    expected_pair_count: int
    pairs: tuple[PairedObservation, ...]
    absolute_delta_summary: DistributionSummary
    overhead_ratio_summary: DistributionSummary | None
    overhead_ratio_summary_unavailable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "expected_pair_count": self.expected_pair_count,
            "pairs": [item.to_dict() for item in self.pairs],
            "absolute_delta_summary": self.absolute_delta_summary.to_dict(),
            "overhead_ratio_summary": (
                self.overhead_ratio_summary.to_dict()
                if self.overhead_ratio_summary is not None
                else None
            ),
            "overhead_ratio_summary_unavailable_reason": (
                self.overhead_ratio_summary_unavailable_reason
            ),
        }


def _round_values(
    values: Mapping[int, Number],
    *,
    field: str,
) -> dict[int, Number]:
    if not isinstance(values, Mapping):
        raise StatisticsError(f"{field} must be a round-to-value mapping")
    result: dict[int, Number] = {}
    for round_index, value in values.items():
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise StatisticsError(
                f"{field} round indices must be non-negative integers"
            )
        result[round_index] = _finite_number(
            value,
            field=f"{field}[{round_index}]",
        )
    return result


def paired_overhead(
    reference_by_round: Mapping[int, Number],
    observed_by_round: Mapping[int, Number],
    *,
    direction: OverheadDirection | str = OverheadDirection.INCREASE,
    expected_pair_count: int = 5,
) -> PairedOverheadSummary:
    """Calculate exact same-round paired overhead.

    Absolute delta is always ``observed - reference``.  For latency-like
    metrics ``overhead_ratio`` is ``(observed - reference) / reference``.
    For throughput it is the degradation convention
    ``(reference - observed) / reference`` so a positive value consistently
    means degradation.  A zero reference affects only the ratio; the absolute
    delta remains available.
    """

    expected = _expected_count(expected_pair_count)
    try:
        parsed_direction = OverheadDirection(direction)
    except (TypeError, ValueError) as error:
        raise StatisticsError(f"unsupported overhead direction: {direction!r}") from error

    reference = _round_values(reference_by_round, field="reference_by_round")
    observed = _round_values(observed_by_round, field="observed_by_round")
    if set(reference) != set(observed):
        missing_observed = sorted(set(reference) - set(observed))
        missing_reference = sorted(set(observed) - set(reference))
        raise StatisticsError(
            "paired overhead requires identical round sets; "
            f"missing_observed={missing_observed}, "
            f"missing_reference={missing_reference}"
        )
    if len(reference) != expected:
        raise StatisticsError(
            f"paired overhead requires exactly {expected} rounds; "
            f"found {len(reference)}"
        )

    pairs: list[PairedObservation] = []
    for round_index in sorted(reference):
        baseline = reference[round_index]
        current = observed[round_index]
        absolute = _finite_result(
            current - baseline,
            field=f"round {round_index} absolute delta",
        )
        if baseline == 0:
            ratio = None
            reason = "overhead ratio is unavailable because reference is zero"
        else:
            numerator = (
                absolute
                if parsed_direction is OverheadDirection.INCREASE
                else baseline - current
            )
            ratio = float(
                _finite_result(
                    numerator / baseline,
                    field=f"round {round_index} overhead ratio",
                )
            )
            reason = None
        pairs.append(
            PairedObservation(
                round_index=round_index,
                reference=baseline,
                observed=current,
                absolute_delta=absolute,
                overhead_ratio=ratio,
                overhead_ratio_unavailable_reason=reason,
            )
        )

    ratio_values = [
        item.overhead_ratio
        for item in pairs
        if item.overhead_ratio is not None
    ]
    if len(ratio_values) != len(pairs):
        ratio_summary = None
        ratio_summary_reason = (
            "paired overhead ratio summary requires a non-zero reference "
            "in every round"
        )
    else:
        ratio_summary = summarize_distribution(ratio_values)
        ratio_summary_reason = None

    return PairedOverheadSummary(
        direction=parsed_direction,
        expected_pair_count=expected,
        pairs=tuple(pairs),
        absolute_delta_summary=summarize_distribution(
            [item.absolute_delta for item in pairs]
        ),
        overhead_ratio_summary=ratio_summary,
        overhead_ratio_summary_unavailable_reason=ratio_summary_reason,
    )


__all__ = [
    "DistributionSummary",
    "Number",
    "OverheadDirection",
    "PairedObservation",
    "PairedOverheadSummary",
    "StatisticsError",
    "paired_overhead",
    "percentile_r7",
    "summarize_distribution",
]
