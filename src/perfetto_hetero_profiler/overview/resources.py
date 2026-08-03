"""Deterministic aggregation of normalized resource metric streams."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ..schema import Availability
from ..schema.metric_catalog import METRIC_CATALOG
from ..schema.records import MetricSample


class ResourceCalculationError(ValueError):
    """Raised when a normalized resource stream violates its contract."""


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _finite_number(value: object, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceCalculationError(f"{field} must be a non-bool number")
    if not math.isfinite(value):
        raise ResourceCalculationError(f"{field} must be finite")
    return value


def _non_bool_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceCalculationError(f"{field} must be a non-bool integer")
    return value


def percentile_r7(values: Sequence[int | float], probability: float) -> float:
    """Return the deterministic Hyndman-Fan type-7 percentile."""

    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or probability < 0
        or probability > 1
    ):
        raise ResourceCalculationError("probability must be finite and in [0, 1]")
    ordered = sorted(
        _finite_number(value, field="percentile value") for value in values
    )
    if not ordered:
        raise ResourceCalculationError("percentile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _display_rule(unit: str) -> dict[str, object]:
    rules: dict[str, tuple[str, int, int, int]] = {
        "ns": ("ms", 1, 1_000_000, 3),
        "bytes": ("MiB", 1, 1_048_576, 3),
        "percent": ("percent", 1, 1, 2),
        "W": ("W", 1, 1, 3),
        "ratio": ("ratio", 1, 1, 6),
    }
    display_unit, numerator, denominator, places = rules.get(
        unit, (unit, 1, 1, 6)
    )
    return {
        "unit": display_unit,
        "scale_numerator": numerator,
        "scale_denominator": denominator,
        "decimal_places": places,
        "rounding": "half_even",
    }


def _canonical_dimensions(dimensions: object) -> str:
    if not isinstance(dimensions, dict):
        raise ResourceCalculationError("metric dimensions must be an object")
    try:
        return json.dumps(
            dimensions,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResourceCalculationError(
            "metric dimensions must be canonical JSON"
        ) from exc


def _metric_contract(metric: MetricSample) -> None:
    definition = METRIC_CATALOG.get(metric.metric_name)
    if definition is None or not metric.metric_name.startswith("resource."):
        raise ResourceCalculationError(
            f"{metric.metric_name!r} is not an official resource metric"
        )
    if metric.unit != definition.unit:
        raise ResourceCalculationError(
            f"{metric.metric_name} unit mismatch: "
            f"{metric.unit!r} != {definition.unit!r}"
        )
    if metric.metric_kind != definition.kind:
        raise ResourceCalculationError(
            f"{metric.metric_name} metric_kind does not match the catalog"
        )
    if metric.scope not in definition.allowed_scopes:
        raise ResourceCalculationError(
            f"{metric.metric_name} scope does not match the catalog"
        )
    _non_bool_int(metric.timestamp_ns, field=f"{metric.metric_name} timestamp_ns")
    availability = _enum_value(metric.availability)
    if availability not in {item.value for item in Availability}:
        raise ResourceCalculationError(
            f"{metric.metric_name} has an invalid availability"
        )
    if availability == Availability.AVAILABLE.value:
        value = _finite_number(
            metric.value, field=f"{metric.metric_name} available value"
        )
        if definition.minimum is not None and value < definition.minimum:
            raise ResourceCalculationError(
                f"{metric.metric_name} value is below its catalog minimum"
            )
        if definition.maximum is not None and value > definition.maximum:
            raise ResourceCalculationError(
                f"{metric.metric_name} value is above its catalog maximum"
            )
    elif metric.value is not None:
        raise ResourceCalculationError(
            f"{metric.metric_name} unavailable sample must have value=null"
        )


def _clock_evidence(
    loaded: object, samples: Sequence[MetricSample]
) -> dict[str, object]:
    domains = tuple(sorted({sample.clock_domain_id for sample in samples}))
    canonical = getattr(loaded, "canonical_clock_domain_id", None)
    if canonical is None:
        canonical_clock = getattr(loaded, "canonical_clock", None)
        canonical = getattr(canonical_clock, "clock_domain_id", None)

    methods = {
        sample.attributes.get("hybrid.alignment_method")
        for sample in samples
        if isinstance(sample.attributes.get("hybrid.alignment_method"), str)
    }
    uncertainties = [
        sample.attributes.get("hybrid.alignment_uncertainty_ns")
        for sample in samples
        if sample.attributes.get("hybrid.alignment_uncertainty_ns") is not None
    ]
    method = next(iter(methods)) if len(methods) == 1 else None
    uncertainty: int | None = None
    if uncertainties:
        parsed = [
            _non_bool_int(value, field="hybrid.alignment_uncertainty_ns")
            for value in uncertainties
        ]
        if any(value < 0 for value in parsed):
            raise ResourceCalculationError(
                "hybrid.alignment_uncertainty_ns must be non-negative"
            )
        uncertainty = max(parsed)

    manifest = getattr(loaded, "manifest", None)
    manifest_attributes = getattr(manifest, "attributes", {})
    raw_offset = (
        manifest_attributes.get("hybrid.alignment_offset_ns")
        if isinstance(manifest_attributes, dict)
        else None
    )
    offset = (
        _non_bool_int(raw_offset, field="hybrid.alignment_offset_ns")
        if raw_offset is not None
        else None
    )
    aligned = (
        len(domains) == 1
        and domains[0] == canonical
        and method is not None
        and uncertainty is not None
    )
    return {
        "domain_ids": list(domains),
        "alignment_status": "aligned" if aligned else "unknown",
        "alignment_method": method,
        "offset_ns": offset if aligned else None,
        "uncertainty_ns": uncertainty if aligned else None,
    }


def _scope(
    sample: MetricSample, *, dimensions: str
) -> dict[str, object]:
    decoded_dimensions = json.loads(dimensions)
    window = decoded_dimensions.get("window")
    if not isinstance(window, str):
        window = None
    return {
        "run_id": sample.run_id,
        "scope_type": str(_enum_value(sample.scope)),
        "observation_layer": "normalized_resource_metric",
        "request_id": sample.request_id,
        "host_id": sample.host_id,
        "device_type": _enum_value(sample.device_type),
        "device_id": sample.device_id,
        "phase": _enum_value(sample.phase),
        "window": window,
    }


def _source(
    metric_name: str, samples: Sequence[MetricSample], *, dimensions: str
) -> dict[str, object]:
    return {
        "source_kind": "normalized_metric_stream",
        "record_ids": [],
        "metric_names": [metric_name],
        "root_id": None,
        "relative_path": "metrics/metrics.jsonl",
        "details": {
            "dimensions": dimensions,
            "sample_timestamps_ns": [sample.timestamp_ns for sample in samples],
        },
    }


def _aggregate(
    *,
    name: str,
    canonical_unit: str,
    value: int | float | None,
    reason: str | None,
    method: str,
    sample_count: int,
    source: dict[str, object],
    scope: dict[str, object],
    clock: dict[str, object],
    formula: str,
    warnings: Iterable[str] = (),
) -> dict[str, object]:
    if value is not None:
        value = _finite_number(value, field=f"{name} aggregate value")
    available = value is not None
    return {
        "name": name,
        "canonical_unit": canonical_unit,
        "availability": (
            Availability.AVAILABLE.value
            if available
            else Availability.NOT_AVAILABLE.value
        ),
        "value": value,
        "unavailable_reason": None if available else reason,
        "aggregation_method": method,
        "sample_count": sample_count,
        "sources": [source],
        "scope": scope,
        "calculation": {
            "method_id": method,
            "formula": formula,
        },
        "clock": clock,
        "quality_warnings": list(warnings),
        "display": _display_rule(canonical_unit),
    }


def _time_weighted_mean(
    samples: Sequence[MetricSample],
) -> tuple[float | None, str | None, int]:
    """Weight each sample after the first by its exact trailing interval."""

    if len(samples) < 2:
        return None, "time weighting requires at least two timestamps", 0
    numerator = 0.0
    denominator = 0
    segment_count = 0
    for previous, current in zip(samples, samples[1:]):
        delta = current.timestamp_ns - previous.timestamp_ns
        if delta <= 0:
            return None, "timestamps do not form a strictly increasing stream", 0
        if (
            isinstance(current.interval_ns, bool)
            or not isinstance(current.interval_ns, int)
            or current.interval_ns <= 0
            or current.interval_ns != delta
        ):
            return None, "intervals do not exactly tile consecutive timestamps", 0
        if _enum_value(current.availability) != Availability.AVAILABLE.value:
            return None, "an interval endpoint is unavailable", 0
        value = _finite_number(current.value, field="time-weighted sample value")
        numerator += value * current.interval_ns
        denominator += current.interval_ns
        segment_count += 1
    coverage = samples[-1].timestamp_ns - samples[0].timestamp_ns
    if denominator <= 0 or denominator != coverage:
        return None, "intervals do not exactly cover the stream", 0
    return numerator / denominator, None, segment_count


def summarize_resources(loaded: object) -> list[dict[str, object]]:
    """Group and aggregate every official ``resource.*`` metric stream."""

    metrics = tuple(getattr(loaded, "metrics", ()))
    groups: dict[
        tuple[str, str, str, str | None, str | None, str],
        list[MetricSample],
    ] = defaultdict(list)
    for metric in metrics:
        if not metric.metric_name.startswith("resource."):
            continue
        _metric_contract(metric)
        key = (
            metric.metric_name,
            metric.host_id,
            str(_enum_value(metric.scope)),
            _enum_value(metric.device_type),
            metric.device_id,
            _canonical_dimensions(metric.dimensions),
        )
        groups[key].append(metric)

    summaries: list[dict[str, object]] = []
    for key in sorted(
        groups,
        key=lambda item: tuple("" if value is None else value for value in item),
    ):
        metric_name, _, _, _, _, dimensions = key
        samples = sorted(groups[key], key=lambda sample: sample.timestamp_ns)
        timestamps = [sample.timestamp_ns for sample in samples]
        if len(timestamps) != len(set(timestamps)):
            raise ResourceCalculationError(
                f"{metric_name} has duplicate timestamps in one resource stream"
            )
        available_samples = [
            sample
            for sample in samples
            if _enum_value(sample.availability) == Availability.AVAILABLE.value
        ]
        values = [
            _finite_number(sample.value, field=f"{metric_name} value")
            for sample in available_samples
        ]
        unit = METRIC_CATALOG[metric_name].unit
        scope = _scope(samples[0], dimensions=dimensions)
        clock = _clock_evidence(loaded, samples)
        source = _source(metric_name, samples, dimensions=dimensions)
        common = {
            "canonical_unit": unit,
            "sample_count": len(values),
            "source": source,
            "scope": scope,
            "clock": clock,
        }
        if values:
            arithmetic = sum(values) / len(values)
            aggregates = [
                _aggregate(
                    name=f"{metric_name}.min",
                    value=min(values),
                    reason=None,
                    method="minimum_v1",
                    formula="min(available values)",
                    **common,
                ),
                _aggregate(
                    name=f"{metric_name}.max",
                    value=max(values),
                    reason=None,
                    method="maximum_v1",
                    formula="max(available values)",
                    **common,
                ),
                _aggregate(
                    name=f"{metric_name}.mean",
                    value=arithmetic,
                    reason=None,
                    method="arithmetic_mean_v1",
                    formula="sum(available values) / available sample count",
                    **common,
                ),
                _aggregate(
                    name=f"{metric_name}.p50",
                    value=percentile_r7(values, 0.50),
                    reason=None,
                    method="percentile_r7_v1",
                    formula="Hyndman-Fan type 7 percentile, p=0.50",
                    **common,
                ),
                _aggregate(
                    name=f"{metric_name}.p95",
                    value=percentile_r7(values, 0.95),
                    reason=None,
                    method="percentile_r7_v1",
                    formula="Hyndman-Fan type 7 percentile, p=0.95",
                    **common,
                ),
            ]
        else:
            aggregates = [
                _aggregate(
                    name=f"{metric_name}.{suffix}",
                    canonical_unit=unit,
                    value=None,
                    reason="no available samples",
                    method=method,
                    sample_count=0,
                    source=source,
                    scope=scope,
                    clock=clock,
                    formula=formula,
                )
                for suffix, method, formula in (
                    ("min", "minimum_v1", "min(available values)"),
                    ("max", "maximum_v1", "max(available values)"),
                    (
                        "mean",
                        "arithmetic_mean_v1",
                        "sum(available values) / available sample count",
                    ),
                    (
                        "p50",
                        "percentile_r7_v1",
                        "Hyndman-Fan type 7 percentile, p=0.50",
                    ),
                    (
                        "p95",
                        "percentile_r7_v1",
                        "Hyndman-Fan type 7 percentile, p=0.95",
                    ),
                )
            ]
        weighted, weighted_reason, weighted_count = _time_weighted_mean(samples)
        aggregates.append(
            _aggregate(
                name=f"{metric_name}.time_weighted_mean",
                canonical_unit=unit,
                value=weighted,
                reason=weighted_reason,
                method="trailing_interval_time_weighted_mean_v1",
                sample_count=weighted_count,
                source=source,
                scope=scope,
                clock=clock,
                formula=(
                    "sum(value[i] * interval_ns[i]) / sum(interval_ns[i]), "
                    "for i=1..n-1 with exact timestamp tiling"
                ),
                warnings=(
                    "the first sample interval is synthetic or unanchored and is excluded",
                ),
            )
        )
        total = len(samples)
        available_count = len(available_samples)
        summaries.append(
            {
                "metric_name": metric_name,
                "canonical_unit": unit,
                "scope": scope,
                "clock": clock,
                "total_sample_count": total,
                "available_sample_count": available_count,
                "unavailable_sample_count": total - available_count,
                "availability_ratio": available_count / total,
                "first_timestamp_ns": timestamps[0] if timestamps else None,
                "last_timestamp_ns": timestamps[-1] if timestamps else None,
                "coverage_ns": (
                    timestamps[-1] - timestamps[0] if timestamps else None
                ),
                "aggregates": aggregates,
                "quality_warnings": (
                    []
                    if available_count == total
                    else ["resource stream contains unavailable samples"]
                ),
            }
        )
    return summaries


__all__ = [
    "ResourceCalculationError",
    "percentile_r7",
    "summarize_resources",
]
