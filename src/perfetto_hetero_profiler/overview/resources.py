"""Deterministic aggregation of normalized resource metric streams."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..schema import Availability
from ..schema.catalog import METRIC_CATALOG
from ..schema.catalog import INTERVAL_RESOURCE_METRICS, display_rule
from ..schema.records import MetricSample


class ResourceCalculationError(ValueError):
    """Raised when a normalized resource stream violates its contract."""


@dataclass(frozen=True, slots=True)
class StageWindow:
    """One marker-proven canonical interval used for resource aggregation."""

    phase: str
    window: str
    request_id: str | None
    start_ns: int | None
    end_ns: int | None
    clock_domain_id: str | None
    host_ids: tuple[str, ...]
    marker_event_ids: tuple[str, ...]
    unavailable_reason: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.unavailable_reason is None
            and self.start_ns is not None
            and self.end_ns is not None
            and self.end_ns > self.start_ns
            and self.clock_domain_id is not None
            and self.request_id is not None
        )


_INTERVAL_RESOURCE_METRICS = INTERVAL_RESOURCE_METRICS


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
        "display": display_rule(canonical_unit),
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


def _stage_scope(
    sample: MetricSample,
    *,
    dimensions: str,
    window: StageWindow,
) -> dict[str, object]:
    scope = _scope(sample, dimensions=dimensions)
    scope.update(
        {
            "request_id": window.request_id,
            "phase": window.phase,
            "window": window.window,
        }
    )
    return scope


def _stage_source(
    metric_name: str,
    samples: Sequence[MetricSample],
    *,
    contributing_samples: Sequence[MetricSample],
    dimensions: str,
    window: StageWindow,
    covered_duration_ns: int | None,
    coverage_ratio: float | None,
    max_interval_ns: int | None,
    method: str,
) -> dict[str, object]:
    source = _source(metric_name, samples, dimensions=dimensions)
    details = source["details"]
    assert isinstance(details, dict)
    details.update(
        {
            "aggregation_scope": "canonical_stage_window",
            "stage": window.phase,
            "window": window.window,
            "stage_start_ns": window.start_ns,
            "stage_end_ns": window.end_ns,
            "stage_duration_ns": (
                window.end_ns - window.start_ns
                if window.start_ns is not None and window.end_ns is not None
                else None
            ),
            "covered_duration_ns": covered_duration_ns,
            "coverage_ratio": coverage_ratio,
            "max_interval_ns": max_interval_ns,
            "coverage_method": method,
            "source_marker_event_ids": list(window.marker_event_ids),
            "stream_first_timestamp_ns": (
                min(sample.timestamp_ns for sample in samples) if samples else None
            ),
            "stream_last_timestamp_ns": (
                max(sample.timestamp_ns for sample in samples) if samples else None
            ),
            "contributing_sample_timestamps_ns": [
                sample.timestamp_ns for sample in contributing_samples
            ],
        }
    )
    return source


def _stage_aggregates(
    *,
    metric_name: str,
    unit: str,
    values: Sequence[int | float],
    weighted_mean: float | None,
    available: bool,
    reason: str | None,
    source: dict[str, object],
    scope: dict[str, object],
    clock: dict[str, object],
    warnings: Sequence[str],
) -> list[dict[str, object]]:
    sample_count = len(values)
    statistics: dict[str, tuple[int | float | None, str, str]] = {
        "min": (
            min(values) if available else None,
            "minimum_v1",
            "min(valid values overlapping the canonical stage window)",
        ),
        "max": (
            max(values) if available else None,
            "maximum_v1",
            "max(valid values overlapping the canonical stage window)",
        ),
        "mean": (
            weighted_mean if available else None,
            "trailing_interval_overlap_weighted_mean_v1",
            (
                "sum(value[i] * overlap_ns[i]) / sum(overlap_ns[i]) over "
                "the canonical stage window"
            ),
        ),
        "p50": (
            percentile_r7(values, 0.50) if available else None,
            "percentile_r7_v1",
            "Hyndman-Fan type 7 percentile of valid stage samples, p=0.50",
        ),
        "p95": (
            percentile_r7(values, 0.95) if available else None,
            "percentile_r7_v1",
            "Hyndman-Fan type 7 percentile of valid stage samples, p=0.95",
        ),
        "time_weighted_mean": (
            weighted_mean if available else None,
            "trailing_interval_time_weighted_mean_v1",
            (
                "sum(value[i] * overlap_ns[i]) / sum(overlap_ns[i]) over "
                "the canonical stage window"
            ),
        ),
    }
    return [
        _aggregate(
            name=f"{metric_name}.{suffix}",
            canonical_unit=unit,
            value=value,
            reason=None if available else reason,
            method=method,
            sample_count=sample_count,
            source=source,
            scope=scope,
            clock=clock,
            formula=formula,
            warnings=warnings,
        )
        for suffix, (value, method, formula) in statistics.items()
    ]


def _point_stage_aggregates(
    *,
    metric_name: str,
    unit: str,
    values: Sequence[int | float],
    available: bool,
    reason: str | None,
    source: dict[str, object],
    scope: dict[str, object],
    clock: dict[str, object],
    warnings: Sequence[str],
) -> list[dict[str, object]]:
    sample_count = len(values)
    arithmetic = math.fsum(values) / len(values) if available else None
    statistics: dict[str, tuple[int | float | None, str, str]] = {
        "min": (
            min(values) if available else None,
            "minimum_v1",
            "min(point samples timestamped inside the canonical stage window)",
        ),
        "max": (
            max(values) if available else None,
            "maximum_v1",
            "max(point samples timestamped inside the canonical stage window)",
        ),
        "mean": (
            arithmetic,
            "arithmetic_mean_v1",
            "sum(valid point samples inside the stage) / valid sample count",
        ),
        "p50": (
            percentile_r7(values, 0.50) if available else None,
            "percentile_r7_v1",
            "Hyndman-Fan type 7 percentile of stage point samples, p=0.50",
        ),
        "p95": (
            percentile_r7(values, 0.95) if available else None,
            "percentile_r7_v1",
            "Hyndman-Fan type 7 percentile of stage point samples, p=0.95",
        ),
        "time_weighted_mean": (
            None,
            "trailing_interval_time_weighted_mean_v1",
            "not applicable to a point-in-time gauge without interpolation",
        ),
    }
    result: list[dict[str, object]] = []
    for suffix, (value, method, formula) in statistics.items():
        suffix_available = available and suffix != "time_weighted_mean"
        result.append(
            _aggregate(
                name=f"{metric_name}.{suffix}",
                canonical_unit=unit,
                value=value if suffix_available else None,
                reason=(
                    None
                    if suffix_available
                    else (
                        "point-in-time gauge does not support interval weighting"
                        if available and suffix == "time_weighted_mean"
                        else reason
                    )
                ),
                method=method,
                sample_count=sample_count if suffix_available else 0,
                source=source,
                scope=scope,
                clock=clock,
                formula=formula,
                warnings=warnings,
            )
        )
    return result


def _stage_summary(
    loaded: object,
    *,
    metric_name: str,
    samples: Sequence[MetricSample],
    dimensions: str,
    window: StageWindow,
) -> dict[str, object]:
    unit = METRIC_CATALOG[metric_name].unit
    scope = _stage_scope(samples[0], dimensions=dimensions, window=window)
    clock = _clock_evidence(loaded, samples)
    warnings: list[str] = []
    reason: str | None = None
    selected: list[MetricSample] = []
    values: list[int | float] = []
    covered_duration_ns: int | None = None
    coverage_ratio: float | None = None
    max_interval_ns: int | None = None
    weighted_mean: float | None = None
    interval_metric = metric_name in _INTERVAL_RESOURCE_METRICS

    if not window.valid:
        reason = "no valid canonical stage window"
        if window.unavailable_reason:
            warnings.append(window.unavailable_reason)
    elif clock["alignment_status"] != "aligned" or window.clock_domain_id not in clock["domain_ids"]:
        reason = "no verified common clock for stage resource aggregation"
    elif window.host_ids and samples[0].host_id not in window.host_ids:
        reason = "no verified same-host marker window for resource stream"
    else:
        assert window.start_ns is not None and window.end_ns is not None
        duration_ns = window.end_ns - window.start_ns
        timestamps = [sample.timestamp_ns for sample in samples]
        monotonic = all(
            current > previous
            for previous, current in zip(timestamps, timestamps[1:])
        )
        if not monotonic:
            reason = "partial stage telemetry coverage"
            warnings.append("resource stream timestamps are not strictly increasing")
        elif interval_metric:
            numerator = 0.0
            available_segments: list[tuple[int, int]] = []
            overlap_seen = False
            interval_problem = False
            for index, sample in enumerate(samples):
                if index == 0:
                    continue
                previous = samples[index - 1]
                delta = sample.timestamp_ns - previous.timestamp_ns
                interval = sample.interval_ns
                implied_start = previous.timestamp_ns
                implied_overlap = max(
                    0,
                    min(sample.timestamp_ns, window.end_ns)
                    - max(implied_start, window.start_ns),
                )
                if (
                    isinstance(interval, bool)
                    or not isinstance(interval, int)
                    or interval <= 0
                    or interval != delta
                ):
                    if implied_overlap > 0:
                        overlap_seen = True
                        interval_problem = True
                    continue
                sample_start = sample.timestamp_ns - interval
                overlap_start = max(sample_start, window.start_ns)
                overlap_end = min(sample.timestamp_ns, window.end_ns)
                overlap_ns = max(0, overlap_end - overlap_start)
                if overlap_ns <= 0:
                    continue
                overlap_seen = True
                selected.append(sample)
                max_interval_ns = max(max_interval_ns or 0, interval)
                if _enum_value(sample.availability) != Availability.AVAILABLE.value:
                    continue
                value = _finite_number(
                    sample.value, field=f"{metric_name} stage value"
                )
                values.append(value)
                numerator += value * overlap_ns
                available_segments.append((overlap_start, overlap_end))

            if available_segments:
                available_segments.sort()
                union_start, union_end = available_segments[0]
                covered_duration_ns = 0
                overlap_problem = False
                for start_ns, end_ns in available_segments[1:]:
                    if start_ns < union_end:
                        overlap_problem = True
                    if start_ns <= union_end:
                        union_end = max(union_end, end_ns)
                    else:
                        covered_duration_ns += union_end - union_start
                        union_start, union_end = start_ns, end_ns
                covered_duration_ns += union_end - union_start
                if overlap_problem:
                    interval_problem = True
                    warnings.append("resource sample intervals overlap")
            else:
                covered_duration_ns = 0
            coverage_ratio = covered_duration_ns / duration_ns
            if max_interval_ns is not None and max_interval_ns > duration_ns:
                warnings.append(
                    "sampling interval exceeds stage duration; value is not stage-exclusive"
                )
            if interval_problem:
                reason = "partial stage telemetry coverage"
                warnings.append("resource intervals do not exactly tile timestamps")
            elif not overlap_seen:
                reason = "no resource sample overlaps canonical stage window"
            elif covered_duration_ns != duration_ns:
                reason = "partial stage telemetry coverage"
            elif not values:
                reason = "partial stage telemetry coverage"
            else:
                weighted_mean = numerator / covered_duration_ns
        else:
            selected = [
                sample
                for sample in samples
                if window.start_ns <= sample.timestamp_ns <= window.end_ns
            ]
            available_samples = [
                sample
                for sample in selected
                if _enum_value(sample.availability) == Availability.AVAILABLE.value
            ]
            values = [
                _finite_number(sample.value, field=f"{metric_name} stage value")
                for sample in available_samples
            ]
            if not selected:
                reason = "no resource sample overlaps canonical stage window"
            elif len(available_samples) != len(selected):
                reason = "partial stage telemetry coverage"
            elif not values:
                reason = "partial stage telemetry coverage"
            coverage_ratio = None
            covered_duration_ns = None
            warnings.append(
                "point-in-time gauge uses only samples timestamped inside the stage; no hold or interpolation"
            )

    available = reason is None
    selected_timestamps = [sample.timestamp_ns for sample in selected]
    source = _stage_source(
        metric_name,
        samples,
        contributing_samples=selected,
        dimensions=dimensions,
        window=window,
        covered_duration_ns=covered_duration_ns,
        coverage_ratio=coverage_ratio,
        max_interval_ns=max_interval_ns,
        method=(
            "trailing_interval_overlap_v1"
            if interval_metric
            else "point_timestamp_inside_stage_v1"
        ),
    )
    if interval_metric:
        aggregates = _stage_aggregates(
            metric_name=metric_name,
            unit=unit,
            values=values,
            weighted_mean=weighted_mean,
            available=available,
            reason=reason,
            source=source,
            scope=scope,
            clock=clock,
            warnings=tuple(sorted(set(warnings))),
        )
    else:
        aggregates = _point_stage_aggregates(
            metric_name=metric_name,
            unit=unit,
            values=values,
            available=available,
            reason=reason,
            source=source,
            scope=scope,
            clock=clock,
            warnings=tuple(sorted(set(warnings))),
        )
    total = len(selected)
    available_count = sum(
        _enum_value(sample.availability) == Availability.AVAILABLE.value
        for sample in selected
    )
    return {
        "metric_name": metric_name,
        "canonical_unit": unit,
        "scope": scope,
        "clock": clock,
        "total_sample_count": total,
        "available_sample_count": available_count,
        "unavailable_sample_count": total - available_count,
        "availability_ratio": available_count / total if total else 0.0,
        "first_timestamp_ns": min(selected_timestamps) if selected_timestamps else None,
        "last_timestamp_ns": max(selected_timestamps) if selected_timestamps else None,
        "coverage_ns": (
            max(selected_timestamps) - min(selected_timestamps)
            if selected_timestamps
            else None
        ),
        "aggregates": aggregates,
        "quality_warnings": sorted(set(warnings)),
    }


def summarize_resources(
    loaded: object,
    *,
    stage_windows: Sequence[StageWindow] = (),
) -> list[dict[str, object]]:
    """Aggregate capture-wide streams and marker-proven stage windows."""

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
        if scope["phase"] is None and scope["window"] is None:
            raw_stream = tuple(groups[key])
            for window in stage_windows:
                summaries.append(
                    _stage_summary(
                        loaded,
                        metric_name=metric_name,
                        samples=raw_stream,
                        dimensions=dimensions,
                        window=window,
                    )
                )
    return summaries


__all__ = [
    "ResourceCalculationError",
    "StageWindow",
    "percentile_r7",
    "summarize_resources",
]
