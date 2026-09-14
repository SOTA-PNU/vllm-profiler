"""Deterministic aggregation of normalized resource metric streams.

Each stream is summarized capture-wide and per marker-proven stage window; the
two never mix.  Interval gauges must exactly tile a window to count as covered,
point gauges use only samples timestamped inside it, and any shortfall stays
unavailable with an explicit reason.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import stat

from ..schema import Availability
from ..schema.catalog import (
    INTERVAL_RESOURCE_METRICS,
    METRIC_CATALOG,
    display_rule,
)
from ..schema.records import MetricSample
from ..support.files import sha256_file


_METRIC_STREAM_PATH = "metrics/metrics.jsonl"
_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
_PARTIAL_COVERAGE = "partial stage telemetry coverage"
_NO_OVERLAP = "no resource sample overlaps canonical stage window"
_NO_AVAILABLE_SAMPLES = "no available samples"
_POINT_GAUGE_WARNING = (
    "point-in-time gauge uses only samples timestamped inside the stage; "
    "no hold or interpolation"
)
_OVERLAP_WEIGHTED_FORMULA = (
    "sum(value[i] * overlap_ns[i]) / sum(overlap_ns[i]) over "
    "the canonical stage window"
)
_TIME_WEIGHTED_FORMULA = (
    "sum(value[i] * interval_ns[i]) / sum(interval_ns[i]), "
    "for i=1..n-1 with exact timestamp tiling"
)
_FIRST_INTERVAL_WARNING = (
    "the first sample interval is synthetic or unanchored and is excluded"
)

_TIME_WEIGHTED_METHOD = "trailing_interval_time_weighted_mean_v1"


def _statistics(
    *,
    order_noun: str,
    percentile_noun: str,
    mean: tuple[str, str],
    time_weighted: tuple[str, str] | None,
) -> dict[str, tuple[str, str]]:
    """Build one context's statistic contract: suffix -> (method_id, formula).

    Order statistics differ only in wording; the means differ substantively.
    """

    table = {
        "min": ("minimum_v1", f"min({order_noun})"),
        "max": ("maximum_v1", f"max({order_noun})"),
        "mean": mean,
        "p50": (
            "percentile_r7_v1",
            f"Hyndman-Fan type 7 percentile{percentile_noun}, p=0.50",
        ),
        "p95": (
            "percentile_r7_v1",
            f"Hyndman-Fan type 7 percentile{percentile_noun}, p=0.95",
        ),
    }
    if time_weighted is not None:
        table["time_weighted_mean"] = time_weighted
    return table


_CAPTURE_STATISTICS = _statistics(
    order_noun="available values",
    percentile_noun="",
    mean=("arithmetic_mean_v1", "sum(available values) / available sample count"),
    time_weighted=None,
)
_INTERVAL_STAGE_STATISTICS = _statistics(
    order_noun="valid values overlapping the canonical stage window",
    percentile_noun=" of valid stage samples",
    mean=("trailing_interval_overlap_weighted_mean_v1", _OVERLAP_WEIGHTED_FORMULA),
    time_weighted=(_TIME_WEIGHTED_METHOD, _OVERLAP_WEIGHTED_FORMULA),
)
_POINT_STAGE_STATISTICS = _statistics(
    order_noun="point samples timestamped inside the canonical stage window",
    percentile_noun=" of stage point samples",
    mean=(
        "arithmetic_mean_v1",
        "sum(valid point samples inside the stage) / valid sample count",
    ),
    time_weighted=(
        _TIME_WEIGHTED_METHOD,
        "not applicable to a point-in-time gauge without interpolation",
    ),
)


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


@dataclass(slots=True)
class _StageCoverage:
    """How much of one stage window a resource stream actually covers."""

    selected: list[MetricSample] = field(default_factory=list)
    values: list[int | float] = field(default_factory=list)
    covered_duration_ns: int | None = None
    coverage_ratio: float | None = None
    max_interval_ns: int | None = None
    weighted_mean: float | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)


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


def _is_available(sample: MetricSample) -> bool:
    return _enum_value(sample.availability) == Availability.AVAILABLE.value


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
    return float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))


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
        if not _is_available(current):
            return None, "an interval endpoint is unavailable", 0
        value = _finite_number(current.value, field="time-weighted sample value")
        numerator += value * current.interval_ns
        denominator += current.interval_ns
        segment_count += 1
    coverage = samples[-1].timestamp_ns - samples[0].timestamp_ns
    if denominator <= 0 or denominator != coverage:
        return None, "intervals do not exactly cover the stream", 0
    return numerator / denominator, None, segment_count


def _union_duration_ns(segments: Sequence[tuple[int, int]]) -> tuple[int, bool]:
    """Return covered duration and whether any two segments actually overlap."""

    ordered = sorted(segments)
    union_start, union_end = ordered[0]
    covered = 0
    overlapped = False
    for start_ns, end_ns in ordered[1:]:
        if start_ns < union_end:
            overlapped = True
        if start_ns <= union_end:
            union_end = max(union_end, end_ns)
        else:
            covered += union_end - union_start
            union_start, union_end = start_ns, end_ns
    return covered + union_end - union_start, overlapped


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
        "calculation": {"method_id": method, "formula": formula},
        "clock": clock,
        "quality_warnings": list(warnings),
        "display": display_rule(canonical_unit),
    }


# The order statistics every aggregation context computes identically.
_ORDER_STATISTICS = {
    "min": min,
    "max": max,
    "p50": lambda values: percentile_r7(values, 0.50),
    "p95": lambda values: percentile_r7(values, 0.95),
}


def _build_aggregates(
    *,
    metric_name: str,
    unit: str,
    values: Sequence[int | float],
    statistics: dict[str, tuple[str, str]],
    mean: int | float | None,
    time_weighted: int | float | None,
    available: bool,
    reason: str | None,
    source: dict[str, object],
    scope: dict[str, object],
    clock: dict[str, object],
    warnings: Sequence[str] = (),
    zero_count_when_unavailable: bool = False,
) -> list[dict[str, object]]:
    """Emit one aggregate per contracted statistic for a single context.

    ``zero_count_when_unavailable`` keeps the point-gauge contract (no samples
    once the stage is unavailable) distinct from the interval contract.
    """

    result: list[dict[str, object]] = []
    for suffix, (method, formula) in statistics.items():
        if suffix == "mean":
            value = mean if available else None
        elif suffix == "time_weighted_mean":
            value = time_weighted if available else None
        else:
            value = _ORDER_STATISTICS[suffix](values) if available else None
        # An available point gauge still has no interval weighting.
        point_gauge_gap = (
            available and suffix == "time_weighted_mean" and time_weighted is None
        )
        result.append(
            _aggregate(
                name=f"{metric_name}.{suffix}",
                canonical_unit=unit,
                value=value,
                reason=(
                    "point-in-time gauge does not support interval weighting"
                    if point_gauge_gap
                    else None
                    if available
                    else reason
                ),
                method=method,
                sample_count=(
                    0
                    if point_gauge_gap
                    or (zero_count_when_unavailable and not available)
                    else len(values)
                ),
                source=source,
                scope=scope,
                clock=clock,
                formula=formula,
                warnings=warnings,
            )
        )
    return result


def _clock_evidence(
    loaded: object, samples: Sequence[MetricSample]
) -> dict[str, object]:
    domains = tuple(sorted({sample.clock_domain_id for sample in samples}))
    canonical = getattr(loaded, "canonical_clock_domain_id", None)
    if canonical is None:
        canonical = getattr(
            getattr(loaded, "canonical_clock", None), "clock_domain_id", None
        )
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
    manifest_attributes = getattr(
        getattr(loaded, "manifest", None), "attributes", {}
    )
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


def _scope(sample: MetricSample, *, dimensions: str) -> dict[str, object]:
    window = json.loads(dimensions).get("window")
    return {
        "run_id": sample.run_id,
        "scope_type": str(_enum_value(sample.scope)),
        "observation_layer": "normalized_resource_metric",
        "request_id": sample.request_id,
        "host_id": sample.host_id,
        "device_type": _enum_value(sample.device_type),
        "device_id": sample.device_id,
        "phase": _enum_value(sample.phase),
        "window": window if isinstance(window, str) else None,
    }


def _metric_stream_reference(loaded: object) -> dict[str, object]:
    """Hash the canonical metric stream; the only filesystem read here."""

    root = getattr(loaded, "root", None)
    if root is None:
        return {
            "root_id": None,
            "artifact_size_bytes": None,
            "artifact_sha256": None,
        }
    path = Path(root) / _METRIC_STREAM_PATH
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ResourceCalculationError(
                "canonical metric stream must be a real regular file"
            )
        digest = sha256_file(path)
        after = path.lstat()
    except OSError as error:
        raise ResourceCalculationError(
            "canonical metric stream cannot be read for provenance"
        ) from error
    if any(
        getattr(before, name) != getattr(after, name) for name in _IDENTITY_FIELDS
    ):
        raise ResourceCalculationError(
            "canonical metric stream changed during provenance hashing"
        )
    root_id = None
    for fingerprint in getattr(loaded, "root_fingerprints", ()):
        if Path(fingerprint.root) == Path(root):
            root_id = fingerprint.root_id
            break
    return {
        "root_id": root_id,
        "artifact_size_bytes": after.st_size,
        "artifact_sha256": digest,
    }


def _source(
    metric_name: str,
    samples: Sequence[MetricSample],
    *,
    dimensions: str,
    stream_reference: dict[str, object],
) -> dict[str, object]:
    return {
        "source_kind": "normalized_metric_stream",
        "record_ids": [],
        "metric_names": [metric_name],
        "root_id": stream_reference["root_id"],
        "relative_path": _METRIC_STREAM_PATH,
        "details": {
            "dimensions": dimensions,
            "artifact_size_bytes": stream_reference["artifact_size_bytes"],
            "artifact_sha256": stream_reference["artifact_sha256"],
            "timestamp_evidence": (
                "reconstruct_from_normalized_metric_stream_timestamp_ns"
            ),
            "stream_sample_count": len(samples),
        },
    }


def _stage_coverage(
    samples: Sequence[MetricSample],
    *,
    metric_name: str,
    window: StageWindow,
    clock: dict[str, object],
    interval_metric: bool,
) -> _StageCoverage:
    """Decide whether this stream can describe the stage at all, then measure."""

    if not window.valid:
        coverage = _StageCoverage(reason="no valid canonical stage window")
        if window.unavailable_reason:
            coverage.warnings.append(window.unavailable_reason)
        return coverage
    if (
        clock["alignment_status"] != "aligned"
        or window.clock_domain_id not in clock["domain_ids"]
    ):
        return _StageCoverage(
            reason="no verified common clock for stage resource aggregation"
        )
    if window.host_ids and samples[0].host_id not in window.host_ids:
        return _StageCoverage(
            reason="no verified same-host marker window for resource stream"
        )
    assert window.start_ns is not None and window.end_ns is not None
    timestamps = [sample.timestamp_ns for sample in samples]
    if not all(
        current > previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        return _StageCoverage(
            reason=_PARTIAL_COVERAGE,
            warnings=["resource stream timestamps are not strictly increasing"],
        )
    if interval_metric:
        # Overlap-weight trailing intervals that exactly tile the window.
        stage_duration_ns = window.end_ns - window.start_ns
        coverage = _StageCoverage()
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
            implied_overlap = max(
                0,
                min(sample.timestamp_ns, window.end_ns)
                - max(previous.timestamp_ns, window.start_ns),
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
            overlap_start = max(sample.timestamp_ns - interval, window.start_ns)
            overlap_end = min(sample.timestamp_ns, window.end_ns)
            overlap_ns = max(0, overlap_end - overlap_start)
            if overlap_ns <= 0:
                continue
            overlap_seen = True
            coverage.selected.append(sample)
            coverage.max_interval_ns = max(coverage.max_interval_ns or 0, interval)
            if not _is_available(sample):
                continue
            value = _finite_number(sample.value, field=f"{metric_name} stage value")
            coverage.values.append(value)
            numerator += value * overlap_ns
            available_segments.append((overlap_start, overlap_end))

        if available_segments:
            covered, overlapped = _union_duration_ns(available_segments)
            coverage.covered_duration_ns = covered
            if overlapped:
                interval_problem = True
                coverage.warnings.append("resource sample intervals overlap")
        else:
            coverage.covered_duration_ns = 0
        coverage.coverage_ratio = coverage.covered_duration_ns / stage_duration_ns
        if coverage.max_interval_ns is not None and coverage.max_interval_ns > stage_duration_ns:
            coverage.warnings.append(
                "sampling interval exceeds stage duration; value is not stage-exclusive"
            )
        if interval_problem:
            coverage.reason = _PARTIAL_COVERAGE
            coverage.warnings.append("resource intervals do not exactly tile timestamps")
        elif not overlap_seen:
            coverage.reason = _NO_OVERLAP
        elif coverage.covered_duration_ns != stage_duration_ns or not coverage.values:
            coverage.reason = _PARTIAL_COVERAGE
        else:
            coverage.weighted_mean = numerator / coverage.covered_duration_ns
        return coverage
    # Point gauge: only samples timestamped inside the stage window.
    assert window.start_ns is not None and window.end_ns is not None
    coverage = _StageCoverage()
    coverage.selected = [
        sample
        for sample in samples
        if window.start_ns <= sample.timestamp_ns <= window.end_ns
    ]
    available_samples = [
        sample for sample in coverage.selected if _is_available(sample)
    ]
    coverage.values = [
        _finite_number(sample.value, field=f"{metric_name} stage value")
        for sample in available_samples
    ]
    if not coverage.selected:
        coverage.reason = _NO_OVERLAP
    elif len(available_samples) != len(coverage.selected) or not coverage.values:
        coverage.reason = _PARTIAL_COVERAGE
    coverage.warnings.append(_POINT_GAUGE_WARNING)
    return coverage


def _stage_source(
    metric_name: str,
    samples: Sequence[MetricSample],
    *,
    coverage: _StageCoverage,
    dimensions: str,
    window: StageWindow,
    method: str,
    stream_reference: dict[str, object],
) -> dict[str, object]:
    source = _source(
        metric_name,
        samples,
        dimensions=dimensions,
        stream_reference=stream_reference,
    )
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
            "covered_duration_ns": coverage.covered_duration_ns,
            "coverage_ratio": coverage.coverage_ratio,
            "max_interval_ns": coverage.max_interval_ns,
            "coverage_method": method,
            "source_marker_event_ids": list(window.marker_event_ids),
            "stream_first_timestamp_ns": (
                min(sample.timestamp_ns for sample in samples) if samples else None
            ),
            "stream_last_timestamp_ns": (
                max(sample.timestamp_ns for sample in samples) if samples else None
            ),
            "contributing_sample_timestamps_ns": [
                sample.timestamp_ns for sample in coverage.selected
            ],
        }
    )
    return source


def _stage_summary(
    loaded: object,
    *,
    metric_name: str,
    samples: Sequence[MetricSample],
    dimensions: str,
    window: StageWindow,
    stream_reference: dict[str, object],
) -> dict[str, object]:
    unit = METRIC_CATALOG[metric_name].unit
    scope = _scope(samples[0], dimensions=dimensions)
    scope.update(
        {
            "request_id": window.request_id,
            "phase": window.phase,
            "window": window.window,
        }
    )
    clock = _clock_evidence(loaded, samples)
    interval_metric = metric_name in INTERVAL_RESOURCE_METRICS
    coverage = _stage_coverage(
        samples,
        metric_name=metric_name,
        window=window,
        clock=clock,
        interval_metric=interval_metric,
    )
    available = coverage.reason is None
    source = _stage_source(
        metric_name,
        samples,
        coverage=coverage,
        dimensions=dimensions,
        window=window,
        method=(
            "trailing_interval_overlap_v1"
            if interval_metric
            else "point_timestamp_inside_stage_v1"
        ),
        stream_reference=stream_reference,
    )
    aggregates = _build_aggregates(
        metric_name=metric_name,
        unit=unit,
        values=coverage.values,
        statistics=(
            _INTERVAL_STAGE_STATISTICS if interval_metric else _POINT_STAGE_STATISTICS
        ),
        mean=(
            coverage.weighted_mean
            if interval_metric
            else (math.fsum(coverage.values) / len(coverage.values) if available else None)
        ),
        time_weighted=coverage.weighted_mean if interval_metric else None,
        available=available,
        reason=coverage.reason,
        source=source,
        scope=scope,
        clock=clock,
        warnings=tuple(sorted(set(coverage.warnings))),
        zero_count_when_unavailable=not interval_metric,
    )
    timestamps = [sample.timestamp_ns for sample in coverage.selected]
    total = len(coverage.selected)
    available_count = sum(_is_available(sample) for sample in coverage.selected)
    return {
        "metric_name": metric_name,
        "canonical_unit": unit,
        "scope": scope,
        "clock": clock,
        "total_sample_count": total,
        "available_sample_count": available_count,
        "unavailable_sample_count": total - available_count,
        "availability_ratio": available_count / total if total else 0.0,
        "first_timestamp_ns": min(timestamps) if timestamps else None,
        "last_timestamp_ns": max(timestamps) if timestamps else None,
        "coverage_ns": (max(timestamps) - min(timestamps) if timestamps else None),
        "aggregates": aggregates,
        "quality_warnings": sorted(set(coverage.warnings)),
    }


def summarize_resources(
    loaded: object,
    *,
    stage_windows: Sequence[StageWindow] = (),
) -> list[dict[str, object]]:
    """Aggregate capture-wide streams and marker-proven stage windows."""

    stream_reference = _metric_stream_reference(loaded)
    # Group validated resource samples by their full stream identity.
    groups: dict[
        tuple[str, str, str, str | None, str | None, str], list[MetricSample]
    ] = defaultdict(list)
    for metric in tuple(getattr(loaded, "metrics", ())):
        if not metric.metric_name.startswith("resource."):
            continue
        _metric_contract(metric)
        groups[
            (
                metric.metric_name,
                metric.host_id,
                str(_enum_value(metric.scope)),
                _enum_value(metric.device_type),
                metric.device_id,
                _canonical_dimensions(metric.dimensions),
            )
        ].append(metric)

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
        available_samples = [sample for sample in samples if _is_available(sample)]
        values = [
            _finite_number(sample.value, field=f"{metric_name} value")
            for sample in available_samples
        ]
        unit = METRIC_CATALOG[metric_name].unit
        scope = _scope(samples[0], dimensions=dimensions)
        clock = _clock_evidence(loaded, samples)
        source = _source(
            metric_name,
            samples,
            dimensions=dimensions,
            stream_reference=stream_reference,
        )
        aggregates = _build_aggregates(
            metric_name=metric_name,
            unit=unit,
            values=values,
            statistics=_CAPTURE_STATISTICS,
            mean=(sum(values) / len(values)) if values else None,
            time_weighted=None,
            available=bool(values),
            reason=_NO_AVAILABLE_SAMPLES,
            source=source,
            scope=scope,
            clock=clock,
        )
        weighted, weighted_reason, weighted_count = _time_weighted_mean(samples)
        aggregates.append(
            _aggregate(
                name=f"{metric_name}.time_weighted_mean",
                canonical_unit=unit,
                value=weighted,
                reason=weighted_reason,
                method=_TIME_WEIGHTED_METHOD,
                sample_count=weighted_count,
                source=source,
                scope=scope,
                clock=clock,
                formula=_TIME_WEIGHTED_FORMULA,
                warnings=(_FIRST_INTERVAL_WARNING,),
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
            summaries.extend(
                _stage_summary(
                    loaded,
                    metric_name=metric_name,
                    samples=raw_stream,
                    dimensions=dimensions,
                    window=window,
                    stream_reference=stream_reference,
                )
                for window in stage_windows
            )
    return summaries


__all__ = [
    "ResourceCalculationError",
    "StageWindow",
    "percentile_r7",
    "summarize_resources",
]
