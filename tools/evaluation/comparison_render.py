"""Repository-only deterministic HTML rendering for Overview comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from perfetto_hetero_profiler.overview.render import (
    OverviewRenderError,
    _document,
    _kpi_value,
    _mapping,
    _sequence,
    _status,
    _table,
    _text,
)


def _comparison_metric_rows(report: Mapping[str, Any]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for metric in _sequence(report.get("metrics", [])):
        if not isinstance(metric, Mapping):
            raise OverviewRenderError("comparison metrics must be objects")
        values = {
            str(value.get("run_id")): value
            for value in _sequence(metric.get("values", []))
            if isinstance(value, Mapping)
        }
        deltas = {
            str(delta.get("run_id")): delta
            for delta in _sequence(metric.get("deltas", []))
            if isinstance(delta, Mapping)
        }
        for run_id in sorted(values):
            value = values[run_id]
            value_kpi = {
                "canonical_unit": metric.get("canonical_unit"),
                "availability": value.get("availability"),
                "value": value.get("value"),
                "unavailable_reason": value.get("unavailable_reason"),
            }
            delta = deltas.get(run_id, {})
            absolute = _mapping(delta.get("absolute"))
            percentage = _mapping(delta.get("percentage"))
            absolute_kpi = {
                "canonical_unit": metric.get("canonical_unit"),
                "availability": absolute.get("availability"),
                "value": absolute.get("value"),
                "unavailable_reason": absolute.get("unavailable_reason"),
            }
            percentage_kpi = {
                "canonical_unit": "percent",
                "availability": percentage.get("availability"),
                "value": percentage.get("value"),
                "unavailable_reason": percentage.get("unavailable_reason"),
                "display": {
                    "unit": "%",
                    "scale_numerator": 1,
                    "scale_denominator": 1,
                    "decimal_places": 3,
                    "rounding": "half_even",
                },
            }
            rows.append(
                (
                    _text(metric.get("section", metric.get("category"))),
                    _text(metric.get("observation_layer")),
                    _text(metric.get("name")),
                    _text(metric.get("direction")),
                    _text(run_id),
                    _kpi_value(value_kpi),
                    _kpi_value(absolute_kpi),
                    _kpi_value(percentage_kpi),
                )
            )
    return rows


def render_comparison_html(report: Mapping[str, Any]) -> str:
    """Render one plain Overview comparison as deterministic offline HTML."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    comparison = _mapping(report.get("comparison"))
    status = comparison.get(
        "comparability", comparison.get("status", "unknown")
    )
    reasons = sorted(
        str(item)
        for item in _sequence(
            comparison.get(
                "comparability_reasons",
                comparison.get("reasons", []),
            )
        )
    )
    header = (
        "<header>"
        "<h1>Heterogeneous profiler comparison</h1>"
        "<p><strong>Independent results dashboard.</strong> This offline "
        "HTML is not Perfetto's built-in Overview.</p>"
        f'<p class="lede">{_status(status)} · Baseline: '
        f"{_text(comparison.get('baseline_run_id'))}</p>"
        "<p>Direction metadata describes the KPI convention only. This report "
        "does not infer a general performance ranking.</p>"
        "</header>"
    )
    eligibility = (
        '<section aria-labelledby="eligibility-heading">'
        '<h2 id="eligibility-heading">Comparison eligibility</h2>'
        f"<p>{_status(status)}</p>"
        + (
            "<ul>"
            + "".join(f"<li>{_text(reason)}</li>" for reason in reasons)
            + "</ul>"
            if reasons
            else '<p class="muted">No reason was supplied.</p>'
        )
        + "</section>"
    )
    run_rows: list[tuple[str, ...]] = []
    for run in _sequence(report.get("runs", [])):
        if not isinstance(run, Mapping):
            raise OverviewRenderError("comparison runs must be objects")
        run_rows.append(
            (
                _text(run.get("run_id")),
                _text(run.get("run_mode")),
                _text(run.get("profile_mode")),
                _text(run.get("profile_kind", run.get("profiler_kind"))),
                _text(run.get("request_sample_count")),
                _text(run.get("clock_alignment_status")),
                _status(
                    "valid"
                    if run.get("source_integrity_valid") is True
                    else "invalid"
                ),
            )
        )
    runs_section = (
        '<section aria-labelledby="runs-heading">'
        '<h2 id="runs-heading">Compared runs</h2>'
        + _table(
            "Run identity and evidence",
            (
                "Run",
                "Mode",
                "Profile mode",
                "Profiler kind",
                "Requests",
                "Clock alignment",
                "Source integrity",
            ),
            run_rows,
        )
        + "</section>"
    )
    metrics_section = (
        '<section aria-labelledby="metrics-heading">'
        '<h2 id="metrics-heading">KPI values and baseline deltas</h2>'
        "<p>Absolute and percentage deltas are shown only when both values are "
        "available and the baseline is non-zero. Latency direction is lower; "
        "throughput direction is higher. No ranking is inferred.</p>"
        + _table(
            "Comparison KPI values",
            (
                "Category",
                "Observation layer",
                "KPI",
                "Direction metadata",
                "Run",
                "Value",
                "Absolute delta",
                "Percentage delta",
            ),
            _comparison_metric_rows(report),
        )
        + "</section>"
    )
    limitations = sorted(
        str(item) for item in _sequence(report.get("limitations", []))
    )
    limitation_section = (
        '<section aria-labelledby="limitations-heading">'
        '<h2 id="limitations-heading">Interpretation cautions</h2>'
        + (
            "<ul>"
            + "".join(f"<li>{_text(item)}</li>" for item in limitations)
            + "</ul>"
            if limitations
            else '<p class="muted">No additional caution was supplied.</p>'
        )
        + "</section>"
    )
    return _document(
        "Overview comparison",
        header
        + eligibility
        + runs_section
        + metrics_section
        + limitation_section,
    )


__all__ = ["render_comparison_html"]
