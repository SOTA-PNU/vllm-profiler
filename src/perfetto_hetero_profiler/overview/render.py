"""Deterministic, self-contained HTML rendering for Phase 6 Overview data."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from html import escape
from html.parser import HTMLParser
from typing import Any


_CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'none'; "
    "connect-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

_FORBIDDEN_TAGS = {
    "script",
    "link",
    "iframe",
    "object",
    "embed",
    "form",
    "base",
}
_URL_ATTRIBUTES = {
    "action",
    "background",
    "cite",
    "data",
    "formaction",
    "href",
    "longdesc",
    "manifest",
    "ping",
    "poster",
    "profile",
    "src",
    "srcset",
    "usemap",
    "xlink:href",
}
_NETWORK_SCHEME_RE = re.compile(r"\b(?:https?|ftp|file)\s*:", re.IGNORECASE)
_URL_TEXT_RE = re.compile(
    r"\b(?:https?|ftp|file)\s*:[^\s<>\"']*", re.IGNORECASE
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9._~-])/"
    r"(?:[A-Za-z0-9._~-]+/)*"
    r"[A-Za-z0-9._~-]+"
)
_EXTERNAL_UI_BOUNDARY_LIMITATION = (
    "this external KPI report is not the Perfetto UI; the matching "
    "trace.pftrace contains a separate timeline Heterogeneous LLM Processing, "
    "not the built-in Overview"
)
_WINDOWS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|\\\\)[^\s<>\"']+"
)
_CSS_NETWORK_RE = re.compile(
    r"(?:url\s*\(|@\s*import\b|\b(?:https?|ftp|file)\s*:|"
    r"expression\s*\(|-moz-binding)",
    re.IGNORECASE,
)
_CONCLUSION_WORD_RE = re.compile(r"\b(?:winner|fastest|best)\b", re.IGNORECASE)
_WORKLOAD_DIGEST_FIELDS = frozenset(
    {"prompt_sha256", "request_body_sha256"}
)
_RECORDED_DIGEST_LABEL = (
    "Recorded (full SHA-256 retained in overview.json)"
)


class OverviewRenderError(ValueError):
    """Raised when plain Overview data cannot be rendered safely."""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _sanitize_string(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _URL_TEXT_RE.sub("[redacted URL]", normalized)
    normalized = _WINDOWS_PATH_RE.sub("[redacted absolute path]", normalized)
    normalized = _ABSOLUTE_PATH_RE.sub("[redacted absolute path]", normalized)
    normalized = _CONCLUSION_WORD_RE.sub("ranking conclusion", normalized)
    return normalized


def _text(value: object) -> str:
    if value is None:
        plain = "Unavailable"
    elif isinstance(value, bool):
        plain = "true" if value else "false"
    elif isinstance(value, (Mapping, list, tuple)):
        plain = json.dumps(
            _sanitized_json(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        plain = str(value)
    return escape(_sanitize_string(plain), quote=True)


def _sanitized_json(value: object) -> Any:
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Mapping):
        return {
            _sanitize_string(str(key)): _sanitized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitized_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise OverviewRenderError("HTML input must not contain NaN or Infinity")
    return value


def _status(value: object) -> str:
    word = str(value) if value is not None else "unknown"
    status_class = {
        "available": "ok",
        "comparable": "ok",
        "complete": "ok",
        "fresh": "ok",
        "matched": "ok",
        "succeeded": "ok",
        "valid": "ok",
        "diagnostic_only": "warn",
        "not_available": "muted",
        "not_collected": "muted",
        "partial": "warn",
        "unknown": "warn",
        "error": "bad",
        "failed": "bad",
        "invalid": "bad",
        "not_comparable": "bad",
    }.get(word, "neutral")
    return (
        f'<span class="status status-{status_class}">'
        f"Status: {_text(word)}</span>"
    )


def _table(
    caption: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    empty_message: str = "No records were supplied.",
) -> str:
    materialized = list(rows)
    head = "".join(f'<th scope="col">{_text(item)}</th>' for item in headers)
    if materialized:
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in materialized
        )
    else:
        body = (
            f'<tr><td colspan="{len(headers)}" class="muted">'
            f"{_text(empty_message)}</td></tr>"
        )
    return (
        '<div class="table-scroll" role="region" '
        f'aria-label="{_text(caption)}">'
        "<table>"
        f"<caption>{_text(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def _flatten(
    value: object, prefix: str = ""
) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, object]] = []
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            item = value[key]
            if isinstance(item, Mapping):
                result.extend(_flatten(item, name))
            else:
                result.append((name, item))
        return result
    return [(prefix or "value", value)]


def _definition_rows(value: object) -> list[tuple[str, str]]:
    return [(_text(key), _text(item)) for key, item in _flatten(value)]


def _public_workload(value: object) -> Mapping[str, Any]:
    """Hide stable request digests from the human-facing HTML report."""

    workload = dict(_mapping(value))
    for field in _WORKLOAD_DIGEST_FIELDS:
        if workload.get(field) is not None:
            workload[field] = _RECORDED_DIGEST_LABEL
    return workload


def _display_rule(kpi: Mapping[str, Any]) -> Mapping[str, Any]:
    configured = kpi.get("display")
    if isinstance(configured, Mapping):
        return configured
    unit = str(kpi.get("canonical_unit", ""))
    defaults: dict[str, tuple[str, int, int, int]] = {
        "ns": ("ms", 1, 1_000_000, 3),
        "bytes": ("MiB", 1, 1_048_576, 3),
        "bytes/s": ("MB/s", 1, 1_000_000, 3),
        "requests/s": ("requests/s", 1, 1, 3),
        "tokens/s": ("tokens/s", 1, 1, 3),
        "percent": ("%", 1, 1, 2),
        "ratio": ("ratio", 1, 1, 6),
        "count": ("count", 1, 1, 0),
        "W": ("W", 1, 1, 3),
    }
    display_unit, numerator, denominator, places = defaults.get(
        unit, (unit, 1, 1, 6)
    )
    return {
        "unit": display_unit,
        "scale_numerator": numerator,
        "scale_denominator": denominator,
        "decimal_places": places,
        "rounding": "half_even",
    }


def _formatted_number(value: object, rule: Mapping[str, Any]) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise OverviewRenderError("available KPI values must be finite numbers")
    numerator = rule.get("scale_numerator", 1)
    denominator = rule.get("scale_denominator", 1)
    places = rule.get("decimal_places", 6)
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator == 0
        or isinstance(places, bool)
        or not isinstance(places, int)
        or places < 0
        or places > 12
    ):
        raise OverviewRenderError("invalid KPI display conversion")
    try:
        converted = Decimal(str(value)) * Decimal(numerator) / Decimal(denominator)
        quantum = Decimal(1).scaleb(-places)
        rounded = converted.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise OverviewRenderError("KPI display conversion failed") from exc
    return format(rounded, f".{places}f")


def _kpi_value(kpi: Mapping[str, Any]) -> str:
    availability = kpi.get("availability")
    if availability != "available":
        reason = kpi.get("unavailable_reason") or "no reason was supplied"
        return f'<span class="unavailable">Unavailable — {_text(reason)}</span>'
    rule = _display_rule(kpi)
    number = _formatted_number(kpi.get("value"), rule)
    return f"{_text(number)} {_text(rule.get('unit', ''))}".rstrip()


def _iter_kpis(report: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    root = report.get("kpis", {})
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes)):
        for item in root:
            if isinstance(item, Mapping):
                yield "uncategorized", item
        return
    if not isinstance(root, Mapping):
        raise OverviewRenderError("kpis must be an object")
    for category in sorted(root, key=str):
        value = root[category]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if not isinstance(item, Mapping):
                    raise OverviewRenderError(
                        f"kpis.{category} entries must be objects"
                    )
                yield str(category), item
        elif isinstance(value, Mapping) and "name" in value:
            yield str(category), value
        elif isinstance(value, Mapping):
            for name in sorted(value, key=str):
                item = value[name]
                if not isinstance(item, Mapping):
                    raise OverviewRenderError(
                        f"kpis.{category}.{name} must be an object"
                    )
                if "name" not in item:
                    copied = dict(item)
                    copied["name"] = str(name)
                    item = copied
                yield str(category), item
        else:
            raise OverviewRenderError(
                f"kpis.{category} must contain KPI objects"
            )


def _scope_label(kpi: Mapping[str, Any]) -> str:
    scope = kpi.get("scope", {})
    if not isinstance(scope, Mapping):
        return "unspecified"
    parts = [
        scope.get("observation_layer"),
        scope.get("scope_type"),
        scope.get("device_type"),
        scope.get("device_id"),
        scope.get("window"),
    ]
    return " · ".join(str(item) for item in parts if item not in (None, ""))


def _kpi_table(caption: str, kpis: Sequence[Mapping[str, Any]]) -> str:
    rows: list[tuple[str, ...]] = []
    for kpi in kpis:
        warnings = _sequence(kpi.get("quality_warnings", []))
        rows.append(
            (
                _text(kpi.get("name", "Unnamed KPI")),
                _kpi_value(kpi),
                _status(kpi.get("availability", "unknown")),
                _text(kpi.get("sample_count", 0)),
                _text(_scope_label(kpi)),
                _text(list(warnings) if warnings else "None recorded"),
            )
        )
    return _table(
        caption,
        ("KPI", "Value", "Availability", "Samples", "Scope / layer", "Warnings"),
        rows,
        empty_message="No KPI is available for this section.",
    )


def _report_run(report: Mapping[str, Any]) -> Mapping[str, Any]:
    run = report.get("run")
    if isinstance(run, Mapping):
        return run
    return {
        "run_id": report.get("run_id"),
        "mode": report.get("run_mode"),
        "profile_mode": report.get("profile_mode"),
        "status": report.get("run_status"),
        "profiler_kind": report.get("profile_kind"),
    }


def _report_hardware(report: Mapping[str, Any]) -> Sequence[Any]:
    return _sequence(report.get("hardware", report.get("hardware_inventory", [])))


def _report_models(report: Mapping[str, Any]) -> Sequence[Any]:
    return _sequence(report.get("models", report.get("model_identity", [])))


def _resource_rows(
    report: Mapping[str, Any],
    *,
    window: str | None,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for summary in _sequence(report.get("resources", [])):
        if not isinstance(summary, Mapping):
            raise OverviewRenderError("resource summaries must be objects")
        scope = _mapping(summary.get("scope"))
        scope_window = scope.get("window")
        if window is None:
            if scope_window in {"prefill", "transfer", "decode"}:
                continue
        elif scope_window != window:
            continue
        stream = " · ".join(
            str(value)
            for value in (
                scope.get("host_id"),
                scope.get("device_type"),
                scope.get("device_id"),
                scope.get("window"),
            )
            if value not in (None, "")
        )
        sample_summary = (
            f"{summary.get('available_sample_count', 0)} available / "
            f"{summary.get('total_sample_count', 0)} total; "
            f"{summary.get('unavailable_sample_count', 0)} unavailable"
        )
        aggregates = _sequence(summary.get("aggregates", []))
        if not aggregates:
            rows.append(
                (
                    _text(summary.get("metric_name")),
                    _text(stream),
                    _text(sample_summary),
                    _text("No aggregate coverage evidence"),
                    _text("No aggregate"),
                    _text("Unavailable"),
                    _text("No aggregate"),
                )
            )
        for aggregate in aggregates:
            if not isinstance(aggregate, Mapping):
                raise OverviewRenderError("resource aggregates must be objects")
            coverage = (
                f"{summary.get('first_timestamp_ns')} … "
                f"{summary.get('last_timestamp_ns')} "
                f"(stream span {summary.get('coverage_ns')} ns)"
            )
            sources = _sequence(aggregate.get("sources", []))
            if sources and isinstance(sources[0], Mapping):
                details = _mapping(sources[0].get("details"))
                ratio = details.get("coverage_ratio")
                if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
                    coverage = (
                        f"{ratio * 100:.3f}% · "
                        f"{details.get('covered_duration_ns')} / "
                        f"{details.get('stage_duration_ns')} ns · "
                        f"max interval {details.get('max_interval_ns')} ns"
                    )
                elif details.get("coverage_method") == "point_timestamp_inside_stage_v1":
                    coverage = (
                        "Point samples inside stage; duration coverage is not applicable"
                    )
            warnings = list(_sequence(aggregate.get("quality_warnings", [])))
            reason = aggregate.get("unavailable_reason")
            if reason:
                warnings.append(reason)
            rows.append(
                (
                    _text(summary.get("metric_name")),
                    _text(stream),
                    _text(sample_summary),
                    _text(coverage),
                    _text(aggregate.get("aggregation_method")),
                    _kpi_value(aggregate),
                    _text(warnings if warnings else "None recorded"),
                )
            )
    return rows


def _provenance_rows(report: Mapping[str, Any]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    all_kpis: list[Mapping[str, Any]] = [
        kpi for _, kpi in _iter_kpis(report)
    ]
    for summary in _sequence(report.get("resources", [])):
        if isinstance(summary, Mapping):
            all_kpis.extend(
                item
                for item in _sequence(summary.get("aggregates", []))
                if isinstance(item, Mapping)
            )
    for kpi in all_kpis:
        calculation = _mapping(kpi.get("calculation"))
        clock = _mapping(kpi.get("clock"))
        sources = _sequence(kpi.get("sources", []))
        source_text = [
            {
                "source_kind": source.get("source_kind"),
                "record_ids": source.get("record_ids", []),
                "metric_names": source.get("metric_names", []),
                "root_id": source.get("root_id"),
                "relative_path": source.get("relative_path"),
            }
            for source in sources
            if isinstance(source, Mapping)
        ]
        rows.append(
            (
                _text(kpi.get("name")),
                _text(kpi.get("aggregation_method")),
                _text(calculation.get("method_id")),
                _text(calculation.get("formula")),
                _text(source_text),
                _text(clock),
            )
        )
    return rows


def _unavailable_rows(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for _, kpi in _iter_kpis(report):
        if kpi.get("availability") != "available":
            rows.append(
                (
                    _text(kpi.get("name")),
                    _text(kpi.get("unavailable_reason") or "no reason was supplied"),
                )
            )
    for summary in _sequence(report.get("resources", [])):
        if isinstance(summary, Mapping):
            for kpi in _sequence(summary.get("aggregates", [])):
                if isinstance(kpi, Mapping) and kpi.get("availability") != "available":
                    rows.append(
                        (
                            _text(kpi.get("name")),
                            _text(
                                kpi.get("unavailable_reason")
                                or "no reason was supplied"
                            ),
                        )
                    )
    return sorted(rows)


def _document(title: str, body: str) -> str:
    html_text = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta http-equiv="Content-Security-Policy" '
        f'content="{_CSP}">\n'
        f"<title>{_text(title)}</title>\n"
        "<style>\n"
        ":root { color-scheme: light dark; font-family: system-ui, sans-serif; }\n"
        "body { margin: 0; background: #f4f6f8; color: #17202a; }\n"
        "main { max-width: 1180px; margin: 0 auto; padding: 1rem; }\n"
        "header, section { background: #fff; border: 1px solid #c8d0d8; "
        "border-radius: .5rem; margin: 0 0 1rem; padding: 1rem; }\n"
        "h1, h2 { line-height: 1.25; margin-top: 0; }\n"
        "h1 { font-size: 1.65rem; } h2 { font-size: 1.25rem; }\n"
        ".lede, .muted { color: #52606d; }\n"
        ".table-scroll { overflow-x: auto; margin-top: .5rem; }\n"
        "table { border-collapse: collapse; min-width: 42rem; width: 100%; }\n"
        "caption { font-weight: 700; text-align: left; padding: .4rem 0; }\n"
        "th, td { border: 1px solid #c8d0d8; padding: .45rem .55rem; "
        "text-align: left; vertical-align: top; }\n"
        "th { background: #edf2f7; }\n"
        ".status { border: 1px solid currentColor; border-radius: 1rem; "
        "display: inline-block; font-weight: 700; padding: .1rem .55rem; }\n"
        ".status-ok { color: #176b3a; } .status-warn { color: #7a5200; }\n"
        ".status-bad { color: #a12622; } .status-muted { color: #52606d; }\n"
        ".status-neutral { color: #334e68; }\n"
        ".unavailable { color: #7a5200; font-weight: 700; }\n"
        "code { overflow-wrap: anywhere; }\n"
        "ul { padding-left: 1.3rem; }\n"
        "@media (prefers-color-scheme: dark) {\n"
        " body { background: #101820; color: #e8eef3; }\n"
        " header, section { background: #18232d; border-color: #52606d; }\n"
        " th { background: #253544; } th, td { border-color: #52606d; }\n"
        " .lede, .muted { color: #b8c5d0; }\n"
        " .status-ok { color: #78d49b; } .status-warn, .unavailable { "
        "color: #ffd166; } .status-bad { color: #ff8a80; }\n"
        " .status-muted, .status-neutral { color: #b8c5d0; }\n"
        "}\n"
        "@media (max-width: 640px) {\n"
        " main { padding: .5rem; } header, section { padding: .75rem; }\n"
        " h1 { font-size: 1.35rem; } table { font-size: .9rem; }\n"
        "}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<main>{body}</main>\n"
        "</body>\n"
        "</html>\n"
    )
    validation = validate_offline_html(html_text)
    if not validation["valid"]:
        raise OverviewRenderError(
            "generated HTML failed offline validation: "
            + "; ".join(validation["issues"])
        )
    return html_text


def render_overview_html(report: Mapping[str, Any]) -> str:
    """Render one plain Overview report as deterministic offline HTML."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    run = _report_run(report)
    run_id = run.get("run_id", "unknown-run")
    workload = _public_workload(report.get("workload"))
    kpi_groups: dict[str, list[Mapping[str, Any]]] = {}
    for category, kpi in _iter_kpis(report):
        kpi_groups.setdefault(category, []).append(kpi)

    header = (
        "<header>"
        "<h1>Heterogeneous profiler Overview</h1>"
        "<p><strong>Independent results dashboard.</strong> This offline "
        "HTML is not Perfetto's built-in Overview.</p>"
        f'<p class="lede">Run {_text(run_id)} · {_status(run.get("status"))}</p>'
        "<p>This report separates request-facing observations from canonical "
        "pipeline observations and preserves unavailable values explicitly.</p>"
        "</header>"
    )
    run_section = (
        '<section aria-labelledby="run-heading">'
        '<h2 id="run-heading">Run and workload information</h2>'
        + _table("Run identity", ("Field", "Value"), _definition_rows(run))
        + _table(
            "Workload configuration",
            ("Field", "Value"),
            _definition_rows(workload),
        )
        + _table(
            "Model identity",
            ("Entry", "Details"),
            (
                (_text(index), _text(item))
                for index, item in enumerate(_report_models(report))
            ),
            empty_message="No model identity was supplied.",
        )
        + _table(
            "Hardware inventory",
            ("Entry", "Details"),
            (
                (_text(index), _text(item))
                for index, item in enumerate(_report_hardware(report))
            ),
            empty_message="No hardware inventory was supplied.",
        )
        + "</section>"
    )
    quality_section = (
        '<section aria-labelledby="quality-heading">'
        '<h2 id="quality-heading">Status and data quality</h2>'
        + _table(
            "Data-quality evidence",
            ("Field", "Value"),
            _definition_rows(report.get("data_quality", {})),
            empty_message="No data-quality evidence was supplied.",
        )
        + "</section>"
    )

    section_specs = (
        (
            "request-facing-heading",
            "Request-facing latency",
            "request_facing_latency",
            "Request-facing latency KPIs",
        ),
        (
            "pipeline-heading",
            "Hybrid pipeline phase breakdown",
            "pipeline_latency",
            "Canonical pipeline latency KPIs",
        ),
        (
            "throughput-heading",
            "Throughput and token count",
            "throughput_and_tokens",
            "Throughput and token KPIs",
        ),
        (
            "transfer-heading",
            "Transfer KPIs",
            "transfer",
            "Transfer KPIs",
        ),
    )
    kpi_sections = "".join(
        f'<section aria-labelledby="{heading_id}">'
        f'<h2 id="{heading_id}">{_text(title)}</h2>'
        + _kpi_table(caption, kpi_groups.get(category, []))
        + "</section>"
        for heading_id, title, category, caption in section_specs
    )
    resource_tables = "".join(
        _table(
            title,
            (
                "Metric",
                "Host / device / window",
                "Samples",
                "Coverage",
                "Aggregation",
                "Value",
                "Warning / unavailable reason",
            ),
            _resource_rows(report, window=window),
            empty_message=f"No {title.lower()} is present for this run.",
        )
        for title, window in (
            ("Prefill resource", "prefill"),
            ("Transfer resource", "transfer"),
            ("Decode resource", "decode"),
            ("Capture-wide resource", None),
        )
    )
    resource_section = (
        '<section aria-labelledby="resource-heading">'
        '<h2 id="resource-heading">CPU, GPU, and NPU resources</h2>'
        "<p>Stage values use canonical marker windows and remain separated by "
        "host and device. Capture-wide values are shown separately and are never "
        "copied into a stage. Missing coverage stays unavailable.</p>"
        + resource_tables
        + "</section>"
    )
    interpretation = report.get("interpretation")
    limitations = (
        interpretation.get("limitations")
        if isinstance(interpretation, Mapping)
        else None
    )
    perfetto_boundary = (
        "<p>This HTML is an external KPI Overview report, not the Perfetto UI. "
        "The matching <code>trace.pftrace</code> is identified below by a "
        "path-free SHA-256 and contains the separate timeline "
        "<code>Heterogeneous LLM Processing</code>. This TrackEvent hierarchy is "
        "not Perfetto's built-in Overview and does not add custom cards there. "
        "Open the trace file in Perfetto UI to inspect timeline tracks, "
        "annotations, and explicit GPU-to-NPU flows.</p>"
        if isinstance(limitations, list)
        and _EXTERNAL_UI_BOUNDARY_LIMITATION in limitations
        else ""
    )
    perfetto_section = (
        '<section aria-labelledby="perfetto-heading">'
        '<h2 id="perfetto-heading">Perfetto trace information</h2>'
        + perfetto_boundary
        + _table(
            "Perfetto reconciliation evidence",
            ("Field", "Value"),
            _definition_rows(report.get("perfetto", {})),
            empty_message="No matching Perfetto evidence was supplied.",
        )
        + "</section>"
    )
    native_section = (
        '<section aria-labelledby="native-heading">'
        '<h2 id="native-heading">Native profiler policy</h2>'
        "<p>Native timestamps remain partial or unaligned unless an explicit "
        "clock transform is present. RBLN Perfetto payloads are validated as "
        "separate native traces and are not merged without a canonical "
        "anchor.</p>"
        + _table(
            "Native profiler evidence",
            ("Entry", "Details"),
            (
                (_text(index), _text(item))
                for index, item in enumerate(
                    _sequence(report.get("native_profiles", []))
                )
            ),
            empty_message="No native profiler capture applies to this run.",
        )
        + "</section>"
    )
    unavailable_section = (
        '<section aria-labelledby="unavailable-heading">'
        '<h2 id="unavailable-heading">Unavailable values and reasons</h2>'
        + _table(
            "Unavailable KPI inventory",
            ("KPI", "Reason"),
            _unavailable_rows(report),
            empty_message="Every reported KPI is available.",
        )
        + "</section>"
    )
    provenance_section = (
        '<section aria-labelledby="provenance-heading">'
        '<h2 id="provenance-heading">Provenance and calculation methods</h2>'
        + _table(
            "KPI formulas and sources",
            (
                "KPI",
                "Aggregation",
                "Method identifier",
                "Formula",
                "Sources",
                "Clock evidence",
            ),
            _provenance_rows(report),
            empty_message="No KPI provenance was supplied.",
        )
        + "</section>"
    )
    interpretation = _mapping(report.get("interpretation"))
    limitations = _sequence(interpretation.get("limitations", []))
    raw_policies = interpretation.get("policies", {})
    if isinstance(raw_policies, Mapping):
        policies = tuple(
            f"{key}: {str(raw_policies[key]).lower()}"
            for key in sorted(raw_policies, key=str)
        )
    else:
        policies = tuple(str(item) for item in _sequence(raw_policies))
    cautions = sorted({str(item) for item in (*limitations, *policies)})
    interpretation_section = (
        '<section aria-labelledby="interpretation-heading">'
        '<h2 id="interpretation-heading">Interpretation cautions</h2>'
        f"<p>Comparison scope: {_text(interpretation.get('comparison_scope', 'unspecified'))}</p>"
        + (
            "<ul>"
            + "".join(f"<li>{_text(item)}</li>" for item in cautions)
            + "</ul>"
            if cautions
            else '<p class="muted">No additional caution was supplied.</p>'
        )
        + "</section>"
    )
    return _document(
        f"Overview — {run_id}",
        (
            header
            + run_section
            + quality_section
            + kpi_sections
            + resource_section
            + perfetto_section
            + native_section
            + unavailable_section
            + provenance_section
            + interpretation_section
        ),
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
            "<ul>" + "".join(f"<li>{_text(reason)}</li>" for reason in reasons) + "</ul>"
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
    limitations = sorted(str(item) for item in _sequence(report.get("limitations", [])))
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
        header + eligibility + runs_section + metrics_section + limitation_section,
    )


class _OfflineHTMLScanner(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.issues: list[str] = []
        self.forbidden_tag_count = 0
        self.url_attribute_count = 0
        self.network_reference_count = 0
        self.absolute_path_count = 0
        self.event_handler_count = 0
        self.csp_values: list[str] = []
        self._style_depth = 0

    def _issue(self, message: str) -> None:
        self.issues.append(message)

    def _scan_text(self, value: str, *, context: str) -> None:
        matches = _NETWORK_SCHEME_RE.findall(value)
        if matches:
            self.network_reference_count += len(matches)
            self._issue(f"{context} contains a network or file scheme")
        paths = _ABSOLUTE_PATH_RE.findall(value)
        windows = _WINDOWS_PATH_RE.findall(value)
        if paths or windows:
            self.absolute_path_count += len(paths) + len(windows)
            self._issue(f"{context} contains a raw absolute path")

    def _scan_css(self, value: str) -> None:
        if _CSS_NETWORK_RE.search(value):
            self.network_reference_count += 1
            self._issue("CSS contains a URL, import, or network-capable expression")
        if "/*" in value or "\\" in value:
            self._issue("CSS comments or escapes are not allowed by the offline policy")
        self._scan_text(value, context="CSS")

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        lowered = tag.casefold()
        if lowered in _FORBIDDEN_TAGS:
            self.forbidden_tag_count += 1
            self._issue(f"forbidden HTML tag: {lowered}")
        if lowered == "style":
            self._style_depth += 1
        normalized = {
            name.casefold(): "" if value is None else value
            for name, value in attrs
        }
        for name, value in normalized.items():
            if name in _URL_ATTRIBUTES:
                self.url_attribute_count += 1
                self._issue(f"URL-bearing attribute is forbidden: {name}")
            if name.startswith("on"):
                self.event_handler_count += 1
                self._issue(f"event-handler attribute is forbidden: {name}")
            if name == "style":
                self._scan_css(value)
            else:
                self._scan_text(value, context=f"attribute {name}")
        if (
            lowered == "meta"
            and normalized.get("http-equiv", "").casefold()
            == "content-security-policy"
        ):
            self.csp_values.append(normalized.get("content", ""))
        if (
            lowered == "meta"
            and normalized.get("http-equiv", "").casefold() == "refresh"
        ):
            self._issue("meta refresh is forbidden")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() == "style":
            self._style_depth = max(0, self._style_depth - 1)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "style":
            self._style_depth = max(0, self._style_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self._scan_css(data)
        else:
            self._scan_text(data, context="text")

    def handle_comment(self, data: str) -> None:
        self._scan_text(data, context="comment")


def _validate_csp(value: str) -> list[str]:
    directives: dict[str, list[str]] = {}
    for segment in value.split(";"):
        words = segment.strip().split()
        if not words:
            continue
        name = words[0].casefold()
        if name in directives:
            return [f"duplicate CSP directive: {name}"]
        directives[name] = [word.casefold() for word in words[1:]]
    issues: list[str] = []
    for name in (
        "default-src",
        "script-src",
        "connect-src",
        "img-src",
        "font-src",
        "object-src",
        "base-uri",
        "form-action",
    ):
        if directives.get(name) != ["'none'"]:
            issues.append(f"CSP {name} must be exactly 'none'")
    if directives.get("style-src") != ["'unsafe-inline'"]:
        issues.append("CSP style-src must allow only inline static CSS")
    return issues


def validate_offline_html(html_text: str) -> dict[str, Any]:
    """Scan HTML for network-capable, active, or path-leaking constructs."""

    if not isinstance(html_text, str):
        raise TypeError("html_text must be a string")
    scanner = _OfflineHTMLScanner()
    try:
        scanner.feed(html_text)
        scanner.close()
    except Exception as exc:
        scanner._issue(f"HTML parser error: {type(exc).__name__}")
    if len(scanner.csp_values) != 1:
        scanner._issue("exactly one Content-Security-Policy meta element is required")
    else:
        scanner.issues.extend(_validate_csp(scanner.csp_values[0]))
    issues = sorted(set(scanner.issues))
    return {
        "valid": not issues,
        "issues": issues,
        "csp_present": len(scanner.csp_values) == 1,
        "forbidden_tag_count": scanner.forbidden_tag_count,
        "url_attribute_count": scanner.url_attribute_count,
        "event_handler_count": scanner.event_handler_count,
        "network_reference_count": scanner.network_reference_count,
        "absolute_path_count": scanner.absolute_path_count,
    }


__all__ = [
    "OverviewRenderError",
    "render_comparison_html",
    "render_overview_html",
    "validate_offline_html",
]
