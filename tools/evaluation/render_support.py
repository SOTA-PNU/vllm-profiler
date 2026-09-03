"""Rendering primitives owned by the repository-only evaluation package."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from html import escape
from importlib import resources
import json
import math
import re
from typing import Any

from perfetto_hetero_profiler.overview.render import (
    OverviewRenderError,
    validate_offline_html,
)


_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; "
    "connect-src 'none'; img-src 'none'; font-src 'none'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'"
)
_URL_TEXT_RE = re.compile(r"\b(?:https?|ftp|file)\s*:[^\s<>\"']*", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9._~-])/(?:[A-Za-z0-9._~-]+/)*[A-Za-z0-9._~-]+"
)
_WINDOWS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|\\\\)[^\s<>\"']+"
)
_CONCLUSION_WORD_RE = re.compile(r"\b(?:winner|fastest|best)\b", re.IGNORECASE)
_STYLE = (
    resources.files("perfetto_hetero_profiler.overview")
    .joinpath("templates/overview.css")
    .read_text(encoding="utf-8")
)


def mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _sanitize_string(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _URL_TEXT_RE.sub("[redacted URL]", normalized)
    normalized = _WINDOWS_PATH_RE.sub("[redacted absolute path]", normalized)
    normalized = _ABSOLUTE_PATH_RE.sub("[redacted absolute path]", normalized)
    return _CONCLUSION_WORD_RE.sub("ranking conclusion", normalized)


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


def text(value: object) -> str:
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


def status(value: object) -> str:
    word = str(value) if value is not None else "unknown"
    status_class = {
        "available": "ok", "comparable": "ok", "complete": "ok",
        "fresh": "ok", "matched": "ok", "succeeded": "ok", "valid": "ok",
        "diagnostic_only": "warn", "not_available": "muted",
        "not_collected": "muted", "partial": "warn", "unknown": "warn",
        "error": "bad", "failed": "bad", "invalid": "bad",
        "not_comparable": "bad",
    }.get(word, "neutral")
    return f'<span class="status status-{status_class}">Status: {text(word)}</span>'


def table(
    caption: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    empty_message: str = "No records were supplied.",
) -> str:
    materialized = list(rows)
    head = "".join(f'<th scope="col">{text(item)}</th>' for item in headers)
    if materialized:
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in materialized
        )
    else:
        body = (
            f'<tr><td colspan="{len(headers)}" class="muted">'
            f"{text(empty_message)}</td></tr>"
        )
    return (
        '<div class="table-scroll" role="region" '
        f'aria-label="{text(caption)}"><table><caption>{text(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _display_rule(kpi: Mapping[str, Any]) -> Mapping[str, Any]:
    configured = kpi.get("display")
    if isinstance(configured, Mapping):
        return configured
    unit = str(kpi.get("canonical_unit", ""))
    defaults = {
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


def kpi_value(kpi: Mapping[str, Any]) -> str:
    if kpi.get("availability") != "available":
        reason = kpi.get("unavailable_reason") or "no reason was supplied"
        return f'<span class="unavailable">Unavailable — {text(reason)}</span>'
    rule = _display_rule(kpi)
    value = kpi.get("value")
    numerator = rule.get("scale_numerator", 1)
    denominator = rule.get("scale_denominator", 1)
    places = rule.get("decimal_places", 6)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator == 0
        or isinstance(places, bool)
        or not isinstance(places, int)
        or not 0 <= places <= 12
    ):
        raise OverviewRenderError("invalid KPI display conversion")
    try:
        converted = Decimal(str(value)) * Decimal(numerator) / Decimal(denominator)
        number = format(
            converted.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN),
            f".{places}f",
        )
    except (InvalidOperation, ValueError, ZeroDivisionError) as error:
        raise OverviewRenderError("KPI display conversion failed") from error
    return f"{text(number)} {text(rule.get('unit', ''))}".rstrip()


def document(title: str, body: str) -> str:
    html_text = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">\n'
        f"<title>{text(title)}</title>\n<style>\n{_STYLE}</style>\n</head>\n"
        f"<body>\n<main>{body}</main>\n</body>\n</html>\n"
    )
    validation = validate_offline_html(html_text)
    if not validation["valid"]:
        raise OverviewRenderError(
            "generated HTML failed offline validation: "
            + "; ".join(validation["issues"])
        )
    return html_text


__all__ = ["document", "kpi_value", "mapping", "sequence", "status", "table", "text"]
