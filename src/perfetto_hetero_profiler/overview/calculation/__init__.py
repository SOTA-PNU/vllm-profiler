"""Deterministic Overview KPI calculation from a validated normalized run.

Assembly is one thin step here; each KPI family owns its own module so that
request-facing, pipeline, throughput, and transfer semantics cannot quietly
borrow each other's evidence.
"""

from __future__ import annotations

from ..resources import ResourceCalculationError, summarize_resources
from .kpi_records import OverviewCalculationError
from .latency import (
    _canonical_stage_windows,
    _pipeline_latency,
    _request_facing_latency,
    union_duration_ns,
)
from .throughput_transfer import _throughput_and_tokens, _transfer_kpis


def calculate_overview_kpis(loaded: object) -> dict[str, object]:
    """Calculate every deterministic Overview KPI from one validated run."""

    metrics = tuple(getattr(loaded, "metrics", ()))
    events = tuple(getattr(loaded, "events", ()))
    request_facing = _request_facing_latency(loaded, metrics)
    pipeline, pairs = _pipeline_latency(loaded, metrics, events)
    stage_windows = _canonical_stage_windows(loaded, pipeline, pairs)
    throughput = _throughput_and_tokens(loaded, metrics)
    transfer = _transfer_kpis(loaded, pipeline, pairs)
    try:
        resources = summarize_resources(loaded, stage_windows=stage_windows)
    except ResourceCalculationError as exc:
        raise OverviewCalculationError(str(exc)) from exc
    return {
        "request_facing_latency": request_facing,
        "pipeline_latency": pipeline,
        "throughput_and_tokens": throughput,
        "transfer": transfer,
        "resource_summaries": resources,
    }


__all__ = [
    "OverviewCalculationError",
    "calculate_overview_kpis",
    "union_duration_ns",
]
