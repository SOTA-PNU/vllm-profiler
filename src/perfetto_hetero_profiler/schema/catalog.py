"""Ordered stage, KPI presentation, and resource contracts.

The normalized metric definitions remain in :mod:`metric_catalog`.  This
module holds the cross-product metadata that is shared by aggregation,
Perfetto planning, trace attributes, and the external Overview.  Tuples are
intentional: their order is part of the deterministic artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Phase
from .metric_catalog import METRIC_CATALOG, METRIC_DEFINITIONS, MetricDefinition


@dataclass(frozen=True, slots=True)
class StageDefinition:
    metric_name: str | None
    start_event: str
    end_event: str
    phase: Phase
    window: str
    track_key: str
    track_name: str
    slice_name: str
    description: str
    end_phase: Phase | None = None
    discriminator: str | None = None
    pipeline_order: int | None = None


STAGE_DEFINITIONS: tuple[StageDefinition, ...] = (
    StageDefinition(
        "latency.e2e", "request_received", "response_done", Phase.REQUEST,
        "request", "request", "Request lifecycle", "Request",
        "End-to-end request lifecycle on the canonical clock.",
        end_phase=Phase.RESPONSE,
    ),
    StageDefinition(
        "latency.prefill", "prefill_start", "prefill_end", Phase.PREFILL,
        "prefill", "gpu_prefill", "GPU Prefill", "GPU Prefill",
        "GPU prefill markers paired without timestamp inference.", pipeline_order=0,
    ),
    StageDefinition(
        "latency.kv_export", "kv_export_start", "kv_export_end", Phase.KV_EXPORT,
        "kv_export", "kv_export", "KV Export", "KV Export",
        "GPU KV export markers paired by explicit request identity.", pipeline_order=1,
    ),
    StageDefinition(
        "transfer.handoff_duration", "kv_handoff_start", "kv_handoff_end",
        Phase.KV_TRANSFER, "handoff", "kv_handoff", "KV Handoff", "KV Handoff",
        "Export-to-transfer handoff paired by explicit correlation identity.",
        discriminator="transfer", pipeline_order=2,
    ),
    StageDefinition(
        "transfer.setup_duration", "kv_transfer_setup_start", "kv_transfer_setup_end",
        Phase.KV_TRANSFER, "transfer_setup", "kv_transfer_setup",
        "KV Transfer Setup", "KV Transfer Setup",
        "Host-side NIXL descriptor and handle preparation.",
        discriminator="transfer", pipeline_order=3,
    ),
    StageDefinition(
        "latency.kv_transfer", "kv_transfer_start", "kv_transfer_end",
        Phase.KV_TRANSFER, "kv_transfer", "kv_transfer", "KV Transfer", "KV Transfer",
        "GPU-to-NPU KV transfer paired by explicit transfer identity.",
        discriminator="transfer", pipeline_order=4,
    ),
    StageDefinition(
        "transfer.wait_duration", "kv_transfer_wait_start", "kv_transfer_wait_end",
        Phase.KV_TRANSFER, "transfer_wait", "kv_transfer_wait", "KV Transfer Wait",
        "KV Transfer Wait", "Observed incomplete-to-done NIXL polling interval.",
        discriminator="transfer", pipeline_order=5,
    ),
    StageDefinition(
        "latency.kv_transform", "kv_transform_start", "kv_transform_end",
        Phase.KV_TRANSFORM, "kv_transform", "kv_transform", "KV Transform",
        "KV Transform", "NPU KV transform markers on the canonical clock.",
        pipeline_order=6,
    ),
    StageDefinition(
        "decode.schedule_wait_duration", "decode_schedule_wait_start",
        "decode_schedule_wait_end", Phase.DECODE, "decode_schedule_wait",
        "decode_schedule_wait", "Decode Scheduling Wait", "Decode Scheduling Wait",
        "Decode-ready to first model-step scheduling interval.",
        discriminator="transfer", pipeline_order=7,
    ),
    StageDefinition(
        "latency.decode", "decode_loop_start", "decode_loop_end", Phase.DECODE,
        "decode", "npu_decode", "NPU Decode", "NPU Decode",
        "NPU decode loop markers on the canonical clock.", pipeline_order=8,
    ),
    StageDefinition(
        None, "decode_step_start", "decode_step_end", Phase.DECODE,
        "decode_step", "npu_decode_step", "NPU Decode Step", "NPU Decode Step",
        "Ordered NPU decode steps with preserved step index.", discriminator="step",
    ),
    StageDefinition(
        "latency.sampling", "sampling_start", "sampling_end", Phase.SAMPLING,
        "sampling", "sampling", "Sampling", "Sampling",
        "Ordered sampling steps with preserved step index.", discriminator="step",
    ),
)

STAGE_BY_TRACK = {stage.track_key: stage for stage in STAGE_DEFINITIONS}
STAGE_BY_METRIC = {
    stage.metric_name: stage
    for stage in STAGE_DEFINITIONS
    if stage.metric_name is not None
}
DERIVED_LATENCY_METRICS = (
    "latency.e2e",
    "latency.prefill",
    "latency.kv_export",
    "latency.kv_transfer",
    "latency.kv_transform",
    "latency.decode",
)
PHASE_RECONCILIATION_METRICS = (*DERIVED_LATENCY_METRICS, "latency.sampling")
PIPELINE_STAGE_ORDER = {
    stage.track_key: stage.pipeline_order
    for stage in STAGE_DEFINITIONS
    if stage.pipeline_order is not None
}


@dataclass(frozen=True, slots=True)
class KpiPresentation:
    section: str
    metric_name: str
    display_name: str
    trace_attribute_key: str | None = None
    trace_value_suffix: str | None = None
    trace_multiplier: int = 1


KPI_SECTION_ORDER = (
    "request_facing_latency",
    "pipeline_latency",
    "throughput_and_tokens",
    "transfer",
)

KPI_PRESENTATIONS: tuple[KpiPresentation, ...] = (
    KpiPresentation("request_facing_latency", "latency.e2e", "Request E2E", "kpi.latency.e2e", "value_ns"),
    KpiPresentation("request_facing_latency", "latency.ttft", "TTFT", "kpi.latency.ttft", "value_ns"),
    KpiPresentation("request_facing_latency", "latency.tpot", "TPOT", "kpi.latency.tpot", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.e2e", "Pipeline E2E", "pipeline.latency.e2e", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.prefill", "Prefill latency", "kpi.latency.prefill", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.kv_export", "KV export latency", "kpi.latency.kv_export", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.kv_transfer", "KV transfer latency", "kpi.latency.kv_transfer", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.kv_transform", "KV transform latency", "kpi.latency.kv_transform", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.decode", "Decode latency", "kpi.latency.decode", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.sampling", "Sampling total", "kpi.latency.sampling", "value_ns"),
    KpiPresentation("pipeline_latency", "latency.wait", "Pipeline wait"),
    KpiPresentation("throughput_and_tokens", "request.count", "Request count", "kpi.request.count", "value_requests"),
    KpiPresentation("throughput_and_tokens", "request.input_tokens", "Input tokens", "kpi.tokens.input", "value_tokens"),
    KpiPresentation("throughput_and_tokens", "request.output_tokens", "Output tokens", "kpi.tokens.output", "value_tokens"),
    KpiPresentation("throughput_and_tokens", "request.total_tokens", "Total tokens", "kpi.tokens.total", "value_tokens"),
    KpiPresentation("throughput_and_tokens", "throughput.requests", "Requests per second", "kpi.throughput.requests", "value_requests_milli_per_second", 1_000),
    KpiPresentation("throughput_and_tokens", "throughput.input_tokens", "Input tokens per second", "kpi.throughput.input_tokens", "value_input_tokens_milli_per_second", 1_000),
    KpiPresentation("throughput_and_tokens", "throughput.output_tokens", "Output tokens per second", "kpi.throughput.output_tokens", "value_output_tokens_milli_per_second", 1_000),
    KpiPresentation("throughput_and_tokens", "throughput.total_tokens", "Total tokens per second", "kpi.throughput.total_tokens", "value_total_tokens_milli_per_second", 1_000),
    KpiPresentation("transfer", "transfer.bytes", "Transferred bytes", "transfer.bytes", "value_bytes"),
    KpiPresentation("transfer", "transfer.duration", "Transfer duration", "transfer.duration", "value_ns"),
    KpiPresentation("transfer", "transfer.effective_bandwidth", "Effective bandwidth", "transfer.effective_bandwidth", "value_bytes_per_second"),
    KpiPresentation("transfer", "transfer.transform_duration", "Transform duration", "transfer.transform_duration", "value_ns"),
    KpiPresentation("transfer", "transfer.wait_duration", "Transfer wait", "kpi.latency.kv_transfer_wait", "value_ns"),
    KpiPresentation("transfer", "transfer.handoff_duration", "KV handoff", "kpi.latency.kv_handoff", "value_ns"),
    KpiPresentation("transfer", "transfer.setup_duration", "Transfer setup", "kpi.latency.kv_transfer_setup", "value_ns"),
    KpiPresentation("transfer", "decode.schedule_wait_duration", "Decode scheduling wait", "kpi.latency.decode_schedule_wait", "value_ns"),
    KpiPresentation("transfer", "transfer.e2e_share", "Transfer E2E share", "transfer.e2e_share", "value_milli_percent", 100_000),
)

KPI_PRESENTATION_BY_IDENTITY = {
    (item.section, item.metric_name): item for item in KPI_PRESENTATIONS
}
TRACE_ATTRIBUTE_LATENCY_IDENTITIES = (
    ("request_facing_latency", "latency.e2e"),
    ("request_facing_latency", "latency.ttft"),
    ("request_facing_latency", "latency.tpot"),
    ("pipeline_latency", "latency.e2e"),
    ("pipeline_latency", "latency.prefill"),
    ("pipeline_latency", "latency.kv_export"),
    ("transfer", "transfer.handoff_duration"),
    ("transfer", "transfer.setup_duration"),
    ("pipeline_latency", "latency.kv_transfer"),
    ("transfer", "transfer.wait_duration"),
    ("pipeline_latency", "latency.kv_transform"),
    ("transfer", "decode.schedule_wait_duration"),
    ("pipeline_latency", "latency.decode"),
    ("pipeline_latency", "latency.sampling"),
)
KPI_SECTION_METRICS = {
    section: frozenset(
        item.metric_name for item in KPI_PRESENTATIONS if item.section == section
    )
    for section in KPI_SECTION_ORDER
}


RESOURCE_AGGREGATIONS: tuple[tuple[str, str], ...] = (
    ("min", "minimum_v1"),
    ("max", "maximum_v1"),
    ("mean", "arithmetic_mean_v1"),
    ("p50", "percentile_r7_v1"),
    ("p95", "percentile_r7_v1"),
    ("time_weighted_mean", "trailing_interval_time_weighted_mean_v1"),
)


@dataclass(frozen=True, slots=True)
class DisplayRule:
    canonical_unit: str
    display_unit: str
    scale_numerator: int
    scale_denominator: int
    decimal_places: int


DISPLAY_RULES: tuple[DisplayRule, ...] = (
    DisplayRule("ns", "ms", 1, 1_000_000, 3),
    DisplayRule("bytes", "MiB", 1, 1_048_576, 3),
    DisplayRule("bytes/s", "MiB/s", 1, 1_048_576, 3),
    DisplayRule("percent", "percent", 1, 1, 2),
    DisplayRule("W", "W", 1, 1, 3),
    DisplayRule("requests", "requests", 1, 1, 0),
    DisplayRule("requests/s", "requests/s", 1, 1, 3),
    DisplayRule("tokens", "tokens", 1, 1, 0),
    DisplayRule("tokens/s", "tokens/s", 1, 1, 3),
    DisplayRule("ratio", "percent", 100, 1, 2),
)
DISPLAY_RULE_BY_UNIT = {item.canonical_unit: item for item in DISPLAY_RULES}


def display_rule(unit: str) -> dict[str, object]:
    rule = DISPLAY_RULE_BY_UNIT.get(unit)
    if rule is None:
        return {
            "unit": unit,
            "scale_numerator": 1,
            "scale_denominator": 1,
            "decimal_places": 6,
            "rounding": "half_even",
        }
    return {
        "unit": rule.display_unit,
        "scale_numerator": rule.scale_numerator,
        "scale_denominator": rule.scale_denominator,
        "decimal_places": rule.decimal_places,
        "rounding": "half_even",
    }


@dataclass(frozen=True, slots=True)
class TraceAttributePresentation:
    section: str
    metric_name: str
    attribute_key: str
    value_suffix: str
    multiplier: int


_TRACE_ATTRIBUTE_ALIASES: tuple[TraceAttributePresentation, ...] = (
    TraceAttributePresentation("pipeline_latency", "latency.kv_export", "transfer.kv_export_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "transfer.duration", "transfer.kv_transfer_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "transfer.transform_duration", "transfer.kv_transform_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "transfer.handoff_duration", "transfer.handoff_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "transfer.setup_duration", "transfer.setup_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "transfer.wait_duration", "transfer.wait_duration", "value_ns", 1),
    TraceAttributePresentation("transfer", "decode.schedule_wait_duration", "transfer.decode_schedule_wait_duration", "value_ns", 1),
)

TRACE_ATTRIBUTE_PRESENTATIONS: tuple[TraceAttributePresentation, ...] = (
    *(
        TraceAttributePresentation(
            item.section,
            item.metric_name,
            item.trace_attribute_key,
            item.trace_value_suffix,
            item.trace_multiplier,
        )
        for item in KPI_PRESENTATIONS
        if item.trace_attribute_key is not None
        and item.trace_value_suffix is not None
    ),
    *_TRACE_ATTRIBUTE_ALIASES,
)

INTERVAL_RESOURCE_METRICS = frozenset(
    {
        "resource.cpu.utilization",
        "resource.gpu.utilization",
        "resource.gpu.power",
        "resource.npu.utilization",
        "resource.npu.power",
    }
)


@dataclass(frozen=True, slots=True)
class ResourcePresentation:
    metric_name: str
    display_name: str
    order: int
    canonical_metric_name: str | None = None


RESOURCE_PRESENTATIONS: tuple[ResourcePresentation, ...] = (
    ResourcePresentation("resource.system.memory_used", "System memory", 0),
    ResourcePresentation("resource.cpu.utilization", "CPU utilization", 1),
    # Input-only display aliases retained for existing Overview artifacts.
    ResourcePresentation(
        "resource.process.cpu_memory",
        "Process CPU memory",
        2,
        "resource.cpu.memory_used",
    ),
    ResourcePresentation(
        "resource.process.memory_used",
        "Process CPU memory",
        2,
        "resource.cpu.memory_used",
    ),
    ResourcePresentation("resource.gpu.memory_used", "GPU memory", 0),
    ResourcePresentation("resource.gpu.power", "GPU power", 1),
    ResourcePresentation("resource.gpu.utilization", "GPU utilization", 2),
    ResourcePresentation("resource.npu.memory_used", "NPU memory", 0),
    ResourcePresentation("resource.npu.power", "NPU power", 1),
    ResourcePresentation("resource.npu.utilization", "NPU utilization", 2),
)

RESOURCE_DISPLAY_NAMES = {
    item.metric_name: item.display_name for item in RESOURCE_PRESENTATIONS
}
RESOURCE_TRACK_ORDER = {
    item.metric_name: item.order for item in RESOURCE_PRESENTATIONS
}


def _require_unique(values: tuple[object, ...], description: str) -> None:
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate {description} in shared catalog")


def validate_catalog_contract(
    *,
    metrics: tuple[MetricDefinition, ...] = METRIC_DEFINITIONS,
    stages: tuple[StageDefinition, ...] = STAGE_DEFINITIONS,
    kpis: tuple[KpiPresentation, ...] = KPI_PRESENTATIONS,
    resources: tuple[ResourcePresentation, ...] = RESOURCE_PRESENTATIONS,
    trace_attributes: tuple[TraceAttributePresentation, ...] = TRACE_ATTRIBUTE_PRESENTATIONS,
) -> None:
    metric_by_name = {item.name: item for item in metrics}
    if len(metric_by_name) != len(metrics):
        raise RuntimeError("duplicate official metric name")
    _require_unique(tuple(item.track_key for item in stages), "stage track")
    _require_unique(
        tuple((item.start_event, item.end_event) for item in stages),
        "stage marker pair",
    )
    orders = tuple(item.pipeline_order for item in stages if item.pipeline_order is not None)
    _require_unique(orders, "pipeline stage order")
    if tuple(sorted(orders)) != tuple(range(len(orders))):
        raise RuntimeError("pipeline stage order must be contiguous from zero")
    _require_unique(
        tuple((item.section, item.metric_name) for item in kpis),
        "KPI presentation",
    )
    _require_unique(
        tuple(item.metric_name for item in resources),
        "resource presentation",
    )
    _require_unique(
        tuple(item.attribute_key for item in trace_attributes),
        "trace attribute key",
    )
    _require_unique(
        tuple(item.canonical_unit for item in DISPLAY_RULES),
        "display rule unit",
    )
    for item in kpis:
        if item.section not in KPI_SECTION_ORDER:
            raise RuntimeError(f"unknown KPI section: {item.section}")
        if (item.trace_attribute_key is None) != (item.trace_value_suffix is None):
            raise RuntimeError(f"incomplete trace attribute contract: {item.metric_name}")
        if item.trace_multiplier <= 0:
            raise RuntimeError(f"invalid trace multiplier: {item.metric_name}")
    for item in trace_attributes:
        if item.section not in KPI_SECTION_ORDER or item.multiplier <= 0:
            raise RuntimeError(f"invalid trace attribute contract: {item.attribute_key}")
    for metric_name in (
        *(item.metric_name for item in stages if item.metric_name),
        *(item.metric_name for item in kpis),
        *(item.canonical_metric_name or item.metric_name for item in resources),
        *(item.metric_name for item in trace_attributes),
    ):
        if metric_name not in metric_by_name:
            raise RuntimeError(f"shared catalog references unknown metric: {metric_name}")
    for item in stages:
        if item.metric_name is None:
            continue
        source_events = metric_by_name[item.metric_name].source_events
        if len(source_events) == 2 and source_events != (item.start_event, item.end_event):
            raise RuntimeError(f"stage marker pair disagrees with metric: {item.metric_name}")


validate_catalog_contract()


__all__ = [
    "DERIVED_LATENCY_METRICS",
    "DISPLAY_RULES",
    "INTERVAL_RESOURCE_METRICS",
    "KPI_PRESENTATION_BY_IDENTITY",
    "KPI_PRESENTATIONS",
    "KPI_SECTION_METRICS",
    "KPI_SECTION_ORDER",
    "PHASE_RECONCILIATION_METRICS",
    "PIPELINE_STAGE_ORDER",
    "RESOURCE_AGGREGATIONS",
    "RESOURCE_DISPLAY_NAMES",
    "RESOURCE_PRESENTATIONS",
    "RESOURCE_TRACK_ORDER",
    "STAGE_BY_METRIC",
    "STAGE_BY_TRACK",
    "STAGE_DEFINITIONS",
    "TRACE_ATTRIBUTE_LATENCY_IDENTITIES",
    "TRACE_ATTRIBUTE_PRESENTATIONS",
    "DisplayRule",
    "KpiPresentation",
    "ResourcePresentation",
    "StageDefinition",
    "TraceAttributePresentation",
    "display_rule",
    "validate_catalog_contract",
]
