"""Ordered stage, KPI presentation, and resource contracts.

The normalized metric definitions remain in :mod:`metric_catalog`.  This
module holds the cross-product metadata that is shared by aggregation,
Perfetto planning, trace attributes, and the external Overview.  Tuples are
intentional: their order is part of the deterministic artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Phase
from .metric_catalog import METRIC_CATALOG, MetricDefinition


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


KPI_SECTION_ORDER = (
    "request_facing_latency",
    "pipeline_latency",
    "throughput_and_tokens",
    "transfer",
)

KPI_PRESENTATIONS: tuple[KpiPresentation, ...] = (
    KpiPresentation("request_facing_latency", "latency.e2e", "Request E2E", "kpi.latency.e2e"),
    KpiPresentation("request_facing_latency", "latency.ttft", "TTFT", "kpi.latency.ttft"),
    KpiPresentation("request_facing_latency", "latency.tpot", "TPOT", "kpi.latency.tpot"),
    KpiPresentation("pipeline_latency", "latency.e2e", "Pipeline E2E", "pipeline.latency.e2e"),
    KpiPresentation("pipeline_latency", "latency.prefill", "Prefill latency", "kpi.latency.prefill"),
    KpiPresentation("pipeline_latency", "latency.kv_export", "KV export latency", "kpi.latency.kv_export"),
    KpiPresentation("pipeline_latency", "latency.kv_transfer", "KV transfer latency", "kpi.latency.kv_transfer"),
    KpiPresentation("pipeline_latency", "latency.kv_transform", "KV transform latency", "kpi.latency.kv_transform"),
    KpiPresentation("pipeline_latency", "latency.decode", "Decode latency", "kpi.latency.decode"),
    KpiPresentation("pipeline_latency", "latency.sampling", "Sampling total", "kpi.latency.sampling"),
    KpiPresentation("pipeline_latency", "latency.wait", "Pipeline wait"),
    KpiPresentation("throughput_and_tokens", "request.count", "Request count"),
    KpiPresentation("throughput_and_tokens", "request.input_tokens", "Input tokens"),
    KpiPresentation("throughput_and_tokens", "request.output_tokens", "Output tokens"),
    KpiPresentation("throughput_and_tokens", "request.total_tokens", "Total tokens"),
    KpiPresentation("throughput_and_tokens", "throughput.requests", "Requests per second"),
    KpiPresentation("throughput_and_tokens", "throughput.input_tokens", "Input tokens per second"),
    KpiPresentation("throughput_and_tokens", "throughput.output_tokens", "Output tokens per second"),
    KpiPresentation("throughput_and_tokens", "throughput.total_tokens", "Total tokens per second"),
    KpiPresentation("transfer", "transfer.bytes", "Transferred bytes"),
    KpiPresentation("transfer", "transfer.duration", "Transfer duration"),
    KpiPresentation("transfer", "transfer.effective_bandwidth", "Effective bandwidth"),
    KpiPresentation("transfer", "transfer.transform_duration", "Transform duration"),
    KpiPresentation("transfer", "transfer.wait_duration", "Transfer wait", "kpi.latency.kv_transfer_wait"),
    KpiPresentation("transfer", "transfer.handoff_duration", "KV handoff", "kpi.latency.kv_handoff"),
    KpiPresentation("transfer", "transfer.setup_duration", "Transfer setup", "kpi.latency.kv_transfer_setup"),
    KpiPresentation("transfer", "decode.schedule_wait_duration", "Decode scheduling wait", "kpi.latency.decode_schedule_wait"),
    KpiPresentation("transfer", "transfer.e2e_share", "Transfer E2E share"),
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


_require_unique(tuple(stage.track_key for stage in STAGE_DEFINITIONS), "stage track")
_require_unique(
    tuple((item.section, item.metric_name) for item in KPI_PRESENTATIONS),
    "KPI presentation",
)
_require_unique(
    tuple(item.metric_name for item in RESOURCE_PRESENTATIONS),
    "resource presentation",
)

for metric_name in (
    *(stage.metric_name for stage in STAGE_DEFINITIONS if stage.metric_name),
    *(item.metric_name for item in KPI_PRESENTATIONS),
    *(
        item.canonical_metric_name or item.metric_name
        for item in RESOURCE_PRESENTATIONS
    ),
):
    if metric_name not in METRIC_CATALOG:
        raise RuntimeError(f"shared catalog references unknown metric: {metric_name}")
