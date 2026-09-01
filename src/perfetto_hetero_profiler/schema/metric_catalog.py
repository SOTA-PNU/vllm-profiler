"""Official metric definitions for schema v1."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import MetricKind, MetricScope


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    kind: MetricKind
    allowed_scopes: tuple[MetricScope, ...]
    description: str
    value_type: str
    minimum: float | None
    maximum: float | None
    derived: bool
    source_events: tuple[str, ...]
    monitor_supported: bool
    detailed_profile_supported: bool
    source_candidates: tuple[str, ...]


def _definition(
    name: str,
    unit: str,
    kind: MetricKind,
    scopes: tuple[MetricScope, ...],
    description: str,
    *,
    value_type: str = "number",
    minimum: float | None = 0,
    maximum: float | None = None,
    derived: bool = False,
    source_events: tuple[str, ...] = (),
    monitor: bool = True,
    detailed: bool = True,
    sources: tuple[str, ...] = (),
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        unit=unit,
        kind=kind,
        allowed_scopes=scopes,
        description=description,
        value_type=value_type,
        minimum=minimum,
        maximum=maximum,
        derived=derived,
        source_events=source_events,
        monitor_supported=monitor,
        detailed_profile_supported=detailed,
        source_candidates=sources,
    )


_REQUEST_PHASE = (MetricScope.REQUEST, MetricScope.PHASE)
_RUN_REQUEST = (MetricScope.RUN, MetricScope.REQUEST)
_RESOURCE = (MetricScope.HOST, MetricScope.PROCESS, MetricScope.DEVICE)

_DEFINITIONS = (
    _definition(
        "latency.e2e",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST,),
        "Request receipt to completed response.",
        derived=True,
        source_events=("request_received", "response_done"),
    ),
    _definition(
        "latency.prefill",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "Prefill start to prefill end.",
        derived=True,
        source_events=("prefill_start", "prefill_end"),
    ),
    _definition(
        "latency.decode",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "Decode loop start to decode loop end.",
        derived=True,
        source_events=("decode_loop_start", "decode_loop_end"),
    ),
    _definition(
        "latency.kv_export",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "KV export interval.",
        derived=True,
        source_events=("kv_export_start", "kv_export_end"),
    ),
    _definition(
        "latency.kv_transform",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "KV representation transform interval.",
        derived=True,
        source_events=("kv_transform_start", "kv_transform_end"),
    ),
    _definition(
        "latency.kv_transfer",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.PHASE, MetricScope.TRANSFER),
        "KV data transfer interval.",
        derived=True,
        source_events=("kv_transfer_start", "kv_transfer_end"),
    ),
    _definition(
        "latency.sampling",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "Sampling start to sampling end.",
        derived=True,
        source_events=("sampling_start", "sampling_end"),
    ),
    _definition(
        "latency.wait",
        "ns",
        MetricKind.DURATION,
        _REQUEST_PHASE,
        "A classified wait interval; dimensions.wait_kind is required by convention.",
        derived=True,
    ),
    _definition(
        "latency.ttft",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST,),
        "Request receipt to first emitted token.",
        derived=True,
        source_events=("request_received", "first_token_emitted"),
    ),
    _definition(
        "latency.tpot",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST,),
        "Average time per output token after the first token.",
        derived=True,
        source_events=("first_token_emitted", "token_emitted"),
    ),
    _definition(
        "throughput.requests",
        "requests/s",
        MetricKind.RATE,
        (MetricScope.RUN,),
        "Completed requests per selected measurement window.",
        derived=True,
    ),
    _definition(
        "throughput.input_tokens",
        "tokens/s",
        MetricKind.RATE,
        (MetricScope.RUN,),
        "Input tokens per selected measurement window.",
        derived=True,
    ),
    _definition(
        "throughput.output_tokens",
        "tokens/s",
        MetricKind.RATE,
        (MetricScope.RUN,),
        "Output tokens per selected measurement window.",
        derived=True,
    ),
    _definition(
        "throughput.total_tokens",
        "tokens/s",
        MetricKind.RATE,
        (MetricScope.RUN,),
        "Input plus output tokens per selected measurement window.",
        derived=True,
    ),
    _definition(
        "request.input_tokens",
        "tokens",
        MetricKind.COUNT,
        _RUN_REQUEST,
        "Number of input tokens.",
        value_type="integer",
    ),
    _definition(
        "request.output_tokens",
        "tokens",
        MetricKind.COUNT,
        _RUN_REQUEST,
        "Number of output tokens.",
        value_type="integer",
    ),
    _definition(
        "request.total_tokens",
        "tokens",
        MetricKind.COUNT,
        _RUN_REQUEST,
        "Input plus output tokens.",
        value_type="integer",
        derived=True,
    ),
    _definition(
        "request.count",
        "requests",
        MetricKind.COUNT,
        (MetricScope.RUN,),
        "Number of requests in the selected window.",
        value_type="integer",
    ),
    _definition(
        "resource.cpu.utilization",
        "percent",
        MetricKind.GAUGE,
        _RESOURCE,
        "CPU utilization.",
        maximum=100,
        sources=("procfs", "psutil-compatible source", "system telemetry"),
    ),
    _definition(
        "resource.cpu.memory_used",
        "bytes",
        MetricKind.GAUGE,
        _RESOURCE,
        "CPU process or host memory in use.",
        sources=("procfs", "system telemetry"),
    ),
    _definition(
        "resource.system.memory_used",
        "bytes",
        MetricKind.GAUGE,
        (MetricScope.HOST,),
        "System memory in use.",
        sources=("procfs", "system telemetry"),
    ),
    _definition(
        "resource.gpu.utilization",
        "percent",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "GPU utilization.",
        maximum=100,
        sources=("NVML",),
    ),
    _definition(
        "resource.gpu.memory_used",
        "bytes",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "GPU memory in use.",
        sources=("NVML",),
    ),
    _definition(
        "resource.gpu.power",
        "W",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "GPU board power.",
        sources=("NVML",),
    ),
    _definition(
        "resource.npu.utilization",
        "percent",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "RBLN NPU utilization.",
        maximum=100,
        sources=("rbln-smi --json",),
    ),
    _definition(
        "resource.npu.memory_used",
        "bytes",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "RBLN NPU memory in use.",
        sources=("rbln-smi --json",),
    ),
    _definition(
        "resource.npu.power",
        "W",
        MetricKind.GAUGE,
        (MetricScope.DEVICE,),
        "RBLN NPU board power.",
        sources=("rbln-smi --json",),
    ),
    _definition(
        "transfer.bytes",
        "bytes",
        MetricKind.COUNT,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Bytes transferred for a KV or other data movement operation.",
        value_type="integer",
    ),
    _definition(
        "transfer.duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Transfer interval duration.",
        derived=True,
        source_events=("kv_transfer_start", "kv_transfer_end"),
    ),
    _definition(
        "transfer.effective_bandwidth",
        "bytes/s",
        MetricKind.RATE,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Transferred bytes divided by non-zero transfer duration.",
        derived=True,
        source_events=("kv_transfer_start", "kv_transfer_end"),
    ),
    _definition(
        "transfer.transform_duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "KV transform duration associated with a transfer.",
        derived=True,
        source_events=("kv_transform_start", "kv_transform_end"),
    ),
    _definition(
        "transfer.wait_duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Classified wait duration associated with a transfer.",
        derived=True,
        source_events=("kv_transfer_wait_start", "kv_transfer_wait_end"),
    ),
    _definition(
        "transfer.handoff_duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Delay from exported KV metadata to transfer setup entry.",
        derived=True,
        source_events=("kv_handoff_start", "kv_handoff_end"),
    ),
    _definition(
        "transfer.setup_duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Host-side transfer descriptor and handle preparation duration.",
        derived=True,
        source_events=(
            "kv_transfer_setup_start",
            "kv_transfer_setup_end",
        ),
    ),
    _definition(
        "decode.schedule_wait_duration",
        "ns",
        MetricKind.DURATION,
        (MetricScope.REQUEST,),
        "Delay from decode-ready KV state to the first decode model step.",
        derived=True,
        source_events=(
            "decode_schedule_wait_start",
            "decode_schedule_wait_end",
        ),
    ),
    _definition(
        "transfer.e2e_share",
        "ratio",
        MetricKind.RATIO,
        (MetricScope.REQUEST, MetricScope.TRANSFER),
        "Transfer duration divided by non-zero E2E latency.",
        maximum=1,
        derived=True,
        source_events=(
            "request_received",
            "kv_transfer_start",
            "kv_transfer_end",
            "response_done",
        ),
    ),
    _definition(
        "hybrid.joined_requests",
        "requests",
        MetricKind.COUNT,
        (MetricScope.RUN,),
        "Requests joined across GPU and NPU sources by an explicit identifier.",
        value_type="integer",
        derived=True,
    ),
    _definition(
        "hybrid.unjoined_requests",
        "requests",
        MetricKind.COUNT,
        (MetricScope.RUN,),
        "Source request groups that could not be joined unambiguously.",
        value_type="integer",
        derived=True,
    ),
    _definition(
        "hybrid.alignment_offset",
        "ns",
        MetricKind.GAUGE,
        (MetricScope.RUN, MetricScope.HOST),
        "Estimated source-clock offset relative to the coordinator clock.",
        minimum=None,
        derived=True,
    ),
    _definition(
        "hybrid.alignment_uncertainty",
        "ns",
        MetricKind.GAUGE,
        (MetricScope.RUN, MetricScope.HOST),
        "Estimated upper bound for hybrid clock alignment error.",
        derived=True,
    ),
)

def validate_metric_definitions(
    definitions: tuple[MetricDefinition, ...],
) -> None:
    """Reject ambiguous metric identities and malformed declarative bounds."""

    names = tuple(item.name for item in definitions)
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate official metric name")
    for item in definitions:
        if not item.name or not item.unit or not item.allowed_scopes:
            raise RuntimeError("metric identity, unit, and scopes must be non-empty")
        if item.minimum is not None and item.maximum is not None:
            if item.minimum > item.maximum:
                raise RuntimeError(f"invalid metric bounds: {item.name}")
        if len(item.source_events) != len(set(item.source_events)):
            raise RuntimeError(f"duplicate source event for metric: {item.name}")


validate_metric_definitions(_DEFINITIONS)

METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = _DEFINITIONS
METRIC_CATALOG: dict[str, MetricDefinition] = {
    definition.name: definition for definition in METRIC_DEFINITIONS
}


__all__ = [
    "METRIC_CATALOG",
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "validate_metric_definitions",
]
