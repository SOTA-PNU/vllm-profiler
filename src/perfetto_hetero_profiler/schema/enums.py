"""String enums used by schema v1."""

from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """Enum whose serialized representation is its string value."""

    def __str__(self) -> str:
        return self.value


class RecordType(StringEnum):
    RUN_MANIFEST = "run_manifest"
    EVENT = "event"
    METRIC = "metric"
    ARTIFACT = "artifact"
    CLOCK_DOMAIN = "clock_domain"
    SYNC_POINT = "sync_point"
    CLOCK_TRANSFORM = "clock_transform"


class RunMode(StringEnum):
    GPU_ONLY = "gpu_only"
    NPU_ONLY = "npu_only"
    HYBRID = "hybrid"


class ProfileMode(StringEnum):
    MONITOR = "monitor"
    DETAILED_PROFILE = "detailed_profile"


class RunStatus(StringEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class Availability(StringEnum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"
    NOT_COLLECTED = "not_collected"
    ERROR = "error"


class ValueOrigin(StringEnum):
    MEASURED = "measured"
    DERIVED = "derived"
    ESTIMATED = "estimated"


class DeviceType(StringEnum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    SYSTEM_MEMORY = "system_memory"
    NETWORK = "network"
    UNKNOWN = "unknown"


class EventType(StringEnum):
    INSTANT = "instant"
    SPAN = "span"


class Phase(StringEnum):
    REQUEST = "request"
    PREFILL = "prefill"
    KV_EXPORT = "kv_export"
    KV_TRANSFORM = "kv_transform"
    KV_TRANSFER = "kv_transfer"
    DECODE = "decode"
    SAMPLING = "sampling"
    RESPONSE = "response"
    SYSTEM = "system"
    SYNCHRONIZATION = "synchronization"


class MetricKind(StringEnum):
    GAUGE = "gauge"
    COUNTER = "counter"
    DURATION = "duration"
    RATE = "rate"
    RATIO = "ratio"
    COUNT = "count"


class MetricScope(StringEnum):
    RUN = "run"
    REQUEST = "request"
    PHASE = "phase"
    HOST = "host"
    PROCESS = "process"
    DEVICE = "device"
    TRANSFER = "transfer"


class ArtifactKind(StringEnum):
    RAW_LOG = "raw_log"
    TORCH_TRACE = "torch_trace"
    NSYS_REPORT = "nsys_report"
    RBLN_REPORT = "rbln_report"
    TELEMETRY = "telemetry"
    EVENT_STREAM = "event_stream"
    METRIC_STREAM = "metric_stream"
    PERFETTO_TRACE = "perfetto_trace"
    OVERVIEW = "overview"
    MANIFEST = "manifest"
    OTHER = "other"


class ClockType(StringEnum):
    MONOTONIC = "monotonic"
    MONOTONIC_RAW = "monotonic_raw"
    REALTIME = "realtime"
    PERF_COUNTER = "perf_counter"
    CUDA = "cuda"
    RBLN = "rbln"
    EXTERNAL = "external"


class SyncMethod(StringEnum):
    NTP = "ntp"
    CHRONY = "chrony"
    RPC_MIDPOINT = "rpc_midpoint"
    SHARED_EVENT = "shared_event"
    MANUAL = "manual"
    DEVICE_CORRELATION = "device_correlation"
