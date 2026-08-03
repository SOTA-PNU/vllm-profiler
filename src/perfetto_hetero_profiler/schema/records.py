"""Dataclass records forming the schema v1 data contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .constants import SCHEMA_VERSION
from .enums import (
    ArtifactKind,
    Availability,
    ClockType,
    DeviceType,
    EventType,
    MetricKind,
    MetricScope,
    Phase,
    ProfileMode,
    RecordType,
    RunMode,
    RunStatus,
    SyncMethod,
    ValueOrigin,
)

Attributes: TypeAlias = dict[str, Any]


@dataclass(kw_only=True)
class ModelDescriptor:
    role: str
    model_id: str
    revision: str | None
    tokenizer_id: str | None
    dtype: str | None


@dataclass(kw_only=True)
class WorkloadDescriptor:
    request_count: int | None
    concurrency: int | None
    request_rate_per_s: float | None
    input_tokens: int | None
    output_tokens: int | None
    max_model_len: int | None
    warmup_requests: int | None


@dataclass(kw_only=True)
class HostDescriptor:
    host_id: str
    role: str
    hostname: str
    operating_system: str
    architecture: str


@dataclass(kw_only=True)
class SoftwareDescriptor:
    name: str
    version: str | None
    role: str
    path: str | None


@dataclass(kw_only=True)
class DeviceDescriptor:
    host_id: str
    device_type: DeviceType
    device_id: str
    vendor: str
    model: str
    status: str
    memory_total_bytes: int | None
    attributes: Attributes = field(default_factory=dict)


@dataclass(kw_only=True)
class RunManifest:
    run_id: str
    mode: RunMode
    profile_mode: ProfileMode
    status: RunStatus
    created_at_unix_ns: int
    models: list[ModelDescriptor]
    workload: WorkloadDescriptor
    hosts: list[HostDescriptor]
    software: list[SoftwareDescriptor]
    devices: list[DeviceDescriptor]
    configuration: Attributes
    attributes: Attributes
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.RUN_MANIFEST


@dataclass(kw_only=True)
class EventRecord:
    run_id: str
    event_id: str
    event_name: str
    event_type: EventType
    phase: Phase
    host_id: str
    clock_domain_id: str
    timestamp_ns: int
    attributes: Attributes
    request_id: str | None = None
    parent_event_id: str | None = None
    process_id: int | None = None
    thread_id: int | None = None
    device_type: DeviceType | None = None
    device_id: str | None = None
    duration_ns: int | None = None
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.EVENT


@dataclass(kw_only=True)
class MetricSample:
    run_id: str
    metric_name: str
    metric_kind: MetricKind
    scope: MetricScope
    host_id: str
    clock_domain_id: str
    timestamp_ns: int
    availability: Availability
    origin: ValueOrigin
    unit: str
    value: int | float | None
    dimensions: Attributes
    attributes: Attributes
    request_id: str | None = None
    phase: Phase | None = None
    device_type: DeviceType | None = None
    device_id: str | None = None
    interval_ns: int | None = None
    reason: str | None = None
    source_event_ids: list[str] | None = None
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.METRIC


@dataclass(kw_only=True)
class ArtifactReference:
    run_id: str
    artifact_id: str
    artifact_kind: ArtifactKind
    relative_path: str
    format: str
    producer: str
    created_at_unix_ns: int
    attributes: Attributes
    host_id: str | None = None
    request_id: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    clock_domain_id: str | None = None
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.ARTIFACT


@dataclass(kw_only=True)
class ClockDomain:
    run_id: str
    clock_domain_id: str
    host_id: str
    clock_type: ClockType
    unit: str
    monotonic: bool
    adjustable: bool
    attributes: Attributes
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.CLOCK_DOMAIN


@dataclass(kw_only=True)
class SyncPoint:
    run_id: str
    sync_point_id: str
    source_clock_domain_id: str
    target_clock_domain_id: str
    source_timestamp_ns: int
    target_timestamp_ns: int
    method: SyncMethod
    uncertainty_ns: int
    attributes: Attributes
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.SYNC_POINT


@dataclass(kw_only=True)
class ClockTransform:
    run_id: str
    transform_id: str
    source_clock_domain_id: str
    target_clock_domain_id: str
    scale: float
    offset_ns: int
    uncertainty_ns: int
    method: SyncMethod
    valid_from_source_ns: int
    valid_to_source_ns: int | None
    attributes: Attributes
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.CLOCK_TRANSFORM


SchemaRecord: TypeAlias = (
    RunManifest
    | EventRecord
    | MetricSample
    | ArtifactReference
    | ClockDomain
    | SyncPoint
    | ClockTransform
)
