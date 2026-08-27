"""Explicit field contracts shared by schema readers and consistency checks.

The dataclasses and committed JSON Schemas remain the authoritative wire-format
implementations.  These compact declarations make their common field inventory,
requiredness, primitive kind, nullability, bounds, and enum values reviewable in
one place without generating classes or schemas at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
from .records import (
    ArtifactReference,
    ClockDomain,
    ClockTransform,
    DeviceDescriptor,
    EventRecord,
    HostDescriptor,
    MetricSample,
    ModelDescriptor,
    RunManifest,
    SoftwareDescriptor,
    SyncPoint,
    WorkloadDescriptor,
)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    required: bool
    value_kind: str
    nullable: bool = False
    enum_type: type[Enum] | None = None
    allowed_values: tuple[object, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    nonempty: bool = False


@dataclass(frozen=True, slots=True)
class RecordFieldContract:
    record_class: type[object]
    schema_filename: str
    fields: tuple[FieldSpec, ...]
    record_type: RecordType | None = None
    schema_definition: str | None = None

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.fields)

    @property
    def required_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.fields if item.required)


def _field(
    name: str,
    value_kind: str,
    *,
    required: bool = True,
    nullable: bool = False,
    enum_type: type[Enum] | None = None,
    allowed_values: tuple[object, ...] = (),
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    nonempty: bool = False,
) -> FieldSpec:
    return FieldSpec(
        name=name,
        required=required,
        value_kind=value_kind,
        nullable=nullable,
        enum_type=enum_type,
        allowed_values=allowed_values,
        minimum=minimum,
        maximum=maximum,
        nonempty=nonempty,
    )


def _string(name: str, *, required: bool = True, nullable: bool = False) -> FieldSpec:
    return _field(
        name,
        "string",
        required=required,
        nullable=nullable,
        nonempty=not nullable,
    )


def _integer(
    name: str,
    *,
    required: bool = True,
    nullable: bool = False,
    minimum: int | None = None,
) -> FieldSpec:
    return _field(
        name,
        "integer",
        required=required,
        nullable=nullable,
        minimum=minimum,
    )


def _enum(
    name: str,
    enum_type: type[Enum],
    *,
    required: bool = True,
    nullable: bool = False,
) -> FieldSpec:
    return _field(
        name,
        "enum",
        required=required,
        nullable=nullable,
        enum_type=enum_type,
    )


def _envelope(record_type: RecordType) -> tuple[FieldSpec, ...]:
    return (
        _string("schema_version"),
        _field("record_type", "enum", allowed_values=(record_type.value,)),
        _string("run_id"),
    )


MODEL_DESCRIPTOR_FIELDS = (
    _field("role", "enum", allowed_values=("served", "prefill", "decode")),
    _string("model_id"),
    _string("revision", nullable=True),
    _string("tokenizer_id", nullable=True),
    _string("dtype", nullable=True),
)
WORKLOAD_DESCRIPTOR_FIELDS = (
    _integer("request_count", nullable=True, minimum=0),
    _integer("concurrency", nullable=True, minimum=0),
    _field("request_rate_per_s", "number", nullable=True, minimum=0),
    _integer("input_tokens", nullable=True, minimum=0),
    _integer("output_tokens", nullable=True, minimum=0),
    _integer("max_model_len", nullable=True, minimum=0),
    _integer("warmup_requests", nullable=True, minimum=0),
)
HOST_DESCRIPTOR_FIELDS = tuple(
    _string(name)
    for name in ("host_id", "role", "hostname", "operating_system", "architecture")
)
SOFTWARE_DESCRIPTOR_FIELDS = (
    _string("name"),
    _string("version", nullable=True),
    _string("role"),
    _string("path", nullable=True),
)
DEVICE_DESCRIPTOR_FIELDS = (
    _string("host_id"),
    _enum("device_type", DeviceType),
    _string("device_id"),
    _string("vendor"),
    _string("model"),
    _string("status"),
    _integer("memory_total_bytes", nullable=True, minimum=0),
    _field("attributes", "object"),
)

RUN_MANIFEST_FIELDS = _envelope(RecordType.RUN_MANIFEST) + (
    _enum("mode", RunMode),
    _enum("profile_mode", ProfileMode),
    _enum("status", RunStatus),
    _integer("created_at_unix_ns", minimum=0),
    _field("models", "array"),
    _field("workload", "object"),
    _field("hosts", "array"),
    _field("software", "array"),
    _field("devices", "array"),
    _field("configuration", "object"),
    _field("attributes", "object"),
)
EVENT_RECORD_FIELDS = _envelope(RecordType.EVENT) + (
    _string("event_id"),
    _string("event_name"),
    _enum("event_type", EventType),
    _enum("phase", Phase),
    _string("host_id"),
    _string("clock_domain_id"),
    _integer("timestamp_ns", minimum=0),
    _field("attributes", "object"),
    _string("request_id", required=False, nullable=True),
    _string("parent_event_id", required=False, nullable=True),
    _integer("process_id", required=False, nullable=True, minimum=0),
    _integer("thread_id", required=False, nullable=True, minimum=0),
    _enum("device_type", DeviceType, required=False, nullable=True),
    _string("device_id", required=False, nullable=True),
    _integer("duration_ns", required=False, nullable=True, minimum=0),
)
METRIC_SAMPLE_FIELDS = _envelope(RecordType.METRIC) + (
    _string("metric_name"),
    _enum("metric_kind", MetricKind),
    _enum("scope", MetricScope),
    _string("host_id"),
    _string("clock_domain_id"),
    _integer("timestamp_ns", minimum=0),
    _enum("availability", Availability),
    _enum("origin", ValueOrigin),
    _string("unit"),
    _field("value", "number", nullable=True),
    _field("dimensions", "object"),
    _field("attributes", "object"),
    _string("request_id", required=False, nullable=True),
    _enum("phase", Phase, required=False, nullable=True),
    _enum("device_type", DeviceType, required=False, nullable=True),
    _string("device_id", required=False, nullable=True),
    _integer("interval_ns", required=False, nullable=True, minimum=0),
    _string("reason", required=False, nullable=True),
    _field("source_event_ids", "array", required=False, nullable=True),
)
ARTIFACT_REFERENCE_FIELDS = _envelope(RecordType.ARTIFACT) + (
    _string("artifact_id"),
    _enum("artifact_kind", ArtifactKind),
    _string("relative_path"),
    _string("format"),
    _string("producer"),
    _integer("created_at_unix_ns", minimum=0),
    _field("attributes", "object"),
    _string("host_id", required=False, nullable=True),
    _string("request_id", required=False, nullable=True),
    _integer("size_bytes", required=False, nullable=True, minimum=0),
    _string("sha256", required=False, nullable=True),
    _string("clock_domain_id", required=False, nullable=True),
)
CLOCK_DOMAIN_FIELDS = _envelope(RecordType.CLOCK_DOMAIN) + (
    _string("clock_domain_id"),
    _string("host_id"),
    _enum("clock_type", ClockType),
    _field("unit", "enum", allowed_values=("ns",)),
    _field("monotonic", "boolean"),
    _field("adjustable", "boolean"),
    _field("attributes", "object"),
)
SYNC_POINT_FIELDS = _envelope(RecordType.SYNC_POINT) + (
    _string("sync_point_id"),
    _string("source_clock_domain_id"),
    _string("target_clock_domain_id"),
    _integer("source_timestamp_ns", minimum=0),
    _integer("target_timestamp_ns", minimum=0),
    _enum("method", SyncMethod),
    _integer("uncertainty_ns", minimum=0),
    _field("attributes", "object"),
)
CLOCK_TRANSFORM_FIELDS = _envelope(RecordType.CLOCK_TRANSFORM) + (
    _string("transform_id"),
    _string("source_clock_domain_id"),
    _string("target_clock_domain_id"),
    _field("scale", "number", minimum=0),
    _integer("offset_ns"),
    _integer("uncertainty_ns", minimum=0),
    _enum("method", SyncMethod),
    _integer("valid_from_source_ns", minimum=0),
    _integer("valid_to_source_ns", nullable=True, minimum=0),
    _field("attributes", "object"),
)


RECORD_FIELD_CONTRACTS: tuple[RecordFieldContract, ...] = (
    RecordFieldContract(ModelDescriptor, "run_manifest.schema.json", MODEL_DESCRIPTOR_FIELDS, schema_definition="model"),
    RecordFieldContract(WorkloadDescriptor, "run_manifest.schema.json", WORKLOAD_DESCRIPTOR_FIELDS, schema_definition="workload"),
    RecordFieldContract(HostDescriptor, "run_manifest.schema.json", HOST_DESCRIPTOR_FIELDS, schema_definition="host"),
    RecordFieldContract(SoftwareDescriptor, "run_manifest.schema.json", SOFTWARE_DESCRIPTOR_FIELDS, schema_definition="software"),
    RecordFieldContract(DeviceDescriptor, "run_manifest.schema.json", DEVICE_DESCRIPTOR_FIELDS, schema_definition="device"),
    RecordFieldContract(RunManifest, "run_manifest.schema.json", RUN_MANIFEST_FIELDS, RecordType.RUN_MANIFEST),
    RecordFieldContract(EventRecord, "event_record.schema.json", EVENT_RECORD_FIELDS, RecordType.EVENT),
    RecordFieldContract(MetricSample, "metric_sample.schema.json", METRIC_SAMPLE_FIELDS, RecordType.METRIC),
    RecordFieldContract(ArtifactReference, "artifact_reference.schema.json", ARTIFACT_REFERENCE_FIELDS, RecordType.ARTIFACT),
    RecordFieldContract(ClockDomain, "clock_domain.schema.json", CLOCK_DOMAIN_FIELDS, RecordType.CLOCK_DOMAIN),
    RecordFieldContract(SyncPoint, "sync_point.schema.json", SYNC_POINT_FIELDS, RecordType.SYNC_POINT),
    RecordFieldContract(ClockTransform, "clock_transform.schema.json", CLOCK_TRANSFORM_FIELDS, RecordType.CLOCK_TRANSFORM),
)

FIELD_CONTRACT_BY_RECORD_TYPE = {
    item.record_type: item
    for item in RECORD_FIELD_CONTRACTS
    if item.record_type is not None
}


def validate_field_contracts(
    contracts: tuple[RecordFieldContract, ...] = RECORD_FIELD_CONTRACTS,
) -> None:
    classes = tuple(item.record_class for item in contracts)
    if len(classes) != len(set(classes)):
        raise RuntimeError("duplicate record class in field contract")
    top_level_types = tuple(
        item.record_type for item in contracts if item.record_type is not None
    )
    if len(top_level_types) != len(set(top_level_types)):
        raise RuntimeError("duplicate record type in field contract")
    for contract in contracts:
        names = contract.field_names
        if len(names) != len(set(names)):
            raise RuntimeError(
                f"duplicate field in {contract.record_class.__name__} contract"
            )
        for item in contract.fields:
            if not item.name or not item.value_kind:
                raise RuntimeError("field contract names and kinds must be non-empty")
            if item.minimum is not None and item.maximum is not None:
                if item.minimum > item.maximum:
                    raise RuntimeError(f"invalid bounds for field {item.name}")


validate_field_contracts()


__all__ = [
    "FIELD_CONTRACT_BY_RECORD_TYPE",
    "FieldSpec",
    "RECORD_FIELD_CONTRACTS",
    "RecordFieldContract",
    "validate_field_contracts",
]
