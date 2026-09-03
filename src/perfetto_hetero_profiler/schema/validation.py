"""Semantic validation for schema v1 records."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .constants import (
    ATTRIBUTE_NAME_RE,
    CANONICAL_EVENT_NAMES,
    EVENT_NAMESPACED_NAME_RE,
    NAMESPACED_NAME_RE,
    SCHEMA_MAJOR_VERSION,
    SEMVER_RE,
)
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
from .jsonschema_runtime import JsonSchemaFailure, validate_structure
from .metric_catalog import METRIC_CATALOG
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
    SchemaRecord,
    SoftwareDescriptor,
    SyncPoint,
    WorkloadDescriptor,
)


class SchemaValidationError(ValueError):
    """A validation error with a stable field path."""

    def __init__(self, field_path: str, message: str):
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


_RECORD_TYPE_BY_CLASS = {
    RunManifest: RecordType.RUN_MANIFEST,
    EventRecord: RecordType.EVENT,
    MetricSample: RecordType.METRIC,
    ArtifactReference: RecordType.ARTIFACT,
    ClockDomain: RecordType.CLOCK_DOMAIN,
    SyncPoint: RecordType.SYNC_POINT,
    ClockTransform: RecordType.CLOCK_TRANSFORM,
}


def _fail(path: str, message: str) -> None:
    raise SchemaValidationError(path, message)


def _nonblank(value: str, path: str) -> None:
    if not value.strip():
        _fail(path, "must be a non-empty string")


def _exact_integer(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")


def _finite(value: int | float, path: str) -> None:
    if not math.isfinite(value):
        _fail(path, "must be finite; NaN and Infinity are not allowed")


def _typed_enum(value: Any, enum_type: type[Enum], path: str) -> None:
    if not isinstance(value, enum_type):
        _fail(path, f"must be one of {[member.value for member in enum_type]}")


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _json_value(item, f"{path}.{key}")
        return
    _fail(path, f"contains non-JSON value of type {type(value).__name__}")


def _mapping(value: dict[str, Any], path: str, *, namespaced: bool = False) -> None:
    for key in value:
        if not isinstance(key, str):
            _fail(path, "object keys must be strings")
        if namespaced and ATTRIBUTE_NAME_RE.fullmatch(key) is None:
            _fail(f"{path}.{key}", "attribute key must use a namespace")
    _json_value(value, path)


def validate_schema_version(version: str, path: str = "schema_version") -> None:
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        _fail(path, "must use MAJOR.MINOR.PATCH")
    if int(match.group(1)) != SCHEMA_MAJOR_VERSION:
        _fail(path, f"unsupported schema major version {match.group(1)}")


def validate_run_id(run_id: str, path: str = "run_id") -> None:
    _nonblank(run_id, path)
    if (
        run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).is_absolute()
        or PureWindowsPath(run_id).is_absolute()
    ):
        _fail(path, "must be a single relative path component without '..'")


def validate_relative_artifact_path(
    value: str, path: str = "artifact.relative_path"
) -> None:
    _nonblank(value, path)
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "\\" in value
        or ".." in posix.parts
        or str(posix) != value
        or value == "."
        or value.endswith("/")
    ):
        _fail(path, "must be a normalized path relative to the run root")


def _validate_record_structure(
    data: Any, expected: RecordType | None = None
) -> RecordType:
    if not isinstance(data, dict):
        _fail("record", "must be an object")
    raw_type = data.get("record_type")
    if raw_type is None:
        _fail("record_type", "required field is missing")
    if expected is None:
        try:
            record_type = RecordType(raw_type)
        except (TypeError, ValueError) as error:
            raise SchemaValidationError(
                "record_type", "is not a supported record type"
            ) from error
    else:
        record_type = expected
    try:
        validate_structure(data, record_type.value)
    except JsonSchemaFailure as error:
        raise SchemaValidationError(error.field_path, error.message) from None
    return record_type


def _validate_envelope(record: SchemaRecord, expected: RecordType, root: str) -> None:
    validate_schema_version(record.schema_version, f"{root}.schema_version")
    if record.record_type is not expected:
        _fail(f"{root}.record_type", f"must be {expected.value}")
    validate_run_id(record.run_id, f"{root}.run_id")


def _validate_model(model: ModelDescriptor, path: str) -> None:
    if not isinstance(model, ModelDescriptor):
        _fail(path, "must be a ModelDescriptor")
    _nonblank(model.model_id, f"{path}.model_id")
    for name in ("revision", "tokenizer_id", "dtype"):
        value = getattr(model, name)
        if value is not None:
            _nonblank(value, f"{path}.{name}")


def _validate_workload(workload: WorkloadDescriptor, path: str) -> None:
    if not isinstance(workload, WorkloadDescriptor):
        _fail(path, "must be a WorkloadDescriptor")
    for name in (
        "request_count",
        "concurrency",
        "input_tokens",
        "output_tokens",
        "max_model_len",
        "warmup_requests",
    ):
        value = getattr(workload, name)
        if value is not None:
            _exact_integer(value, f"{path}.{name}")
    if workload.request_rate_per_s is not None:
        _finite(workload.request_rate_per_s, f"{path}.request_rate_per_s")


def _validate_host(host: HostDescriptor, path: str) -> None:
    if not isinstance(host, HostDescriptor):
        _fail(path, "must be a HostDescriptor")
    for name in ("host_id", "role", "hostname", "operating_system", "architecture"):
        _nonblank(getattr(host, name), f"{path}.{name}")


def _validate_software(software: SoftwareDescriptor, path: str) -> None:
    if not isinstance(software, SoftwareDescriptor):
        _fail(path, "must be a SoftwareDescriptor")
    _nonblank(software.name, f"{path}.name")
    _nonblank(software.role, f"{path}.role")
    for name in ("version", "path"):
        value = getattr(software, name)
        if value is not None:
            _nonblank(value, f"{path}.{name}")


def _validate_device(device: DeviceDescriptor, path: str) -> None:
    if not isinstance(device, DeviceDescriptor):
        _fail(path, "must be a DeviceDescriptor")
    _typed_enum(device.device_type, DeviceType, f"{path}.device_type")
    for name in ("host_id", "device_id", "vendor", "model", "status"):
        _nonblank(getattr(device, name), f"{path}.{name}")
    if device.memory_total_bytes is not None:
        _exact_integer(device.memory_total_bytes, f"{path}.memory_total_bytes")
    _mapping(device.attributes, f"{path}.attributes", namespaced=True)


def _validate_manifest(record: RunManifest) -> None:
    root = "run_manifest"
    _validate_envelope(record, RecordType.RUN_MANIFEST, root)
    _typed_enum(record.mode, RunMode, f"{root}.mode")
    _typed_enum(record.profile_mode, ProfileMode, f"{root}.profile_mode")
    _typed_enum(record.status, RunStatus, f"{root}.status")
    _exact_integer(record.created_at_unix_ns, f"{root}.created_at_unix_ns")
    if not isinstance(record.models, list):
        _fail(f"{root}.models", "must be an array")
    for index, model in enumerate(record.models):
        _validate_model(model, f"{root}.models[{index}]")
    _validate_workload(record.workload, f"{root}.workload")
    if not isinstance(record.hosts, list):
        _fail(f"{root}.hosts", "must be an array")
    for index, host in enumerate(record.hosts):
        _validate_host(host, f"{root}.hosts[{index}]")
    host_ids = [host.host_id for host in record.hosts]
    if len(set(host_ids)) != len(host_ids):
        _fail(f"{root}.hosts", "host_id values must be unique")
    if not isinstance(record.software, list):
        _fail(f"{root}.software", "must be an array")
    for index, software in enumerate(record.software):
        _validate_software(software, f"{root}.software[{index}]")
    if not isinstance(record.devices, list):
        _fail(f"{root}.devices", "must be an array")
    for index, device in enumerate(record.devices):
        _validate_device(device, f"{root}.devices[{index}]")
        if device.host_id not in host_ids:
            _fail(
                f"{root}.devices[{index}].host_id",
                "must reference a host in the manifest",
            )
    device_keys = [(device.host_id, device.device_id) for device in record.devices]
    if len(set(device_keys)) != len(device_keys):
        _fail(f"{root}.devices", "(host_id, device_id) values must be unique")
    device_types = {device.device_type for device in record.devices}
    if record.mode is RunMode.GPU_ONLY and (
        DeviceType.GPU not in device_types or DeviceType.NPU in device_types
    ):
        _fail(f"{root}.devices", "gpu_only requires GPU and excludes NPU devices")
    if record.mode is RunMode.NPU_ONLY and (
        DeviceType.NPU not in device_types or DeviceType.GPU in device_types
    ):
        _fail(f"{root}.devices", "npu_only requires NPU and excludes GPU devices")
    if record.mode is RunMode.HYBRID and not {
        DeviceType.GPU,
        DeviceType.NPU,
    }.issubset(device_types):
        _fail(f"{root}.devices", "hybrid requires both GPU and NPU devices")
    model_roles = {model.role for model in record.models}
    if record.mode is RunMode.HYBRID and not {"prefill", "decode"}.issubset(
        model_roles
    ):
        _fail(f"{root}.models", "hybrid requires prefill and decode model roles")
    _mapping(record.configuration, f"{root}.configuration")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_event(record: EventRecord) -> None:
    root = "event"
    _validate_envelope(record, RecordType.EVENT, root)
    for name in ("event_id", "event_name", "host_id", "clock_domain_id"):
        _nonblank(getattr(record, name), f"{root}.{name}")
    if record.parent_event_id is not None:
        _nonblank(record.parent_event_id, f"{root}.parent_event_id")
        if record.parent_event_id == record.event_id:
            _fail(f"{root}.parent_event_id", "must differ from event_id")
    if (
        record.event_name not in CANONICAL_EVENT_NAMES
        and EVENT_NAMESPACED_NAME_RE.fullmatch(record.event_name) is None
    ):
        _fail(
            f"{root}.event_name",
            "must be canonical or use an approved custom-event namespace",
        )
    _typed_enum(record.event_type, EventType, f"{root}.event_type")
    _typed_enum(record.phase, Phase, f"{root}.phase")
    _exact_integer(record.timestamp_ns, f"{root}.timestamp_ns")
    if record.event_type is EventType.INSTANT and record.duration_ns is not None:
        _fail(f"{root}.duration_ns", "must be null for an instant event")
    if record.event_type is EventType.SPAN and record.duration_ns is None:
        _fail(f"{root}.duration_ns", "must be an integer")
    for name in ("process_id", "thread_id", "duration_ns"):
        value = getattr(record, name)
        if value is not None:
            _exact_integer(value, f"{root}.{name}")
    if (record.device_type is None) != (record.device_id is None):
        _fail(
            f"{root}.device_id",
            "device_type and device_id must either both be set or both be null",
        )
    if record.device_type is not None:
        _typed_enum(record.device_type, DeviceType, f"{root}.device_type")
        _nonblank(record.device_id, f"{root}.device_id")
    if record.request_id is not None:
        _nonblank(record.request_id, f"{root}.request_id")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_metric(record: MetricSample) -> None:
    root = "metric"
    _validate_envelope(record, RecordType.METRIC, root)
    for name in ("metric_name", "host_id", "clock_domain_id", "unit"):
        _nonblank(getattr(record, name), f"{root}.{name}")
    _typed_enum(record.metric_kind, MetricKind, f"{root}.metric_kind")
    _typed_enum(record.scope, MetricScope, f"{root}.scope")
    _typed_enum(record.availability, Availability, f"{root}.availability")
    _typed_enum(record.origin, ValueOrigin, f"{root}.origin")
    _exact_integer(record.timestamp_ns, f"{root}.timestamp_ns")
    if record.availability is Availability.AVAILABLE:
        if record.value is None:
            _fail(f"{root}.value", "must be a number")
        _finite(record.value, f"{root}.value")
    else:
        if record.value is not None:
            _fail(
                f"{root}.value",
                f"must be null when availability={record.availability.value}",
            )
        if record.reason is None or not record.reason.strip():
            _fail(f"{root}.reason", "is required when metric is unavailable")
    if record.interval_ns is not None:
        _exact_integer(record.interval_ns, f"{root}.interval_ns")
    if record.request_id is not None:
        _nonblank(record.request_id, f"{root}.request_id")
    if (record.device_type is None) != (record.device_id is None):
        _fail(
            f"{root}.device_id",
            "device_type and device_id must either both be set or both be null",
        )
    if record.device_type is not None:
        _typed_enum(record.device_type, DeviceType, f"{root}.device_type")
        _nonblank(record.device_id, f"{root}.device_id")
    if record.source_event_ids is not None:
        if not isinstance(record.source_event_ids, list):
            _fail(f"{root}.source_event_ids", "must be an array or null")
        for index, event_id in enumerate(record.source_event_ids):
            _nonblank(event_id, f"{root}.source_event_ids[{index}]")
    definition = METRIC_CATALOG.get(record.metric_name)
    if definition is None:
        if NAMESPACED_NAME_RE.fullmatch(record.metric_name) is None:
            _fail(f"{root}.metric_name", "is not in the official metric catalog")
    else:
        if record.unit != definition.unit:
            _fail(
                f"{root}.unit",
                f"must be {definition.unit!r} for {record.metric_name}",
            )
        if record.metric_kind is not definition.kind:
            _fail(
                f"{root}.metric_kind",
                f"must be {definition.kind.value} for {record.metric_name}",
            )
        if record.scope not in definition.allowed_scopes:
            _fail(f"{root}.scope", f"is not allowed for {record.metric_name}")
        if record.availability is Availability.AVAILABLE:
            if definition.value_type == "integer" and (
                not isinstance(record.value, int) or isinstance(record.value, bool)
            ):
                _fail(f"{root}.value", "must be an integer for this metric")
            assert record.value is not None
            if definition.minimum is not None and record.value < definition.minimum:
                _fail(f"{root}.value", f"must be >= {definition.minimum}")
            if definition.maximum is not None and record.value > definition.maximum:
                _fail(f"{root}.value", f"must be <= {definition.maximum}")
    if record.availability is Availability.AVAILABLE:
        assert record.value is not None
        maximum = (
            100
            if record.unit == "percent"
            else 1
            if record.unit == "ratio"
            else None
        )
        if maximum is not None and record.value > maximum:
            _fail(f"{root}.value", f"must be <= {maximum}")
        nonnegative_units = {
            "percent",
            "ratio",
            "bytes",
            "bytes/s",
            "ns",
            "W",
            "count",
            "tokens",
            "tokens/s",
            "requests",
            "requests/s",
        }
        if record.unit in nonnegative_units and record.value < 0:
            _fail(f"{root}.value", "must be >= 0")
    _mapping(record.dimensions, f"{root}.dimensions")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_artifact(record: ArtifactReference) -> None:
    root = "artifact"
    _validate_envelope(record, RecordType.ARTIFACT, root)
    _typed_enum(record.artifact_kind, ArtifactKind, f"{root}.artifact_kind")
    for name in ("artifact_id", "format", "producer"):
        _nonblank(getattr(record, name), f"{root}.{name}")
    validate_relative_artifact_path(record.relative_path, f"{root}.relative_path")
    _exact_integer(record.created_at_unix_ns, f"{root}.created_at_unix_ns")
    if record.size_bytes is not None:
        _exact_integer(record.size_bytes, f"{root}.size_bytes")
    for name in ("host_id", "request_id", "clock_domain_id"):
        value = getattr(record, name)
        if value is not None:
            _nonblank(value, f"{root}.{name}")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_clock_domain(record: ClockDomain) -> None:
    root = "clock_domain"
    _validate_envelope(record, RecordType.CLOCK_DOMAIN, root)
    _typed_enum(record.clock_type, ClockType, f"{root}.clock_type")
    for name in ("clock_domain_id", "host_id"):
        _nonblank(getattr(record, name), f"{root}.{name}")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_sync_point(record: SyncPoint) -> None:
    root = "sync_point"
    _validate_envelope(record, RecordType.SYNC_POINT, root)
    _typed_enum(record.method, SyncMethod, f"{root}.method")
    for name in (
        "sync_point_id",
        "source_clock_domain_id",
        "target_clock_domain_id",
    ):
        _nonblank(getattr(record, name), f"{root}.{name}")
    if record.source_clock_domain_id == record.target_clock_domain_id:
        _fail(
            f"{root}.target_clock_domain_id",
            "must differ from source_clock_domain_id",
        )
    for name in ("source_timestamp_ns", "target_timestamp_ns", "uncertainty_ns"):
        _exact_integer(getattr(record, name), f"{root}.{name}")
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


def _validate_clock_transform(record: ClockTransform) -> None:
    root = "clock_transform"
    _validate_envelope(record, RecordType.CLOCK_TRANSFORM, root)
    _typed_enum(record.method, SyncMethod, f"{root}.method")
    for name in (
        "transform_id",
        "source_clock_domain_id",
        "target_clock_domain_id",
    ):
        _nonblank(getattr(record, name), f"{root}.{name}")
    if record.source_clock_domain_id == record.target_clock_domain_id:
        _fail(
            f"{root}.target_clock_domain_id",
            "must differ from source_clock_domain_id",
        )
    _finite(record.scale, f"{root}.scale")
    if record.scale <= 0:
        _fail(f"{root}.scale", "must be a finite positive number")
    for name in ("offset_ns", "uncertainty_ns", "valid_from_source_ns"):
        _exact_integer(getattr(record, name), f"{root}.{name}")
    if record.valid_to_source_ns is not None:
        _exact_integer(record.valid_to_source_ns, f"{root}.valid_to_source_ns")
        if record.valid_to_source_ns < record.valid_from_source_ns:
            _fail(
                f"{root}.valid_to_source_ns",
                "must be >= valid_from_source_ns",
            )
    _mapping(record.attributes, f"{root}.attributes", namespaced=True)


_SEMANTIC_VALIDATORS = {
    RunManifest: _validate_manifest,
    EventRecord: _validate_event,
    MetricSample: _validate_metric,
    ArtifactReference: _validate_artifact,
    ClockDomain: _validate_clock_domain,
    SyncPoint: _validate_sync_point,
    ClockTransform: _validate_clock_transform,
}


def _validate_record_semantics(record: SchemaRecord) -> None:
    validator = _SEMANTIC_VALIDATORS.get(type(record))
    if validator is None:
        _fail("record", f"unsupported record class {type(record).__name__}")
    validator(record)  # type: ignore[arg-type]


def _validate_typed_record(record: SchemaRecord, data: dict[str, Any]) -> None:
    expected = _RECORD_TYPE_BY_CLASS.get(type(record))
    if expected is None:
        _fail("record", f"unsupported record class {type(record).__name__}")
    _validate_record_structure(data, expected)
    _validate_record_semantics(record)


def validate_record(record: SchemaRecord) -> None:
    """Validate one typed schema record or raise ``SchemaValidationError``."""
    from .serialization import _primitive

    _validate_typed_record(record, _primitive(record))


def validate_record_dict(data: dict[str, Any]) -> None:
    """Validate one strict record dictionary."""
    from .serialization import record_from_dict

    record_from_dict(data)
