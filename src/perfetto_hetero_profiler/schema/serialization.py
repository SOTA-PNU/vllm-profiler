"""Strict JSON and JSONL serialization for schema v1 records."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

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
from .validation import SchemaValidationError, validate_record, validate_schema_version


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def record_to_dict(record: SchemaRecord) -> dict[str, Any]:
    """Validate and convert a record into JSON-compatible primitives."""
    validate_record(record)
    if record.schema_version != SCHEMA_VERSION:
        raise SchemaValidationError(
            "schema_version",
            f"writer only supports schema version {SCHEMA_VERSION}",
        )
    return _primitive(record)


def _strict_fields(
    data: Any,
    cls: type[Any],
    path: str,
    *,
    required: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaValidationError(path, "must be an object")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SchemaValidationError(f"{path}.{unknown[0]}", "unknown field")
    required_names = allowed if required is None else required
    missing = sorted(required_names - set(data))
    if missing:
        raise SchemaValidationError(f"{path}.{missing[0]}", "required field is missing")
    return data


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise SchemaValidationError(
            path, f"must be one of {[member.value for member in enum_type]}"
        ) from error


def _model(data: Any, path: str) -> ModelDescriptor:
    values = _strict_fields(data, ModelDescriptor, path)
    return ModelDescriptor(**values)


def _workload(data: Any, path: str) -> WorkloadDescriptor:
    values = _strict_fields(data, WorkloadDescriptor, path)
    return WorkloadDescriptor(**values)


def _host(data: Any, path: str) -> HostDescriptor:
    values = _strict_fields(data, HostDescriptor, path)
    return HostDescriptor(**values)


def _software(data: Any, path: str) -> SoftwareDescriptor:
    values = _strict_fields(data, SoftwareDescriptor, path)
    return SoftwareDescriptor(**values)


def _device(data: Any, path: str) -> DeviceDescriptor:
    values = dict(_strict_fields(data, DeviceDescriptor, path))
    values["device_type"] = _enum(
        DeviceType, values["device_type"], f"{path}.device_type"
    )
    return DeviceDescriptor(**values)


_TOP_LEVEL_REQUIRED: dict[RecordType, set[str]] = {
    RecordType.RUN_MANIFEST: {
        "schema_version",
        "record_type",
        "run_id",
        "mode",
        "profile_mode",
        "status",
        "created_at_unix_ns",
        "models",
        "workload",
        "hosts",
        "software",
        "devices",
        "configuration",
        "attributes",
    },
    RecordType.EVENT: {
        "schema_version",
        "record_type",
        "run_id",
        "event_id",
        "event_name",
        "event_type",
        "phase",
        "host_id",
        "clock_domain_id",
        "timestamp_ns",
        "attributes",
    },
    RecordType.METRIC: {
        "schema_version",
        "record_type",
        "run_id",
        "metric_name",
        "metric_kind",
        "scope",
        "host_id",
        "clock_domain_id",
        "timestamp_ns",
        "availability",
        "origin",
        "unit",
        "value",
        "dimensions",
        "attributes",
    },
    RecordType.ARTIFACT: {
        "schema_version",
        "record_type",
        "run_id",
        "artifact_id",
        "artifact_kind",
        "relative_path",
        "format",
        "producer",
        "created_at_unix_ns",
        "attributes",
    },
    RecordType.CLOCK_DOMAIN: {
        "schema_version",
        "record_type",
        "run_id",
        "clock_domain_id",
        "host_id",
        "clock_type",
        "unit",
        "monotonic",
        "adjustable",
        "attributes",
    },
    RecordType.SYNC_POINT: {
        "schema_version",
        "record_type",
        "run_id",
        "sync_point_id",
        "source_clock_domain_id",
        "target_clock_domain_id",
        "source_timestamp_ns",
        "target_timestamp_ns",
        "method",
        "uncertainty_ns",
        "attributes",
    },
    RecordType.CLOCK_TRANSFORM: {
        "schema_version",
        "record_type",
        "run_id",
        "transform_id",
        "source_clock_domain_id",
        "target_clock_domain_id",
        "scale",
        "offset_ns",
        "uncertainty_ns",
        "method",
        "valid_from_source_ns",
        "valid_to_source_ns",
        "attributes",
    },
}


def _top(data: Any, cls: type[Any], record_type: RecordType) -> dict[str, Any]:
    return dict(
        _strict_fields(
            data,
            cls,
            record_type.value,
            required=_TOP_LEVEL_REQUIRED[record_type],
        )
    )


def _manifest(data: dict[str, Any]) -> RunManifest:
    values = _top(data, RunManifest, RecordType.RUN_MANIFEST)
    values["record_type"] = RecordType.RUN_MANIFEST
    values["mode"] = _enum(RunMode, values["mode"], "run_manifest.mode")
    values["profile_mode"] = _enum(
        ProfileMode, values["profile_mode"], "run_manifest.profile_mode"
    )
    values["status"] = _enum(RunStatus, values["status"], "run_manifest.status")
    if not isinstance(values["models"], list):
        raise SchemaValidationError("run_manifest.models", "must be an array")
    values["models"] = [
        _model(item, f"run_manifest.models[{index}]")
        for index, item in enumerate(values["models"])
    ]
    values["workload"] = _workload(values["workload"], "run_manifest.workload")
    if not isinstance(values["hosts"], list):
        raise SchemaValidationError("run_manifest.hosts", "must be an array")
    values["hosts"] = [
        _host(item, f"run_manifest.hosts[{index}]")
        for index, item in enumerate(values["hosts"])
    ]
    if not isinstance(values["software"], list):
        raise SchemaValidationError("run_manifest.software", "must be an array")
    values["software"] = [
        _software(item, f"run_manifest.software[{index}]")
        for index, item in enumerate(values["software"])
    ]
    if not isinstance(values["devices"], list):
        raise SchemaValidationError("run_manifest.devices", "must be an array")
    values["devices"] = [
        _device(item, f"run_manifest.devices[{index}]")
        for index, item in enumerate(values["devices"])
    ]
    return RunManifest(**values)


def _event(data: dict[str, Any]) -> EventRecord:
    values = _top(data, EventRecord, RecordType.EVENT)
    values["record_type"] = RecordType.EVENT
    values["event_type"] = _enum(EventType, values["event_type"], "event.event_type")
    values["phase"] = _enum(Phase, values["phase"], "event.phase")
    if values.get("device_type") is not None:
        values["device_type"] = _enum(
            DeviceType, values["device_type"], "event.device_type"
        )
    return EventRecord(**values)


def _metric(data: dict[str, Any]) -> MetricSample:
    values = _top(data, MetricSample, RecordType.METRIC)
    values["record_type"] = RecordType.METRIC
    values["metric_kind"] = _enum(
        MetricKind, values["metric_kind"], "metric.metric_kind"
    )
    values["scope"] = _enum(MetricScope, values["scope"], "metric.scope")
    values["availability"] = _enum(
        Availability, values["availability"], "metric.availability"
    )
    values["origin"] = _enum(ValueOrigin, values["origin"], "metric.origin")
    if values.get("phase") is not None:
        values["phase"] = _enum(Phase, values["phase"], "metric.phase")
    if values.get("device_type") is not None:
        values["device_type"] = _enum(
            DeviceType, values["device_type"], "metric.device_type"
        )
    return MetricSample(**values)


def _artifact(data: dict[str, Any]) -> ArtifactReference:
    values = _top(data, ArtifactReference, RecordType.ARTIFACT)
    values["record_type"] = RecordType.ARTIFACT
    values["artifact_kind"] = _enum(
        ArtifactKind, values["artifact_kind"], "artifact.artifact_kind"
    )
    return ArtifactReference(**values)


def _clock_domain(data: dict[str, Any]) -> ClockDomain:
    values = _top(data, ClockDomain, RecordType.CLOCK_DOMAIN)
    values["record_type"] = RecordType.CLOCK_DOMAIN
    values["clock_type"] = _enum(
        ClockType, values["clock_type"], "clock_domain.clock_type"
    )
    return ClockDomain(**values)


def _sync_point(data: dict[str, Any]) -> SyncPoint:
    values = _top(data, SyncPoint, RecordType.SYNC_POINT)
    values["record_type"] = RecordType.SYNC_POINT
    values["method"] = _enum(SyncMethod, values["method"], "sync_point.method")
    return SyncPoint(**values)


def _clock_transform(data: dict[str, Any]) -> ClockTransform:
    values = _top(data, ClockTransform, RecordType.CLOCK_TRANSFORM)
    values["record_type"] = RecordType.CLOCK_TRANSFORM
    values["method"] = _enum(
        SyncMethod, values["method"], "clock_transform.method"
    )
    return ClockTransform(**values)


_READERS = {
    RecordType.RUN_MANIFEST: _manifest,
    RecordType.EVENT: _event,
    RecordType.METRIC: _metric,
    RecordType.ARTIFACT: _artifact,
    RecordType.CLOCK_DOMAIN: _clock_domain,
    RecordType.SYNC_POINT: _sync_point,
    RecordType.CLOCK_TRANSFORM: _clock_transform,
}


def record_from_dict(data: dict[str, Any]) -> SchemaRecord:
    """Construct and validate a typed record selected by ``record_type``."""
    if not isinstance(data, dict):
        raise SchemaValidationError("record", "must be an object")
    if "schema_version" not in data:
        raise SchemaValidationError("schema_version", "required field is missing")
    validate_schema_version(data["schema_version"])
    if "record_type" not in data:
        raise SchemaValidationError("record_type", "required field is missing")
    record_type = _enum(RecordType, data["record_type"], "record_type")
    record = _READERS[record_type](data)
    validate_record(record)
    return record


def record_to_json(
    record: SchemaRecord, *, sort_keys: bool = True, indent: int | None = None
) -> str:
    """Serialize one record, rejecting NaN and Infinity."""
    return json.dumps(
        record_to_dict(record),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=sort_keys,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def record_from_json(text: str) -> SchemaRecord:
    """Parse and validate one JSON record."""
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise SchemaValidationError("json", str(error)) from error
    return record_from_dict(data)


def write_json(
    path: str | Path,
    record: SchemaRecord,
    *,
    overwrite: bool = False,
    sort_keys: bool = True,
) -> None:
    """Write one UTF-8 JSON record with a final newline."""
    output_path = Path(path)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        output.write(record_to_json(record, sort_keys=sort_keys, indent=2))
        output.write("\n")


def read_json(path: str | Path) -> SchemaRecord:
    """Read one UTF-8 JSON record."""
    return record_from_json(Path(path).read_text(encoding="utf-8"))


def write_jsonl(
    path: str | Path,
    records: Iterable[SchemaRecord],
    *,
    overwrite: bool = False,
    sort_keys: bool = True,
) -> None:
    """Write one compact record per line with a final newline."""
    output_path = Path(path)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(record_to_json(record, sort_keys=sort_keys))
            output.write("\n")


def read_jsonl(path: str | Path) -> list[SchemaRecord]:
    """Read and validate every record in a UTF-8 JSONL file."""
    records: list[SchemaRecord] = []
    event_ids: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise SchemaValidationError(
                    f"line {line_number}", "blank JSONL lines are not allowed"
                )
            try:
                record = record_from_json(line)
            except SchemaValidationError as error:
                raise SchemaValidationError(
                    f"line {line_number}.{error.field_path}", error.message
                ) from error
            if isinstance(record, EventRecord):
                event_key = (record.run_id, record.event_id)
                if event_key in event_ids:
                    raise SchemaValidationError(
                        f"line {line_number}.event.event_id",
                        "must be unique within a run",
                    )
                event_ids.add(event_key)
            records.append(record)
    return records
