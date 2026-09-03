"""Strict JSON and JSONL serialization for schema v1 records."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
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
from .validation import (
    SchemaValidationError,
    _validate_record_semantics,
    _validate_record_structure,
    _validate_typed_record,
)


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def record_to_dict(record: SchemaRecord) -> dict[str, Any]:
    """Validate and convert a record into JSON-compatible primitives."""
    data = _primitive(record)
    _validate_typed_record(record, data)
    if record.schema_version != SCHEMA_VERSION:
        raise SchemaValidationError(
            "schema_version",
            f"writer only supports schema version {SCHEMA_VERSION}",
        )
    return data


def _manifest(data: dict[str, Any]) -> RunManifest:
    values = dict(data)
    values["record_type"] = RecordType.RUN_MANIFEST
    values["mode"] = RunMode(values["mode"])
    values["profile_mode"] = ProfileMode(values["profile_mode"])
    values["status"] = RunStatus(values["status"])
    values["models"] = [ModelDescriptor(**item) for item in values["models"]]
    values["workload"] = WorkloadDescriptor(**values["workload"])
    values["hosts"] = [HostDescriptor(**item) for item in values["hosts"]]
    values["software"] = [SoftwareDescriptor(**item) for item in values["software"]]
    values["devices"] = [
        DeviceDescriptor(
            **{**item, "device_type": DeviceType(item["device_type"])}
        )
        for item in values["devices"]
    ]
    return RunManifest(**values)


def _event(data: dict[str, Any]) -> EventRecord:
    values = dict(data)
    values["record_type"] = RecordType.EVENT
    values["event_type"] = EventType(values["event_type"])
    values["phase"] = Phase(values["phase"])
    if values.get("device_type") is not None:
        values["device_type"] = DeviceType(values["device_type"])
    return EventRecord(**values)


def _metric(data: dict[str, Any]) -> MetricSample:
    values = dict(data)
    values["record_type"] = RecordType.METRIC
    values["metric_kind"] = MetricKind(values["metric_kind"])
    values["scope"] = MetricScope(values["scope"])
    values["availability"] = Availability(values["availability"])
    values["origin"] = ValueOrigin(values["origin"])
    if values.get("phase") is not None:
        values["phase"] = Phase(values["phase"])
    if values.get("device_type") is not None:
        values["device_type"] = DeviceType(values["device_type"])
    return MetricSample(**values)


def _artifact(data: dict[str, Any]) -> ArtifactReference:
    values = dict(data)
    values["record_type"] = RecordType.ARTIFACT
    values["artifact_kind"] = ArtifactKind(values["artifact_kind"])
    return ArtifactReference(**values)


def _clock_domain(data: dict[str, Any]) -> ClockDomain:
    values = dict(data)
    values["record_type"] = RecordType.CLOCK_DOMAIN
    values["clock_type"] = ClockType(values["clock_type"])
    return ClockDomain(**values)


def _sync_point(data: dict[str, Any]) -> SyncPoint:
    values = dict(data)
    values["record_type"] = RecordType.SYNC_POINT
    values["method"] = SyncMethod(values["method"])
    return SyncPoint(**values)


def _clock_transform(data: dict[str, Any]) -> ClockTransform:
    values = dict(data)
    values["record_type"] = RecordType.CLOCK_TRANSFORM
    values["method"] = SyncMethod(values["method"])
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
    record_type = _validate_record_structure(data)
    record = _READERS[record_type](data)
    _validate_record_semantics(record)
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
