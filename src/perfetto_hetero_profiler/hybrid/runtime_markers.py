"""Strict ingestion of append-only per-process hybrid runtime markers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..schema import (
    CANONICAL_EVENT_NAMES,
    SCHEMA_VERSION,
    DeviceType,
    EventRecord,
    EventType,
    Phase,
    SchemaValidationError,
    validate_record,
)


CANONICAL_MARKER_PHASES = {
    "request_received": Phase.REQUEST,
    "prefill_start": Phase.PREFILL,
    "prefill_end": Phase.PREFILL,
    "kv_export_start": Phase.KV_EXPORT,
    "kv_export_end": Phase.KV_EXPORT,
    "kv_transfer_start": Phase.KV_TRANSFER,
    "kv_transfer_end": Phase.KV_TRANSFER,
    "kv_transform_start": Phase.KV_TRANSFORM,
    "kv_transform_end": Phase.KV_TRANSFORM,
    "decode_loop_start": Phase.DECODE,
    "decode_step_start": Phase.DECODE,
    "decode_step_end": Phase.DECODE,
    "sampling_start": Phase.SAMPLING,
    "sampling_end": Phase.SAMPLING,
    "decode_loop_end": Phase.DECODE,
    "first_token_emitted": Phase.RESPONSE,
    "token_emitted": Phase.RESPONSE,
    "response_done": Phase.RESPONSE,
}

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "event_name",
        "timestamp_ns",
        "host_id",
        "clock_domain_id",
        "process_role",
        "pid",
        "thread_id",
        "request_id",
        "phase",
        "source",
        "attributes",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "correlation_id",
        "remote_request_id_suffix",
        "transfer_id",
        "sequence",
    }
)
_RESERVED_ATTRIBUTES = frozenset(
    {
        "hybrid.process_role",
        "hybrid.source",
        "hybrid.correlation_id",
        "hybrid.remote_request_id_suffix",
        "hybrid.transfer_id",
        "hybrid.marker_sequence",
        "hybrid.marker_file_index",
        "hybrid.marker_line_number",
    }
)
_SAFE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_FORBIDDEN_ATTRIBUTE_SEGMENTS = frozenset(
    {
        "address",
        "completion_text",
        "kv_data",
        "kv_tensor",
        "pointer",
        "prompt",
        "raw_pointer",
        "response_text",
        "tensor",
        "token_ids",
        "token_text",
    }
)

ProcessDeviceMap = Mapping[str, tuple[DeviceType, str] | None]


class RuntimeMarkerIngestError(ValueError):
    """A stable file/line-scoped error in a raw runtime marker stream."""

    def __init__(self, path: Path, line_number: int, message: str):
        self.path = Path(path)
        self.line_number = line_number
        self.message = message
        location = str(self.path)
        if line_number > 0:
            location = f"{location}:{line_number}"
        super().__init__(f"{location}: {message}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _nonempty_string(
    value: object,
    *,
    path: Path,
    line_number: int,
    field: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"{field} must be a non-empty string",
        )
    return value


def _nonnegative_integer(
    value: object,
    *,
    path: Path,
    line_number: int,
    field: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"{field} must be a non-negative integer",
        )
    return value


def _validate_safe_attributes(
    attributes: object,
    *,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(attributes, dict):
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            "attributes must be an object",
        )
    reserved = sorted(set(attributes) & _RESERVED_ATTRIBUTES)
    if reserved:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"attributes contains reserved key {reserved[0]!r}",
        )

    def validate_nested(value: object, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise RuntimeMarkerIngestError(
                        path,
                        line_number,
                        "attribute keys must be strings",
                    )
                nested_path = (*key_path, key)
                normalized_segments = {
                    segment.lower().replace("-", "_")
                    for segment in key.split(".")
                }
                forbidden = sorted(
                    normalized_segments & _FORBIDDEN_ATTRIBUTE_SEGMENTS
                )
                if forbidden:
                    raise RuntimeMarkerIngestError(
                        path,
                        line_number,
                        f"attribute {'.'.join(nested_path)!r} contains forbidden "
                        f"payload field {forbidden[0]!r}",
                    )
                validate_nested(nested, nested_path)
        elif isinstance(value, list):
            for nested in value:
                validate_nested(nested, key_path)

    validate_nested(attributes)
    return dict(attributes)


def _optional_identifier(
    row: dict[str, object],
    name: str,
    *,
    path: Path,
    line_number: int,
) -> str | None:
    value = row.get(name)
    if value is None:
        return None
    return _nonempty_string(
        value,
        path=path,
        line_number=line_number,
        field=name,
    )


def _event_id(
    *,
    file_index: int,
    line_number: int,
    event_name: str,
    timestamp_ns: int,
    process_id: int,
    thread_id: int,
) -> str:
    identity = (
        f"{file_index}\0{line_number}\0{event_name}\0{timestamp_ns}"
        f"\0{process_id}\0{thread_id}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"runtime-marker-{digest}"


def _parse_marker(
    row: object,
    *,
    path: Path,
    file_index: int,
    line_number: int,
    run_id: str,
    expected_host_id: str,
    expected_clock_domain_id: str,
    process_devices: ProcessDeviceMap,
) -> EventRecord:
    if not isinstance(row, dict):
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            "marker must be a JSON object",
        )
    keys = set(row)
    missing = sorted(_REQUIRED_FIELDS - keys)
    if missing:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"required field is missing: {missing[0]}",
        )
    unknown = sorted(keys - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if unknown:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"unknown field: {unknown[0]}",
        )
    if row["schema_version"] != SCHEMA_VERSION:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"schema_version must be {SCHEMA_VERSION}",
        )

    event_name = _nonempty_string(
        row["event_name"],
        path=path,
        line_number=line_number,
        field="event_name",
    )
    if event_name not in CANONICAL_EVENT_NAMES:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"event_name is not canonical: {event_name}",
        )
    expected_phase = CANONICAL_MARKER_PHASES[event_name]
    phase_text = _nonempty_string(
        row["phase"],
        path=path,
        line_number=line_number,
        field="phase",
    )
    if phase_text != expected_phase.value:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"phase {phase_text!r} does not match {event_name!r}; "
            f"expected {expected_phase.value!r}",
        )

    timestamp_ns = _nonnegative_integer(
        row["timestamp_ns"],
        path=path,
        line_number=line_number,
        field="timestamp_ns",
    )
    process_id = _nonnegative_integer(
        row["pid"],
        path=path,
        line_number=line_number,
        field="pid",
    )
    thread_id = _nonnegative_integer(
        row["thread_id"],
        path=path,
        line_number=line_number,
        field="thread_id",
    )
    host_id = _nonempty_string(
        row["host_id"],
        path=path,
        line_number=line_number,
        field="host_id",
    )
    clock_domain_id = _nonempty_string(
        row["clock_domain_id"],
        path=path,
        line_number=line_number,
        field="clock_domain_id",
    )
    if host_id != expected_host_id:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"host_id {host_id!r} does not match expected {expected_host_id!r}",
        )
    if clock_domain_id != expected_clock_domain_id:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            "clock_domain_id "
            f"{clock_domain_id!r} does not match expected "
            f"{expected_clock_domain_id!r}",
        )
    process_role = _nonempty_string(
        row["process_role"],
        path=path,
        line_number=line_number,
        field="process_role",
    )
    source = _nonempty_string(
        row["source"],
        path=path,
        line_number=line_number,
        field="source",
    )
    request_id = _nonempty_string(
        row["request_id"],
        path=path,
        line_number=line_number,
        field="request_id",
    )
    correlation_id = _optional_identifier(
        row,
        "correlation_id",
        path=path,
        line_number=line_number,
    )
    remote_suffix = _optional_identifier(
        row,
        "remote_request_id_suffix",
        path=path,
        line_number=line_number,
    )
    if remote_suffix is not None and _SAFE_SUFFIX_RE.fullmatch(remote_suffix) is None:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            "remote_request_id_suffix must be 1-64 safe identifier characters",
        )
    transfer_id = _optional_identifier(
        row,
        "transfer_id",
        path=path,
        line_number=line_number,
    )
    sequence = row.get("sequence")
    if sequence is not None:
        sequence = _nonnegative_integer(
            sequence,
            path=path,
            line_number=line_number,
            field="sequence",
        )

    attributes = _validate_safe_attributes(
        row["attributes"],
        path=path,
        line_number=line_number,
    )
    attributes.update(
        {
            "hybrid.process_role": process_role,
            "hybrid.source": source,
            "hybrid.marker_file_index": file_index,
            "hybrid.marker_line_number": line_number,
        }
    )
    if correlation_id is not None:
        attributes["hybrid.correlation_id"] = correlation_id
    if remote_suffix is not None:
        attributes["hybrid.remote_request_id_suffix"] = remote_suffix
    if transfer_id is not None:
        attributes["hybrid.transfer_id"] = transfer_id
    if sequence is not None:
        attributes["hybrid.marker_sequence"] = sequence

    device = process_devices.get(process_role)
    if device is not None and (
        not isinstance(device, tuple)
        or len(device) != 2
        or not isinstance(device[0], DeviceType)
        or not isinstance(device[1], str)
        or not device[1]
    ):
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"invalid process device mapping for role {process_role!r}",
        )
    device_type = device[0] if device is not None else None
    device_id = device[1] if device is not None else None
    event = EventRecord(
        run_id=run_id,
        event_id=_event_id(
            file_index=file_index,
            line_number=line_number,
            event_name=event_name,
            timestamp_ns=timestamp_ns,
            process_id=process_id,
            thread_id=thread_id,
        ),
        event_name=event_name,
        event_type=EventType.INSTANT,
        phase=expected_phase,
        host_id=host_id,
        clock_domain_id=clock_domain_id,
        timestamp_ns=timestamp_ns,
        request_id=request_id,
        process_id=process_id,
        thread_id=thread_id,
        device_type=device_type,
        device_id=device_id,
        attributes=attributes,
    )
    try:
        validate_record(event)
    except SchemaValidationError as error:
        raise RuntimeMarkerIngestError(
            path,
            line_number,
            f"normalized event is invalid: {error}",
        ) from error
    return event


def ingest_runtime_marker_files(
    paths: Iterable[str | Path],
    *,
    run_id: str,
    expected_host_id: str,
    expected_clock_domain_id: str,
    process_devices: ProcessDeviceMap | None = None,
) -> tuple[EventRecord, ...]:
    """Read, validate, normalize, and timestamp-sort process marker streams.

    The input files remain immutable. Any malformed marker raises a
    ``RuntimeMarkerIngestError`` so callers cannot silently mark an incomplete
    source run as succeeded.
    """

    marker_paths = tuple(sorted((Path(path) for path in paths), key=str))
    if not marker_paths:
        raise RuntimeMarkerIngestError(
            Path("<runtime-markers>"),
            0,
            "at least one marker file is required",
        )
    devices = process_devices or {}
    events: list[EventRecord] = []
    event_ids: set[str] = set()
    for file_index, path in enumerate(marker_paths):
        if not path.is_file():
            raise RuntimeMarkerIngestError(path, 0, "marker file does not exist")
        line_count = 0
        try:
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    line_count = line_number
                    if not line.strip():
                        raise RuntimeMarkerIngestError(
                            path,
                            line_number,
                            "blank JSONL lines are not allowed",
                        )
                    try:
                        row = json.loads(line, parse_constant=_reject_constant)
                    except (json.JSONDecodeError, ValueError) as error:
                        raise RuntimeMarkerIngestError(
                            path,
                            line_number,
                            f"invalid JSON: {error}",
                        ) from error
                    event = _parse_marker(
                        row,
                        path=path,
                        file_index=file_index,
                        line_number=line_number,
                        run_id=run_id,
                        expected_host_id=expected_host_id,
                        expected_clock_domain_id=expected_clock_domain_id,
                        process_devices=devices,
                    )
                    if event.event_id in event_ids:
                        raise RuntimeMarkerIngestError(
                            path,
                            line_number,
                            f"duplicate generated event_id {event.event_id}",
                        )
                    event_ids.add(event.event_id)
                    events.append(event)
        except RuntimeMarkerIngestError:
            raise
        except (OSError, UnicodeError) as error:
            raise RuntimeMarkerIngestError(path, line_count, str(error)) from error
        if line_count == 0:
            raise RuntimeMarkerIngestError(path, 0, "marker file is empty")
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.timestamp_ns,
                event.event_id,
            ),
        )
    )
