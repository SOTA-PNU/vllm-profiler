"""Nsight Systems SQLite native trace conversion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import re
import sqlite3
from typing import Final

from .loader import LoadedHybridRun, SourceRunMetadata
from .model import SliceSpec, TrackSpec
from .native_details import (
    NativeDetailError,
    NativeDetailResult,
    NativeDetailSummary,
    _ClockBridge,
    _NSYS_GLOBAL_PID_MASK,
    _NSYS_REQUIRED_TABLES,
    _NativeSlice,
    _artifact_path,
    _attach_explicit_flows,
    _clock_bridge,
    _non_bool_int_or_none,
    _nsys_annotations,
    _nsys_api_category,
    _positive_duration,
    _read_alignment,
    _stable_file_identity,
    _stable_token,
    _stable_uint64,
    _validate_mapped_interval,
)


SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS: Final = ("3.16.1",)
_NSYS_EXPORT_SCHEMA_VERSION_KEY: Final = "EXPORT_SCHEMA_VERSION"
_NSYS_EXPORT_SCHEMA_VERSION_RE: Final = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
)


def nsys_detail_result(
    loaded: LoadedHybridRun,
    source: SourceRunMetadata,
    *,
    native_clock_domain: str,
    native_timestamp_unit: str,
    host_boundary_uncertainty_ns: int,
) -> NativeDetailResult:
    sqlite_artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.format == "sqlite"
            and artifact.relative_path.endswith(".sqlite")
        ),
        key=lambda item: item.relative_path,
    )
    report_artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.format == "nsys-rep"
        ),
        key=lambda item: item.relative_path,
    )
    if len(sqlite_artifacts) != 1 or len(report_artifacts) != 1:
        raise NativeDetailError(
            "gpu_nsys requires exactly one existing SQLite export and report"
        )
    sqlite_artifact = sqlite_artifacts[0]
    sqlite_path = _artifact_path(source, sqlite_artifact)
    _stable_file_identity(sqlite_path, sqlite_artifact)
    _stable_file_identity(
        _artifact_path(source, report_artifacts[0]), report_artifacts[0]
    )
    alignment = _read_alignment(source)
    bridge = _clock_bridge(
        loaded,
        source,
        alignment,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        host_boundary_uncertainty_ns=host_boundary_uncertainty_ns,
    )

    uri = f"file:{sqlite_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        _validate_nsys_sqlite_preamble(connection)
        start_rows = connection.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchall()
        if (
            len(start_rows) != 1
            or isinstance(start_rows[0][0], bool)
            or not isinstance(start_rows[0][0], int)
        ):
            raise NativeDetailError("Nsight session start time is invalid")
        session_unix_ns = start_rows[0][0]
        strings = {
            int(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT id, value FROM StringIds ORDER BY id"
            )
        }
        process_names = {
            int(row[0]): (int(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT globalPid, pid, name FROM PROCESSES ORDER BY globalPid"
            )
        }
        slices, tracks, counts, metadata_count = _read_nsys_rows(
            loaded,
            connection,
            strings=strings,
            process_names=process_names,
            session_unix_ns=session_unix_ns,
            bridge=bridge,
        )
    finally:
        connection.close()
    _stable_file_identity(sqlite_path, sqlite_artifact)

    slices, flows = _attach_explicit_flows(
        loaded.manifest.run_id,
        "gpu_nsys",
        slices,
    )
    valid_interval, mapped_interval = _validate_mapped_interval(
        alignment,
        bridge,
        tuple(item.spec for item in slices),
        (),
    )
    summary = NativeDetailSummary(
        profiler_type="gpu_nsys",
        source_role=source.source_role,
        support_status="converted_from_existing_official_sqlite_export",
        alignment_status="partial_derived",
        alignment_method=(
            "nsight_utcEpochNs_plus_native_ns_plus_recorded_host_clock_samples"
        ),
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=len(slices),
        emitted_slice_count=len(slices),
        emitted_instant_count=0,
        emitted_flow_count=len(flows),
        metadata_only_event_count=metadata_count,
        skipped_event_count=0,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=bridge.uncertainty_ns,
        clock_offset_ns=bridge.offset_ns,
        observed_offset_half_range_ns=bridge.observed_half_range_ns,
        native_epoch_base_ns=session_unix_ns,
        clock_sample_offsets_ns=bridge.sample_offsets_ns,
        canonical_transform_offset_ns=bridge.canonical_offset_ns,
        clock_formula=(
            "canonical_ns = utcEpochNs + activity.start "
            "- clock_offset_ns + canonical_transform_offset_ns"
        ),
        alignment_valid_interval_ns=valid_interval,
        mapped_event_interval_ns=mapped_interval,
        event_counts=tuple(sorted(counts.items())),
        artifact_count=2,
        artifact_sha256=(
            report_artifacts[0].sha256,
            sqlite_artifact.sha256,
        ),
        notes=(
            "existing SQLite export is opened read-only and immutable",
            "API-to-device flows require one unique official correlationId",
            "Unix/monotonic samples are non-atomic; alignment remains partial",
            "reported uncertainty is not a proven clock-error bound",
        ),
    )
    return NativeDetailResult(
        tracks=tuple(tracks.values()),
        slices=tuple(item.spec for item in slices),
        flows=flows,
        summaries=(summary,),
    )


def _validate_nsys_sqlite_preamble(connection: sqlite3.Connection) -> str:
    """Validate integrity, export schema, and required tables in that order."""

    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
    except sqlite3.DatabaseError as error:
        raise NativeDetailError(
            "Nsight SQLite quick_check could not be completed"
        ) from error
    if quick_check != [("ok",)]:
        raise NativeDetailError("Nsight SQLite quick_check failed")

    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    except sqlite3.DatabaseError as error:
        raise NativeDetailError(
            "Nsight SQLite table inventory could not be read"
        ) from error

    version = _read_nsys_export_schema_version(connection, tables=tables)
    missing = sorted(_NSYS_REQUIRED_TABLES - tables)
    if missing:
        raise NativeDetailError(
            f"Nsight SQLite lacks required tables: {missing}"
        )
    return version


def _read_nsys_export_schema_version(
    connection: sqlite3.Connection,
    *,
    tables: set[str],
) -> str:
    """Return the one explicitly supported official export schema version."""

    if "META_DATA_EXPORT" not in tables:
        raise _nsys_schema_version_error(
            "META_DATA_EXPORT table is missing",
            (),
        )
    try:
        table_info = connection.execute(
            "PRAGMA table_info(META_DATA_EXPORT)"
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise _nsys_schema_version_error(
            "META_DATA_EXPORT fields could not be read",
            (),
        ) from error
    columns = {
        row[1]
        for row in table_info
        if len(row) > 1 and isinstance(row[1], str)
    }
    missing_columns = sorted({"name", "value"} - columns)
    if missing_columns:
        raise _nsys_schema_version_error(
            f"META_DATA_EXPORT field is missing: {missing_columns}",
            (),
        )
    try:
        rows = connection.execute(
            "SELECT value FROM META_DATA_EXPORT WHERE name = ?",
            (_NSYS_EXPORT_SCHEMA_VERSION_KEY,),
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION row could not be read",
            (),
        ) from error
    observed = tuple(row[0] for row in rows if len(row) == 1)
    if len(rows) != len(observed):
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION row shape is invalid",
            observed,
        )
    if not observed:
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION row is missing",
            (),
        )
    if len(observed) != 1:
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION is duplicated or conflicting",
            observed,
        )
    version = observed[0]
    if not isinstance(version, str):
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION has a non-text value",
            observed,
        )
    if not version.strip():
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION is empty",
            observed,
        )
    if _NSYS_EXPORT_SCHEMA_VERSION_RE.fullmatch(version) is None:
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION format is invalid",
            observed,
        )
    if version not in SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS:
        raise _nsys_schema_version_error(
            "EXPORT_SCHEMA_VERSION is unsupported",
            observed,
        )
    return version


def _nsys_schema_version_error(
    reason: str,
    observed: tuple[object, ...],
) -> NativeDetailError:
    observed_values = [
        _safe_observed_schema_version(value) for value in observed
    ] or ["<missing>"]
    return NativeDetailError(
        "Nsight SQLite export schema version validation failed: "
        f"{reason}; observed={observed_values}; "
        f"supported={list(SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS)}"
    )


def _safe_observed_schema_version(value: object) -> str:
    if isinstance(value, str):
        if (
            len(value) <= 64
            and all(character.isprintable() for character in value)
            and "/" not in value
            and "\\" not in value
        ):
            return value
        return f"<malformed text length={len(value)}>"
    if value is None:
        return "<NULL>"
    if isinstance(value, bool):
        return f"<INTEGER {int(value)}>"
    if isinstance(value, int):
        return f"<INTEGER {value}>"
    if isinstance(value, float):
        return f"<REAL {value!r}>"
    if isinstance(value, bytes):
        return f"<BLOB length={len(value)}>"
    return f"<{type(value).__name__}>"


def _read_nsys_rows(
    loaded: LoadedHybridRun,
    connection: sqlite3.Connection,
    *,
    strings: Mapping[int, str],
    process_names: Mapping[int, tuple[int, str]],
    session_unix_ns: int,
    bridge: _ClockBridge,
) -> tuple[
    list[_NativeSlice],
    dict[str, TrackSpec],
    Counter[str],
    int,
]:
    root_key = "native.gpu_nsys"
    tracks: dict[str, TrackSpec] = {
        root_key: TrackSpec(
            key=root_key,
            uuid=_stable_uint64(loaded.manifest.run_id, "track", root_key),
            name="GPU Nsight native details (partial alignment)",
            kind="group",
            description=(
                "Nsight activities from the immutable official SQLite export; "
                "timestamps use session UTC evidence and partial clock samples."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=5,
        )
    }
    category_order = {
        "NVTX ranges": 0,
        "CUDA Runtime API": 1,
        "CUDA Driver API": 2,
        "CUDA kernels": 3,
        "CUDA memcpy": 4,
        "CUDA memset": 5,
    }
    category_keys: dict[str, str] = {}
    result: list[_NativeSlice] = []
    counts: Counter[str] = Counter()

    def ensure_track(
        category: str,
        identity: str,
        lane_name: str,
    ) -> str:
        category_key = category_keys.get(category)
        if category_key is None:
            category_key = (
                f"{root_key}.category.{_stable_token(category)}"
            )
            category_keys[category] = category_key
            tracks[category_key] = TrackSpec(
                key=category_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", category_key
                ),
                name=category,
                kind="group",
                description=f"Nsight {category}.",
                parent_key=root_key,
                child_ordering="lexicographic",
                sibling_order_rank=category_order[category],
            )
        leaf_key = (
            f"{category_key}.lane.{_stable_token(identity)}"
        )
        if leaf_key not in tracks:
            tracks[leaf_key] = TrackSpec(
                key=leaf_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", leaf_key
                ),
                name=lane_name,
                kind="slice",
                description="Original Nsight thread or CUDA stream identity.",
                parent_key=category_key,
            )
        return leaf_key

    runtime_rows = connection.execute(
        """
        SELECT start, end, eventClass, globalTid, correlationId, nameId,
               returnValue
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start, end, eventClass, globalTid, correlationId, nameId
        """
    )
    for start, end, event_class, global_tid, correlation, name_id, return_value in runtime_rows:
        category = _nsys_api_category(event_class)
        lane = f"Nsight globalTid {global_tid}"
        leaf = ensure_track(category, f"tid:{global_tid}", lane)
        name = strings.get(int(name_id), f"StringId {name_id}")
        annotations = _nsys_annotations(
            bridge,
            native_start_ns=int(start),
            values={
                "category": category,
                "global_tid": global_tid,
                "correlation_id": correlation,
                "return_value": return_value,
                "event_class": event_class,
            },
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=name,
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight API"),
                    annotations=annotations,
                ),
                category=category,
                correlation_id=_non_bool_int_or_none(correlation),
                endpoint_kind="host_api",
                correlation_scope=(
                    f"nsight-process:{int(global_tid) & _NSYS_GLOBAL_PID_MASK}"
                ),
            )
        )
        counts[category] += 1

    kernel_rows = connection.execute(
        """
        SELECT start, end, deviceId, contextId, streamId, correlationId,
               globalPid, demangledName, shortName, gridX, gridY, gridZ,
               blockX, blockY, blockZ, registersPerThread,
               staticSharedMemory, dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start, end, deviceId, contextId, streamId, correlationId
        """
    )
    for row in kernel_rows:
        (
            start,
            end,
            device,
            context,
            stream,
            correlation,
            global_pid,
            demangled_name,
            short_name,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            registers,
            static_shared,
            dynamic_shared,
        ) = row
        category = "CUDA kernels"
        lane = (
            f"GPU {device} / context {context} / stream {stream}"
        )
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            lane,
        )
        process = process_names.get(int(global_pid)) if global_pid is not None else None
        values = {
            "category": category,
            "device": device,
            "context": context,
            "stream": stream,
            "correlation_id": correlation,
            "global_pid": global_pid,
            "pid": process[0] if process else None,
            "process_name": process[1] if process else None,
            "grid": f"{grid_x},{grid_y},{grid_z}",
            "block": f"{block_x},{block_y},{block_z}",
            "registers_per_thread": registers,
            "static_shared_memory": static_shared,
            "dynamic_shared_memory": dynamic_shared,
        }
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=strings.get(
                        int(demangled_name),
                        strings.get(int(short_name), f"StringId {short_name}"),
                    ),
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight kernel"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values=values,
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    memcpy_rows = connection.execute(
        """
        SELECT m.start, m.end, m.deviceId, m.contextId, m.streamId,
               m.correlationId, m.globalPid, m.bytes, m.copyKind,
               copy.label, m.srcKind, src.label, m.dstKind, dst.label
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS copy ON copy.id = m.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS src ON src.id = m.srcKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS dst ON dst.id = m.dstKind
        ORDER BY m.start, m.end, m.deviceId, m.contextId, m.streamId,
                 m.correlationId
        """
    )
    for row in memcpy_rows:
        (
            start,
            end,
            device,
            context,
            stream,
            correlation,
            global_pid,
            byte_count,
            copy_kind,
            copy_label,
            src_kind,
            src_label,
            dst_kind,
            dst_label,
        ) = row
        category = "CUDA memcpy"
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            f"GPU {device} / context {context} / stream {stream}",
        )
        process = process_names.get(int(global_pid)) if global_pid is not None else None
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=f"Memcpy {copy_label or copy_kind}",
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight memcpy"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "device": device,
                            "context": context,
                            "stream": stream,
                            "correlation_id": correlation,
                            "global_pid": global_pid,
                            "pid": process[0] if process else None,
                            "process_name": process[1] if process else None,
                            "bytes": byte_count,
                            "copy_kind": copy_kind,
                            "copy_label": copy_label,
                            "source_kind": src_kind,
                            "source_label": src_label,
                            "destination_kind": dst_kind,
                            "destination_label": dst_label,
                        },
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    memset_rows = connection.execute(
        """
        SELECT start, end, deviceId, contextId, streamId, correlationId,
               globalPid, value, bytes, memKind
        FROM CUPTI_ACTIVITY_KIND_MEMSET
        ORDER BY start, end, deviceId, contextId, streamId, correlationId
        """
    )
    for start, end, device, context, stream, correlation, global_pid, value, byte_count, mem_kind in memset_rows:
        category = "CUDA memset"
        leaf = ensure_track(
            category,
            f"device:{device}:context:{context}:stream:{stream}",
            f"GPU {device} / context {context} / stream {stream}",
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name="Memset",
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "Nsight memset"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "device": device,
                            "context": context,
                            "stream": stream,
                            "correlation_id": correlation,
                            "global_pid": global_pid,
                            "value": value,
                            "bytes": byte_count,
                            "memory_kind": mem_kind,
                        },
                    ),
                ),
                category=category,
                correlation_id=(
                    _non_bool_int_or_none(correlation)
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
                endpoint_kind="device",
                correlation_scope=(
                    f"nsight-process:{int(global_pid)}"
                    if _non_bool_int_or_none(global_pid) is not None
                    else None
                ),
            )
        )
        counts[category] += 1

    metadata_count = 0
    nvtx_rows = connection.execute(
        """
        SELECT start, end, eventType, rangeId, text, globalTid, textId,
               domainId
        FROM NVTX_EVENTS
        ORDER BY start, end, eventType, globalTid, rangeId
        """
    )
    for start, end, event_type, range_id, text, global_tid, text_id, domain_id in nvtx_rows:
        if end is None or int(end) <= int(start):
            metadata_count += 1
            continue
        category = "NVTX ranges"
        leaf = ensure_track(
            category,
            f"tid:{global_tid}:domain:{domain_id}",
            f"Nsight globalTid {global_tid} / domain {domain_id}",
        )
        name = (
            str(text)
            if text
            else strings.get(int(text_id), f"NVTX range {range_id}")
        )
        result.append(
            _NativeSlice(
                spec=SliceSpec(
                    track_key=leaf,
                    name=name,
                    timestamp_ns=bridge.unix_to_canonical(
                        session_unix_ns + int(start)
                    ),
                    duration_ns=_positive_duration(start, end, "NVTX range"),
                    annotations=_nsys_annotations(
                        bridge,
                        native_start_ns=int(start),
                        values={
                            "category": category,
                            "event_type": event_type,
                            "range_id": range_id,
                            "global_tid": global_tid,
                            "domain_id": domain_id,
                        },
                    ),
                ),
                category=category,
                correlation_id=None,
                endpoint_kind="annotation",
            )
        )
        counts[category] += 1
    return result, tracks, counts, metadata_count


__all__ = [
    "SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS",
    "nsys_detail_result",
]
