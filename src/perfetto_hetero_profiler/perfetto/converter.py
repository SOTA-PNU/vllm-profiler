"""Transactional product conversion from normalized runs to Perfetto bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping

from ..schema import SCHEMA_VERSION
from .artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    build_manifest,
    validate_manifest,
    verify_stored_sidecar,
    write_json_exclusive,
)
from .loader import LoadedHybridRun, load_hybrid_run
from .native_details import (
    NativeDetailResult,
    augment_trace_plan,
    build_native_detail_plan,
    native_validation_metadata,
    request_focused_plan,
)
from .planner import NativeProfileEnvelope, PlanBuildResult, build_trace_plan
from .tooling import (
    PERFETTO_PACKAGE_VERSION,
    PERFETTO_UPSTREAM_REVISION,
    PERFETTO_WHEEL_FILENAME,
    PERFETTO_WHEEL_SHA256,
    PERFETTO_WHEEL_SOURCE,
    PROTOBUF_PACKAGE_VERSION,
    TRACE_PROCESSOR_RELEASE,
    TRACE_PROCESSOR_REVISION,
    TRACE_PROCESSOR_RPC_API_VERSION,
    TRACE_PROCESSOR_SHA256,
    TRACE_PROCESSOR_SIZE_BYTES,
    TRACE_PROCESSOR_SOURCE,
    ToolchainRuntime,
    resolve_toolchain,
)
from .validation import validate_native_perfetto_trace, validate_trace
from .writer import write_trace
from .timeline_summary import build_timeline_summary_context
from .trace_attributes import trace_attribute_validation_report


TRACE_NAME = "trace.pftrace"
REQUEST_FOCUSED_TRACE_NAME = "trace.request-focused.pftrace"
CONVERSION_MANIFEST_NAME = "conversion_manifest.json"
TRACE_VALIDATION_NAME = "trace_validation.json"
TRACE_ATTRIBUTE_VALIDATION_NAME = "trace_attributes_validation.json"
REQUEST_FOCUSED_VALIDATION_NAME = "trace.request-focused.validation.json"
RBLN_NATIVE_TRACE_NAME = "trace.rbln-native.pftrace"
RBLN_NATIVE_VALIDATION_NAME = "trace.rbln-native.validation.json"
CONVERSION_RECORD_TYPE = "perfetto_conversion_manifest"
OUTPUT_ROOT_ID = "conversion"

_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class PerfettoConversionError(RuntimeError):
    """A conversion could not be completed without weakening its contract."""


@dataclass(frozen=True, slots=True)
class PerfettoConversionConfig:
    """Inputs for one immutable, no-overwrite Perfetto conversion."""

    run_directory: Path
    output_directory: Path | None = None
    trace_processor_path: Path | None = None
    include_native_details: bool = False
    request_focused: bool = False


def plan_perfetto_conversion(
    config: PerfettoConversionConfig,
) -> dict[str, Any]:
    """Validate the full input and toolchain and return a write-free plan."""

    loaded, planning, native, toolchain, output = _prepare(config)
    focused_metadata: dict[str, Any] | None = None
    if config.request_focused:
        focused = request_focused_plan(planning.plan)
        focused_native_validation = (
            native_validation_metadata(
                focused,
                native,
                filtered_subset=True,
            )
            if config.include_native_details
            else None
        )
        if (
            focused_native_validation is not None
            and focused_native_validation["valid"] is not True
        ):
            raise PerfettoConversionError(
                "request-focused native detail plan validation failed"
            )
        focused_metadata = {
            "track_count": len(focused.tracks),
            "slice_count": len(focused.slices),
            "instant_count": len(focused.instants),
            "counter_count": len(focused.counters),
            "flow_count": len(focused.flows),
            "resource_telemetry_included": False,
            "timestamp_rebased": False,
            **_native_request_membership_metadata(native),
            "native_validation": focused_native_validation,
        }
    return {
        "status": "planned",
        "dry_run": True,
        "run_id": loaded.manifest.run_id,
        "output_directory": os.fspath(output),
        "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
        "input_validation": _input_validation_metadata(loaded),
        "counts": _conversion_counts(planning, native),
        "trace_mapping": _timeline_summary_mapping_metadata(planning, native),
        "native_profiles": _native_profile_metadata(loaded, native),
        "separate_native_traces": [
            trace.metadata for trace in native.separate_traces
        ],
        "include_native_details": config.include_native_details,
        "request_focused_trace": config.request_focused,
        "request_focused_plan": focused_metadata,
        "toolchain": _toolchain_metadata(toolchain),
        "overwrite": False,
        "hardware_execution": False,
    }


def convert_perfetto(
    config: PerfettoConversionConfig,
) -> dict[str, Any]:
    """Generate, validate, inventory, and atomically publish one trace bundle."""

    loaded, planning, native, toolchain, output = _prepare(config)
    parent = output.parent
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.perfetto-staging-",
            dir=parent,
        )
    )
    published = False
    try:
        trace_path = staging / TRACE_NAME
        write_trace(planning.plan, trace_path)
        trace_path.chmod(0o644)
        _fsync_file(trace_path)
        trace_size, trace_sha256 = _stable_file_identity(trace_path)

        trace_validation = validate_trace(
            planning.plan,
            trace_path,
            toolchain=toolchain,
        )
        if config.include_native_details:
            native_validation = native_validation_metadata(
                planning.plan,
                native,
            )
            if native_validation["valid"] is not True:
                raise PerfettoConversionError(
                    "native detail plan validation failed"
                )
            trace_validation["native_details"] = native_validation
        write_json_exclusive(
            staging / TRACE_VALIDATION_NAME,
            trace_validation,
        )
        attribute_validation = trace_attribute_validation_report(
            planning.plan.trace_attributes,
            trace_validation,
        )
        if attribute_validation["valid"] is not True:
            raise PerfettoConversionError(
                "trace attribute metadata validation failed"
            )
        write_json_exclusive(
            staging / TRACE_ATTRIBUTE_VALIDATION_NAME,
            attribute_validation,
        )

        request_trace: dict[str, object] | None = None
        request_validation: dict[str, Any] | None = None
        if config.request_focused:
            focused_plan = request_focused_plan(planning.plan)
            focused_path = staging / REQUEST_FOCUSED_TRACE_NAME
            write_trace(focused_plan, focused_path)
            focused_path.chmod(0o644)
            _fsync_file(focused_path)
            focused_size, focused_sha256 = _stable_file_identity(focused_path)
            request_validation = validate_trace(
                focused_plan,
                focused_path,
                toolchain=toolchain,
            )
            if config.include_native_details:
                focused_native_validation = native_validation_metadata(
                    focused_plan,
                    native,
                    filtered_subset=True,
                )
                if focused_native_validation["valid"] is not True:
                    raise PerfettoConversionError(
                        "request-focused native detail validation failed"
                    )
                request_validation["native_details"] = (
                    focused_native_validation
                )
            write_json_exclusive(
                staging / REQUEST_FOCUSED_VALIDATION_NAME,
                request_validation,
            )
            request_trace = {
                "root_id": OUTPUT_ROOT_ID,
                "relative_path": REQUEST_FOCUSED_TRACE_NAME,
                "format": "perfetto_protobuf",
                "size_bytes": focused_size,
                "sha256": focused_sha256,
                "timestamp_rebased": False,
                "resource_telemetry_included": False,
                **_native_request_membership_metadata(native),
            }

        separate_native_traces: list[dict[str, object]] = []
        for view in native.separate_traces:
            if (
                view.output_name != RBLN_NATIVE_TRACE_NAME
                or view.validation_name != RBLN_NATIVE_VALIDATION_NAME
            ):
                raise PerfettoConversionError(
                    "unsupported separate native trace output contract"
                )
            native_trace_path = staging / view.output_name
            _write_bytes_exclusive(native_trace_path, view.payload)
            native_trace_path.chmod(0o644)
            _fsync_file(native_trace_path)
            native_size, native_sha256 = _stable_file_identity(
                native_trace_path
            )
            if (native_size, native_sha256) != (
                view.size_bytes,
                view.sha256,
            ):
                raise PerfettoConversionError(
                    "native trace differs from immutable source"
                )
            native_trace_validation = validate_native_perfetto_trace(
                native_trace_path,
                toolchain=toolchain,
                profiler_type=view.profiler_type,
                expected_size_bytes=view.size_bytes,
                expected_sha256=view.sha256,
                expected_slice_count=view.expected_slice_count,
                expected_track_count=view.expected_track_count,
                expected_flow_count=view.expected_flow_count,
            )
            if native_trace_validation["valid"] is not True:
                raise PerfettoConversionError(
                    "separate native Perfetto trace validation failed"
                )
            write_json_exclusive(
                staging / view.validation_name,
                native_trace_validation,
            )
            separate_native_traces.append(
                {
                    "root_id": OUTPUT_ROOT_ID,
                    **view.metadata,
                    "validation": {
                        "root_id": OUTPUT_ROOT_ID,
                        "relative_path": view.validation_name,
                        "valid": True,
                        "mismatches": [],
                        "counts": native_trace_validation["counts"],
                    },
                    "byte_identical_to_source": True,
                }
            )

        conversion_manifest = _conversion_manifest(
            loaded=loaded,
            planning=planning,
            native=native,
            toolchain=toolchain,
            trace_size=trace_size,
            trace_sha256=trace_sha256,
            trace_validation=trace_validation,
            attribute_validation=attribute_validation,
            request_trace=request_trace,
            request_validation=request_validation,
            separate_native_traces=separate_native_traces,
        )
        write_json_exclusive(
            staging / CONVERSION_MANIFEST_NAME,
            conversion_manifest,
        )

        roots = _artifact_roots(loaded, staging)
        artifact_manifest = build_manifest(
            roots,
            output_root_id=OUTPUT_ROOT_ID,
            required_artifacts=_required_artifacts(config, native),
        )
        manifest_path = staging / ARTIFACT_MANIFEST_NAME
        write_json_exclusive(manifest_path, artifact_manifest)
        artifact_validation = validate_manifest(
            manifest_path,
            roots,
            output_root_id=OUTPUT_ROOT_ID,
        )
        if (
            artifact_validation.get("valid") is not True
            or artifact_validation.get("mismatches") != []
        ):
            raise PerfettoConversionError(
                "detached artifact validation found a mismatch"
            )
        write_json_exclusive(
            staging / ARTIFACT_VALIDATION_NAME,
            artifact_validation,
        )
        verify_stored_sidecar(
            manifest_path,
            roots,
            output_root_id=OUTPUT_ROOT_ID,
        )

        after = load_hybrid_run(loaded.root)
        _assert_input_unchanged(loaded, after)
        _publish_directory_no_replace(staging, output)
        published = True

        published_roots = _artifact_roots(after, output)
        published_validation = verify_stored_sidecar(
            output / ARTIFACT_MANIFEST_NAME,
            published_roots,
            output_root_id=OUTPUT_ROOT_ID,
        )
        final_size, final_sha256 = _stable_file_identity(output / TRACE_NAME)
        if (final_size, final_sha256) != (trace_size, trace_sha256):
            raise PerfettoConversionError(
                "published trace differs from the validated staging trace"
            )
        return {
            "status": "succeeded",
            "dry_run": False,
            "run_id": loaded.manifest.run_id,
            "output_directory": os.fspath(output),
            "trace": {
                "path": os.fspath(output / TRACE_NAME),
                "size_bytes": final_size,
                "sha256": final_sha256,
            },
            "counts": _conversion_counts(planning, native),
            "trace_mapping": _timeline_summary_mapping_metadata(
                planning,
                native,
            ),
            "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
            "clock_alignment_status": (
                (
                    "partial_derived"
                    if native.emitted_event_count
                    else "partial"
                )
                if loaded.native_envelopes
                else "not_applicable"
            ),
            "native_profiles": _native_profile_metadata(loaded, native),
            "input_validation": _input_validation_metadata(after),
            "trace_validation": {
                "valid": trace_validation["valid"],
                "query_count": len(trace_validation["queries"]),
                "mismatches": trace_validation["mismatches"],
            },
            "trace_attribute_validation": {
                "valid": attribute_validation["valid"],
                "attribute_count": attribute_validation["attribute_count"],
                "integer_count": attribute_validation["integer_count"],
                "string_count": attribute_validation["string_count"],
                "mismatches": attribute_validation["mismatches"],
            },
            "artifact_validation": {
                "valid": published_validation["valid"],
                "checked": published_validation["checked"],
                "mismatches": published_validation["mismatches"],
                "manifest_sha256": published_validation["manifest_sha256"],
            },
            "toolchain": _toolchain_metadata(toolchain),
            "request_focused_trace": (
                {
                    "path": os.fspath(
                        output / REQUEST_FOCUSED_TRACE_NAME
                    ),
                    "size_bytes": request_trace["size_bytes"],
                    "sha256": request_trace["sha256"],
                }
                if request_trace is not None
                else None
            ),
            "separate_native_traces": separate_native_traces,
            "hardware_execution": False,
        }
    finally:
        if not published:
            _remove_owned_staging(staging, parent=parent, output_name=output.name)


def _prepare(
    config: PerfettoConversionConfig,
) -> tuple[
    LoadedHybridRun,
    PlanBuildResult,
    NativeDetailResult,
    ToolchainRuntime,
    Path,
]:
    if not isinstance(config, PerfettoConversionConfig):
        raise TypeError("config must be PerfettoConversionConfig")
    if not isinstance(config.include_native_details, bool):
        raise TypeError("include_native_details must be bool")
    if not isinstance(config.request_focused, bool):
        raise TypeError("request_focused must be bool")
    loaded = load_hybrid_run(config.run_directory)
    output = _output_path(loaded, config.output_directory)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    toolchain = resolve_toolchain(config.trace_processor_path)
    correlations = {
        event.attributes.get("hybrid.correlation_id")
        for event in loaded.events
        if event.event_name == "request_received"
    }
    timeline_summary = (
        build_timeline_summary_context(loaded)
        if len(correlations) == 1
        else None
    )
    planning = build_trace_plan(
        loaded.manifest,
        loaded.events,
        loaded.metrics,
        canonical_clock_domain_id=loaded.canonical_clock_domain_id,
        native_envelopes=loaded.native_envelopes,
        timeline_summary=timeline_summary,
    )
    native = (
        build_native_detail_plan(loaded, planning.plan)
        if config.include_native_details
        else NativeDetailResult()
    )
    if native.summaries:
        planning = replace(
            planning,
            plan=augment_trace_plan(planning.plan, native),
        )
    return loaded, planning, native, toolchain, output


def _output_path(
    loaded: LoadedHybridRun,
    requested: Path | None,
) -> Path:
    output = (
        loaded.root.with_name(f"{loaded.root.name}-perfetto")
        if requested is None
        else _absolute_without_resolving(Path(requested))
    )
    if requested is None:
        output = _absolute_without_resolving(output)
    parent = output.parent
    try:
        parent_stat = parent.lstat()
    except OSError as error:
        raise PerfettoConversionError(
            f"output parent cannot be inspected: {parent}: {error}"
        ) from error
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise PerfettoConversionError(
            f"output parent must be a real directory: {parent}"
        )
    if not output.name or output.name in {".", ".."}:
        raise PerfettoConversionError("output directory name is unsafe")

    output_resolved = output.resolve(strict=False)
    for fingerprint in loaded.root_fingerprints:
        source_resolved = fingerprint.root.resolve(strict=True)
        if (
            output_resolved == source_resolved
            or output_resolved in source_resolved.parents
            or source_resolved in output_resolved.parents
        ):
            raise PerfettoConversionError(
                "output directory must not overlap an immutable input root"
            )
    return output


def _absolute_without_resolving(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def _input_validation_metadata(loaded: LoadedHybridRun) -> dict[str, Any]:
    return {
        "valid": True,
        "closeout_manifest_sha256": loaded.closeout_manifest_sha256,
        "closeout_artifact_count": loaded.closeout_artifact_count,
        "roots": [
            fingerprint.metadata
            for fingerprint in sorted(
                loaded.root_fingerprints,
                key=lambda item: item.root_id,
            )
        ],
    }


def _native_request_membership_metadata(
    native: NativeDetailResult,
) -> dict[str, object]:
    if native.emitted_event_count:
        return {
            "native_selection_policy": (
                "mapped_point_overlap_with_partial_alignment"
            ),
            "native_request_membership_proven": False,
            "native_request_membership_status": (
                "not_proven_partial_alignment_point_overlap"
            ),
        }
    if native.separate_traces:
        return {
            "native_selection_policy": (
                "not_applicable_separate_unaligned_native_trace"
            ),
            "native_request_membership_proven": None,
            "native_request_membership_status": (
                "not_applicable_separate_unaligned_native_trace"
            ),
        }
    return {
        "native_selection_policy": "not_applicable_no_emitted_native_events",
        "native_request_membership_proven": None,
        "native_request_membership_status": (
            "not_applicable_no_emitted_native_events"
        ),
    }


def _native_profile_metadata(
    loaded: LoadedHybridRun,
    native: NativeDetailResult | None = None,
) -> list[dict[str, Any]]:
    source_by_role = loaded.source_by_role
    detail_by_type = {
        item.profiler_type: item
        for item in (() if native is None else native.summaries)
    }
    separate_types = {
        item.profiler_type
        for item in (() if native is None else native.separate_traces)
    }
    values: list[dict[str, Any]] = []
    for envelope in sorted(
        loaded.native_envelopes,
        key=lambda item: (item.source_role, item.profiler_type),
    ):
        source = source_by_role[envelope.source_role]
        native_artifacts = [
            artifact
            for artifact in source.artifacts
            if (
                artifact.clock_domain_id == envelope.native_clock_domain
                or (
                    envelope.profiler_type == "gpu_nsys"
                    and artifact.format == "sqlite"
                    and artifact.relative_path.endswith(".sqlite")
                )
            )
        ]
        value = {
            **_envelope_metadata(envelope),
            "artifact_references": [
                {
                    "root_id": envelope.source_role,
                    "relative_path": artifact.relative_path,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                }
                for artifact in sorted(
                    native_artifacts,
                    key=lambda item: item.relative_path,
                )
            ],
        }
        detail = detail_by_type.get(envelope.profiler_type)
        if detail is not None:
            value.update(detail.metadata)
            value["native_event_alignment"] = detail.alignment_status
            value["separate_native_trace_published"] = (
                envelope.profiler_type in separate_types
            )
            if envelope.profiler_type == "npu_rbln":
                value.update(
                    {
                        "opaque_rbln_pb": False,
                        "rbln_pb_classification": (
                            "perfetto_compatible_rbln_trace"
                        ),
                        "rbln_pb_structure_analysis": (
                            "official_perfetto_protobuf_schema"
                        ),
                        "rbln_pb_raw_bytes_embedded": False,
                        "rbln_pb_raw_bytes_embedded_in_canonical_trace": False,
                        "rbln_pb_canonical_merge": False,
                    }
                )
        values.append(value)
    return values


def _envelope_metadata(
    envelope: NativeProfileEnvelope,
) -> dict[str, Any]:
    result = asdict(envelope)
    result["native_details_emitted"] = False
    result["native_event_alignment"] = "unaligned"
    if envelope.profiler_type == "npu_rbln":
        result["opaque_rbln_pb"] = False
        result["rbln_pb_classification"] = "perfetto_compatible_rbln_trace"
        result["rbln_pb_structure_analysis"] = (
            "deferred_to_official_trace_processor"
        )
        result["rbln_pb_raw_bytes_embedded"] = False
        result["rbln_pb_raw_bytes_embedded_in_canonical_trace"] = False
        result["rbln_pb_canonical_merge"] = False
    return result


def _toolchain_metadata(toolchain: ToolchainRuntime) -> dict[str, Any]:
    return {
        "python_package": {
            "name": "perfetto",
            "version": PERFETTO_PACKAGE_VERSION,
            "wheel_filename": PERFETTO_WHEEL_FILENAME,
            "wheel_sha256": PERFETTO_WHEEL_SHA256,
            "source": PERFETTO_WHEEL_SOURCE,
            "upstream_revision": PERFETTO_UPSTREAM_REVISION,
        },
        "protobuf": {
            "name": "protobuf",
            "version": PROTOBUF_PACKAGE_VERSION,
        },
        "trace_processor": {
            **toolchain.metadata,
            "release": TRACE_PROCESSOR_RELEASE,
            "revision": TRACE_PROCESSOR_REVISION,
            "size_bytes": TRACE_PROCESSOR_SIZE_BYTES,
            "rpc_api_version": TRACE_PROCESSOR_RPC_API_VERSION,
            "sha256": TRACE_PROCESSOR_SHA256,
            "source": TRACE_PROCESSOR_SOURCE,
        },
        "fetch_latest_trace_processor": False,
        "network_required_after_preparation": False,
    }


def _conversion_manifest(
    *,
    loaded: LoadedHybridRun,
    planning: PlanBuildResult,
    native: NativeDetailResult,
    toolchain: ToolchainRuntime,
    trace_size: int,
    trace_sha256: str,
    trace_validation: Mapping[str, Any],
    attribute_validation: Mapping[str, Any],
    request_trace: Mapping[str, object] | None,
    request_validation: Mapping[str, Any] | None,
    separate_native_traces: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": CONVERSION_RECORD_TYPE,
        "status": "succeeded",
        "run_id": loaded.manifest.run_id,
        "source_mode": loaded.manifest.mode.value,
        "source_profile_mode": loaded.manifest.profile_mode.value,
        "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
        "clock_policy": {
            "canonical_events": "direct",
            "available_resource_metrics": "direct",
            "unavailable_resource_metrics": "omitted_not_zero_filled",
            "native_profiler_events": (
                "partial_derived_with_explicit_uncertainty"
                if native.emitted_event_count
                else (
                    "separate_native_clock_trace_without_rebase"
                    if native.separate_traces
                    else "unaligned_not_emitted"
                )
            ),
            "host_api_capture_boundary": (
                "partial_envelope"
                if loaded.native_envelopes
                else "not_applicable"
            ),
            "timestamp_fallback": False,
        },
        "flow_policy": {
            "explicit_correlation_required": True,
            "timestamp_proximity_fallback": False,
            "emitted_flow_count": len(planning.plan.flows),
            "representative_location": "detailed_tracks_only",
            "timeline_summary_flow_copy_count": 0,
        },
        "trace_mapping": _timeline_summary_mapping_metadata(planning, native),
        "trace": {
            "root_id": OUTPUT_ROOT_ID,
            "relative_path": TRACE_NAME,
            "format": "perfetto_protobuf",
            "size_bytes": trace_size,
            "sha256": trace_sha256,
        },
        "counts": _conversion_counts(planning, native),
        "track_names": sorted(track.name for track in planning.plan.tracks),
        "input_validation": _input_validation_metadata(loaded),
        "native_profiles": _native_profile_metadata(loaded, native),
        "separate_native_traces": separate_native_traces,
        "rbln_pb_policy": {
            "classification": (
                "perfetto_compatible_rbln_trace"
                if any(
                    item.profiler_type == "npu_rbln"
                    for item in native.summaries
                )
                else (
                    "perfetto_compatible_rbln_trace"
                    if any(
                        envelope.profiler_type == "npu_rbln"
                        for envelope in loaded.native_envelopes
                    )
                    else "not_applicable"
                )
            ),
            "structure_analysis": (
                "official_perfetto_protobuf_schema"
                if any(
                    item.profiler_type == "npu_rbln"
                    for item in native.summaries
                )
                else (
                    "deferred_to_official_trace_processor"
                    if any(
                        envelope.profiler_type == "npu_rbln"
                        for envelope in loaded.native_envelopes
                    )
                    else "not_applicable"
                )
            ),
            "raw_bytes_embedded": False,
            "raw_bytes_embedded_in_canonical_trace": False,
            "canonical_merge": False,
            "separate_native_trace_published": bool(
                native.separate_traces
            ),
        },
        "trace_validation": {
            "root_id": OUTPUT_ROOT_ID,
            "relative_path": TRACE_VALIDATION_NAME,
            "valid": trace_validation["valid"],
            "query_count": len(trace_validation["queries"]),
            "mismatches": trace_validation["mismatches"],
        },
        "trace_attribute_validation": {
            "root_id": OUTPUT_ROOT_ID,
            "relative_path": TRACE_ATTRIBUTE_VALIDATION_NAME,
            "valid": attribute_validation["valid"],
            "attribute_count": attribute_validation["attribute_count"],
            "integer_count": attribute_validation["integer_count"],
            "string_count": attribute_validation["string_count"],
            "mismatches": attribute_validation["mismatches"],
        },
        "toolchain": _toolchain_metadata(toolchain),
        "request_focused_trace": request_trace,
        "request_focused_validation": (
            {
                "root_id": OUTPUT_ROOT_ID,
                "relative_path": REQUEST_FOCUSED_VALIDATION_NAME,
                "valid": request_validation["valid"],
                "query_count": len(request_validation["queries"]),
                "mismatches": request_validation["mismatches"],
            }
            if request_validation is not None
            else None
        ),
        "determinism": {
            "protobuf_deterministic_serialization": True,
            "stable_track_and_flow_identifiers": True,
            "volatile_runtime_values_in_trace": False,
            "output_overwrite": False,
        },
        "hardware_execution": False,
    }


def _timeline_summary_mapping_metadata(
    planning: PlanBuildResult,
    native: NativeDetailResult | None = None,
) -> dict[str, Any]:
    plan = planning.plan
    tracks = plan.track_by_key
    summary_tracks = sorted(
        (
            {
                "key": track.key,
                "uuid": track.uuid,
                "name": track.name,
                "kind": track.kind,
                "unit": track.unit,
                "parent_key": track.parent_key,
                "child_ordering": track.child_ordering,
                "sibling_order_rank": track.sibling_order_rank,
            }
            for track in plan.tracks
            if track.key.startswith("summary.")
        ),
        key=lambda item: item["key"],
    )
    resource_tracks = sorted(
        (
            {
                "key": track.key,
                "uuid": track.uuid,
                "name": track.name,
                "kind": track.kind,
                "unit": track.unit,
                "parent_key": track.parent_key,
                "child_ordering": track.child_ordering,
                "sibling_order_rank": track.sibling_order_rank,
            }
            for track in plan.tracks
            if track.key.startswith("telemetry.resources")
            or (
                track.parent_key is not None
                and track.parent_key.startswith("telemetry.resources")
            )
        ),
        key=lambda item: item["key"],
    )
    root = tracks.get("summary.root")
    if root is None:
        return {
            "kind": "normalized_events",
            "mapping_version": plan.mapping_version,
            "source_identity_sha256": plan.source_identity_sha256,
            "root_track": None,
            "timeline_summary_hierarchy": [],
            "resource_telemetry_hierarchy": resource_tracks,
            "trace_attributes": [
                {"key": item.key, "value": item.value}
                for item in plan.trace_attributes
            ],
            "ordering_policy": {
                "descriptor_metadata": (
                    "TrackDescriptor.child_ordering_and_sibling_order_rank"
                ),
                "ui_guarantee": "hint_not_absolute_cross_version_order",
            },
            "flow_policy": {
                "representative_location": "detailed_tracks_only",
                "explicit_correlation_required": True,
                "timestamp_proximity_fallback": False,
            },
            "native_clock_policy": {
                "native_details_emitted": bool(
                    native is not None and native.emitted_event_count
                ),
                "unaligned_native_events_inferred": False,
                "partial_derived_native_events": bool(
                    native is not None and native.emitted_event_count
                ),
                "host_api_boundary_only": not bool(
                    native is not None and native.emitted_event_count
                ),
            },
        }
    kpi_mappings: list[dict[str, Any]] = []
    for counter in sorted(
        (
            item
            for item in plan.counters
            if item.track_key.startswith("summary.kpi:")
        ),
        key=lambda item: item.track_key,
    ):
        track = tracks[counter.track_key]
        annotations = dict(counter.annotations)
        kpi_mappings.append(
            {
                "identity": annotations["hetero.kpi_identity"],
                "canonical_name": annotations["hetero.kpi_name"],
                "track_key": track.key,
                "track_uuid": track.uuid,
                "track_name": track.name,
                "parent_key": track.parent_key,
                "canonical_unit": track.unit,
                "availability": annotations["hetero.availability"],
                "anchor_event_id": annotations["hetero.anchor_event_id"],
                "source_event_ids_json": annotations[
                    "hetero.source_event_ids_json"
                ],
            }
        )
    data_quality = [
        item
        for item in plan.instants
        if item.track_key == "summary.data_quality"
        and item.name == "Data Quality status"
    ]
    if len(data_quality) != 1:
        raise PerfettoConversionError(
            "timeline summary plan must have one Data Quality status instant"
        )
    quality = dict(data_quality[0].annotations)
    return {
        "kind": "timeline_summary",
        "mapping_version": plan.mapping_version,
        "source_identity_sha256": plan.source_identity_sha256,
        "root_track": {
            "key": root.key,
            "name": root.name,
            "uuid": root.uuid,
            "uuid_derivation": (
                "sha256(run_id NUL "
                "'track:' + mapping_version + ':' + source_identity_sha256 "
                "NUL track_key) first_8_bytes_big_endian signed_63_bit_nonzero"
            ),
        },
        "timeline_summary_hierarchy": summary_tracks,
        "resource_telemetry_hierarchy": resource_tracks,
        "trace_attributes": [
            {"key": item.key, "value": item.value}
            for item in plan.trace_attributes
        ],
        "ordering_policy": {
            "descriptor_metadata": (
                "TrackDescriptor.child_ordering_and_sibling_order_rank"
            ),
            "ui_guarantee": "hint_not_absolute_cross_version_order",
        },
        "kpi_counter_mapping": kpi_mappings,
        "unavailable_handling": {
            "counter_policy": "omitted_not_zero_filled",
            "count": quality["hetero.unavailable_kpi_count"],
            "reasons_json": quality["hetero.unavailable_kpis_json"],
        },
        "resource_grouping": {
            "policy": quality["hetero.resource_grouping"],
            "device_identity_preserved": True,
            "counter_samples_copied": False,
            "time_scope": quality["hetero.resource_time_scope"],
            "pre_request_sample_count": quality[
                "hetero.pre_request_resource_sample_count"
            ],
            "pre_request_duration_ns": quality[
                "hetero.pre_request_resource_duration_ns"
            ],
            "warmup_interval_status": quality[
                "hetero.warmup_interval_status"
            ],
        },
        "flow_policy": {
            "representative_location": "detailed_tracks_only",
            "explicit_correlation_required": True,
            "timestamp_proximity_fallback": False,
        },
        "native_clock_policy": {
            "native_details_emitted": bool(
                native is not None and native.emitted_event_count
            ),
            "unaligned_native_events_inferred": False,
            "partial_derived_native_events": bool(
                native is not None and native.emitted_event_count
            ),
            "host_api_boundary_only": not bool(
                native is not None and native.emitted_event_count
            ),
            "rbln_pb_structure_analysis": quality[
                "hetero.rbln_pb_structure_analysis"
            ],
        },
    }


def _conversion_counts(
    planning: PlanBuildResult,
    native: NativeDetailResult,
) -> dict[str, int]:
    result = asdict(planning.metadata)
    result.update(
        {
            "emitted_track_count": len(planning.plan.tracks),
            "emitted_slice_count": len(planning.plan.slices),
            "emitted_instant_count": len(planning.plan.instants),
            "emitted_counter_count": len(planning.plan.counters),
            "emitted_flow_count": len(planning.plan.flows),
            "native_detail_track_count": len(native.tracks),
            "native_detail_slice_count": len(native.slices),
            "native_detail_instant_count": len(native.instants),
            "native_detail_flow_count": len(native.flows),
            "separate_native_trace_count": len(native.separate_traces),
        }
    )
    return result


def _artifact_roots(
    loaded: LoadedHybridRun,
    output_root: Path,
) -> dict[str, Path]:
    roots = {
        fingerprint.root_id: fingerprint.root
        for fingerprint in loaded.root_fingerprints
    }
    roots[OUTPUT_ROOT_ID] = output_root
    return roots


def _required_artifacts(
    config: PerfettoConversionConfig,
    native: NativeDetailResult,
) -> tuple[tuple[str, str], ...]:
    required = [
        ("coordinator", "result.json"),
        (OUTPUT_ROOT_ID, CONVERSION_MANIFEST_NAME),
        (OUTPUT_ROOT_ID, TRACE_NAME),
        (OUTPUT_ROOT_ID, TRACE_VALIDATION_NAME),
        (OUTPUT_ROOT_ID, TRACE_ATTRIBUTE_VALIDATION_NAME),
        ("gpu", "manifest.json"),
        ("hybrid", "artifacts/artifacts.jsonl"),
        ("hybrid", "clocks/clock_domains.jsonl"),
        ("hybrid", "clocks/transforms.jsonl"),
        ("hybrid", "events/events.jsonl"),
        ("hybrid", "manifest.json"),
        ("hybrid", "metrics/metrics.jsonl"),
        ("npu", "manifest.json"),
        ("recovery", "recovery_result.json"),
    ]
    if config.request_focused:
        required.extend(
            (
                (OUTPUT_ROOT_ID, REQUEST_FOCUSED_TRACE_NAME),
                (OUTPUT_ROOT_ID, REQUEST_FOCUSED_VALIDATION_NAME),
            )
        )
    for trace in native.separate_traces:
        required.extend(
            (
                (OUTPUT_ROOT_ID, trace.output_name),
                (OUTPUT_ROOT_ID, trace.validation_name),
            )
        )
    return tuple(required)


def _assert_input_unchanged(
    before: LoadedHybridRun,
    after: LoadedHybridRun,
) -> None:
    before_identity = (
        before.manifest.run_id,
        before.closeout_manifest_sha256,
        before.closeout_artifact_count,
        tuple(
            (
                item.root_id,
                item.file_count,
                item.fingerprint_sha256,
            )
            for item in before.root_fingerprints
        ),
    )
    after_identity = (
        after.manifest.run_id,
        after.closeout_manifest_sha256,
        after.closeout_artifact_count,
        tuple(
            (
                item.root_id,
                item.file_count,
                item.fingerprint_sha256,
            )
            for item in after.root_fingerprints
        ),
    )
    if after_identity != before_identity:
        raise PerfettoConversionError(
            "immutable input fingerprint changed during conversion"
        )


def _stable_file_identity(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise PerfettoConversionError(f"output is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    state = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    if state != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PerfettoConversionError(f"output changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise PerfettoConversionError("native trace payload must be non-empty")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short native trace write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(staging: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PerfettoConversionError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(staging),
        _AT_FDCWD,
        os.fsencode(output),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"output already exists: {output}")
        if error_number in {errno.ENOSYS, errno.EINVAL}:
            raise PerfettoConversionError(
                "atomic no-replace directory publication is unsupported"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(output),
        )
    _fsync_directory(output.parent)


def _remove_owned_staging(
    staging: Path,
    *,
    parent: Path,
    output_name: str,
) -> None:
    expected_prefix = f".{output_name}.perfetto-staging-"
    if (
        staging.parent != parent
        or not staging.name.startswith(expected_prefix)
        or staging.is_symlink()
    ):
        raise PerfettoConversionError(
            f"refusing to remove an unowned staging path: {staging}"
        )
    if staging.is_dir():
        shutil.rmtree(staging)


__all__ = [
    "CONVERSION_MANIFEST_NAME",
    "CONVERSION_RECORD_TYPE",
    "PerfettoConversionConfig",
    "PerfettoConversionError",
    "REQUEST_FOCUSED_TRACE_NAME",
    "REQUEST_FOCUSED_VALIDATION_NAME",
    "TRACE_NAME",
    "TRACE_VALIDATION_NAME",
    "convert_perfetto",
    "plan_perfetto_conversion",
]
