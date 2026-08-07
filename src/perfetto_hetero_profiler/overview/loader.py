"""Strict read-only loaders for Phase 6 Overview inputs.

The Overview layer deliberately reuses the Phase 5 normalized-run loader and
trace planner.  This module adds the missing boundary around a *published*
Perfetto bundle: exact file layout, detached-manifest freshness, source
fingerprint matching, trace identity, and a fresh official Trace Processor
reconciliation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
from typing import Any, Callable, TypeVar

from ..perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    verify_stored_sidecar,
)
from ..perfetto.converter import (
    CONVERSION_MANIFEST_NAME,
    OUTPUT_ROOT_ID as PERFETTO_ROOT_ID,
    RBLN_NATIVE_TRACE_NAME,
    RBLN_NATIVE_VALIDATION_NAME,
    TRACE_NAME,
    TRACE_VALIDATION_NAME,
)
from ..perfetto.loader import LoadedHybridRun
from ..perfetto.native_details import (
    augment_trace_plan,
    build_native_detail_plan,
    native_validation_metadata,
)
from ..perfetto.planner import PlanBuildResult, build_trace_plan
from ..perfetto.tooling import ToolchainRuntime, resolve_toolchain
from ..perfetto.timeline_summary import (
    LEGACY_MAPPING_VERSION,
    TIMELINE_SUMMARY_MAPPING_VERSION,
    build_timeline_summary_context,
)
from ..perfetto.validation import validate_trace


_EXPECTED_PERFETTO_FILES = frozenset(
    {
        ARTIFACT_MANIFEST_NAME,
        ARTIFACT_VALIDATION_NAME,
        CONVERSION_MANIFEST_NAME,
        TRACE_NAME,
        TRACE_VALIDATION_NAME,
    }
)
_EXPECTED_RBLN_PERFETTO_FILES = frozenset(
    {
        *_EXPECTED_PERFETTO_FILES,
        RBLN_NATIVE_TRACE_NAME,
        RBLN_NATIVE_VALIDATION_NAME,
    }
)
_JSON_VALUE = TypeVar("_JSON_VALUE")


class OverviewInputError(RuntimeError):
    """An immutable Overview input failed its strict contract."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Stable path-free identity for a regular input file."""

    relative_path: str
    size_bytes: int
    sha256: str
    mtime_ns: int
    mode: int

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "mtime_ns": self.mtime_ns,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class PerfettoBundleIdentity:
    """Path-free identity for an exact five-file Phase 5 output."""

    files: tuple[FileIdentity, ...]
    inventory_sha256: str

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "file_count": len(self.files),
            "inventory_sha256": self.inventory_sha256,
            "files": [item.metadata for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class LoadedPerfettoBundle:
    """A matching Phase 5 Perfetto output, freshly reconciled."""

    root: Path
    conversion_manifest: dict[str, Any]
    stored_trace_validation: dict[str, Any]
    fresh_trace_validation: dict[str, Any]
    artifact_validation: dict[str, Any]
    identity: PerfettoBundleIdentity
    planning: PlanBuildResult
    toolchain: ToolchainRuntime

    @property
    def trace_path(self) -> Path:
        return self.root / TRACE_NAME


def _absolute_without_resolving(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def _require_real_directory(path: str | Path, *, description: str) -> Path:
    candidate = _absolute_without_resolving(Path(path))
    current = Path(candidate.anchor)
    try:
        file_stat = current.lstat()
    except OSError as error:
        raise OverviewInputError(
            f"{description} cannot be inspected: {current}: {error}"
        ) from error
    for index, part in enumerate(candidate.parts[1:]):
        current = current / part
        try:
            file_stat = current.lstat()
        except OSError as error:
            raise OverviewInputError(
                f"{description} cannot be inspected: {current}: {error}"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise OverviewInputError(
                f"{description} must not use a symlink component"
            )
        if (
            index < len(candidate.parts[1:]) - 1
            and not stat.S_ISDIR(file_stat.st_mode)
        ):
            raise OverviewInputError(
                f"{description} parent component must be a directory"
            )
    if not stat.S_ISDIR(file_stat.st_mode):
        raise OverviewInputError(f"{description} must be a directory")
    return candidate


def _same_state(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    return all(getattr(before, name) == getattr(after, name) for name in fields)


def _stable_regular_file(
    path: Path,
    *,
    relative_path: str,
) -> FileIdentity:
    try:
        before = path.lstat()
    except OSError as error:
        raise OverviewInputError(f"input file cannot be inspected: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OverviewInputError(f"input must be a real regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.lstat()
    except OSError as error:
        raise OverviewInputError(f"input file cannot be hashed: {path}") from error
    if not _same_state(before, after):
        raise OverviewInputError(f"input changed while it was read: {path}")
    return FileIdentity(
        relative_path=relative_path,
        size_bytes=after.st_size,
        sha256=digest.hexdigest(),
        mtime_ns=after.st_mtime_ns,
        mode=stat.S_IMODE(after.st_mode),
    )


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _stable_read(
    path: Path,
    reader: Callable[[Path], _JSON_VALUE],
    *,
    description: str,
) -> _JSON_VALUE:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OverviewInputError(f"{description} must be a real regular file")
    try:
        value = reader(path)
        after = path.lstat()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise OverviewInputError(f"invalid {description}: {error}") from error
    if not _same_state(before, after):
        raise OverviewInputError(f"{description} changed while it was read")
    return value


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    def read(candidate: Path) -> Any:
        return json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )

    value = _stable_read(path, read, description=description)
    if not isinstance(value, dict):
        raise OverviewInputError(f"{description} must be a JSON object")
    return value


def _exact_perfetto_files(root: Path) -> tuple[FileIdentity, ...]:
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise OverviewInputError("Perfetto directory cannot be enumerated") from error
    actual_names = {entry.name for entry in entries}
    if actual_names not in {
        _EXPECTED_PERFETTO_FILES,
        _EXPECTED_RBLN_PERFETTO_FILES,
    }:
        expected = (
            _EXPECTED_RBLN_PERFETTO_FILES
            if RBLN_NATIVE_TRACE_NAME in actual_names
            or RBLN_NATIVE_VALIDATION_NAME in actual_names
            else _EXPECTED_PERFETTO_FILES
        )
        missing = sorted(expected - actual_names)
        unexpected = sorted(actual_names - expected)
        raise OverviewInputError(
            "Perfetto directory must match exactly one supported file set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(
        _stable_regular_file(entry, relative_path=entry.name)
        for entry in entries
    )


def _bundle_identity(root: Path) -> PerfettoBundleIdentity:
    files = _exact_perfetto_files(root)
    canonical = json.dumps(
        [item.metadata for item in files],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return PerfettoBundleIdentity(
        files=files,
        inventory_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _normalized_input_metadata(loaded: LoadedHybridRun) -> dict[str, Any]:
    return {
        "valid": True,
        "closeout_manifest_sha256": loaded.closeout_manifest_sha256,
        "closeout_artifact_count": loaded.closeout_artifact_count,
        "roots": [
            item.metadata
            for item in sorted(
                loaded.root_fingerprints,
                key=lambda fingerprint: fingerprint.root_id,
            )
        ],
    }


def normalized_identity(loaded: LoadedHybridRun) -> tuple[object, ...]:
    """Return the exact immutable identity used by Phase 5 conversion."""

    return (
        loaded.manifest.run_id,
        loaded.closeout_manifest_sha256,
        loaded.closeout_artifact_count,
        tuple(
            (
                item.root_id,
                item.file_count,
                item.fingerprint_sha256,
            )
            for item in loaded.root_fingerprints
        ),
    )


def read_validated_source_json(
    loaded: LoadedHybridRun,
    *,
    root_id: str,
    relative_path: str,
) -> dict[str, Any]:
    """Read one fixed JSON object under an already validated closeout root."""

    if not isinstance(relative_path, str) or "\\" in relative_path:
        raise OverviewInputError("source JSON path must be a POSIX relative path")
    parts = PurePosixPath(relative_path).parts
    if (
        not parts
        or PurePosixPath(relative_path).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(relative_path).as_posix() != relative_path
    ):
        raise OverviewInputError("source JSON path is unsafe")
    roots = {
        fingerprint.root_id: fingerprint.root
        for fingerprint in loaded.root_fingerprints
    }
    if root_id not in roots:
        raise OverviewInputError(f"unknown closeout root id: {root_id!r}")
    current = roots[root_id]
    for index, part in enumerate(parts):
        current = current / part
        try:
            file_stat = current.lstat()
        except OSError as error:
            raise OverviewInputError(
                f"validated source JSON cannot be inspected: {root_id}:{relative_path}"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise OverviewInputError("validated source JSON path uses a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(file_stat.st_mode):
            raise OverviewInputError("validated source JSON parent is not a directory")
    return _read_json_object(
        current,
        description=f"validated source JSON {root_id}:{relative_path}",
    )


def perfetto_identity(perfetto_root: str | Path) -> PerfettoBundleIdentity:
    """Snapshot an exact published Perfetto bundle without executing TP."""

    root = _require_real_directory(perfetto_root, description="Perfetto output")
    return _bundle_identity(root)


def _assert_no_overlap(loaded: LoadedHybridRun, perfetto_root: Path) -> None:
    perfetto_resolved = perfetto_root.resolve(strict=True)
    for fingerprint in loaded.root_fingerprints:
        source = fingerprint.root.resolve(strict=True)
        if (
            perfetto_resolved == source
            or perfetto_resolved in source.parents
            or source in perfetto_resolved.parents
        ):
            raise OverviewInputError(
                "Perfetto output must not overlap a normalized source root"
            )


def _artifact_roots(
    loaded: LoadedHybridRun,
    perfetto_root: Path,
) -> dict[str, Path]:
    roots = {
        fingerprint.root_id: fingerprint.root
        for fingerprint in loaded.root_fingerprints
    }
    roots[PERFETTO_ROOT_ID] = perfetto_root
    return roots


def _mapping_version(manifest: dict[str, Any]) -> str:
    mapping = manifest.get("trace_mapping")
    if mapping is None:
        return LEGACY_MAPPING_VERSION
    if not isinstance(mapping, dict):
        raise OverviewInputError("Perfetto trace_mapping must be an object")
    version = mapping.get("mapping_version")
    if version not in {
        LEGACY_MAPPING_VERSION,
        TIMELINE_SUMMARY_MAPPING_VERSION,
    }:
        raise OverviewInputError(
            f"unsupported Perfetto trace mapping version: {version!r}"
        )
    return version


def _expected_query_count(
    mapping_version: str,
    manifest: dict[str, Any] | None = None,
) -> int:
    if mapping_version == LEGACY_MAPPING_VERSION:
        base = 10
    elif mapping_version == TIMELINE_SUMMARY_MAPPING_VERSION:
        base = 15
    else:
        raise OverviewInputError(
            f"unsupported Perfetto trace mapping version: {mapping_version!r}"
        )
    if manifest is None:
        return base
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise OverviewInputError("Perfetto conversion counts are invalid")
    native_count = 0
    for field in ("native_detail_slice_count", "native_detail_instant_count"):
        value = counts.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OverviewInputError(
                f"Perfetto conversion {field} is invalid"
            )
        native_count += value
    # Trace validation adds exactly one native-event semantics query whenever
    # converted native slices or instants are present. RBLN's separate,
    # unaligned trace does not change the canonical trace query inventory.
    return base + int(native_count > 0)


def _require_manifest_match(
    loaded: LoadedHybridRun,
    manifest: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "record_type",
        "status",
        "run_id",
        "source_mode",
        "source_profile_mode",
        "canonical_clock_domain_id",
        "input_validation",
        "trace",
        "trace_validation",
        "counts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise OverviewInputError(
            f"Perfetto conversion manifest is missing {missing[0]!r}"
        )
    expected = {
        "status": "succeeded",
        "run_id": loaded.manifest.run_id,
        "source_mode": loaded.manifest.mode.value,
        "source_profile_mode": loaded.manifest.profile_mode.value,
        "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
        "input_validation": _normalized_input_metadata(loaded),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise OverviewInputError(
                f"Perfetto conversion manifest {field} does not match source"
            )
    mapping_version = _mapping_version(manifest)
    trace = manifest.get("trace")
    if not isinstance(trace, dict) or trace.get("root_id") != PERFETTO_ROOT_ID:
        raise OverviewInputError("Perfetto trace root does not match contract")
    if trace.get("relative_path") != TRACE_NAME:
        raise OverviewInputError("Perfetto trace relative path does not match contract")
    trace_validation = manifest.get("trace_validation")
    if (
        not isinstance(trace_validation, dict)
        or trace_validation.get("root_id") != PERFETTO_ROOT_ID
        or trace_validation.get("relative_path") != TRACE_VALIDATION_NAME
        or trace_validation.get("valid") is not True
        or trace_validation.get("mismatches") != []
        or trace_validation.get("query_count")
        != _expected_query_count(mapping_version, manifest)
    ):
        raise OverviewInputError(
            "Perfetto conversion trace-validation summary is invalid"
        )


def _require_trace_identity(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    trace_identity = _stable_regular_file(
        root / TRACE_NAME,
        relative_path=TRACE_NAME,
    )
    trace = manifest["trace"]
    if (
        trace.get("size_bytes") != trace_identity.size_bytes
        or trace.get("sha256") != trace_identity.sha256
    ):
        raise OverviewInputError(
            "Perfetto trace size/SHA-256 differs from conversion manifest"
        )


def _require_stored_trace_validation(
    loaded: LoadedHybridRun,
    value: dict[str, Any],
    *,
    manifest: dict[str, Any],
) -> None:
    if value.get("valid") is not True or value.get("mismatches") != []:
        raise OverviewInputError("stored Perfetto trace validation is not valid")
    if value.get("run_id") != loaded.manifest.run_id:
        raise OverviewInputError("stored Perfetto trace validation run mismatch")
    if value.get("canonical_clock_domain_id") != loaded.canonical_clock_domain_id:
        raise OverviewInputError("stored Perfetto clock domain mismatch")
    queries = value.get("queries")
    mapping_version = _mapping_version(manifest)
    if (
        not isinstance(queries, list)
        or len(queries) != _expected_query_count(mapping_version, manifest)
        or any(
            not isinstance(query, dict) or query.get("matched") is not True
            for query in queries
        )
    ):
        raise OverviewInputError("stored Perfetto SQL validation is incomplete")
    trace = value.get("trace")
    if (
        not isinstance(trace, dict)
        or trace.get("size_bytes") != manifest["trace"].get("size_bytes")
        or trace.get("sha256") != manifest["trace"].get("sha256")
    ):
        raise OverviewInputError("stored Perfetto validation trace identity mismatch")


def load_matching_perfetto(
    loaded: LoadedHybridRun,
    perfetto_root: str | Path,
    *,
    trace_processor_path: Path | None = None,
) -> LoadedPerfettoBundle:
    """Load and freshly reconcile one exact matching Phase 5 output."""

    if not isinstance(loaded, LoadedHybridRun):
        raise TypeError("loaded must be a LoadedHybridRun")
    root = _require_real_directory(perfetto_root, description="Perfetto output")
    _assert_no_overlap(loaded, root)
    identity_before = _bundle_identity(root)
    manifest = _read_json_object(
        root / CONVERSION_MANIFEST_NAME,
        description="Perfetto conversion manifest",
    )
    stored_validation = _read_json_object(
        root / TRACE_VALIDATION_NAME,
        description="stored Perfetto trace validation",
    )
    _require_manifest_match(loaded, manifest)
    _require_trace_identity(root, manifest)
    _require_stored_trace_validation(
        loaded,
        stored_validation,
        manifest=manifest,
    )

    try:
        artifact_validation = verify_stored_sidecar(
            root / ARTIFACT_MANIFEST_NAME,
            _artifact_roots(loaded, root),
            output_root_id=PERFETTO_ROOT_ID,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise OverviewInputError(
            f"Perfetto detached artifact validation failed: {error}"
        ) from error
    if (
        artifact_validation.get("valid") is not True
        or artifact_validation.get("mismatches") != []
    ):
        raise OverviewInputError(
            "Perfetto detached artifact validation found mismatches"
        )

    toolchain = resolve_toolchain(trace_processor_path)
    mapping_version = _mapping_version(manifest)
    planning = build_trace_plan(
        loaded.manifest,
        loaded.events,
        loaded.metrics,
        canonical_clock_domain_id=loaded.canonical_clock_domain_id,
        native_envelopes=loaded.native_envelopes,
        timeline_summary=(
            build_timeline_summary_context(loaded)
            if mapping_version == TIMELINE_SUMMARY_MAPPING_VERSION
            else None
        ),
    )
    native = None
    if "native_details" in stored_validation:
        native = build_native_detail_plan(loaded, planning.plan)
        if native.summaries:
            planning = replace(
                planning,
                plan=augment_trace_plan(planning.plan, native),
            )
    fresh_validation = validate_trace(
        planning.plan,
        root / TRACE_NAME,
        toolchain=toolchain,
    )
    if native is not None:
        fresh_validation["native_details"] = native_validation_metadata(
            planning.plan,
            native,
        )
    if fresh_validation != stored_validation:
        raise OverviewInputError(
            "fresh official Trace Processor result differs from stored Phase 5 "
            "validation"
        )
    identity_after = _bundle_identity(root)
    if identity_after != identity_before:
        raise OverviewInputError(
            "Perfetto bundle changed while it was validated"
        )
    return LoadedPerfettoBundle(
        root=root,
        conversion_manifest=manifest,
        stored_trace_validation=stored_validation,
        fresh_trace_validation=fresh_validation,
        artifact_validation=artifact_validation,
        identity=identity_after,
        planning=planning,
        toolchain=toolchain,
    )


def reconciliation_summary(bundle: LoadedPerfettoBundle) -> dict[str, Any]:
    """Return deterministic, path-free evidence suitable for Overview JSON."""

    report = bundle.fresh_trace_validation
    queries = report["queries"]
    query_summaries = [
        {
            "name": query["name"],
            "row_count": query["row_count"],
            "rows_sha256": query["rows_sha256"],
            "expected_row_count": query["expected_row_count"],
            "expected_rows_sha256": query["expected_rows_sha256"],
            "matched": query["matched"],
        }
        for query in sorted(queries, key=lambda item: item["name"])
    ]
    toolchain = report["toolchain"]
    safe_toolchain = {
        name: toolchain[name]
        for name in (
            "filename",
            "version",
            "sha256",
            "perfetto_package_version",
            "protobuf_package_version",
            "trace_processor_rpc_api_version",
        )
        if name in toolchain
    }
    report_count_names = (
        "annotations",
        "counters",
        "dangling_flows",
        "flows",
        "import_errors",
        "native_policy",
        "process",
        "slices",
        "step_annotations",
        "tracks",
    )
    return {
        "valid": True,
        "trace": dict(report["trace"]),
        # Keep the Phase 6 external-report schema backward-compatible. New
        # mapping-specific query counts remain available in ``queries``.
        "counts": {
            name: report["counts"][name] for name in report_count_names
        },
        "query_count": len(query_summaries),
        "queries": query_summaries,
        "mismatches": [],
        "flow_endpoint_reconciliation": report[
            "flow_endpoint_reconciliation"
        ],
        "artifact_validation": {
            "valid": bundle.artifact_validation["valid"],
            "checked": bundle.artifact_validation["checked"],
            "mismatches": bundle.artifact_validation["mismatches"],
            "manifest_sha256": bundle.artifact_validation["manifest_sha256"],
        },
        "toolchain": safe_toolchain,
    }


def phase_duration_reconciliation(
    bundle: LoadedPerfettoBundle,
) -> list[dict[str, Any]]:
    """Reconcile event-planned integer phase durations with TP slice rows."""

    mapping = (
        ("latency.e2e", "request", "Request"),
        ("latency.prefill", "gpu_prefill", "GPU Prefill"),
        ("latency.kv_export", "kv_export", "KV Export"),
        ("latency.kv_transfer", "kv_transfer", "KV Transfer"),
        ("latency.kv_transform", "kv_transform", "KV Transform"),
        ("latency.decode", "npu_decode", "NPU Decode"),
        ("latency.sampling", "sampling", "Sampling"),
    )
    expected: dict[str, list[int]] = {
        slice_name: [] for _, _, slice_name in mapping
    }
    detail_keys = {
        track_key: slice_name for _, track_key, slice_name in mapping
    }
    for item in bundle.planning.plan.slices:
        slice_name = detail_keys.get(item.track_key)
        if slice_name is not None and item.name == slice_name:
            expected[slice_name].append(item.duration_ns)
    slice_query = next(
        query
        for query in bundle.fresh_trace_validation["queries"]
        if query["name"] == "slices"
    )
    detail_track_names = {
        bundle.planning.plan.track_by_key[track_key].name: slice_name
        for _, track_key, slice_name in mapping
    }
    actual: dict[str, list[int]] = {
        slice_name: [] for _, _, slice_name in mapping
    }
    summary_query = next(
        (
            query
            for query in bundle.fresh_trace_validation["queries"]
            if query["name"] == "timeline_summary_slices"
        ),
        None,
    )
    summary_rows = Counter(
        (
            row.get("track_name"),
            row.get("slice_name"),
            row.get("ts"),
            row.get("dur"),
        )
        for row in (
            summary_query.get("rows", [])
            if isinstance(summary_query, dict)
            else []
        )
    )
    for row in slice_query["rows"]:
        identity = (
            row.get("track_name"),
            row.get("slice_name"),
            row.get("ts"),
            row.get("dur"),
        )
        if summary_rows[identity]:
            summary_rows[identity] -= 1
            continue
        name = detail_track_names.get(row.get("track_name"))
        duration = row.get("dur")
        if (
            name in actual
            and isinstance(duration, int)
            and not isinstance(duration, bool)
        ):
            actual[name].append(duration)
    values: list[dict[str, Any]] = []
    for kpi_name, _, slice_name in mapping:
        expected_values = sorted(expected[slice_name])
        actual_values = sorted(actual[slice_name])
        values.append(
            {
                "kpi_name": kpi_name,
                "slice_name": slice_name,
                "slice_count": len(expected_values),
                "event_duration_ns": sum(expected_values),
                "perfetto_duration_ns": sum(actual_values),
                "matched": expected_values == actual_values,
            }
        )
    return values


def assert_perfetto_unchanged(
    before: LoadedPerfettoBundle,
    after: LoadedPerfettoBundle,
) -> None:
    """Reject mutation of a Phase 5 input during Overview publication."""

    if before.identity != after.identity:
        raise OverviewInputError(
            "immutable Perfetto bundle changed during Overview generation"
        )


__all__ = [
    "FileIdentity",
    "LoadedPerfettoBundle",
    "OverviewInputError",
    "PerfettoBundleIdentity",
    "assert_perfetto_unchanged",
    "load_matching_perfetto",
    "normalized_identity",
    "phase_duration_reconciliation",
    "perfetto_identity",
    "read_validated_source_json",
    "reconciliation_summary",
]
