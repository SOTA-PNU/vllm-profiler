"""Strict read-only loaders for Overview inputs.

The boundary around a *published* Perfetto bundle: symlink-free TOCTOU-checked
reads, the exact supported file set and its inventory identity, agreement with
the normalized run, and path-free evidence.  :func:`load_matching_perfetto`
runs the official Trace Processor and re-verifies identity afterwards.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    REQUEST_FOCUSED_TRACE_NAME,
    REQUEST_FOCUSED_VALIDATION_NAME,
    RBLN_NATIVE_TRACE_NAME,
    RBLN_NATIVE_VALIDATION_NAME,
    TRACE_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
    TRACE_VALIDATION_NAME,
)
from ..perfetto.loader import LoadedHybridRun
from ..perfetto.model import base_track_key
from ..perfetto.native_details import (
    augment_trace_plan,
    build_native_detail_plan,
    native_validation_metadata,
)
from ..perfetto.planner import PlanBuildResult, build_trace_plan
from ..perfetto.timeline_summary import (
    LEGACY_MAPPING_VERSION,
    TIMELINE_SUMMARY_MAPPING_VERSION,
    build_timeline_summary_context,
)
from ..perfetto.tooling import ToolchainRuntime, resolve_toolchain
from ..perfetto.trace_attributes import TRACE_ATTRIBUTE_NAMESPACE
from ..perfetto.validation import summarize_trace_validation, validate_trace
from ..schema.catalog import PHASE_RECONCILIATION_METRICS, STAGE_BY_METRIC
from ..support.files import sha256_file


_JSON_VALUE = TypeVar("_JSON_VALUE")
_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")


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


def _absolute_without_resolving(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def require_real_directory(path: str | Path, *, description: str) -> Path:
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
    return all(
        getattr(before, name) == getattr(after, name) for name in _IDENTITY_FIELDS
    )


def _stable_regular_file(path: Path, *, relative_path: str) -> FileIdentity:
    try:
        before = path.lstat()
    except OSError as error:
        raise OverviewInputError(f"input file cannot be inspected: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OverviewInputError(f"input must be a real regular file: {path}")
    try:
        digest = sha256_file(path)
        after = path.lstat()
    except OSError as error:
        raise OverviewInputError(f"input file cannot be hashed: {path}") from error
    if not _same_state(before, after):
        raise OverviewInputError(f"input changed while it was read: {path}")
    return FileIdentity(
        relative_path=relative_path,
        size_bytes=after.st_size,
        sha256=digest,
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


def read_json_object(path: Path, *, description: str) -> dict[str, Any]:
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
    return read_json_object(
        current,
        description=f"validated source JSON {root_id}:{relative_path}",
    )


_EXPECTED_PERFETTO_FILES = frozenset(
    {
        ARTIFACT_MANIFEST_NAME,
        ARTIFACT_VALIDATION_NAME,
        CONVERSION_MANIFEST_NAME,
        TRACE_NAME,
        TRACE_VALIDATION_NAME,
    }
)
_OPTIONAL_FILE_GROUPS = (
    (
        {RBLN_NATIVE_TRACE_NAME, RBLN_NATIVE_VALIDATION_NAME},
        "RBLN native trace",
    ),
    ({TRACE_ATTRIBUTE_VALIDATION_NAME}, "trace attributes"),
    (
        {REQUEST_FOCUSED_TRACE_NAME, REQUEST_FOCUSED_VALIDATION_NAME},
        "request-focused trace",
    ),
)


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    """Exact path-free file-set identity, proven by one inventory hash."""

    files: tuple[FileIdentity, ...]
    inventory_sha256: str

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "file_count": len(self.files),
            "inventory_sha256": self.inventory_sha256,
            "files": [item.metadata for item in self.files],
        }


PerfettoBundleIdentity = BundleIdentity  # historical public name


def inventory_identity(files: tuple[FileIdentity, ...]) -> BundleIdentity:
    """Hash an exact file inventory into a stable, path-free identity."""
    payload = json.dumps(
        [item.metadata for item in files],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return BundleIdentity(
        files=files,
        inventory_sha256=hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class LoadedPerfettoBundle:
    """A matching Perfetto output, freshly reconciled."""

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


def _exact_perfetto_files(root: Path) -> tuple[FileIdentity, ...]:
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise OverviewInputError("Perfetto directory cannot be enumerated") from error
    actual_names = {entry.name for entry in entries}
    expected = set(_EXPECTED_PERFETTO_FILES)
    partial_groups: list[str] = []
    for names, description in _OPTIONAL_FILE_GROUPS:
        present = names & actual_names
        if present and present != names:
            partial_groups.append(description)
        if present:
            expected.update(names)
    if partial_groups or actual_names != expected:
        missing = sorted(expected - actual_names)
        unexpected = sorted(actual_names - expected)
        raise OverviewInputError(
            "Perfetto directory must match exactly one supported file set; "
            f"missing={missing}, unexpected={unexpected}, "
            f"partial_groups={sorted(partial_groups)}"
        )
    return tuple(
        _stable_regular_file(entry, relative_path=entry.name)
        for entry in entries
    )


def _bundle_identity(root: Path) -> BundleIdentity:
    return inventory_identity(_exact_perfetto_files(root))


def perfetto_identity(perfetto_root: str | Path) -> PerfettoBundleIdentity:
    """Snapshot an exact published Perfetto bundle without executing TP."""

    root = require_real_directory(perfetto_root, description="Perfetto output")
    return _bundle_identity(root)


def assert_perfetto_unchanged(
    before: LoadedPerfettoBundle,
    after: LoadedPerfettoBundle,
) -> None:
    """Reject mutation of a conversion input during Overview publication."""

    if before.identity != after.identity:
        raise OverviewInputError(
            "immutable Perfetto bundle changed during Overview generation"
        )


_REQUIRED_MANIFEST_FIELDS = {
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
_REQUIRED_QUERY_FIELDS = {
    "name",
    "sql",
    "columns",
    "row_count",
    "rows_sha256",
    "expected_row_count",
    "expected_rows_sha256",
    "matched",
}
_OPTIONAL_QUERY_FIELDS = {"rows", "rows_sha256_method"}
_NATIVE_COUNT_FIELDS = (
    "native_detail_slice_count",
    "native_detail_instant_count",
)


def normalized_input_metadata(loaded: LoadedHybridRun) -> dict[str, Any]:
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
    """Return the exact immutable identity used by Perfetto conversion."""

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


def _artifact_roots(loaded: LoadedHybridRun, perfetto_root: Path) -> dict[str, Path]:
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
    for field in _NATIVE_COUNT_FIELDS:
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


def _require_manifest_match(loaded: LoadedHybridRun, manifest: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(manifest))
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
        "input_validation": normalized_input_metadata(loaded),
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
    attribute_validation = manifest.get("trace_attribute_validation")
    if attribute_validation is not None and (
        not isinstance(attribute_validation, dict)
        or attribute_validation.get("root_id") != PERFETTO_ROOT_ID
        or attribute_validation.get("relative_path")
        != TRACE_ATTRIBUTE_VALIDATION_NAME
        or attribute_validation.get("valid") is not True
        or attribute_validation.get("mismatches") != []
    ):
        raise OverviewInputError(
            "Perfetto trace-attribute validation summary is invalid"
        )


def _require_trace_identity(root: Path, manifest: dict[str, Any]) -> None:
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
            not isinstance(query, dict)
            or query.get("matched") is not True
            or not _REQUIRED_QUERY_FIELDS.issubset(query)
            or set(query) - _REQUIRED_QUERY_FIELDS - _OPTIONAL_QUERY_FIELDS
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


def _fresh_validation_matches_stored(
    fresh: dict[str, Any],
    stored: dict[str, Any],
) -> bool:
    """Compare fresh full-row evidence with compact or historical reports."""

    fresh_queries = fresh.get("queries")
    stored_queries = stored.get("queries")
    if not isinstance(fresh_queries, list) or not isinstance(stored_queries, list):
        return False
    if len(fresh_queries) != len(stored_queries):
        return False
    if all(
        isinstance(query, dict) and "rows" not in query
        for query in stored_queries
    ):
        return summarize_trace_validation(fresh) == stored
    try:
        comparable_fresh = dict(fresh)
        comparable_fresh["queries"] = [
            {key: fresh_query[key] for key in stored_query}
            for fresh_query, stored_query in zip(
                fresh_queries, stored_queries
            )
            if isinstance(fresh_query, dict)
            and isinstance(stored_query, dict)
        ]
    except KeyError:
        return False
    return (
        len(comparable_fresh["queries"]) == len(stored_queries)
        and comparable_fresh == stored
    )


_PHASE_MAPPING = tuple(
    (
        metric_name,
        STAGE_BY_METRIC[metric_name].track_key,
        STAGE_BY_METRIC[metric_name].slice_name,
    )
    for metric_name in PHASE_RECONCILIATION_METRICS
)
_SLICE_IDENTITY_KEYS = ("track_name", "slice_name", "ts", "dur")
_PHASE_SLICE_BY_TRACK_KEY = {
    track_key: slice_name for _, track_key, slice_name in _PHASE_MAPPING
}
_SAFE_TOOLCHAIN_FIELDS = (
    "filename",
    "version",
    "sha256",
    "perfetto_package_version",
    "protobuf_package_version",
    "trace_processor_rpc_api_version",
)
# Keep the external-report schema backward-compatible. New mapping-specific
# query counts remain available in ``queries``.
_REPORT_COUNT_NAMES = (
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


def reconciliation_summary(bundle: LoadedPerfettoBundle) -> dict[str, Any]:
    """Return deterministic, path-free evidence suitable for Overview JSON."""

    report = bundle.fresh_trace_validation
    query_summaries = [
        {
            "name": query["name"],
            "row_count": query["row_count"],
            "rows_sha256": query["rows_sha256"],
            "expected_row_count": query["expected_row_count"],
            "expected_rows_sha256": query["expected_rows_sha256"],
            "matched": query["matched"],
        }
        for query in sorted(report["queries"], key=lambda item: item["name"])
    ]
    toolchain = report["toolchain"]
    return {
        "valid": True,
        "trace": dict(report["trace"]),
        "counts": {
            name: report["counts"][name] for name in _REPORT_COUNT_NAMES
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
        "toolchain": {
            name: toolchain[name]
            for name in _SAFE_TOOLCHAIN_FIELDS
            if name in toolchain
        },
    }


def phase_duration_reconciliation(bundle: LoadedPerfettoBundle) -> list[dict[str, Any]]:
    """Reconcile event-planned integer phase durations with TP slice rows."""

    expected: dict[str, list[int]] = {
        slice_name: [] for _, _, slice_name in _PHASE_MAPPING
    }
    for item in bundle.planning.plan.slices:
        slice_name = _PHASE_SLICE_BY_TRACK_KEY.get(base_track_key(item.track_key))
        if slice_name is not None:
            expected[slice_name].append(item.duration_ns)
    slice_query = next(
        query
        for query in bundle.fresh_trace_validation["queries"]
        if query["name"] == "slices"
    )
    # One phase can occupy several concurrent lanes, each with its own track
    # name, so the map comes from the planned tracks.  Resolved before the
    # compact-report shortcut so both report shapes fail the same way.
    detail_track_names = {
        track.name: _PHASE_SLICE_BY_TRACK_KEY[base_track_key(track.key)]
        for track in bundle.planning.plan.tracks
        if base_track_key(track.key) in _PHASE_SLICE_BY_TRACK_KEY
    }
    slice_rows = slice_query.get("rows")
    if not isinstance(slice_rows, list):
        if slice_query.get("matched") is not True:
            raise OverviewInputError(
                "compact Perfetto slice validation did not match its plan"
            )
        # Large reports omit inline rows. The fresh validator already compared
        # every canonical slice row and duplicate multiplicity literally with
        # this exact plan, so the phase subset is identical without retaining
        # millions of unrelated native slices.
        return [
            {
                "kpi_name": kpi_name,
                "slice_name": slice_name,
                "slice_count": len(expected[slice_name]),
                "event_duration_ns": sum(expected[slice_name]),
                "perfetto_duration_ns": sum(expected[slice_name]),
                "matched": True,
            }
            for kpi_name, _, slice_name in _PHASE_MAPPING
        ]
    # Timeline-summary rows can restate a canonical slice; on legacy mappings
    # those restatements are consumed once each so a phase is not double counted.
    summary_query = next(
        (
            query
            for query in bundle.fresh_trace_validation["queries"]
            if query["name"] == "timeline_summary_slices"
        ),
        None,
    )
    duplicated_summary_rows: Counter = Counter()
    if bundle.planning.plan.mapping_version != TIMELINE_SUMMARY_MAPPING_VERSION:
        duplicated_summary_rows.update(
            _slice_identity(row)
            for row in (
                summary_query.get("rows", [])
                if isinstance(summary_query, dict)
                else []
            )
        )
    actual: dict[str, list[int]] = {
        slice_name: [] for _, _, slice_name in _PHASE_MAPPING
    }
    for row in slice_rows:
        identity = _slice_identity(row)
        if duplicated_summary_rows[identity]:
            duplicated_summary_rows[identity] -= 1
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
    for kpi_name, _, slice_name in _PHASE_MAPPING:
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


def _slice_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identify one Perfetto slice row; missing columns compare as None."""

    return tuple(row.get(key) for key in _SLICE_IDENTITY_KEYS)


def load_matching_perfetto(
    loaded: LoadedHybridRun,
    perfetto_root: str | Path,
    *,
    trace_processor_path: Path | None = None,
) -> LoadedPerfettoBundle:
    """Load and freshly reconcile one exact matching Perfetto output."""

    if not isinstance(loaded, LoadedHybridRun):
        raise TypeError("loaded must be a LoadedHybridRun")
    root = require_real_directory(perfetto_root, description="Perfetto output")
    _assert_no_overlap(loaded, root)
    identity_before = _bundle_identity(root)
    manifest = read_json_object(
        root / CONVERSION_MANIFEST_NAME,
        description="Perfetto conversion manifest",
    )
    stored_validation = read_json_object(
        root / TRACE_VALIDATION_NAME,
        description="stored Perfetto trace validation",
    )
    has_attribute_validation = TRACE_ATTRIBUTE_VALIDATION_NAME in {
        item.name for item in root.iterdir()
    }
    if has_attribute_validation:
        attributes = read_json_object(
            root / TRACE_ATTRIBUTE_VALIDATION_NAME,
            description="stored Perfetto trace attribute validation",
        )
        if (
            attributes.get("valid") is not True
            or attributes.get("mismatches") != []
            or attributes.get("namespace") != TRACE_ATTRIBUTE_NAMESPACE
        ):
            raise OverviewInputError(
                "stored Perfetto trace attribute validation is invalid"
            )
    _require_manifest_match(loaded, manifest)
    if has_attribute_validation != (
        manifest.get("trace_attribute_validation") is not None
    ):
        raise OverviewInputError(
            "Perfetto trace-attribute validation file and manifest differ"
        )
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
    if not _fresh_validation_matches_stored(
        fresh_validation,
        stored_validation,
    ):
        raise OverviewInputError(
            "fresh official Trace Processor result differs from stored conversion "
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
    "read_json_object",
    "read_validated_source_json",
    "reconciliation_summary",
    "require_real_directory",
]
