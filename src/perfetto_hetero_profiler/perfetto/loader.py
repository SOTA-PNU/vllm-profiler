"""Strict, read-only loader for normalized succeeded hybrid runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, TypeVar

from ..hybrid.join import validate_marker_order
from ..schema import (
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactReference,
    ClockDomain,
    ClockTransform,
    DeviceType,
    EventRecord,
    MetricSample,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    record_from_dict,
    validate_detached_artifact_manifest,
)
from .planner import NativeProfileEnvelope


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_DESCRIPTOR_FIELDS = {
    "device_type",
    "host_ids",
    "ingested_at_unix_ns",
    "profile_mode",
    "source_artifacts",
    "source_clock_domains",
    "source_manifest_sha256",
    "source_path",
    "source_role",
    "source_run_id",
    "source_status",
}
_ALIGNMENT_FIELDS = {
    "alignment_method",
    "alignment_status",
    "anchor_count",
    "anchors",
    "canonical_clock_domain",
    "host_boundary_uncertainty_ns",
    "native_capture_end",
    "native_capture_start",
    "native_clock_domain",
    "native_timestamp_unit",
    "offset_ns",
    "profiler_type",
    "reason",
    "timestamp_fallback",
    "unaligned_profiler_events",
    "uncertainty_ns",
    "valid_interval_monotonic_ns",
}
_CLOSEOUT_ROOT_IDS = ("coordinator", "gpu", "hybrid", "npu", "recovery")

_RecordT = TypeVar("_RecordT")


class PerfettoInputError(RuntimeError):
    """A normalized input or its detached integrity evidence is invalid."""


@dataclass(frozen=True, slots=True)
class RootFingerprint:
    """Validated path plus deterministic content identity for one closeout root."""

    root_id: str
    root: Path
    file_count: int
    fingerprint_sha256: str

    @property
    def inventory_sha256(self) -> str:
        """Alias clarifying that this hashes the validated closeout rows."""

        return self.fingerprint_sha256

    @property
    def content_sha256(self) -> str:
        """Compatibility alias for callers treating the inventory as content."""

        return self.fingerprint_sha256

    @property
    def metadata(self) -> dict[str, object]:
        """Return path-free fingerprint metadata."""

        return {
            "root_id": self.root_id,
            "file_count": self.file_count,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceRunMetadata:
    """One validated sibling source, located solely from its source run id."""

    source_role: str
    source_run_id: str
    root: Path
    manifest: RunManifest
    clock_domains: tuple[ClockDomain, ...]
    artifacts: tuple[ArtifactReference, ...]
    manifest_sha256: str
    fingerprint: RootFingerprint


@dataclass(frozen=True, slots=True)
class LoadedHybridRun:
    """Immutable normalized input used by deterministic Perfetto conversion."""

    root: Path
    manifest: RunManifest
    clock_domains: tuple[ClockDomain, ...]
    transforms: tuple[ClockTransform, ...]
    events: tuple[EventRecord, ...]
    metrics: tuple[MetricSample, ...]
    artifacts: tuple[ArtifactReference, ...]
    canonical_clock: ClockDomain
    sources: tuple[SourceRunMetadata, ...]
    root_fingerprints: tuple[RootFingerprint, ...]
    closeout_manifest_sha256: str
    closeout_artifact_count: int
    native_envelopes: tuple[NativeProfileEnvelope, ...]

    @property
    def canonical_clock_domain_id(self) -> str:
        return self.canonical_clock.clock_domain_id

    @property
    def source_by_role(self) -> dict[str, SourceRunMetadata]:
        return {source.source_role: source for source in self.sources}

    @property
    def fingerprint_by_root_id(self) -> dict[str, RootFingerprint]:
        return {
            fingerprint.root_id: fingerprint
            for fingerprint in self.root_fingerprints
        }


@dataclass(frozen=True, slots=True)
class _SourceData:
    source_role: str
    source_run_id: str
    root: Path
    manifest: RunManifest
    clock_domains: tuple[ClockDomain, ...]
    artifacts: tuple[ArtifactReference, ...]
    manifest_sha256: str


def _is_non_bool_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_run_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise PerfettoInputError(f"{field} is not a safe run id")
    return value


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PerfettoInputError(f"{field} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or normalized in {"", "."}
        or normalized != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PerfettoInputError(f"{field} must be a safe POSIX relative path")
    return normalized


def _absolute_without_resolving(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def _require_real_directory(path: Path, *, description: str) -> Path:
    candidate = _absolute_without_resolving(path)
    try:
        file_stat = candidate.lstat()
    except OSError as error:
        raise PerfettoInputError(
            f"{description} cannot be inspected: {candidate}: {error}"
        ) from error
    if stat.S_ISLNK(file_stat.st_mode):
        raise PerfettoInputError(f"{description} must not be a symlink: {candidate}")
    if not stat.S_ISDIR(file_stat.st_mode):
        raise PerfettoInputError(f"{description} is not a directory: {candidate}")
    return candidate


def _checked_file(root: Path, relative_path: object, *, field: str) -> Path:
    normalized = _safe_relative_path(relative_path, field=field)
    current = root
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            file_stat = current.lstat()
        except OSError as error:
            raise PerfettoInputError(
                f"{field} cannot be inspected: {current}: {error}"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise PerfettoInputError(f"{field} must not use a symlink: {current}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(file_stat.st_mode):
                raise PerfettoInputError(
                    f"{field} parent is not a directory: {current}"
                )
        elif not stat.S_ISREG(file_stat.st_mode):
            raise PerfettoInputError(f"{field} is not a regular file: {current}")
    return current


def _same_file_state(before: Any, after: Any) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _stable_read(
    path: Path,
    reader: Callable[[Path], _RecordT],
    *,
    description: str,
) -> _RecordT:
    before = path.lstat()
    try:
        value = reader(path)
    except PerfettoInputError:
        raise
    except Exception as error:
        raise PerfettoInputError(f"{description} is invalid: {error}") from error
    after = path.lstat()
    if not _same_file_state(before, after):
        raise PerfettoInputError(f"{description} changed while it was read: {path}")
    return value


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_plain_json_object(path: Path, *, description: str) -> dict[str, Any]:
    def read(candidate: Path) -> dict[str, Any]:
        try:
            value = json.loads(
                candidate.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {token}")
                ),
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise PerfettoInputError(f"{description} is invalid: {error}") from error
        if not isinstance(value, dict):
            raise PerfettoInputError(f"{description} must be a JSON object")
        return value

    return _stable_read(path, read, description=description)


def _typed_tuple(
    rows: Sequence[object],
    expected_type: type[_RecordT],
    *,
    description: str,
) -> tuple[_RecordT, ...]:
    invalid_index = next(
        (
            index
            for index, row in enumerate(rows)
            if not isinstance(row, expected_type)
        ),
        None,
    )
    if invalid_index is not None:
        raise PerfettoInputError(
            f"{description}[{invalid_index}] has an unexpected record type"
        )
    return tuple(rows)  # type: ignore[return-value]


def _read_schema_json(
    root: Path,
    relative_path: str,
    expected_type: type[_RecordT],
) -> _RecordT:
    path = _checked_file(root, relative_path, field=relative_path)
    value = _stable_read(path, read_json, description=relative_path)
    if not isinstance(value, expected_type):
        raise PerfettoInputError(f"{relative_path} has an unexpected record type")
    return value


def _read_schema_jsonl(
    root: Path,
    relative_path: str,
    expected_type: type[_RecordT],
) -> tuple[_RecordT, ...]:
    path = _checked_file(root, relative_path, field=relative_path)
    rows = _stable_read(path, read_jsonl, description=relative_path)
    return _typed_tuple(rows, expected_type, description=relative_path)


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PerfettoInputError(f"cannot hash artifact {path}: {error}") from error
    after = path.lstat()
    if not _same_file_state(before, after):
        raise PerfettoInputError(f"artifact changed while it was hashed: {path}")
    return digest.hexdigest()


def _validate_artifacts(
    *,
    root: Path,
    run_id: str,
    artifacts: tuple[ArtifactReference, ...],
    clock_domain_ids: set[str],
    host_ids: set[str],
) -> None:
    artifact_ids: set[str] = set()
    relative_paths: set[str] = set()
    for artifact in artifacts:
        if artifact.run_id != run_id:
            raise PerfettoInputError("artifact run_id does not match its manifest")
        if artifact.artifact_id in artifact_ids:
            raise PerfettoInputError(
                f"duplicate artifact id: {artifact.artifact_id!r}"
            )
        artifact_ids.add(artifact.artifact_id)
        relative_path = _safe_relative_path(
            artifact.relative_path,
            field=f"artifact {artifact.artifact_id!r} relative_path",
        )
        if relative_path in relative_paths:
            raise PerfettoInputError(f"duplicate artifact path: {relative_path}")
        relative_paths.add(relative_path)
        path = _checked_file(
            root,
            relative_path,
            field=f"artifact {artifact.artifact_id!r}",
        )
        if (
            not _is_non_bool_int(artifact.size_bytes)
            or artifact.size_bytes < 0
        ):
            raise PerfettoInputError(
                f"artifact {artifact.artifact_id!r} has no valid size"
            )
        if (
            not isinstance(artifact.sha256, str)
            or _SHA256_RE.fullmatch(artifact.sha256) is None
        ):
            raise PerfettoInputError(
                f"artifact {artifact.artifact_id!r} has no valid SHA-256"
            )
        actual_size = path.lstat().st_size
        if actual_size != artifact.size_bytes:
            raise PerfettoInputError(
                f"artifact size mismatch: {relative_path}: expected "
                f"{artifact.size_bytes}, got {actual_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != artifact.sha256:
            raise PerfettoInputError(
                f"artifact SHA-256 mismatch: {relative_path}: expected "
                f"{artifact.sha256}, got {actual_sha256}"
            )
        if (
            artifact.clock_domain_id is not None
            and artifact.clock_domain_id not in clock_domain_ids
        ):
            raise PerfettoInputError(
                f"artifact {artifact.artifact_id!r} references an unknown clock"
            )
        if artifact.host_id is not None and artifact.host_id not in host_ids:
            raise PerfettoInputError(
                f"artifact {artifact.artifact_id!r} references an unknown host"
            )


def _validate_source_descriptor(
    *,
    hybrid_root: Path,
    role: str,
    expected_device_type: DeviceType,
) -> _SourceData:
    descriptor_path = _checked_file(
        hybrid_root,
        f"sources/{role}-source.json",
        field=f"{role} source descriptor",
    )
    descriptor = _read_plain_json_object(
        descriptor_path,
        description=f"{role} source descriptor",
    )
    if set(descriptor) != _SOURCE_DESCRIPTOR_FIELDS:
        raise PerfettoInputError(
            f"{role} source descriptor fields do not match the contract"
        )
    if descriptor["source_role"] != role:
        raise PerfettoInputError(f"{role} source descriptor role mismatch")
    if descriptor["device_type"] != expected_device_type.value:
        raise PerfettoInputError(f"{role} source descriptor device mismatch")
    if descriptor["source_status"] != RunStatus.SUCCEEDED.value:
        raise PerfettoInputError(f"{role} source did not succeed")
    if not isinstance(descriptor["source_path"], str):
        raise PerfettoInputError(f"{role} source_path must be a string")

    source_run_id = _safe_run_id(
        descriptor["source_run_id"],
        field=f"{role} source_run_id",
    )
    # source_path is historical provenance only.  The trusted location is the
    # safe sibling derived from source_run_id.
    source_root = _require_real_directory(
        hybrid_root.parent / source_run_id,
        description=f"{role} sibling source root",
    )
    if source_root.parent != hybrid_root.parent:
        raise PerfettoInputError(f"{role} source root is not a direct sibling")

    manifest = _read_schema_json(source_root, "manifest.json", RunManifest)
    if manifest.run_id != source_run_id or source_root.name != source_run_id:
        raise PerfettoInputError(f"{role} source manifest identity mismatch")
    if manifest.status is not RunStatus.SUCCEEDED:
        raise PerfettoInputError(f"{role} source manifest did not succeed")
    expected_mode = (
        RunMode.GPU_ONLY if expected_device_type is DeviceType.GPU
        else RunMode.NPU_ONLY
    )
    if manifest.mode is not expected_mode:
        raise PerfettoInputError(f"{role} source manifest mode mismatch")
    if descriptor["profile_mode"] != manifest.profile_mode.value:
        raise PerfettoInputError(f"{role} source profile mode mismatch")
    expected_hosts = sorted(host.host_id for host in manifest.hosts)
    if descriptor["host_ids"] != expected_hosts:
        raise PerfettoInputError(f"{role} source host ids mismatch")
    if expected_device_type not in {
        device.device_type for device in manifest.devices
    }:
        raise PerfettoInputError(
            f"{role} source manifest lacks a {expected_device_type.value} device"
        )

    manifest_path = _checked_file(
        source_root,
        "manifest.json",
        field=f"{role} source manifest",
    )
    manifest_sha256 = _sha256_file(manifest_path)
    if descriptor["source_manifest_sha256"] != manifest_sha256:
        raise PerfettoInputError(f"{role} source manifest SHA-256 mismatch")

    clocks = _read_schema_jsonl(
        source_root,
        "clocks/clock_domains.jsonl",
        ClockDomain,
    )
    if not clocks:
        raise PerfettoInputError(f"{role} source has no clock domain")
    clock_ids = [clock.clock_domain_id for clock in clocks]
    if len(set(clock_ids)) != len(clock_ids):
        raise PerfettoInputError(f"{role} source clock ids are not unique")
    if descriptor["source_clock_domains"] != clock_ids:
        raise PerfettoInputError(f"{role} source clock descriptor mismatch")
    for clock in clocks:
        if clock.run_id != source_run_id:
            raise PerfettoInputError(f"{role} source clock run_id mismatch")

    artifacts = _read_schema_jsonl(
        source_root,
        "artifacts/artifacts.jsonl",
        ArtifactReference,
    )
    raw_descriptor_artifacts = descriptor["source_artifacts"]
    if not isinstance(raw_descriptor_artifacts, list):
        raise PerfettoInputError(f"{role} source_artifacts must be an array")
    descriptor_artifacts: list[ArtifactReference] = []
    for index, raw_artifact in enumerate(raw_descriptor_artifacts):
        if not isinstance(raw_artifact, dict):
            raise PerfettoInputError(
                f"{role} source_artifacts[{index}] must be an object"
            )
        try:
            artifact = record_from_dict(raw_artifact)
        except (TypeError, ValueError) as error:
            raise PerfettoInputError(
                f"{role} source_artifacts[{index}] is invalid: {error}"
            ) from error
        if not isinstance(artifact, ArtifactReference):
            raise PerfettoInputError(
                f"{role} source_artifacts[{index}] is not an artifact record"
            )
        descriptor_artifacts.append(artifact)
    if tuple(descriptor_artifacts) != artifacts:
        raise PerfettoInputError(
            f"{role} source descriptor artifacts differ from source index"
        )
    _validate_artifacts(
        root=source_root,
        run_id=source_run_id,
        artifacts=artifacts,
        clock_domain_ids=set(clock_ids),
        host_ids=set(expected_hosts),
    )
    return _SourceData(
        source_role=role,
        source_run_id=source_run_id,
        root=source_root,
        manifest=manifest,
        clock_domains=clocks,
        artifacts=artifacts,
        manifest_sha256=manifest_sha256,
    )


def _validate_normalized_records(
    *,
    manifest: RunManifest,
    clocks: tuple[ClockDomain, ...],
    transforms: tuple[ClockTransform, ...],
    events: tuple[EventRecord, ...],
    metrics: tuple[MetricSample, ...],
    artifacts: tuple[ArtifactReference, ...],
) -> ClockDomain:
    if manifest.status is not RunStatus.SUCCEEDED:
        raise PerfettoInputError("only succeeded runs can be converted")
    if manifest.mode is not RunMode.HYBRID:
        raise PerfettoInputError("only normalized hybrid runs can be converted")

    configured = manifest.configuration.get("canonical_clock_domain_id")
    if not isinstance(configured, str) or not configured:
        raise PerfettoInputError("manifest has no canonical clock domain id")
    canonical = [
        clock
        for clock in clocks
        if clock.attributes.get("hybrid.role") == "canonical"
    ]
    if len(canonical) != 1 or canonical[0].clock_domain_id != configured:
        raise PerfettoInputError(
            "run must define exactly one configured canonical clock"
        )
    canonical_clock = canonical[0]
    if (
        canonical_clock.unit != "ns"
        or not canonical_clock.monotonic
        or canonical_clock.adjustable
    ):
        raise PerfettoInputError("canonical clock must be monotonic nanoseconds")

    host_ids = {host.host_id for host in manifest.hosts}
    device_keys = {
        (device.host_id, device.device_type, device.device_id)
        for device in manifest.devices
    }
    clock_ids: set[str] = set()
    for clock in clocks:
        if clock.run_id != manifest.run_id:
            raise PerfettoInputError("clock run_id does not match manifest")
        if clock.clock_domain_id in clock_ids:
            raise PerfettoInputError("clock domain ids must be unique")
        clock_ids.add(clock.clock_domain_id)
        if clock.host_id not in host_ids:
            raise PerfettoInputError("clock domain references an unknown host")

    transform_ids: set[str] = set()
    for transform in transforms:
        if transform.run_id != manifest.run_id:
            raise PerfettoInputError("clock transform run_id mismatch")
        if transform.transform_id in transform_ids:
            raise PerfettoInputError("clock transform ids must be unique")
        transform_ids.add(transform.transform_id)
        if (
            transform.source_clock_domain_id not in clock_ids
            or transform.target_clock_domain_id != configured
        ):
            raise PerfettoInputError(
                "clock transform does not map a known source to canonical"
            )

    event_ids: set[str] = set()
    for event in events:
        if event.run_id != manifest.run_id:
            raise PerfettoInputError("event run_id does not match manifest")
        if event.event_id in event_ids:
            raise PerfettoInputError(f"duplicate event id: {event.event_id!r}")
        event_ids.add(event.event_id)
        if event.clock_domain_id != configured:
            raise PerfettoInputError(
                f"event {event.event_id!r} is not on the canonical clock"
            )
        if event.host_id not in host_ids:
            raise PerfettoInputError(
                f"event {event.event_id!r} references an unknown host"
            )
        if event.device_id is not None and (
            event.host_id,
            event.device_type,
            event.device_id,
        ) not in device_keys:
            raise PerfettoInputError(
                f"event {event.event_id!r} references an unknown device"
            )
    for event in events:
        if (
            event.parent_event_id is not None
            and event.parent_event_id not in event_ids
        ):
            raise PerfettoInputError(
                f"event {event.event_id!r} has an unknown parent"
            )

    marker_validation = validate_marker_order(events)
    if marker_validation.status != "valid":
        issue_statuses = tuple(
            issue.status for issue in marker_validation.ordering_issues
        )
        raise PerfettoInputError(
            "canonical marker contract is not valid: "
            f"status={marker_validation.status}, "
            f"missing={marker_validation.missing_markers!r}, "
            f"duplicates={marker_validation.duplicate_markers!r}, "
            f"pairing={marker_validation.pairing_issues!r}, "
            f"ordering={issue_statuses!r}"
        )

    for metric in metrics:
        if metric.run_id != manifest.run_id:
            raise PerfettoInputError("metric run_id does not match manifest")
        if metric.clock_domain_id != configured:
            raise PerfettoInputError(
                f"metric {metric.metric_name!r} is not on the canonical clock"
            )
        if metric.host_id not in host_ids:
            raise PerfettoInputError(
                f"metric {metric.metric_name!r} references an unknown host"
            )
        if metric.device_id is not None and (
            metric.host_id,
            metric.device_type,
            metric.device_id,
        ) not in device_keys:
            raise PerfettoInputError(
                f"metric {metric.metric_name!r} references an unknown device"
            )

    artifact_clock_ids = clock_ids | {
        original
        for clock in clocks
        if isinstance(
            original := clock.attributes.get(
                "hybrid.original_clock_domain_id"
            ),
            str,
        )
        and original
    }
    for artifact in artifacts:
        if artifact.run_id != manifest.run_id:
            raise PerfettoInputError("artifact run_id does not match manifest")
        if (
            artifact.clock_domain_id is not None
            and artifact.clock_domain_id not in artifact_clock_ids
        ):
            raise PerfettoInputError("artifact references an unknown clock")
        if artifact.host_id is not None and artifact.host_id not in host_ids:
            raise PerfettoInputError("artifact references an unknown host")
    return canonical_clock


def _validate_anchor(anchor: object, *, kind: str) -> Mapping[str, Any]:
    if not isinstance(anchor, dict):
        raise PerfettoInputError(f"{kind} profiler anchor must be an object")
    if anchor.get("kind") != kind:
        raise PerfettoInputError(f"expected {kind} profiler anchor")
    for field in ("before_monotonic_ns", "after_monotonic_ns"):
        value = anchor.get(field)
        if not _is_non_bool_int(value) or value < 0:
            raise PerfettoInputError(f"{kind}.{field} must be non-negative int")
    if anchor["after_monotonic_ns"] < anchor["before_monotonic_ns"]:
        raise PerfettoInputError(f"{kind} host boundary is reversed")
    if anchor.get("http_status") != 200:
        raise PerfettoInputError(f"{kind} did not return HTTP 200")
    return anchor


def _native_envelope(
    source: _SourceData,
    *,
    canonical_clock: ClockDomain,
    hybrid_clocks: tuple[ClockDomain, ...],
    transforms: tuple[ClockTransform, ...],
) -> NativeProfileEnvelope | None:
    alignment_artifacts = [
        artifact
        for artifact in source.artifacts
        if artifact.relative_path == "clocks/profiler_alignment.json"
    ]
    if not alignment_artifacts:
        if source.manifest.profile_mode is ProfileMode.DETAILED_PROFILE:
            raise PerfettoInputError(
                f"{source.source_role} detailed source lacks profiler alignment"
            )
        return None
    if len(alignment_artifacts) != 1:
        raise PerfettoInputError(
            f"{source.source_role} source has multiple profiler alignments"
        )
    if source.manifest.profile_mode is not ProfileMode.DETAILED_PROFILE:
        raise PerfettoInputError(
            f"{source.source_role} monitor source has profiler alignment"
        )

    alignment_path = _checked_file(
        source.root,
        "clocks/profiler_alignment.json",
        field=f"{source.source_role} profiler alignment",
    )
    alignment = _read_plain_json_object(
        alignment_path,
        description=f"{source.source_role} profiler alignment",
    )
    if set(alignment) != _ALIGNMENT_FIELDS:
        raise PerfettoInputError("profiler alignment fields do not match contract")
    if (
        alignment["alignment_status"] != "partial"
        or alignment["alignment_method"] != "host_api_boundary_bracket"
        or alignment["unaligned_profiler_events"] is not True
        or alignment["timestamp_fallback"] is not False
        or alignment["offset_ns"] is not None
        or alignment["uncertainty_ns"] is not None
    ):
        raise PerfettoInputError(
            "native profiler must retain explicit partial unaligned policy"
        )
    if alignment["canonical_clock_domain"] not in {
        clock.clock_domain_id for clock in source.clock_domains
    }:
        raise PerfettoInputError("profiler boundary source clock is unknown")
    native_clock_domain = alignment["native_clock_domain"]
    if (
        not isinstance(native_clock_domain, str)
        or not native_clock_domain
        or native_clock_domain
        not in {clock.clock_domain_id for clock in source.clock_domains}
        or native_clock_domain == alignment["canonical_clock_domain"]
    ):
        raise PerfettoInputError("profiler native clock domain is invalid")
    native_timestamp_unit = alignment["native_timestamp_unit"]
    profiler_type = alignment["profiler_type"]
    if not isinstance(native_timestamp_unit, str) or not native_timestamp_unit:
        raise PerfettoInputError("profiler native timestamp unit is invalid")
    if not isinstance(profiler_type, str) or not profiler_type:
        raise PerfettoInputError("profiler type is invalid")
    if not profiler_type.startswith(f"{source.source_role}_"):
        raise PerfettoInputError("profiler type does not match source role")

    anchors = alignment["anchors"]
    if (
        alignment["anchor_count"] != 2
        or not isinstance(anchors, list)
        or len(anchors) != 2
    ):
        raise PerfettoInputError("profiler alignment requires two host anchors")
    start_anchor = _validate_anchor(anchors[0], kind="profiler_start_api")
    stop_anchor = _validate_anchor(anchors[1], kind="profiler_stop_api")
    interval = alignment["valid_interval_monotonic_ns"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(not _is_non_bool_int(value) or value < 0 for value in interval)
    ):
        raise PerfettoInputError("profiler valid interval is invalid")
    source_start, source_end = interval
    if (
        source_start != start_anchor["before_monotonic_ns"]
        or source_end != stop_anchor["after_monotonic_ns"]
        or source_end < source_start
    ):
        raise PerfettoInputError(
            "profiler valid interval does not match the outer API bracket"
        )
    uncertainty = alignment["host_boundary_uncertainty_ns"]
    if not _is_non_bool_int(uncertainty) or uncertainty < 0:
        raise PerfettoInputError("profiler host boundary uncertainty is invalid")

    matching_transforms = [
        transform
        for transform in transforms
        if transform.target_clock_domain_id == canonical_clock.clock_domain_id
        and transform.attributes.get("hybrid.source_role")
        == source.source_role
    ]
    if len(matching_transforms) != 1:
        raise PerfettoInputError(
            f"{source.source_role} has no unique host-to-canonical transform"
        )
    transform = matching_transforms[0]
    source_clock = next(
        (
            clock
            for clock in hybrid_clocks
            if clock.clock_domain_id == transform.source_clock_domain_id
        ),
        None,
    )
    if (
        source_clock is None
        or source_clock.attributes.get("hybrid.original_clock_domain_id")
        != alignment["canonical_clock_domain"]
        or source_clock.attributes.get("hybrid.source_role")
        != source.source_role
    ):
        raise PerfettoInputError(
            "profiler host boundary lacks explicit source clock provenance"
        )
    if transform.scale != 1.0:
        raise PerfettoInputError(
            "native host envelope requires an exact unit-scale transform"
        )
    timestamp_ns = source_start + transform.offset_ns
    end_ns = source_end + transform.offset_ns
    if timestamp_ns < 0 or end_ns < timestamp_ns:
        raise PerfettoInputError("profiler host envelope maps outside canonical time")

    native_artifacts = [
        artifact
        for artifact in source.artifacts
        if artifact.clock_domain_id == native_clock_domain
    ]
    if not native_artifacts:
        raise PerfettoInputError("profiler alignment has no native artifact")
    is_rbln = profiler_type == "npu_rbln"
    detail_artifacts = [
        artifact
        for artifact in source.artifacts
        if artifact.relative_path == "summary/detailed_profile.json"
    ]
    if len(detail_artifacts) != 1:
        raise PerfettoInputError("detailed source lacks one profile summary")
    detail_path = _checked_file(
        source.root,
        "summary/detailed_profile.json",
        field=f"{source.source_role} detailed profile summary",
    )
    detail = _read_plain_json_object(
        detail_path,
        description=f"{source.source_role} detailed profile summary",
    )
    if detail.get("kind") != profiler_type or detail.get("enabled") is not True:
        raise PerfettoInputError("detailed profile summary identity mismatch")
    _validate_native_detail_files(
        source=source,
        native_artifacts=native_artifacts,
        value=detail.get("files"),
    )
    if is_rbln:
        # Phase 4B closeout metadata predates the verified Perfetto-compatible
        # interpretation of RBLN PB files.  Accept that immutable legacy
        # provenance as input, while also accepting the corrected capture
        # policy emitted by newer runs.  Neither form proves canonical clock
        # alignment; Phase 6C validates the PB with Trace Processor separately.
        legacy_metadata = (
            detail.get("format") == "vendor_rbln_pb"
            and detail.get("structural_parse") == "not_available"
        )
        corrected_metadata = (
            detail.get("format") == "perfetto_trace_protobuf"
            and detail.get("structural_parse")
            == "deferred_to_official_trace_processor"
        )
        if (
            not (legacy_metadata or corrected_metadata)
            or detail.get("report_count") != len(native_artifacts)
            or any(
                artifact.artifact_kind is not ArtifactKind.RBLN_REPORT
                or artifact.format != "vendor-rbln-pb"
                or not artifact.relative_path.endswith(".pb")
                for artifact in native_artifacts
            )
        ):
            raise PerfettoInputError(
                "RBLN native reports do not preserve a supported Perfetto "
                "trace provenance policy"
            )

    return NativeProfileEnvelope(
        profiler_type=profiler_type,
        source_role=source.source_role,
        timestamp_ns=timestamp_ns,
        duration_ns=end_ns - timestamp_ns,
        alignment_status="partial",
        alignment_method="host_api_boundary_bracket",
        uncertainty_ns=uncertainty + transform.uncertainty_ns,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        artifact_count=len(native_artifacts),
        # Retained as a schema-compatibility field.  RBLN PB is an official
        # Perfetto-compatible trace; its unresolved property is clock
        # alignment, not file opacity.
        opaque_rbln_pb=False,
    )


def _validate_native_detail_files(
    *,
    source: _SourceData,
    native_artifacts: list[ArtifactReference],
    value: object,
) -> None:
    if not isinstance(value, list) or len(value) != len(native_artifacts):
        raise PerfettoInputError(
            "detailed profile file list does not match native artifacts"
        )
    expected = {
        artifact.relative_path: artifact for artifact in native_artifacts
    }
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "inode",
            "mtime_ns",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise PerfettoInputError(
                f"detailed profile files[{index}] does not match contract"
            )
        relative_path = _safe_relative_path(
            item["path"],
            field=f"detailed profile files[{index}].path",
        )
        if relative_path in seen or relative_path not in expected:
            raise PerfettoInputError(
                f"detailed profile files[{index}] is duplicate or unknown"
            )
        seen.add(relative_path)
        artifact = expected[relative_path]
        if (
            item["sha256"] != artifact.sha256
            or item["size_bytes"] != artifact.size_bytes
            or not _is_non_bool_int(item["size_bytes"])
            or item["size_bytes"] <= 0
        ):
            raise PerfettoInputError(
                f"detailed profile files[{index}] integrity mismatch"
            )
        if (
            not _is_non_bool_int(item["inode"])
            or item["inode"] <= 0
            or not _is_non_bool_int(item["mtime_ns"])
            or item["mtime_ns"] < 0
        ):
            raise PerfettoInputError(
                f"detailed profile files[{index}] metadata is invalid"
            )
        path = _checked_file(
            source.root,
            relative_path,
            field=f"detailed profile files[{index}]",
        )
        file_stat = path.lstat()
        if (
            item["inode"] != file_stat.st_ino
            or item["mtime_ns"] != file_stat.st_mtime_ns
        ):
            raise PerfettoInputError(
                f"detailed profile files[{index}] disk metadata mismatch"
            )
    if seen != set(expected):
        raise PerfettoInputError(
            "detailed profile file list omits a native artifact"
        )


def _fingerprints_from_closeout(
    manifest: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
) -> tuple[RootFingerprint, ...]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PerfettoInputError("closeout manifest artifacts must be an array")
    grouped: dict[str, list[dict[str, Any]]] = {
        root_id: [] for root_id in sorted(roots)
    }
    for item in artifacts:
        if not isinstance(item, dict) or item.get("root_id") not in grouped:
            raise PerfettoInputError("closeout manifest artifact root is invalid")
        grouped[item["root_id"]].append(item)

    values: list[RootFingerprint] = []
    for root_id in sorted(grouped):
        rows = sorted(
            grouped[root_id],
            key=lambda item: item["relative_path"],
        )
        payload = json.dumps(
            [
                {
                    "relative_path": item["relative_path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "mtime_ns": item["mtime_ns"],
                }
                for item in rows
            ],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        values.append(
            RootFingerprint(
                root_id=root_id,
                root=roots[root_id],
                file_count=len(rows),
                fingerprint_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(values)


def _validate_closeout(
    *,
    run_id: str,
    hybrid_root: Path,
    gpu_root: Path,
    npu_root: Path,
) -> tuple[
    tuple[RootFingerprint, ...],
    str,
    int,
]:
    roots = {
        "coordinator": _require_real_directory(
            hybrid_root.parent / f"{run_id}-coordinator",
            description="coordinator root",
        ),
        "gpu": gpu_root,
        "hybrid": hybrid_root,
        "npu": npu_root,
        "recovery": _require_real_directory(
            hybrid_root.parent / f"{run_id}-closeout-recovery",
            description="closeout recovery root",
        ),
    }
    recovery_root = roots["recovery"]
    manifest_path = _checked_file(
        recovery_root,
        "artifact_manifest.json",
        field="closeout artifact manifest",
    )
    validation_path = _checked_file(
        recovery_root,
        "artifact_manifest_validation.json",
        field="closeout artifact validation",
    )
    recovery_result_path = _checked_file(
        recovery_root,
        "recovery_result.json",
        field="closeout recovery result",
    )
    try:
        recomputed = validate_detached_artifact_manifest(
            manifest_path,
            roots,
            report_path=None,
        )
    except (ArtifactIntegrityError, OSError, ValueError) as error:
        raise PerfettoInputError(
            f"detached closeout artifact manifest is invalid: {error}"
        ) from error
    if recomputed.get("valid") is not True or recomputed.get("mismatches") != []:
        raise PerfettoInputError(
            "detached closeout artifact validation found mismatches"
        )
    stored = _read_plain_json_object(
        validation_path,
        description="stored closeout artifact validation",
    )
    if stored != recomputed:
        raise PerfettoInputError(
            "stored closeout validation differs from pure recomputation"
        )
    recovery_result = _read_plain_json_object(
        recovery_result_path,
        description="closeout recovery result",
    )
    if (
        recovery_result.get("schema_version") != SCHEMA_VERSION
        or recovery_result.get("record_type") != "closeout_recovery_result"
        or recovery_result.get("source_run_id") != run_id
        or recovery_result.get("success") is not True
        or recovery_result.get("hardware_rerun") is not False
        or recovery_result.get("postprocess_only") is not True
    ):
        raise PerfettoInputError("closeout recovery result is not successful")
    closeout_manifest = _read_plain_json_object(
        manifest_path,
        description="closeout artifact manifest",
    )
    if closeout_manifest.get("root_ids") != list(_CLOSEOUT_ROOT_IDS):
        raise PerfettoInputError("closeout root ids do not match hybrid contract")
    artifact_count = closeout_manifest.get("artifact_count")
    if (
        not _is_non_bool_int(artifact_count)
        or artifact_count != recomputed.get("checked")
    ):
        raise PerfettoInputError("closeout artifact count is invalid")
    return (
        _fingerprints_from_closeout(closeout_manifest, roots=roots),
        recomputed["manifest_sha256"],
        artifact_count,
    )


def load_hybrid_run(run_root: str | Path) -> LoadedHybridRun:
    """Load and fully validate one succeeded normalized hybrid run read-only."""

    root = _require_real_directory(Path(run_root), description="hybrid run root")
    manifest = _read_schema_json(root, "manifest.json", RunManifest)
    _safe_run_id(manifest.run_id, field="manifest.run_id")
    if root.name != manifest.run_id:
        raise PerfettoInputError("run directory name does not match manifest run_id")

    clocks = _read_schema_jsonl(
        root,
        "clocks/clock_domains.jsonl",
        ClockDomain,
    )
    transforms = _read_schema_jsonl(
        root,
        "clocks/transforms.jsonl",
        ClockTransform,
    )
    events = _read_schema_jsonl(root, "events/events.jsonl", EventRecord)
    metrics = _read_schema_jsonl(root, "metrics/metrics.jsonl", MetricSample)
    artifacts = _read_schema_jsonl(
        root,
        "artifacts/artifacts.jsonl",
        ArtifactReference,
    )
    for role in ("gpu", "npu"):
        matches = [
            artifact
            for artifact in artifacts
            if artifact.relative_path == f"sources/{role}-source.json"
        ]
        if (
            len(matches) != 1
            or matches[0].artifact_kind is not ArtifactKind.MANIFEST
            or matches[0].format != "json"
        ):
            raise PerfettoInputError(
                f"{role} source descriptor is not a unique manifest artifact"
            )
    canonical_clock = _validate_normalized_records(
        manifest=manifest,
        clocks=clocks,
        transforms=transforms,
        events=events,
        metrics=metrics,
        artifacts=artifacts,
    )
    _validate_artifacts(
        root=root,
        run_id=manifest.run_id,
        artifacts=artifacts,
        clock_domain_ids={
            value
            for clock in clocks
            for value in (
                clock.clock_domain_id,
                clock.attributes.get("hybrid.original_clock_domain_id"),
            )
            if isinstance(value, str) and value
        },
        host_ids={host.host_id for host in manifest.hosts},
    )

    gpu_source = _validate_source_descriptor(
        hybrid_root=root,
        role="gpu",
        expected_device_type=DeviceType.GPU,
    )
    npu_source = _validate_source_descriptor(
        hybrid_root=root,
        role="npu",
        expected_device_type=DeviceType.NPU,
    )
    if manifest.configuration.get("gpu_source_run") != gpu_source.source_run_id:
        raise PerfettoInputError("manifest GPU source run does not match descriptor")
    if manifest.configuration.get("npu_source_run") != npu_source.source_run_id:
        raise PerfettoInputError("manifest NPU source run does not match descriptor")

    root_fingerprints, closeout_sha256, closeout_count = _validate_closeout(
        run_id=manifest.run_id,
        hybrid_root=root,
        gpu_root=gpu_source.root,
        npu_root=npu_source.root,
    )
    fingerprints = {
        item.root_id: item for item in root_fingerprints
    }
    source_metadata = tuple(
        SourceRunMetadata(
            source_role=source.source_role,
            source_run_id=source.source_run_id,
            root=source.root,
            manifest=source.manifest,
            clock_domains=source.clock_domains,
            artifacts=source.artifacts,
            manifest_sha256=source.manifest_sha256,
            fingerprint=fingerprints[source.source_role],
        )
        for source in (gpu_source, npu_source)
    )

    native_envelopes = tuple(
        envelope
        for source in (gpu_source, npu_source)
        if (
            envelope := _native_envelope(
                source,
                canonical_clock=canonical_clock,
                hybrid_clocks=clocks,
                transforms=transforms,
            )
        )
        is not None
    )
    expected_envelopes = (
        1 if manifest.profile_mode is ProfileMode.DETAILED_PROFILE else 0
    )
    if len(native_envelopes) != expected_envelopes:
        raise PerfettoInputError(
            "hybrid profile mode and native envelope count disagree"
        )
    expected_alignment_status = (
        "partial" if native_envelopes else "not_applicable"
    )
    if (
        manifest.attributes.get("hybrid.profiler_alignment_status")
        != expected_alignment_status
    ):
        raise PerfettoInputError(
            "manifest profiler alignment status disagrees with evidence"
        )

    return LoadedHybridRun(
        root=root,
        manifest=manifest,
        clock_domains=clocks,
        transforms=transforms,
        events=events,
        metrics=metrics,
        artifacts=artifacts,
        canonical_clock=canonical_clock,
        sources=source_metadata,
        root_fingerprints=root_fingerprints,
        closeout_manifest_sha256=closeout_sha256,
        closeout_artifact_count=closeout_count,
        native_envelopes=native_envelopes,
    )


def load_normalized_hybrid_run(run_root: str | Path) -> LoadedHybridRun:
    """Explicit alias for :func:`load_hybrid_run`."""

    return load_hybrid_run(run_root)


__all__ = [
    "LoadedHybridRun",
    "PerfettoInputError",
    "RootFingerprint",
    "SourceRunMetadata",
    "load_hybrid_run",
    "load_normalized_hybrid_run",
]
