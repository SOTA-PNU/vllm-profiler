"""Pure, self-reference-free artifact inventories for Perfetto conversion.

The conversion output is published separately from its immutable
source run.  Its artifact manifest inventories every regular file under the
logical roots supplied by the caller, except for the manifest itself and its
stored validation sidecar in the designated output root.

This module deliberately performs no directory publication and no validation
report writes.  Callers can therefore build and validate in a staging
directory, write the returned validation report once with
``write_json_exclusive``, and publish the complete directory afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
import uuid

from ..schema.constants import SCHEMA_VERSION


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_VALIDATION_NAME = "artifact_manifest_validation.json"

MANIFEST_RECORD_TYPE = "detached_artifact_manifest"
VALIDATION_RECORD_TYPE = "detached_artifact_validation"

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "output_root_id",
    "artifact_count",
    "root_ids",
    "required_artifacts",
    "detached_outputs",
    "artifacts",
}
_ARTIFACT_FIELDS = {
    "root_id",
    "relative_path",
    "size_bytes",
    "sha256",
    "mtime_ns",
    "mode",
}
_PATH_FIELDS = {"root_id", "relative_path"}
_VALIDATION_FIELDS = {
    "schema_version",
    "record_type",
    "valid",
    "checked",
    "mismatches",
    "manifest_sha256",
}


class ArtifactInventoryError(RuntimeError):
    """An artifact root, manifest, or stored validation is unsafe or invalid."""


# Short alias for callers that do not need to distinguish inventory failures.
ArtifactError = ArtifactInventoryError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactInventoryError(
            f"value is not deterministic finite JSON: {error}"
        ) from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    """Atomically write deterministic JSON without replacing an existing path."""

    output = Path(path)
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ArtifactInventoryError(
            f"JSON output parent must be a real directory: {parent}"
        )
    if not output.name or output.name in {".", ".."}:
        raise ArtifactInventoryError("JSON output name is unsafe")

    payload = _json_bytes(value)
    temporary = parent / (
        f".{output.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short deterministic JSON write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {output}") from error
        temporary.unlink()
        _fsync_directory(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactInventoryError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise ArtifactInventoryError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or normalized == "."
        or normalized != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactInventoryError(f"{field} must be a safe relative path")
    return normalized


def _normalize_roots(
    roots: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not roots:
        raise ArtifactInventoryError("at least one artifact root is required")

    normalized: dict[str, Path] = {}
    resolved: dict[str, Path] = {}
    for root_id, value in roots.items():
        if not isinstance(root_id, str) or _ROOT_ID_RE.fullmatch(root_id) is None:
            raise ArtifactInventoryError(
                f"invalid artifact root id: {root_id!r}"
            )
        root = Path(value)
        if root.is_symlink() or not root.is_dir():
            raise ArtifactInventoryError(
                f"artifact root must be a real directory: {root}"
            )
        normalized[root_id] = root
        resolved[root_id] = root.resolve()

    items = sorted(resolved.items())
    for index, (left_id, left) in enumerate(items):
        for right_id, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ArtifactInventoryError(
                    "artifact roots must not overlap: "
                    f"{left_id!r} and {right_id!r}"
                )
    return normalized


def _detached_outputs(output_root_id: str) -> frozenset[tuple[str, str]]:
    return frozenset(
        {
            (output_root_id, ARTIFACT_MANIFEST_NAME),
            (output_root_id, ARTIFACT_VALIDATION_NAME),
        }
    )


def _stable_file_record(
    root_id: str,
    root: Path,
    path: Path,
) -> dict[str, Any]:
    relative_path = _safe_relative_path(
        path.relative_to(root).as_posix(),
        field=f"artifact {root_id!r} relative_path",
    )
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactInventoryError(f"artifact inventory rejects symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactInventoryError(
            f"artifact inventory rejects non-regular file: {path}"
        )
    digest = _sha256_file(path)
    after = path.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_mode",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ArtifactInventoryError(
            f"artifact changed while it was inventoried: {path}"
        )
    return {
        "root_id": root_id,
        "relative_path": relative_path,
        "size_bytes": after.st_size,
        "sha256": digest,
        "mtime_ns": after.st_mtime_ns,
        "mode": stat.S_IMODE(after.st_mode),
    }


def _inventory(
    roots: Mapping[str, Path],
    *,
    exclusions: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    def raise_walk_error(error: OSError) -> None:
        raise ArtifactInventoryError(
            f"artifact root traversal failed: {error}"
        ) from error

    records: list[dict[str, Any]] = []
    for root_id, root in sorted(roots.items()):
        for current, directory_names, file_names in os.walk(
            root,
            topdown=True,
            onerror=raise_walk_error,
            followlinks=False,
        ):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for name in tuple(directory_names):
                child = current_path / name
                child_stat = child.lstat()
                if stat.S_ISLNK(child_stat.st_mode):
                    raise ArtifactInventoryError(
                        f"artifact inventory rejects symlink: {child}"
                    )
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise ArtifactInventoryError(
                        f"artifact inventory rejects non-directory entry: {child}"
                    )
            for name in file_names:
                path = current_path / name
                relative_path = _safe_relative_path(
                    path.relative_to(root).as_posix(),
                    field=f"artifact {root_id!r} relative_path",
                )
                if (root_id, relative_path) in exclusions:
                    file_stat = path.lstat()
                    if stat.S_ISLNK(file_stat.st_mode):
                        raise ArtifactInventoryError(
                            f"detached output must not be a symlink: {path}"
                        )
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise ArtifactInventoryError(
                            f"detached output must be a regular file: {path}"
                        )
                    continue
                records.append(_stable_file_record(root_id, root, path))
    return sorted(
        records,
        key=lambda item: (item["root_id"], item["relative_path"]),
    )


def _normalize_required(
    required_artifacts: Iterable[tuple[str, str]],
    *,
    root_ids: set[str],
    exclusions: frozenset[tuple[str, str]],
) -> list[dict[str, str]]:
    required: set[tuple[str, str]] = set()
    for index, item in enumerate(required_artifacts):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ArtifactInventoryError(
                f"required_artifacts[{index}] must be a (root_id, path) tuple"
            )
        root_id, raw_path = item
        if root_id not in root_ids:
            raise ArtifactInventoryError(
                f"required artifact uses unknown root id: {root_id!r}"
            )
        relative_path = _safe_relative_path(
            raw_path,
            field=f"required_artifacts[{index}].relative_path",
        )
        key = (root_id, relative_path)
        if key in exclusions:
            raise ArtifactInventoryError(
                "manifest and validation sidecar cannot be required artifacts"
            )
        required.add(key)
    return [
        {"root_id": root_id, "relative_path": relative_path}
        for root_id, relative_path in sorted(required)
    ]


def _normalize_exclusions(
    exclusions: Iterable[tuple[str, str]],
    *,
    root_ids: set[str],
) -> frozenset[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    for index, item in enumerate(exclusions):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ArtifactInventoryError(
                f"exclusions[{index}] must be a (root_id, path) tuple"
            )
        root_id, raw_path = item
        if root_id not in root_ids:
            raise ArtifactInventoryError(
                f"exclusion uses unknown root id: {root_id!r}"
            )
        normalized.add(
            (
                root_id,
                _safe_relative_path(
                    raw_path,
                    field=f"exclusions[{index}].relative_path",
                ),
            )
        )
    return frozenset(normalized)


def snapshot_roots(
    roots: Mapping[str, str | Path],
    *,
    exclusions: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Return a deterministic logical inventory and its aggregate SHA-256.

    The aggregate covers canonical JSON for the sorted artifact records only.
    Those records contain logical root IDs and relative paths, never host
    absolute paths, so equal relocated root families have equal snapshots when
    their file metadata is otherwise identical.
    """

    normalized = _normalize_roots(roots)
    normalized_exclusions = _normalize_exclusions(
        exclusions,
        root_ids=set(normalized),
    )
    artifacts = _inventory(
        normalized,
        exclusions=normalized_exclusions,
    )
    canonical = json.dumps(
        artifacts,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_manifest(
    roots: Mapping[str, str | Path],
    *,
    output_root_id: str,
    required_artifacts: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Build a deterministic manifest without writing or mutating any root."""

    normalized = _normalize_roots(roots)
    if output_root_id not in normalized:
        raise ArtifactInventoryError(
            f"output_root_id is not an artifact root: {output_root_id!r}"
        )
    exclusions = _detached_outputs(output_root_id)
    artifacts = _inventory(normalized, exclusions=exclusions)
    required = _normalize_required(
        required_artifacts,
        root_ids=set(normalized),
        exclusions=exclusions,
    )
    artifact_keys = {
        (item["root_id"], item["relative_path"]) for item in artifacts
    }
    missing = [
        item
        for item in required
        if (item["root_id"], item["relative_path"]) not in artifact_keys
    ]
    if missing:
        item = missing[0]
        raise ArtifactInventoryError(
            "required artifact is missing: "
            f"{item['root_id']}:{item['relative_path']}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": MANIFEST_RECORD_TYPE,
        "output_root_id": output_root_id,
        "artifact_count": len(artifacts),
        "root_ids": sorted(normalized),
        "required_artifacts": required,
        "detached_outputs": [
            {"root_id": root_id, "relative_path": relative_path}
            for root_id, relative_path in sorted(exclusions)
        ],
        "artifacts": artifacts,
    }


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactInventoryError(
            f"{description} must be a real regular file: {path}"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactInventoryError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactInventoryError(f"{description} must be an object")
    return value


def _manifest_path(
    manifest_path: str | Path,
    roots: Mapping[str, Path],
    *,
    output_root_id: str,
) -> Path:
    path = Path(manifest_path)
    expected = roots[output_root_id] / ARTIFACT_MANIFEST_NAME
    try:
        matches = path.resolve(strict=True) == expected.resolve(strict=True)
    except OSError as error:
        raise ArtifactInventoryError(
            f"artifact manifest path cannot be resolved: {error}"
        ) from error
    if not matches or path.is_symlink():
        raise ArtifactInventoryError(
            f"artifact manifest must be {expected}"
        )
    return path


def _path_entries(
    value: object,
    *,
    field: str,
    root_ids: set[str],
) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ArtifactInventoryError(f"{field} must be an array")
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _PATH_FIELDS:
            raise ArtifactInventoryError(
                f"{field}[{index}] does not match contract"
            )
        root_id = item["root_id"]
        if not isinstance(root_id, str) or root_id not in root_ids:
            raise ArtifactInventoryError(
                f"{field}[{index}] uses unknown root id: {root_id!r}"
            )
        relative_path = _safe_relative_path(
            item["relative_path"],
            field=f"{field}[{index}].relative_path",
        )
        key = (root_id, relative_path)
        if key in seen:
            raise ArtifactInventoryError(
                f"duplicate {field} entry: {root_id}:{relative_path}"
            )
        seen.add(key)
        result.append(key)
    if result != sorted(result):
        raise ArtifactInventoryError(f"{field} must be deterministically sorted")
    return result


def _manifest_entries(
    manifest: dict[str, Any],
    *,
    root_ids: set[str],
    output_root_id: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    frozenset[tuple[str, str]],
]:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ArtifactInventoryError(
            "artifact manifest fields do not match contract"
        )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["record_type"] != MANIFEST_RECORD_TYPE
    ):
        raise ArtifactInventoryError("artifact manifest version/type mismatch")
    if manifest["output_root_id"] != output_root_id:
        raise ArtifactInventoryError("artifact manifest output root mismatch")
    if manifest["root_ids"] != sorted(root_ids):
        raise ArtifactInventoryError(
            "artifact manifest root ids do not match inputs"
        )

    detached_keys = _path_entries(
        manifest["detached_outputs"],
        field="detached_outputs",
        root_ids=root_ids,
    )
    expected_exclusions = _detached_outputs(output_root_id)
    if set(detached_keys) != set(expected_exclusions):
        raise ArtifactInventoryError(
            "only the output manifest and validation sidecar may be excluded"
        )

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list):
        raise ArtifactInventoryError("artifacts must be an array")
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != _ARTIFACT_FIELDS:
            raise ArtifactInventoryError(
                f"artifacts[{index}] does not match contract"
            )
        root_id = item["root_id"]
        if not isinstance(root_id, str) or root_id not in root_ids:
            raise ArtifactInventoryError(
                f"artifacts[{index}] uses unknown root id: {root_id!r}"
            )
        relative_path = _safe_relative_path(
            item["relative_path"],
            field=f"artifacts[{index}].relative_path",
        )
        key = (root_id, relative_path)
        if key in expected_exclusions:
            raise ArtifactInventoryError(
                f"detached output is self-inventoried: {root_id}:{relative_path}"
            )
        if key in entries:
            raise ArtifactInventoryError(
                f"duplicate artifact entry: {root_id}:{relative_path}"
            )
        for name in ("size_bytes", "mtime_ns", "mode"):
            field_value = item[name]
            if (
                not isinstance(field_value, int)
                or isinstance(field_value, bool)
                or field_value < 0
            ):
                raise ArtifactInventoryError(
                    f"artifacts[{index}].{name} must be a non-negative integer"
                )
        if item["mode"] > 0o7777:
            raise ArtifactInventoryError(
                f"artifacts[{index}].mode is outside POSIX permission bits"
            )
        if (
            not isinstance(item["sha256"], str)
            or _SHA256_RE.fullmatch(item["sha256"]) is None
        ):
            raise ArtifactInventoryError(
                f"artifacts[{index}].sha256 is invalid"
            )
        entries[key] = item
        ordered_keys.append(key)
    if ordered_keys != sorted(ordered_keys):
        raise ArtifactInventoryError("artifacts must be deterministically sorted")

    artifact_count = manifest["artifact_count"]
    if (
        not isinstance(artifact_count, int)
        or isinstance(artifact_count, bool)
        or artifact_count != len(entries)
    ):
        raise ArtifactInventoryError(
            "artifact_count does not match artifacts"
        )

    required_keys = _path_entries(
        manifest["required_artifacts"],
        field="required_artifacts",
        root_ids=root_ids,
    )
    excluded_required = set(required_keys) & set(expected_exclusions)
    if excluded_required:
        raise ArtifactInventoryError(
            "detached outputs cannot be required artifacts"
        )
    missing_required = set(required_keys) - set(entries)
    if missing_required:
        root_id, relative_path = sorted(missing_required)[0]
        raise ArtifactInventoryError(
            f"required artifact is not inventoried: {root_id}:{relative_path}"
        )
    return entries, expected_exclusions


def validate_manifest(
    manifest_path: str | Path,
    roots: Mapping[str, str | Path],
    *,
    output_root_id: str,
) -> dict[str, Any]:
    """Purely validate a manifest against current disk state."""

    normalized = _normalize_roots(roots)
    if output_root_id not in normalized:
        raise ArtifactInventoryError(
            f"output_root_id is not an artifact root: {output_root_id!r}"
        )
    path = _manifest_path(
        manifest_path,
        normalized,
        output_root_id=output_root_id,
    )
    manifest = _read_json_object(path, description="artifact manifest")
    expected, exclusions = _manifest_entries(
        manifest,
        root_ids=set(normalized),
        output_root_id=output_root_id,
    )
    actual = {
        (item["root_id"], item["relative_path"]): item
        for item in _inventory(normalized, exclusions=exclusions)
    }

    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(expected) - set(actual)):
        mismatches.append(
            {
                "root_id": key[0],
                "relative_path": key[1],
                "reason": "missing",
            }
        )
    for key in sorted(set(actual) - set(expected)):
        mismatches.append(
            {
                "root_id": key[0],
                "relative_path": key[1],
                "reason": "unexpected",
            }
        )
    for key in sorted(set(expected) & set(actual)):
        expected_item = expected[key]
        actual_item = actual[key]
        changed = [
            field
            for field in ("size_bytes", "sha256", "mtime_ns", "mode")
            if expected_item[field] != actual_item[field]
        ]
        if changed:
            mismatches.append(
                {
                    "root_id": key[0],
                    "relative_path": key[1],
                    "reason": "changed",
                    "fields": changed,
                    "expected": {
                        field: expected_item[field] for field in changed
                    },
                    "actual": {
                        field: actual_item[field] for field in changed
                    },
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": VALIDATION_RECORD_TYPE,
        "valid": not mismatches,
        "checked": len(expected),
        "mismatches": mismatches,
        "manifest_sha256": _sha256_file(path),
    }


def _validate_stored_report(value: dict[str, Any]) -> None:
    if set(value) != _VALIDATION_FIELDS:
        raise ArtifactInventoryError(
            "stored validation fields do not match contract"
        )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["record_type"] != VALIDATION_RECORD_TYPE
    ):
        raise ArtifactInventoryError("stored validation version/type mismatch")
    if not isinstance(value["valid"], bool):
        raise ArtifactInventoryError("stored validation valid must be boolean")
    if (
        not isinstance(value["checked"], int)
        or isinstance(value["checked"], bool)
        or value["checked"] < 0
    ):
        raise ArtifactInventoryError(
            "stored validation checked must be non-negative integer"
        )
    if not isinstance(value["mismatches"], list):
        raise ArtifactInventoryError(
            "stored validation mismatches must be an array"
        )
    if (
        not isinstance(value["manifest_sha256"], str)
        or _SHA256_RE.fullmatch(value["manifest_sha256"]) is None
    ):
        raise ArtifactInventoryError(
            "stored validation manifest_sha256 is invalid"
        )


def verify_stored_sidecar(
    manifest_path: str | Path,
    roots: Mapping[str, str | Path],
    *,
    output_root_id: str,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable sidecar against the manifest and a fresh report."""

    normalized = _normalize_roots(roots)
    if output_root_id not in normalized:
        raise ArtifactInventoryError(
            f"output_root_id is not an artifact root: {output_root_id!r}"
        )
    manifest = _manifest_path(
        manifest_path,
        normalized,
        output_root_id=output_root_id,
    )
    expected_sidecar = normalized[output_root_id] / ARTIFACT_VALIDATION_NAME
    sidecar = (
        expected_sidecar if sidecar_path is None else Path(sidecar_path)
    )
    try:
        matches = (
            sidecar.resolve(strict=True)
            == expected_sidecar.resolve(strict=True)
        )
    except OSError as error:
        raise ArtifactInventoryError(
            f"stored validation path cannot be resolved: {error}"
        ) from error
    if not matches or sidecar.is_symlink():
        raise ArtifactInventoryError(
            f"stored validation must be {expected_sidecar}"
        )

    stored = _read_json_object(
        sidecar,
        description="stored artifact validation",
    )
    _validate_stored_report(stored)
    actual_manifest_sha256 = _sha256_file(manifest)
    if stored["manifest_sha256"] != actual_manifest_sha256:
        raise ArtifactInventoryError(
            "stored validation manifest SHA-256 does not match the manifest"
        )
    fresh = validate_manifest(
        manifest,
        normalized,
        output_root_id=output_root_id,
    )
    if stored != fresh:
        raise ArtifactInventoryError(
            "stored validation does not match fresh disk validation"
        )
    if fresh["valid"] is not True:
        raise ArtifactInventoryError(
            "stored and fresh artifact validation are not valid"
        )
    return fresh


# A descriptive alias for code that treats both files as one published bundle.
verify_published_bundle = verify_stored_sidecar


__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "ARTIFACT_VALIDATION_NAME",
    "ArtifactError",
    "ArtifactInventoryError",
    "MANIFEST_RECORD_TYPE",
    "VALIDATION_RECORD_TYPE",
    "build_manifest",
    "snapshot_roots",
    "validate_manifest",
    "verify_published_bundle",
    "verify_stored_sidecar",
    "write_json_exclusive",
]
