"""Detached, non-circular integrity inventories for finalized artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

from .constants import SCHEMA_VERSION


DETACHED_MANIFEST_NAME = "artifact_manifest.json"
DETACHED_VALIDATION_NAME = "artifact_manifest_validation.json"
RECOVERY_RESULT_NAME = "recovery_result.json"
RECOVERY_ROOT_ID = "recovery"

_MANIFEST_RECORD_TYPE = "detached_artifact_manifest"
_VALIDATION_RECORD_TYPE = "detached_artifact_validation"
_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactIntegrityError(RuntimeError):
    """An invalid inventory contract or unsafe recovery output."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        if path.is_file() and path.read_bytes() == payload:
            return
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short detached artifact metadata write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output already exists: {path}"
                ) from error
            temporary.unlink()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise ArtifactIntegrityError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or normalized == "."
        or normalized != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactIntegrityError(f"{field} must be a safe relative path")
    return normalized


def _normalize_roots(roots: Mapping[str, str | Path]) -> dict[str, Path]:
    if not roots:
        raise ArtifactIntegrityError("at least one artifact root is required")
    normalized: dict[str, Path] = {}
    resolved: dict[str, Path] = {}
    for root_id, value in roots.items():
        if not isinstance(root_id, str) or _ROOT_ID_RE.fullmatch(root_id) is None:
            raise ArtifactIntegrityError(f"invalid artifact root id: {root_id!r}")
        path = Path(value)
        if not path.is_dir() or path.is_symlink():
            raise ArtifactIntegrityError(
                f"artifact root must be a real directory: {path}"
            )
        normalized[root_id] = path
        resolved[root_id] = path.resolve()
    items = sorted(resolved.items())
    for index, (left_id, left) in enumerate(items):
        for right_id, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ArtifactIntegrityError(
                    "artifact roots must not overlap: "
                    f"{left_id!r} and {right_id!r}"
                )
    return normalized


def _detached_exclusions() -> frozenset[tuple[str, str]]:
    return frozenset(
        {
            (RECOVERY_ROOT_ID, DETACHED_MANIFEST_NAME),
            (RECOVERY_ROOT_ID, DETACHED_VALIDATION_NAME),
        }
    )


def _inventory(
    roots: Mapping[str, Path],
    *,
    exclusions: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root_id, root in sorted(roots.items()):
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ArtifactIntegrityError(
                    f"artifact inventory rejects symlinks: {path}"
                )
            if not path.is_file():
                continue
            relative_path = path.relative_to(root).as_posix()
            if (root_id, relative_path) in exclusions:
                continue
            stat = path.stat()
            records.append(
                {
                    "root_id": root_id,
                    "relative_path": relative_path,
                    "size_bytes": stat.st_size,
                    "sha256": _sha256_file(path),
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return records


def _normalize_required(
    required_artifacts: Iterable[tuple[str, str]],
    *,
    root_ids: set[str],
) -> list[dict[str, str]]:
    required: set[tuple[str, str]] = set()
    for index, item in enumerate(required_artifacts):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ArtifactIntegrityError(
                f"required_artifacts[{index}] must be a (root_id, path) tuple"
            )
        root_id, relative_path = item
        if root_id not in root_ids:
            raise ArtifactIntegrityError(
                f"required artifact uses unknown root id: {root_id!r}"
            )
        required.add(
            (
                root_id,
                _safe_relative_path(
                    relative_path,
                    field=f"required_artifacts[{index}].relative_path",
                ),
            )
        )
    required.add((RECOVERY_ROOT_ID, RECOVERY_RESULT_NAME))
    return [
        {"root_id": root_id, "relative_path": relative_path}
        for root_id, relative_path in sorted(required)
    ]


def build_detached_artifact_manifest(
    roots: Mapping[str, str | Path],
    *,
    required_artifacts: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Build a deterministic manifest without writing or mutating any root."""

    normalized = _normalize_roots(roots)
    if RECOVERY_ROOT_ID not in normalized:
        raise ArtifactIntegrityError(
            f"artifact roots must include {RECOVERY_ROOT_ID!r}"
        )
    exclusions = _detached_exclusions()
    artifacts = _inventory(normalized, exclusions=exclusions)
    required = _normalize_required(
        required_artifacts,
        root_ids=set(normalized),
    )
    keys = {
        (item["root_id"], item["relative_path"]) for item in artifacts
    }
    missing_required = [
        item
        for item in required
        if (item["root_id"], item["relative_path"]) not in keys
    ]
    if missing_required:
        item = missing_required[0]
        raise ArtifactIntegrityError(
            "required artifact is missing: "
            f"{item['root_id']}:{item['relative_path']}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": _MANIFEST_RECORD_TYPE,
        "artifact_count": len(artifacts),
        "root_ids": sorted(normalized),
        "required_artifacts": required,
        "detached_outputs": [
            {"root_id": root_id, "relative_path": relative_path}
            for root_id, relative_path in sorted(exclusions)
        ],
        "artifacts": artifacts,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"invalid artifact manifest: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError("artifact manifest must be an object")
    expected_keys = {
        "schema_version",
        "record_type",
        "artifact_count",
        "root_ids",
        "required_artifacts",
        "detached_outputs",
        "artifacts",
    }
    if set(value) != expected_keys:
        raise ArtifactIntegrityError("artifact manifest fields do not match contract")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["record_type"] != _MANIFEST_RECORD_TYPE
    ):
        raise ArtifactIntegrityError("artifact manifest version/type mismatch")
    return value


def _manifest_entries(
    value: dict[str, Any],
    *,
    root_ids: set[str],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    frozenset[tuple[str, str]],
]:
    listed_roots = value["root_ids"]
    if (
        not isinstance(listed_roots, list)
        or listed_roots != sorted(root_ids)
    ):
        raise ArtifactIntegrityError("artifact manifest root ids do not match inputs")

    detached = value["detached_outputs"]
    if not isinstance(detached, list):
        raise ArtifactIntegrityError("detached_outputs must be an array")
    detached_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(detached):
        if not isinstance(item, dict) or set(item) != {
            "root_id",
            "relative_path",
        }:
            raise ArtifactIntegrityError(
                f"detached_outputs[{index}] does not match contract"
            )
        root_id = item["root_id"]
        if root_id not in root_ids:
            raise ArtifactIntegrityError(
                f"detached output uses unknown root id: {root_id!r}"
            )
        detached_keys.add(
            (
                root_id,
                _safe_relative_path(
                    item["relative_path"],
                    field=f"detached_outputs[{index}].relative_path",
                ),
            )
        )
    if detached_keys != set(_detached_exclusions()):
        raise ArtifactIntegrityError(
            "only the manifest and detached validation report may be excluded"
        )

    artifacts = value["artifacts"]
    if not isinstance(artifacts, list):
        raise ArtifactIntegrityError("artifacts must be an array")
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    expected_fields = {
        "root_id",
        "relative_path",
        "size_bytes",
        "sha256",
        "mtime_ns",
    }
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ArtifactIntegrityError(
                f"artifacts[{index}] does not match contract"
            )
        root_id = item["root_id"]
        if root_id not in root_ids:
            raise ArtifactIntegrityError(
                f"artifact uses unknown root id: {root_id!r}"
            )
        relative_path = _safe_relative_path(
            item["relative_path"],
            field=f"artifacts[{index}].relative_path",
        )
        for field in ("size_bytes", "mtime_ns"):
            field_value = item[field]
            if (
                not isinstance(field_value, int)
                or isinstance(field_value, bool)
                or field_value < 0
            ):
                raise ArtifactIntegrityError(
                    f"artifacts[{index}].{field} must be non-negative integer"
                )
        if (
            not isinstance(item["sha256"], str)
            or _SHA256_RE.fullmatch(item["sha256"]) is None
        ):
            raise ArtifactIntegrityError(
                f"artifacts[{index}].sha256 is invalid"
            )
        key = (root_id, relative_path)
        if key in entries:
            raise ArtifactIntegrityError(
                f"duplicate artifact entry: {root_id}:{relative_path}"
            )
        entries[key] = item
    artifact_count = value["artifact_count"]
    if (
        not isinstance(artifact_count, int)
        or isinstance(artifact_count, bool)
        or artifact_count != len(entries)
    ):
        raise ArtifactIntegrityError("artifact_count does not match artifacts")

    required = value["required_artifacts"]
    if not isinstance(required, list):
        raise ArtifactIntegrityError("required_artifacts must be an array")
    required_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(required):
        if not isinstance(item, dict) or set(item) != {
            "root_id",
            "relative_path",
        }:
            raise ArtifactIntegrityError(
                f"required_artifacts[{index}] does not match contract"
            )
        root_id = item["root_id"]
        if root_id not in root_ids:
            raise ArtifactIntegrityError(
                f"required artifact uses unknown root id: {root_id!r}"
            )
        required_keys.add(
            (
                root_id,
                _safe_relative_path(
                    item["relative_path"],
                    field=f"required_artifacts[{index}].relative_path",
                ),
            )
        )
    if (RECOVERY_ROOT_ID, RECOVERY_RESULT_NAME) not in required_keys:
        raise ArtifactIntegrityError("recovery_result.json must be required")
    missing_required = required_keys - set(entries)
    if missing_required:
        root_id, relative_path = sorted(missing_required)[0]
        raise ArtifactIntegrityError(
            f"required artifact is not inventoried: {root_id}:{relative_path}"
        )
    return entries, frozenset(detached_keys)


def validate_detached_artifact_manifest(
    manifest_path: str | Path,
    roots: Mapping[str, str | Path],
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every listed artifact from disk and detect unexpected files."""

    path = Path(manifest_path)
    normalized = _normalize_roots(roots)
    value = _read_manifest(path)
    expected, exclusions = _manifest_entries(
        value,
        root_ids=set(normalized),
    )
    actual_rows = _inventory(normalized, exclusions=exclusions)
    actual = {
        (item["root_id"], item["relative_path"]): item
        for item in actual_rows
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
            for field in ("size_bytes", "sha256", "mtime_ns")
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "record_type": _VALIDATION_RECORD_TYPE,
        "valid": not mismatches,
        "checked": len(expected),
        "mismatches": mismatches,
        "manifest_sha256": _sha256_file(path),
    }
    if report_path is not None:
        _write_json_atomic(Path(report_path), report, overwrite=True)
    return report


def create_detached_recovery(
    output_directory: str | Path,
    source_roots: Mapping[str, str | Path],
    recovery_result: Mapping[str, Any],
    *,
    required_artifacts: Iterable[tuple[str, str]] = (),
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Publish one immutable recovery result with a detached validation report."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"recovery output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"recovery output parent is missing: {output.parent}")
    if recovery_result.get("success") is not True:
        raise ArtifactIntegrityError("failed recovery cannot be published")
    if recovery_result.get("hardware_rerun") is not False:
        raise ArtifactIntegrityError("detached recovery must be metadata-only")
    normalized_sources = _normalize_roots(source_roots)
    if RECOVERY_ROOT_ID in normalized_sources:
        raise ArtifactIntegrityError(
            f"source root id {RECOVERY_ROOT_ID!r} is reserved"
        )
    resolved_output = output.resolve()
    for root_id, source in normalized_sources.items():
        resolved_source = source.resolve()
        if (
            resolved_output == resolved_source
            or resolved_output in resolved_source.parents
            or resolved_source in resolved_output.parents
        ):
            raise ArtifactIntegrityError(
                "recovery output must be outside every source root: "
                f"{root_id!r}"
            )

    staging = output.with_name(
        f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    recovery_path = staging / RECOVERY_RESULT_NAME
    _write_json_atomic(recovery_path, recovery_result, overwrite=False)
    roots: dict[str, Path] = {
        **normalized_sources,
        RECOVERY_ROOT_ID: staging,
    }
    manifest = build_detached_artifact_manifest(
        roots,
        required_artifacts=required_artifacts,
    )
    manifest_path = staging / DETACHED_MANIFEST_NAME
    validation_path = staging / DETACHED_VALIDATION_NAME
    _write_json_atomic(manifest_path, manifest, overwrite=False)
    validation = validate_detached_artifact_manifest(
        manifest_path,
        roots,
        report_path=validation_path,
    )
    if not validation["valid"]:
        raise ArtifactIntegrityError(
            f"detached artifact validation failed: {validation['mismatches']}"
        )
    if output.exists():
        raise FileExistsError(f"recovery output already exists: {output}")
    staging.rename(output)

    final_roots = {**normalized_sources, RECOVERY_ROOT_ID: output}
    final_validation = validate_detached_artifact_manifest(
        output / DETACHED_MANIFEST_NAME,
        final_roots,
        report_path=output / DETACHED_VALIDATION_NAME,
    )
    if not final_validation["valid"]:
        raise ArtifactIntegrityError(
            "published detached recovery failed validation: "
            f"{final_validation['mismatches']}"
        )
    return output, manifest, final_validation
