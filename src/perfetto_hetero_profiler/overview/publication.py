"""No-overwrite transactional publication for Overview bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from ..support.files import sha256_file

from ..perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    build_manifest,
    validate_manifest,
    verify_stored_sidecar,
    write_json_exclusive,
)


OVERVIEW_OUTPUT_ROOT_ID = "overview"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")


class OverviewPublicationError(RuntimeError):
    """An Overview output could not be safely published."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode deterministic, finite JSON with the repository line policy."""

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


def _absolute_without_resolving(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            file_stat = current.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise OverviewPublicationError(
                f"output path component cannot be inspected: {current}"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise OverviewPublicationError(
                f"output path must not contain a symlink component: {current}"
            )


def validate_output_path(
    output: str | Path,
    *,
    immutable_roots: Sequence[Path],
) -> Path:
    """Validate a direct child output and reject input/output overlap."""

    candidate = _absolute_without_resolving(Path(output))
    if _SAFE_NAME_RE.fullmatch(candidate.name) is None:
        raise OverviewPublicationError("output directory name is unsafe")
    parent = candidate.parent
    _reject_symlink_components(parent)
    try:
        parent_stat = parent.lstat()
    except OSError as error:
        raise OverviewPublicationError(
            f"output parent cannot be inspected: {parent}"
        ) from error
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise OverviewPublicationError("output parent must be a real directory")
    if os.path.lexists(candidate):
        raise FileExistsError(f"output already exists: {candidate}")

    output_resolved = candidate.resolve(strict=False)
    for root in immutable_roots:
        source = Path(root).resolve(strict=True)
        if (
            output_resolved == source
            or output_resolved in source.parents
            or source in output_resolved.parents
        ):
            raise OverviewPublicationError(
                "output directory must not overlap an immutable input"
            )
    return candidate


def _safe_payload_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PurePosixPath(value).parts != (value,)
        or value in {".", ".."}
    ):
        raise OverviewPublicationError(f"unsafe output filename: {value!r}")
    if value in {ARTIFACT_MANIFEST_NAME, ARTIFACT_VALIDATION_NAME}:
        raise OverviewPublicationError(
            "detached manifest outputs are managed by the publisher"
        )
    return value


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    if not isinstance(data, bytes):
        raise TypeError("payload data must be bytes")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


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
        raise OverviewPublicationError(
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
            raise OverviewPublicationError(
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
    expected_prefix = f".{output_name}.overview-staging-"
    if (
        staging.parent != parent
        or not staging.name.startswith(expected_prefix)
        or staging.is_symlink()
    ):
        raise OverviewPublicationError(
            f"refusing to remove an unowned staging path: {staging}"
        )
    if staging.is_dir():
        shutil.rmtree(staging)


def _file_metadata(path: Path) -> dict[str, Any]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OverviewPublicationError(f"published output is not regular: {path}")
    digest = sha256_file(path)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise OverviewPublicationError(f"published output changed: {path}")
    return {
        "relative_path": path.name,
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def _verify_exact_output(
    root: Path,
    payload_names: Sequence[str],
) -> None:
    expected = {
        *payload_names,
        ARTIFACT_MANIFEST_NAME,
        ARTIFACT_VALIDATION_NAME,
    }
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if {entry.name for entry in entries} != expected or len(entries) != len(expected):
        raise OverviewPublicationError(
            "published Overview bundle does not contain exactly five files"
        )
    for entry in entries:
        file_stat = entry.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise OverviewPublicationError(
                "published Overview entries must be real regular files"
            )


def publish_bundle(
    output: Path,
    *,
    payloads: Mapping[str, bytes],
    validate_staging: Callable[[Path], None] | None = None,
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Write, inventory, validate, and atomically publish an exact bundle.

    ``output`` must already have passed :func:`validate_output_path`.
    Payloads must contain exactly the semantic JSON, HTML, and validation
    files.  The detached manifest and sidecar are generated here and are never
    self-inventoried.
    """

    if len(payloads) != 3:
        raise OverviewPublicationError(
            "an Overview bundle must have exactly three semantic payloads"
        )
    normalized_payloads = {
        _safe_payload_name(name): data for name, data in payloads.items()
    }
    if len(normalized_payloads) != 3:
        raise OverviewPublicationError("output payload names must be unique")
    parent = output.parent
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.overview-staging-",
            dir=parent,
        )
    )
    published = False
    try:
        for name in sorted(normalized_payloads):
            _write_bytes_exclusive(staging / name, normalized_payloads[name])

        roots = {OVERVIEW_OUTPUT_ROOT_ID: staging}
        required = tuple(
            (OVERVIEW_OUTPUT_ROOT_ID, name)
            for name in sorted(normalized_payloads)
        )
        manifest = build_manifest(
            roots,
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
            required_artifacts=required,
        )
        manifest_path = staging / ARTIFACT_MANIFEST_NAME
        write_json_exclusive(manifest_path, manifest)
        artifact_validation = validate_manifest(
            manifest_path,
            roots,
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
        if (
            artifact_validation.get("valid") is not True
            or artifact_validation.get("mismatches") != []
        ):
            raise OverviewPublicationError(
                "detached artifact validation found a mismatch"
            )
        write_json_exclusive(
            staging / ARTIFACT_VALIDATION_NAME,
            artifact_validation,
        )
        verify_stored_sidecar(
            manifest_path,
            roots,
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
        _verify_exact_output(staging, tuple(normalized_payloads))
        _fsync_directory(staging)
        if validate_staging is not None:
            validate_staging(staging)

        if before_publish is not None:
            before_publish()
        _publish_directory_no_replace(staging, output)
        published = True

        _verify_exact_output(output, tuple(normalized_payloads))
        published_validation = verify_stored_sidecar(
            output / ARTIFACT_MANIFEST_NAME,
            {OVERVIEW_OUTPUT_ROOT_ID: output},
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
        return {
            "artifact_validation": published_validation,
            "files": [
                _file_metadata(output / name)
                for name in sorted(
                    {
                        *normalized_payloads,
                        ARTIFACT_MANIFEST_NAME,
                        ARTIFACT_VALIDATION_NAME,
                    }
                )
            ],
        }
    finally:
        if not published:
            _remove_owned_staging(
                staging,
                parent=parent,
                output_name=output.name,
            )


__all__ = [
    "OVERVIEW_OUTPUT_ROOT_ID",
    "OverviewPublicationError",
    "canonical_json_bytes",
    "publish_bundle",
    "validate_output_path",
]
