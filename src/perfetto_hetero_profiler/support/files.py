"""Content hashing and deterministic file-tree fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(
    root: str | Path,
    paths: Iterable[Path],
    *,
    include_hash: bool = True,
    include_mtime: bool = False,
) -> list[dict[str, object]]:
    """Return path-sorted, path-free identities for files below ``root``."""

    root_path = Path(root)
    result: list[dict[str, object]] = []
    for path in sorted(Path(value) for value in paths):
        stat = path.stat()
        row: dict[str, object] = {
            "relative_path": path.relative_to(root_path).as_posix(),
            "size_bytes": stat.st_size,
        }
        if include_mtime:
            row["mtime_ns"] = stat.st_mtime_ns
        if include_hash:
            row["sha256"] = sha256_file(path)
        result.append(row)
    return result


def fingerprint_tree(
    root: str | Path,
    *,
    pattern: str = "*",
    include_hash: bool = True,
    include_mtime: bool = False,
) -> list[dict[str, object]]:
    """Fingerprint regular files selected by one recursive glob."""

    root_path = Path(root)
    paths = (
        path
        for path in root_path.rglob(pattern)
        if path.is_file()
    )
    return fingerprint_files(
        root_path,
        paths,
        include_hash=include_hash,
        include_mtime=include_mtime,
    )
