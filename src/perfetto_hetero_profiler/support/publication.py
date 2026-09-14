"""Filesystem primitives shared by no-overwrite artifact publishers."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory_no_replace(
    staging: Path,
    output: Path,
    *,
    error_type: type[RuntimeError],
) -> None:
    """Atomically publish one staged directory without replacing a target."""

    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise error_type("atomic no-replace directory publication is unavailable")
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
            raise error_type("atomic no-replace directory publication is unsupported")
        raise OSError(error_number, os.strerror(error_number), os.fspath(output))
    fsync_directory(output.parent)


__all__ = ["fsync_directory", "publish_directory_no_replace"]
