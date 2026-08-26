"""Pinned official Perfetto toolchain discovery and integrity validation.

The Python package and Trace Processor values in this module are release
provenance, not mutable defaults.  In particular, resolving a missing binary
uses the version pinned by the installed official ``perfetto`` package and
never opts in to the "latest" endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any

from ..support.files import sha256_file


PERFETTO_PACKAGE_VERSION = "0.57.2"
PERFETTO_WHEEL_FILENAME = "perfetto-0.57.2-py3-none-any.whl"
PERFETTO_WHEEL_SHA256 = (
    "307896cee046778e632b4361a704417ad1ecd6f48b07da0a6178681f919311fa"
)
PERFETTO_WHEEL_SOURCE = (
    "https://files.pythonhosted.org/packages/3d/59/"
    "7252b92e3e5738a6bf88e4579007b7df4c8995f0c95b75e3682473d2153b/"
    "perfetto-0.57.2-py3-none-any.whl"
)
PERFETTO_UPSTREAM_REVISION = "1fc6b4daa5a9fb3b44ea019cfb81edde3f39e242"

PROTOBUF_PACKAGE_VERSION = "6.33.6"

TRACE_PROCESSOR_RELEASE = "v56.1"
TRACE_PROCESSOR_FILENAME = "trace_processor_shell"
TRACE_PROCESSOR_SOURCE = (
    "https://commondatastorage.googleapis.com/"
    "perfetto-luci-artifacts/v56.1/linux-amd64/trace_processor_shell"
)
TRACE_PROCESSOR_SIZE_BYTES = 13_687_824
TRACE_PROCESSOR_SHA256 = (
    "becb22d3f2c51dc234627a3ffd5b066602575b50ad4eb082815156f1bc7cb65a"
)
TRACE_PROCESSOR_REVISION = "c794fceabe584dc9172e5512aaaeecc21019a635"
TRACE_PROCESSOR_VERSION = "v56.1-c794fceab"
TRACE_PROCESSOR_VERSION_LINE = (
    "Perfetto v56.1-c794fceab "
    "(c794fceabe584dc9172e5512aaaeecc21019a635)"
)
TRACE_PROCESSOR_RPC_API_VERSION = 14
TRACE_PROCESSOR_VERSION_OUTPUT = (
    f"{TRACE_PROCESSOR_VERSION_LINE}\n"
    f"Trace Processor RPC API version: {TRACE_PROCESSOR_RPC_API_VERSION}"
)
TRACE_PROCESSOR_VERSION_TIMEOUT_SECONDS = 10

_EXPECTED_PLATFORM = "linux"
_EXPECTED_MACHINE = "x86_64"
_EXPECTED_ARCH = "linux-amd64"


class ToolchainValidationError(RuntimeError):
    """The installed Perfetto toolchain does not match the pinned release."""


@dataclass(frozen=True, slots=True)
class ToolchainRuntime:
    """Validated runtime-only toolchain state.

    ``binary_path`` is intentionally kept out of
    :attr:`manifest_metadata`: absolute cache and virtual-environment paths
    are host-specific and would make otherwise deterministic output differ.
    """

    binary_path: Path
    perfetto_package_version: str
    protobuf_package_version: str
    trace_processor_version: str
    trace_processor_version_output: str
    trace_processor_rpc_api_version: int

    @property
    def trace_processor_path(self) -> Path:
        """Compatibility alias identifying the validated executable."""

        return self.binary_path

    @property
    def manifest_metadata(self) -> dict[str, str]:
        """Return deterministic, path-free Trace Processor provenance."""

        return {
            "filename": TRACE_PROCESSOR_FILENAME,
            "version": self.trace_processor_version,
            "sha256": TRACE_PROCESSOR_SHA256,
            "source": TRACE_PROCESSOR_SOURCE,
        }

    @property
    def metadata(self) -> dict[str, object]:
        """Return path-free metadata suitable for detached manifests."""

        return dict(self.manifest_metadata)

    def to_manifest(self) -> dict[str, object]:
        """Return a fresh deterministic manifest value."""

        return self.metadata


def _installed_version(distribution: str, expected: str) -> str:
    try:
        actual = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as error:
        raise ToolchainValidationError(
            f"required distribution {distribution}=={expected} is not installed"
        ) from error
    if actual != expected:
        raise ToolchainValidationError(
            f"{distribution} version mismatch: expected {expected}, got {actual}"
        )
    return actual


def _validate_official_manifest() -> None:
    if sys.platform.lower() != _EXPECTED_PLATFORM:
        raise ToolchainValidationError(
            "the pinned Trace Processor artifact is only approved for "
            f"{_EXPECTED_PLATFORM}-{_EXPECTED_MACHINE}; got {sys.platform}"
        )
    machine = platform.machine().lower()
    if machine != _EXPECTED_MACHINE:
        raise ToolchainValidationError(
            "the pinned Trace Processor artifact is only approved for "
            f"{_EXPECTED_PLATFORM}-{_EXPECTED_MACHINE}; got "
            f"{sys.platform}-{machine}"
        )

    try:
        from perfetto.prebuilts.manifests.trace_processor_shell import (
            TRACE_PROCESSOR_SHELL_MANIFEST,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise ToolchainValidationError(
            "the installed perfetto package does not expose its official "
            "Trace Processor manifest"
        ) from error

    entries = [
        entry
        for entry in TRACE_PROCESSOR_SHELL_MANIFEST
        if entry.get("arch") == _EXPECTED_ARCH
    ]
    if len(entries) != 1:
        raise ToolchainValidationError(
            "the installed perfetto package does not contain exactly one "
            f"{_EXPECTED_ARCH} Trace Processor manifest entry"
        )
    entry: dict[str, Any] = entries[0]
    expected = {
        "arch": _EXPECTED_ARCH,
        "file_name": TRACE_PROCESSOR_FILENAME,
        "file_size": TRACE_PROCESSOR_SIZE_BYTES,
        "url": TRACE_PROCESSOR_SOURCE,
        "sha256": TRACE_PROCESSOR_SHA256,
        "platform": _EXPECTED_PLATFORM,
        "machine": [_EXPECTED_MACHINE],
    }
    mismatches = [
        f"{key}: expected {value!r}, got {entry.get(key)!r}"
        for key, value in expected.items()
        if entry.get(key) != value
    ]
    if mismatches:
        raise ToolchainValidationError(
            "installed perfetto Trace Processor manifest mismatch: "
            + "; ".join(mismatches)
        )


def _absolute_without_resolving(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.absolute()


def _validate_regular_binary(path: Path) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ToolchainValidationError(
            f"Trace Processor binary does not exist: {path}"
        ) from error
    except OSError as error:
        raise ToolchainValidationError(
            f"cannot inspect Trace Processor binary {path}: {error}"
        ) from error

    if stat.S_ISLNK(file_stat.st_mode):
        raise ToolchainValidationError(
            f"Trace Processor binary must not be a symlink: {path}"
        )
    if not stat.S_ISREG(file_stat.st_mode):
        raise ToolchainValidationError(
            f"Trace Processor binary is not a regular file: {path}"
        )
    if file_stat.st_size != TRACE_PROCESSOR_SIZE_BYTES:
        raise ToolchainValidationError(
            "Trace Processor size mismatch: expected "
            f"{TRACE_PROCESSOR_SIZE_BYTES}, got {file_stat.st_size} ({path})"
        )
    if not os.access(path, os.X_OK):
        raise ToolchainValidationError(
            f"Trace Processor binary is not executable: {path}"
        )
    return file_stat


def _sha256_file(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as error:
        raise ToolchainValidationError(
            f"cannot hash Trace Processor binary {path}: {error}"
        ) from error


def _same_file_state(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )


def _validate_version(path: Path) -> str:
    try:
        completed = subprocess.run(
            [os.fspath(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=TRACE_PROCESSOR_VERSION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ToolchainValidationError(
            "Trace Processor version check timed out after "
            f"{TRACE_PROCESSOR_VERSION_TIMEOUT_SECONDS}s: {path}"
        ) from error
    except OSError as error:
        raise ToolchainValidationError(
            f"cannot execute Trace Processor version check {path}: {error}"
        ) from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise ToolchainValidationError(
            "Trace Processor version check failed with exit code "
            f"{completed.returncode}{suffix}"
        )
    actual = completed.stdout.replace("\r\n", "\n").strip()
    if actual != TRACE_PROCESSOR_VERSION_OUTPUT:
        raise ToolchainValidationError(
            "Trace Processor version output mismatch: expected "
            f"{TRACE_PROCESSOR_VERSION_OUTPUT!r}, got {actual!r}"
        )
    return actual


def _resolve_pinned_binary() -> Path:
    try:
        from perfetto.trace_processor.platform import PlatformDelegate
    except (ImportError, ModuleNotFoundError) as error:
        raise ToolchainValidationError(
            "the installed perfetto package does not expose PlatformDelegate"
        ) from error
    try:
        resolved = PlatformDelegate().get_shell_path(
            bin_path=None,
            fetch_latest=False,
        )
    except Exception as error:
        raise ToolchainValidationError(
            "failed to resolve the package-pinned Trace Processor with "
            "fetch_latest=False"
        ) from error
    if not resolved:
        raise ToolchainValidationError(
            "the official PlatformDelegate returned an empty binary path"
        )
    return Path(resolved)


def resolve_toolchain(binary_path: Path | None = None) -> ToolchainRuntime:
    """Resolve and validate the exact official Perfetto toolchain.

    When ``binary_path`` is absent, the official package
    :class:`PlatformDelegate` performs pinned resolution with
    ``fetch_latest=False``.  The returned path is then subject to the same
    regular-file, symlink, size, SHA-256, executability, and version checks as
    an explicitly supplied path.
    """

    perfetto_version = _installed_version(
        "perfetto",
        PERFETTO_PACKAGE_VERSION,
    )
    protobuf_version = _installed_version(
        "protobuf",
        PROTOBUF_PACKAGE_VERSION,
    )
    _validate_official_manifest()

    candidate = binary_path if binary_path is not None else _resolve_pinned_binary()
    path = _absolute_without_resolving(Path(candidate))
    before = _validate_regular_binary(path)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != TRACE_PROCESSOR_SHA256:
        raise ToolchainValidationError(
            "Trace Processor SHA-256 mismatch: expected "
            f"{TRACE_PROCESSOR_SHA256}, got {actual_sha256} ({path})"
        )
    version_output = _validate_version(path)
    after = _validate_regular_binary(path)
    if not _same_file_state(before, after):
        raise ToolchainValidationError(
            f"Trace Processor binary changed during validation: {path}"
        )

    return ToolchainRuntime(
        binary_path=path,
        perfetto_package_version=perfetto_version,
        protobuf_package_version=protobuf_version,
        trace_processor_version=TRACE_PROCESSOR_VERSION,
        trace_processor_version_output=version_output,
        trace_processor_rpc_api_version=TRACE_PROCESSOR_RPC_API_VERSION,
    )


__all__ = [
    "PERFETTO_PACKAGE_VERSION",
    "PERFETTO_UPSTREAM_REVISION",
    "PERFETTO_WHEEL_FILENAME",
    "PERFETTO_WHEEL_SHA256",
    "PERFETTO_WHEEL_SOURCE",
    "PROTOBUF_PACKAGE_VERSION",
    "TRACE_PROCESSOR_FILENAME",
    "TRACE_PROCESSOR_RELEASE",
    "TRACE_PROCESSOR_REVISION",
    "TRACE_PROCESSOR_RPC_API_VERSION",
    "TRACE_PROCESSOR_SHA256",
    "TRACE_PROCESSOR_SIZE_BYTES",
    "TRACE_PROCESSOR_SOURCE",
    "TRACE_PROCESSOR_VERSION",
    "TRACE_PROCESSOR_VERSION_LINE",
    "TRACE_PROCESSOR_VERSION_OUTPUT",
    "TRACE_PROCESSOR_VERSION_TIMEOUT_SECONDS",
    "ToolchainRuntime",
    "ToolchainValidationError",
    "resolve_toolchain",
]
