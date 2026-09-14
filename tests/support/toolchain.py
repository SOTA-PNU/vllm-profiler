"""Availability gate for tests that launch the pinned Trace Processor."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from perfetto_hetero_profiler.perfetto.tooling import (
    TRACE_PROCESSOR_FILENAME,
    TRACE_PROCESSOR_RELEASE,
)

REQUIRE_TRACE_PROCESSOR_ENV = "HETERO_TESTS_REQUIRE_TRACE_PROCESSOR"
_TestClass = TypeVar("_TestClass", bound=type[unittest.TestCase])


@dataclass(frozen=True)
class TraceProcessorCapability:
    binary_path: Path
    issues: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.issues

    @property
    def reason(self) -> str:
        return "; ".join(self.issues)


def trace_processor_path() -> Path:
    """Return the dedicated binary path for the active Python environment."""

    return (
        Path(sys.prefix)
        / "bin"
        / f"{TRACE_PROCESSOR_FILENAME}-{TRACE_PROCESSOR_RELEASE}"
    )


def _localhost_socket_issue() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError as error:
        return f"localhost socket permission denied: {error}"
    except OSError as error:
        return f"localhost socket unavailable: {error}"
    return None


def trace_processor_capability(
    *,
    require_binary: bool = True,
    require_socket: bool = True,
    dependency_error: str | None = None,
) -> TraceProcessorCapability:
    """Inspect independent package, binary, executable and socket capabilities."""

    binary = trace_processor_path()
    issues: list[str] = []
    if importlib.util.find_spec("perfetto") is None:
        issues.append("Perfetto Python package is unavailable")
    if dependency_error:
        issues.append(f"Perfetto writer dependency is unavailable: {dependency_error}")
    if require_binary:
        if not binary.is_file():
            issues.append(f"pinned Trace Processor binary is missing: {binary}")
        elif not os.access(binary, os.X_OK):
            issues.append(f"pinned Trace Processor binary is not executable: {binary}")
    if require_socket:
        socket_issue = _localhost_socket_issue()
        if socket_issue is not None:
            issues.append(socket_issue)
    return TraceProcessorCapability(binary_path=binary, issues=tuple(issues))


def _required() -> bool:
    return os.getenv(REQUIRE_TRACE_PROCESSOR_ENV) == "1"


def trace_processor_test_class(
    *,
    require_binary: bool = True,
    require_socket: bool = True,
    dependency_error: str | None = None,
):
    """Skip an integration class, or fail explicitly in required-toolchain mode."""

    capability = trace_processor_capability(
        require_binary=require_binary,
        require_socket=require_socket,
        dependency_error=dependency_error,
    )

    def decorate(test_class: _TestClass) -> _TestClass:
        if capability.available:
            return test_class
        reason = f"Trace Processor integration unavailable: {capability.reason}"
        if not _required():
            return unittest.skip(reason)(test_class)

        @classmethod
        def fail_required_toolchain(cls) -> None:
            raise RuntimeError(
                f"{REQUIRE_TRACE_PROCESSOR_ENV}=1 requires the integration "
                f"toolchain: {capability.reason}"
            )

        test_class.setUpClass = fail_required_toolchain
        return test_class

    return decorate


def require_trace_processor(
    test_case: unittest.TestCase,
    *,
    require_socket: bool = True,
) -> Path:
    """Return the binary path or make one test skip/fail according to the gate."""

    capability = trace_processor_capability(require_socket=require_socket)
    if capability.available:
        return capability.binary_path
    reason = f"Trace Processor integration unavailable: {capability.reason}"
    if _required():
        test_case.fail(f"{REQUIRE_TRACE_PROCESSOR_ENV}=1: {reason}")
    test_case.skipTest(reason)
    raise AssertionError("skipTest unexpectedly returned")
