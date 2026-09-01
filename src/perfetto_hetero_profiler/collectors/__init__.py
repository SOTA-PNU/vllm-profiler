"""Collector lifecycle, command, process, and system telemetry primitives."""

from .base import BaseCollector, CollectorError, CollectorState
from .command import CommandSpec, build_environment, mask_command, mask_environment
from .process import CommandResult, ManagedProcess
from .system import SystemTelemetryCollector

__all__ = [
    "BaseCollector",
    "CollectorError",
    "CollectorState",
    "CommandResult",
    "CommandSpec",
    "ManagedProcess",
    "SystemTelemetryCollector",
    "build_environment",
    "mask_command",
    "mask_environment",
]
