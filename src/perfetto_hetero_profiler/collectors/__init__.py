"""Collector lifecycle, command, process, and system telemetry primitives."""

from .base import BaseCollector, CollectorError, CollectorState
from .command import CommandSpec, build_environment, mask_command, mask_environment
from .process import CommandResult, ManagedProcess
from .system import (
    CpuTimes,
    ProcTelemetryCollector,
    parse_meminfo,
    parse_process_rss,
    parse_proc_stat,
)

__all__ = [
    "BaseCollector",
    "CollectorError",
    "CollectorState",
    "CommandResult",
    "CommandSpec",
    "CpuTimes",
    "ManagedProcess",
    "ProcTelemetryCollector",
    "build_environment",
    "mask_command",
    "mask_environment",
    "parse_meminfo",
    "parse_process_rss",
    "parse_proc_stat",
]
