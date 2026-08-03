"""Configuration records for one NPU-only collection run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ...schema import ProfileMode, RunPaths
from ..command import CommandSpec, DEFAULT_ENV_ALLOWLIST


MIN_SAMPLE_INTERVAL_MS = 100
DEFAULT_SAMPLE_INTERVAL_MS = 1000


@dataclass(frozen=True)
class NpuDeviceInfo:
    index: int
    name: str
    status: str = "unknown"
    memory_total_bytes: int | None = None
    firmware_version: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("NPU device index must be non-negative")
        if not self.name.strip():
            raise ValueError("NPU device name must be non-empty")

    @property
    def device_id(self) -> str:
        return f"npu-{self.index}"


@dataclass(frozen=True)
class NpuRunConfig:
    run_root: Path
    run_id: str
    profile_mode: ProfileMode
    sample_interval_ms: int
    command: tuple[str, ...]
    cwd: Path | None = None
    host_id: str = "host-0"
    timeout_sec: float | None = None
    model_id: str = "unspecified"
    device_ids: tuple[int, ...] = ()
    npu_devices: tuple[NpuDeviceInfo, ...] = ()
    allow_detailed_execution: bool = False
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    env_allowlist: tuple[str, ...] = DEFAULT_ENV_ALLOWLIST

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))
        if self.sample_interval_ms < MIN_SAMPLE_INTERVAL_MS:
            raise ValueError(
                f"sample interval must be >= {MIN_SAMPLE_INTERVAL_MS} ms"
            )
        if not self.command:
            raise ValueError("child command is required")
        if not self.host_id.strip():
            raise ValueError("host id must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if any(index < 0 for index in self.device_ids):
            raise ValueError("device id must be a non-negative integer")
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("device ids must be unique")
        if self.npu_devices and self.device_ids:
            present = {device.index for device in self.npu_devices}
            missing = set(self.device_ids) - present
            if missing:
                raise ValueError(
                    f"configured NPU devices do not include device {min(missing)}"
                )
        RunPaths(self.run_root, self.run_id)
        self.command_spec

    @property
    def paths(self) -> RunPaths:
        return RunPaths(self.run_root, self.run_id)

    @property
    def command_spec(self) -> CommandSpec:
        return CommandSpec(
            argv=self.command,
            cwd=self.cwd,
            env_overrides=self.env_overrides,
            env_allowlist=self.env_allowlist,
            timeout_sec=self.timeout_sec,
        )
