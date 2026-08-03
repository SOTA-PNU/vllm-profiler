"""Configuration records for one GPU-only collection run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...schema import ProfileMode, RunPaths
from ..command import CommandSpec


MIN_SAMPLE_INTERVAL_MS = 100
DEFAULT_SAMPLE_INTERVAL_MS = 1000


@dataclass(frozen=True)
class GpuDeviceInfo:
    index: int
    name: str
    memory_total_bytes: int | None = None

    @property
    def device_id(self) -> str:
        return f"gpu-{self.index}"


@dataclass(frozen=True)
class GpuRunConfig:
    run_root: Path
    run_id: str
    profile_mode: ProfileMode
    sample_interval_ms: int
    command: tuple[str, ...]
    cwd: Path | None = None
    host_alias: str = "host-0"
    timeout_sec: float | None = None
    model_id: str = "unspecified"
    gpu_devices: tuple[GpuDeviceInfo, ...] = ()

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
        if not self.host_alias.strip():
            raise ValueError("host alias must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
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
            timeout_sec=self.timeout_sec,
        )
