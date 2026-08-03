"""Configuration for Phase 4 hybrid bundle alignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..schema import RunPaths


class AlignmentMethod(str, Enum):
    SAME_CLOCK_DOMAIN = "same-clock-domain"
    LOCAL = "local"
    FAKE = "fake"


@dataclass(frozen=True)
class HybridMergeConfig:
    run_root: Path
    run_id: str
    gpu_run: Path
    npu_run: Path
    alignment_method: AlignmentMethod = AlignmentMethod.SAME_CLOCK_DOMAIN
    max_uncertainty_ns: int = 1_000_000
    coordinator_host_id: str = "hybrid-coordinator"
    canonical_clock_domain_id: str = "hybrid-canonical"
    probe_count: int = 7
    minimum_probe_samples: int = 5
    fake_offset_ns: int = 0
    fake_delay_ns: int = 100_000
    fake_jitter_ns: int = 0
    fake_asymmetry_ns: int = 0
    allow_non_fake_sources: bool = False

    def __post_init__(self) -> None:
        for name in ("run_root", "gpu_run", "npu_run"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not self.run_root.is_absolute():
            raise ValueError("run_root must be absolute")
        if not self.gpu_run.is_absolute() or not self.npu_run.is_absolute():
            raise ValueError("source run paths must be absolute")
        if self.gpu_run == self.npu_run:
            raise ValueError("GPU and NPU source runs must differ")
        if not isinstance(self.alignment_method, AlignmentMethod):
            object.__setattr__(
                self, "alignment_method", AlignmentMethod(self.alignment_method)
            )
        if self.max_uncertainty_ns < 0:
            raise ValueError("max_uncertainty_ns must be non-negative")
        if self.probe_count < 1:
            raise ValueError("probe_count must be positive")
        if not 1 <= self.minimum_probe_samples <= self.probe_count:
            raise ValueError(
                "minimum_probe_samples must be between 1 and probe_count"
            )
        if self.fake_delay_ns < 0 or self.fake_jitter_ns < 0:
            raise ValueError("fake delay and jitter must be non-negative")
        if not self.coordinator_host_id.strip():
            raise ValueError("coordinator_host_id must be non-empty")
        if not self.canonical_clock_domain_id.strip():
            raise ValueError("canonical_clock_domain_id must be non-empty")
        RunPaths(self.run_root, self.run_id)

    @property
    def paths(self) -> RunPaths:
        return RunPaths(self.run_root, self.run_id)


def build_hybrid_plan(config: HybridMergeConfig) -> dict[str, object]:
    """Describe a merge without reading sources or probing clocks."""
    return {
        "executes": False,
        "run_id": config.run_id,
        "run_directory": str(config.paths.root),
        "sources": {
            "gpu": str(config.gpu_run),
            "npu": str(config.npu_run),
        },
        "alignment": {
            "method": config.alignment_method.value,
            "canonical_clock_domain_id": config.canonical_clock_domain_id,
            "maximum_uncertainty_ns": config.max_uncertainty_ns,
            "probe_count": config.probe_count,
            "minimum_probe_samples": config.minimum_probe_samples,
            "probe_executes": False,
        },
        "validation": {
            "source_schema": True,
            "request_join": True,
            "marker_ordering": True,
            "artifact_integrity": True,
            "source_policy": (
                "fake_or_explicitly_allowed"
                if config.allow_non_fake_sources
                else "fake_only"
            ),
        },
        "creates_output": False,
        "creates_perfetto_trace": False,
    }
