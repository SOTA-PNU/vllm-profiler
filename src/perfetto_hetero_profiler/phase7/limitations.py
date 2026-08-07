"""Stable Phase 7B limitation inventory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Limitation:
    limitation_id: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "limitation_id": self.limitation_id,
            "status": self.status,
            "detail": self.detail,
        }


REQUIRED_LIMITATIONS = (
    Limitation("fixed_model_partition", "limited", "Only Qwen3-0.6B with one fixed GPU-prefill/NPU-decode partition is evaluated."),
    Limitation("formal_sample_count", "limited", "Each condition has five formal observations."),
    Limitation("native_partial_clock", "partial", "GPU Torch, Nsight, and NPU Torch use partial-derived clock evidence."),
    Limitation("perfetto_ui_plugin", "not_implemented", "HTML is external and no Perfetto UI plugin is implemented."),
    Limitation("rbln_canonical_clock_anchor", "not_available", "RBLN PB is a valid Perfetto trace but lacks a trustworthy canonical anchor."),
    Limitation("reference_runtime_markers", "partial", "Reference disables resource and detailed profilers while existing runtime markers remain enabled."),
    Limitation("resource_sampling_interval", "limited", "Sampling can miss peaks shorter than the configured interval."),
    Limitation("single_detailed_profiler", "by_design", "Only one detailed profiler is enabled in a trial."),
    Limitation("transfer_setup_wait_marker", "not_available", "No independent transfer setup/wait marker exists."),
)


def limitation_inventory() -> tuple[dict[str, str], ...]:
    ids = [item.limitation_id for item in REQUIRED_LIMITATIONS]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("limitation inventory must be sorted and unique")
    return tuple(item.to_dict() for item in REQUIRED_LIMITATIONS)


__all__ = ["Limitation", "REQUIRED_LIMITATIONS", "limitation_inventory"]
