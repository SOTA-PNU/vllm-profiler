"""Legacy wire identifiers retained for existing artifact compatibility."""

from __future__ import annotations

from typing import Final


LEGACY_MEASURED_WINDOW: Final = "measured_smoke"
LEGACY_MEASURED_WINDOW_AGGREGATION: Final = "measured_smoke_window_v1"
LEGACY_MEASURED_COUNT_AGGREGATION: Final = "measured_smoke_count_v1"
LEGACY_MEASURED_WINDOW_SCOPE: Final = "measured_smoke_window"
LEGACY_SINGLE_REQUEST_SCOPE: Final = "single_request_smoke"
LEGACY_GPU_COLLECTION_SUMMARY: Final = "summary/smoke.json"
LEGACY_GPU_COLLECTION_PRODUCER: Final = "gpu-vllm-smoke"
LEGACY_NPU_COLLECTION_PRODUCER: Final = "npu-runtime-smoke"
LEGACY_GPU_NSYS_OUTPUT: Final = "raw/gpu/nsys/vllm-smoke"


__all__ = [
    "LEGACY_GPU_COLLECTION_PRODUCER",
    "LEGACY_GPU_COLLECTION_SUMMARY",
    "LEGACY_GPU_NSYS_OUTPUT",
    "LEGACY_MEASURED_COUNT_AGGREGATION",
    "LEGACY_MEASURED_WINDOW",
    "LEGACY_MEASURED_WINDOW_AGGREGATION",
    "LEGACY_MEASURED_WINDOW_SCOPE",
    "LEGACY_NPU_COLLECTION_PRODUCER",
    "LEGACY_SINGLE_REQUEST_SCOPE",
]
