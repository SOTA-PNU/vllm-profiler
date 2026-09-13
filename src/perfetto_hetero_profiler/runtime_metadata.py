"""Truthful, campaign-agnostic runtime metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .schema import RunManifest, RunMode


class RuntimeMetadataError(ValueError):
    """Runtime metadata is internally inconsistent or unsupported."""


def _token_field(values: Iterable[int | None], field: str) -> dict[str, Any]:
    rows = list(values)
    for value in rows:
        if value is None or isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeMetadataError(f"{field} observations must be exact integers")
        if value < 0:
            raise RuntimeMetadataError(f"{field} observations must be non-negative")
    if not rows:
        return {
            "uniform": None,
            "value": None,
            "minimum": None,
            "maximum": None,
            "sample_count": 0,
            "availability": "not_available",
            "reason": "no measured request observation was available",
        }
    uniform = len(set(rows)) == 1
    return {
        "uniform": uniform,
        "value": rows[0] if uniform else None,
        "minimum": min(rows),
        "maximum": max(rows),
        "sample_count": len(rows),
        "availability": "available",
        "reason": None,
    }


def measured_token_metadata(observations: Iterable[object]) -> dict[str, Any]:
    """Summarize measured request token evidence without inventing one value."""

    rows = list(observations)
    return {
        "evidence": "measured_requests",
        "input_tokens": _token_field(
            (getattr(item, "input_tokens", None) for item in rows), "input token"
        ),
        "output_tokens": _token_field(
            (getattr(item, "output_tokens", None) for item in rows), "output token"
        ),
    }


def measured_request_concurrency(observations: Iterable[object]) -> int | None:
    """Return maximum overlap from measured request intervals."""

    events: list[tuple[int, int]] = []
    for item in observations:
        start = getattr(item, "received_ns", None)
        end = getattr(item, "done_ns", None)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
        ):
            raise RuntimeMetadataError(
                "measured request concurrency requires valid request intervals"
            )
        events.extend(((start, 1), (end, -1)))
    if not events:
        return None
    current = maximum = 0
    # A request ending exactly when another starts is not overlapping.
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def model_identity_metadata(
    *,
    served_model_id: str,
    revision: str | None,
    tokenizer_id: str | None,
    dtype: str | None,
    size_label: str | None,
    exact_parameter_count: int | None,
    metadata_origin: str,
) -> dict[str, Any]:
    """Return only explicitly declared model identity fields."""

    return {
        "served_model_id": served_model_id,
        "revision": revision,
        "tokenizer_id": tokenizer_id,
        "dtype": dtype,
        "size_label": size_label,
        "size_label_origin": metadata_origin if size_label is not None else None,
        "exact_parameter_count": exact_parameter_count,
        "exact_parameter_count_status": (
            "declared" if exact_parameter_count is not None else "not_available"
        ),
        "metadata_origin": metadata_origin,
        "path_inference_used": False,
    }


def topology_metadata(mode: RunMode) -> dict[str, Any]:
    """Describe device roles without implying a layer-ratio partition."""

    if mode is RunMode.GPU_ONLY:
        return {
            "mode": mode.value,
            "partition_strategy": "all_gpu",
            "prefill_device_type": "gpu",
            "decode_device_type": "gpu",
            "layer_partition_ratio": None,
            "layer_partition_ratio_status": "not_applicable",
            "transfer_direction": "not_applicable",
        }
    if mode is RunMode.NPU_ONLY:
        return {
            "mode": mode.value,
            "partition_strategy": "all_npu",
            "prefill_device_type": "npu",
            "decode_device_type": "npu",
            "layer_partition_ratio": None,
            "layer_partition_ratio_status": "not_applicable",
            "transfer_direction": "not_applicable",
        }
    return {
        "mode": mode.value,
        "partition_strategy": "gpu_prefill_npu_decode",
        "prefill_device_type": "gpu",
        "decode_device_type": "npu",
        "layer_partition_ratio": None,
        "layer_partition_ratio_status": "not_applicable",
        "transfer_direction": "gpu_to_npu",
        "source_device_type": "gpu",
        "destination_device_type": "npu",
        "producer_role": "kv_producer",
        "consumer_role": "kv_consumer",
    }


def transfer_dimensions() -> dict[str, str]:
    """Canonical dimensions for GPU-produced, NPU-consumed KV transfer."""

    return {
        "transfer.source_device_type": "gpu",
        "transfer.destination_device_type": "npu",
        "transfer.direction": "gpu_to_npu",
        "transfer.producer_role": "kv_producer",
        "transfer.consumer_role": "kv_consumer",
    }


def _nested(value: Mapping[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise RuntimeMetadataError(
                "source manifest is missing runtime metadata: " + ".".join(parts)
            )
        current = current[part]
    return current


def merged_runtime_metadata(
    gpu: RunManifest, npu: RunManifest
) -> dict[str, Any] | None:
    """Fail closed on source metadata mismatch and return Hybrid metadata.

    Metadata predating this optional extension remains readable when it is absent
    from both sources.  A partially upgraded pair is ambiguous and is rejected.
    """

    gpu_runtime = gpu.configuration.get("runtime_metadata")
    npu_runtime = npu.configuration.get("runtime_metadata")
    if gpu_runtime is None and npu_runtime is None:
        return None
    if (
        (gpu_runtime is not None and not isinstance(gpu_runtime, Mapping))
        or (npu_runtime is not None and not isinstance(npu_runtime, Mapping))
    ):
        raise RuntimeMetadataError("runtime_metadata must be an object")
    gpu_contract = (
        gpu_runtime.get("source_merge_contract")
        if isinstance(gpu_runtime, Mapping)
        else None
    )
    npu_contract = (
        npu_runtime.get("source_merge_contract")
        if isinstance(npu_runtime, Mapping)
        else None
    )
    if gpu_contract is None and npu_contract is None:
        return None
    if gpu_runtime is None or npu_runtime is None or gpu_contract != npu_contract:
        raise RuntimeMetadataError(
            "runtime_metadata must be present in both source manifests or neither"
        )
    assert isinstance(gpu_runtime, Mapping)
    assert isinstance(npu_runtime, Mapping)
    if gpu_contract != "truthful_runtime_metadata_v1":
        raise RuntimeMetadataError("source runtime metadata contract is unsupported")
    for field in ("model_identity", "token_counts", "max_num_seqs_capacity"):
        gpu_value = _nested(gpu_runtime, field)
        npu_value = _nested(npu_runtime, field)
        if gpu_value != npu_value:
            raise RuntimeMetadataError(f"source manifest {field} mismatch")
    scalar_fields = (
        ("actual concurrency", gpu.workload.concurrency, npu.workload.concurrency),
        ("input tokens", gpu.workload.input_tokens, npu.workload.input_tokens),
        ("output tokens", gpu.workload.output_tokens, npu.workload.output_tokens),
        ("max model length", gpu.workload.max_model_len, npu.workload.max_model_len),
        (
            "max_num_seqs capacity",
            gpu_runtime["max_num_seqs_capacity"],
            npu_runtime["max_num_seqs_capacity"],
        ),
    )
    for label, left, right in scalar_fields:
        if left != right:
            raise RuntimeMetadataError(f"source manifest {label} mismatch")
    model_fields = ("model_id", "revision", "tokenizer_id", "dtype")
    if not gpu.models or not npu.models:
        raise RuntimeMetadataError("source manifest model identity is missing")
    for field in model_fields:
        if getattr(gpu.models[0], field) != getattr(npu.models[0], field):
            raise RuntimeMetadataError(f"source manifest model {field} mismatch")
    expected = (
        (gpu, "gpu", "kv_producer", RunMode.GPU_ONLY),
        (npu, "npu", "kv_consumer", RunMode.NPU_ONLY),
    )
    for manifest, role, transfer_role, mode in expected:
        if manifest.attributes.get("hybrid.source_role") != role:
            raise RuntimeMetadataError(f"{role} source partition role mismatch")
        if manifest.attributes.get("hybrid.transfer_role") != transfer_role:
            raise RuntimeMetadataError(f"{role} source transfer role mismatch")
        if manifest.attributes.get("hybrid.transfer_direction") != "gpu_to_npu":
            raise RuntimeMetadataError(f"{role} source transfer direction mismatch")
        runtime = _nested(manifest.configuration, "runtime_metadata")
        if _nested(runtime, "topology") != topology_metadata(mode):
            raise RuntimeMetadataError(f"{role} source topology mismatch")
        identity = _nested(runtime, "model_identity")
        model = manifest.models[0]
        for metadata_field, descriptor_field in (
            ("served_model_id", "model_id"),
            ("revision", "revision"),
            ("tokenizer_id", "tokenizer_id"),
            ("dtype", "dtype"),
        ):
            if _nested(identity, metadata_field) != getattr(model, descriptor_field):
                raise RuntimeMetadataError(
                    f"{role} source model identity disagrees with descriptor"
                )
        for token_field, workload_value in (
            ("input_tokens", manifest.workload.input_tokens),
            ("output_tokens", manifest.workload.output_tokens),
        ):
            if _nested(runtime, "token_counts", token_field, "value") != workload_value:
                raise RuntimeMetadataError(
                    f"{role} source token metadata disagrees with workload"
                )
    return {
        "model_identity": dict(gpu_runtime["model_identity"]),
        "token_counts": dict(gpu_runtime["token_counts"]),
        "max_num_seqs_capacity": gpu_runtime["max_num_seqs_capacity"],
        "topology": topology_metadata(RunMode.HYBRID),
    }


def lifecycle_metadata(
    *,
    started_at_unix_ns: int,
    finished_at_unix_ns: int,
    terminal_statuses: Sequence[str],
    measured_request_count: int,
    shutdown_integrity: str,
    cleanup_status: str,
) -> dict[str, Any]:
    """Validate and normalize one terminal lifecycle summary."""

    if finished_at_unix_ns < started_at_unix_ns:
        raise RuntimeMetadataError("lifecycle finished time precedes started time")
    if not terminal_statuses or len(set(terminal_statuses)) != 1:
        raise RuntimeMetadataError("lifecycle terminal status mismatch")
    if measured_request_count < 0:
        raise RuntimeMetadataError("measured request count must be non-negative")
    return {
        "started_at_unix_ns": started_at_unix_ns,
        "finished_at_unix_ns": finished_at_unix_ns,
        "duration_ns": finished_at_unix_ns - started_at_unix_ns,
        "terminal_status": terminal_statuses[0],
        "measured_request_count": measured_request_count,
        "shutdown_integrity": shutdown_integrity,
        "cleanup_status": cleanup_status,
    }
