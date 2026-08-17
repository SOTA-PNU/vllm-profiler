"""Assembly of a path-free single-run Overview report."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..hybrid.join import validate_marker_groups
from ..perfetto.loader import LoadedHybridRun
from ..schema import Availability
from .calculation import calculate_overview_kpis
from .loader import (
    LoadedPerfettoBundle,
    OverviewInputError,
    read_validated_source_json,
    reconciliation_summary,
)


class OverviewReportError(RuntimeError):
    """A validated source could not be represented without inventing data."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _canonical_key(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_model_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise OverviewReportError("model identity must be a non-empty string")
    if Path(value).is_absolute() or "/" in value or "\\" in value:
        result = Path(value.replace("\\", "/")).name
    else:
        result = value
    if not result or result in {".", ".."}:
        raise OverviewReportError("model identity cannot be safely redacted")
    return result


def _sort_sources(kpi: dict[str, Any]) -> None:
    sources = kpi.get("sources")
    if not isinstance(sources, list):
        raise OverviewReportError("KPI sources must be an array")
    for source in sources:
        if not isinstance(source, dict):
            raise OverviewReportError("KPI source must be an object")
        for name in ("record_ids", "metric_names"):
            values = source.get(name)
            if isinstance(values, list):
                source[name] = sorted(set(values))
    kpi["sources"] = sorted(sources, key=_canonical_key)
    warnings = kpi.get("quality_warnings")
    if isinstance(warnings, list):
        kpi["quality_warnings"] = sorted(set(warnings))


def _canonicalize_calculation(
    calculated: dict[str, object],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = {}
    for section_name in (
        "request_facing_latency",
        "pipeline_latency",
        "throughput_and_tokens",
        "transfer",
    ):
        raw = calculated.get(section_name)
        if not isinstance(raw, list):
            raise OverviewReportError(f"{section_name} must be a KPI array")
        values = [dict(item) for item in raw]
        for item in values:
            _sort_sources(item)
        sections[section_name] = sorted(
            values,
            key=lambda item: (
                str(item.get("name")),
                _canonical_key(item.get("scope")),
                str(item.get("aggregation_method")),
            ),
        )

    raw_resources = calculated.get("resource_summaries")
    if not isinstance(raw_resources, list):
        raise OverviewReportError("resource_summaries must be an array")
    resources = [dict(item) for item in raw_resources]
    for resource in resources:
        aggregates = resource.get("aggregates")
        if not isinstance(aggregates, list):
            raise OverviewReportError("resource aggregates must be an array")
        canonical_aggregates = [dict(item) for item in aggregates]
        for item in canonical_aggregates:
            _sort_sources(item)
        resource["aggregates"] = sorted(
            canonical_aggregates,
            key=lambda item: (
                str(item.get("aggregation_method")),
                str(item.get("name")),
            ),
        )
        warnings = resource.get("quality_warnings")
        if isinstance(warnings, list):
            resource["quality_warnings"] = sorted(set(warnings))
    resources.sort(
        key=lambda item: (
            str(item.get("metric_name")),
            _canonical_key(item.get("scope")),
        )
    )
    return sections, resources


def _kpi_value(
    sections: Mapping[str, Sequence[Mapping[str, Any]]],
    name: str,
) -> int | float | None:
    matches = [
        item
        for values in sections.values()
        for item in values
        if item.get("name") == name
        and item.get("scope", {}).get("observation_layer")
        == "request_facing_client"
    ]
    if len(matches) != 1:
        return None
    value = matches[0].get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _command_option(command: object, name: str) -> int | None:
    if not isinstance(command, list) or any(
        not isinstance(item, str) for item in command
    ):
        return None
    values = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == name
    ]
    if not values:
        return None
    if len(values) != 1:
        raise OverviewReportError(f"command has duplicate {name}")
    try:
        value = int(values[0])
    except ValueError as error:
        raise OverviewReportError(f"command {name} is not an integer") from error
    if value < 0:
        raise OverviewReportError(f"command {name} must be non-negative")
    return value


def _equal_command_option(
    provenance: Mapping[str, Any],
    name: str,
) -> int | None:
    values = [
        _command_option(provenance.get(key), name)
        for key in ("prefill_command", "decode_command")
    ]
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    if len(concrete) != 2 or concrete[0] != concrete[1]:
        raise OverviewReportError(
            f"prefill/decode command {name} does not match"
        )
    return concrete[0]


def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OverviewReportError(f"{field} must be a non-negative integer")
    return value


def _optional_finite_number(
    value: object,
    *,
    field: str,
) -> int | float | None:
    import math

    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise OverviewReportError(f"{field} must be a finite non-negative number")
    return value


def _optional_sha(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OverviewReportError(f"{field} must be a lowercase SHA-256")
    return value


def _workload(
    loaded: LoadedHybridRun,
    sections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    try:
        provenance = read_validated_source_json(
            loaded,
            root_id="coordinator",
            relative_path="provenance.json",
        )
    except OverviewInputError:
        provenance = {}
    raw_workload = provenance.get("workload")
    if not isinstance(raw_workload, dict):
        raw_workload = {}
    measured = _optional_nonnegative_int(
        raw_workload.get("measured"),
        field="workload.measured",
    )
    count = _kpi_value(sections, "request.count")
    request_count = (
        int(count)
        if isinstance(count, int) and not isinstance(count, bool)
        else loaded.manifest.workload.request_count
    )
    if measured is not None and request_count is not None and measured != request_count:
        raise OverviewReportError(
            "coordinator measured count disagrees with normalized request.count"
        )
    input_tokens = _kpi_value(sections, "request.input_tokens")
    output_tokens = _kpi_value(sections, "request.output_tokens")
    total_tokens = _kpi_value(sections, "request.total_tokens")
    return {
        "request_count": request_count,
        "input_tokens": int(input_tokens) if isinstance(input_tokens, int) else None,
        "output_tokens": (
            int(output_tokens) if isinstance(output_tokens, int) else None
        ),
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
        "concurrency": loaded.manifest.workload.concurrency,
        "request_rate_per_s": loaded.manifest.workload.request_rate_per_s,
        "warmup_requests": _optional_nonnegative_int(
            raw_workload.get("warmup", loaded.manifest.workload.warmup_requests),
            field="workload.warmup",
        ),
        "max_output_tokens": _optional_nonnegative_int(
            raw_workload.get("max_tokens"),
            field="workload.max_tokens",
        ),
        "temperature": _optional_finite_number(
            raw_workload.get("temperature"),
            field="workload.temperature",
        ),
        "retry_count": _optional_nonnegative_int(
            raw_workload.get("retry"),
            field="workload.retry",
        ),
        "prompt_sha256": _optional_sha(
            raw_workload.get("prompt_sha256"),
            field="workload.prompt_sha256",
        ),
        "request_body_sha256": _optional_sha(
            raw_workload.get("request_body_sha256"),
            field="workload.request_body_sha256",
        ),
        "offline": (
            provenance.get("offline")
            if isinstance(provenance.get("offline"), bool)
            else None
        ),
        "max_model_len": (
            _equal_command_option(provenance, "--max-model-len")
            or loaded.manifest.workload.max_model_len
        ),
        "block_size": _equal_command_option(provenance, "--block-size"),
    }


def _models(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    values = [
        {
            "role": model.role,
            "model_id": _safe_model_id(model.model_id),
            "revision": model.revision,
            "dtype": model.dtype,
        }
        for model in loaded.manifest.models
    ]
    return sorted(values, key=_canonical_key)


def _hardware(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    values = [
        {
            "device_type": device.device_type.value,
            "device_id": device.device_id,
            "vendor": device.vendor,
            "model": device.model,
            "memory_total_bytes": device.memory_total_bytes,
        }
        for device in loaded.manifest.devices
    ]
    return sorted(values, key=_canonical_key)


def _metric_integer(
    loaded: LoadedHybridRun,
    name: str,
    *,
    default: int = 0,
) -> int:
    matches = [metric for metric in loaded.metrics if metric.metric_name == name]
    if len(matches) != 1:
        return default
    metric = matches[0]
    if (
        metric.availability is not Availability.AVAILABLE
        or isinstance(metric.value, bool)
        or not isinstance(metric.value, int)
    ):
        return default
    return metric.value


def _data_quality(
    loaded: LoadedHybridRun,
    perfetto: LoadedPerfettoBundle,
) -> dict[str, Any]:
    marker = validate_marker_groups(loaded.events)
    resource_metrics = [
        metric
        for metric in loaded.metrics
        if metric.metric_name.startswith("resource.")
    ]
    available_resources = sum(
        metric.availability is Availability.AVAILABLE
        for metric in resource_metrics
    )
    pipeline_e2e = [
        metric
        for metric in loaded.metrics
        if metric.metric_name == "latency.e2e"
        and metric.dimensions.get("hybrid.join_method") == "correlation_id"
    ]
    join_methods = sorted(
        {
            str(metric.dimensions["hybrid.join_method"])
            for metric in pipeline_e2e
        }
    )
    join_method = join_methods[0] if len(join_methods) == 1 else "not_available"
    attributes = loaded.manifest.attributes
    profiler_kind = attributes.get("hybrid.phase4b2b_profile_kind", "unknown")
    native_alignment = attributes.get(
        "hybrid.profiler_alignment_status",
        "not_available",
    )
    if not isinstance(profiler_kind, str) or not profiler_kind:
        raise OverviewReportError("profiler kind is invalid")
    if not isinstance(native_alignment, str) or not native_alignment:
        raise OverviewReportError("native profiler alignment status is invalid")
    offset = attributes.get("hybrid.alignment_offset_ns")
    uncertainty = attributes.get("hybrid.alignment_uncertainty_ns")
    if isinstance(offset, bool) or not isinstance(offset, int):
        offset = None
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, int)
        or uncertainty < 0
    ):
        uncertainty = None
    alignment_method = loaded.manifest.configuration.get("alignment_method")
    if not isinstance(alignment_method, str) or not alignment_method:
        alignment_method = None
    alignment_status = (
        "aligned"
        if alignment_method is not None
        and offset is not None
        and uncertainty is not None
        else "not_available"
    )
    roots = [
        item.metadata
        for item in sorted(
            loaded.root_fingerprints,
            key=lambda fingerprint: fingerprint.root_id,
        )
    ]
    has_rbln = any(
        envelope.profiler_type == "npu_rbln"
        for envelope in loaded.native_envelopes
    )
    limitations = {
        "one measured request cannot support a general performance conclusion",
        "capture modes are diagnostic observations rather than repeated benchmarks",
    }
    if loaded.native_envelopes:
        limitations.add(
            "native profiler events remain partial and unaligned internally"
        )
    if has_rbln:
        limitations.add(
            "RBLN Perfetto events require a separate native-relative timeline "
            "until a canonical clock anchor exists"
        )
    return {
        "run_status": loaded.manifest.status.value,
        "canonical_marker_count": len(loaded.events),
        "marker_validation": {
            "status": marker.status,
            "missing_count": len(marker.missing_markers),
            "duplicate_count": len(marker.duplicate_markers),
            "pairing_violation_count": len(marker.pairing_issues),
            "order_violation_count": len(marker.ordering_issues),
        },
        "request_join": {
            "joined_count": _metric_integer(loaded, "hybrid.joined_requests"),
            "unjoined_count": _metric_integer(
                loaded,
                "hybrid.unjoined_requests",
            ),
            "method": join_method,
        },
        "alignment": {
            "status": alignment_status,
            "method": alignment_method,
            "offset_ns": offset,
            "uncertainty_ns": uncertainty,
        },
        "resource_samples": {
            "total": len(resource_metrics),
            "available": available_resources,
            "unavailable": len(resource_metrics) - available_resources,
        },
        "profiler": {
            "kind": profiler_kind,
            "native_alignment_status": native_alignment,
        },
        "source_artifact_validation": {
            "valid": True,
            "closeout_artifact_count": loaded.closeout_artifact_count,
            "closeout_manifest_sha256": loaded.closeout_manifest_sha256,
            "roots": roots,
        },
        "perfetto_sql_validation": {
            "valid": perfetto.fresh_trace_validation["valid"],
            "query_count": len(perfetto.fresh_trace_validation["queries"]),
            "mismatches": list(perfetto.fresh_trace_validation["mismatches"]),
        },
        "trace_sha256": perfetto.fresh_trace_validation["trace"]["sha256"],
        "per_sample_stream_preserved": (
            attributes.get("hybrid.per_sample_stream_preserved") is True
        ),
        "cleanup_complete": attributes.get("hybrid.cleanup_complete") is True,
        "rbln_pb_policy": {
            "classification": (
                "perfetto_compatible_rbln_trace"
                if has_rbln
                else "not_present"
            ),
            "structure_analysis": (
                "deferred_to_perfetto_conversion"
                if has_rbln
                else "not_applicable"
            ),
            "raw_bytes_embedded": False,
        },
        "sample_limitations": sorted(limitations),
    }


def _native_profiles(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for envelope in loaded.native_envelopes:
        value = asdict(envelope)
        if envelope.profiler_type == "npu_rbln":
            value["opaque_rbln_pb"] = False
        value["native_event_alignment"] = "unaligned"
        value["structure_analysis"] = (
            "deferred_to_perfetto_conversion"
            if envelope.profiler_type == "npu_rbln"
            else "not_applicable"
        )
        values.append(value)
    return sorted(values, key=_canonical_key)


def _interpretation() -> dict[str, Any]:
    return {
        "comparison_scope": "single_request_diagnostic_capture",
        "benchmark_claim_allowed": False,
        "limitations": sorted(
            {
                "one request per capture is not a statistical benchmark",
                "profiler capture kinds may impose different overhead",
                "native profiler clocks are not inferred beyond host API envelopes",
                "no hardware winner or general performance ranking is supported",
                (
                    "this external KPI report is not the Perfetto UI; the "
                    "matching trace.pftrace contains a separate timeline "
                    "Heterogeneous LLM Processing, not the built-in Overview"
                ),
            }
        ),
        "policies": {
            "request_observation_layers_separate": True,
            "timestamp_proximity_join": False,
            "unavailable_zero_fill": False,
            "native_clock_inference": False,
            "rbln_pb_parsing": False,
            "resource_device_aggregation": False,
        },
    }


def build_overview_report(
    loaded: LoadedHybridRun,
    perfetto: LoadedPerfettoBundle,
) -> dict[str, Any]:
    """Build a deterministic plain-dict report from validated immutable inputs."""

    if loaded.manifest.run_id != perfetto.conversion_manifest["run_id"]:
        raise OverviewReportError("normalized and Perfetto run IDs differ")
    calculated = calculate_overview_kpis(loaded)
    sections, resources = _canonicalize_calculation(calculated)
    attributes = loaded.manifest.attributes
    profiler_kind = attributes.get("hybrid.phase4b2b_profile_kind")
    if not isinstance(profiler_kind, str) or not profiler_kind:
        profiler_kind = "unknown"
    return {
        "schema_version": "1.0.0",
        "record_type": "overview_report",
        "run": {
            "run_id": loaded.manifest.run_id,
            "mode": loaded.manifest.mode.value,
            "profile_mode": loaded.manifest.profile_mode.value,
            "status": loaded.manifest.status.value,
            "profiler_kind": profiler_kind,
            "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
        },
        "workload": _workload(loaded, sections),
        "models": _models(loaded),
        "hardware": _hardware(loaded),
        "kpis": sections,
        "resources": resources,
        "data_quality": _data_quality(loaded, perfetto),
        "perfetto": reconciliation_summary(perfetto),
        "native_profiles": _native_profiles(loaded),
        "interpretation": _interpretation(),
    }


__all__ = [
    "OverviewReportError",
    "build_overview_report",
]
