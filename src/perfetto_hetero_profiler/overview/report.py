"""Assembly of a path-free single-run Overview report.

One job: turn validated immutable inputs into the exact published document —
fail-closed coercion of coordinator provenance, the deterministic ordering that
makes ``overview.json`` byte-stable, the run inventory, and the data-quality
block stating what the capture proves and what it does not.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..hybrid.join import validate_marker_groups
from ..perfetto.compatibility import LEGACY_PROFILE_KIND_ATTRIBUTE
from ..perfetto.loader import LoadedHybridRun
from ..schema import Availability
from .calculation import calculate_overview_kpis
from .loader import (
    LoadedPerfettoBundle,
    OverviewInputError,
    normalized_input_metadata,
    read_validated_source_json,
    reconciliation_summary,
)


OVERVIEW_SCHEMA_VERSION = "1.0.0"
OVERVIEW_RECORD_TYPE = "overview_report"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RBLN_PROFILER = "npu_rbln"
_SHARED_COMMAND_KEYS = ("prefill_command", "decode_command")
_KPI_SECTION_NAMES = (
    "request_facing_latency",
    "pipeline_latency",
    "throughput_and_tokens",
    "transfer",
)
_BASE_LIMITATIONS = {
    "one measured request cannot support a general performance conclusion",
    "capture modes are diagnostic observations rather than repeated benchmarks",
}
_NATIVE_LIMITATION = (
    "native profiler events remain partial and unaligned internally"
)
_RBLN_LIMITATION = (
    "RBLN Perfetto events require a separate native-relative timeline "
    "until a canonical clock anchor exists"
)


class OverviewReportError(RuntimeError):
    """A validated source could not be represented without inventing data."""


def _canonical_key(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")



def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OverviewReportError(f"{field} must be a non-negative integer")
    return value


def _optional_finite_number(value: object, *, field: str) -> int | float | None:
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


def _ordered_copies(raw: object, *, error: str, key) -> list[dict[str, Any]]:
    """Copy a KPI-like array, canonicalize each entry's sources, then sort."""

    if not isinstance(raw, list):
        raise OverviewReportError(error)
    values = [dict(item) for item in raw]
    for item in values:
        _sort_sources(item)
    return sorted(values, key=key)


def canonicalize_calculation(
    calculated: dict[str, object],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return the KPI sections and resource streams in publication order."""

    sections = {
        name: _ordered_copies(
            calculated.get(name),
            error=f"{name} must be a KPI array",
            key=lambda item: (
                str(item.get("name")),
                _canonical_key(item.get("scope")),
                str(item.get("aggregation_method")),
            ),
        )
        for name in _KPI_SECTION_NAMES
    }
    raw_resources = calculated.get("resource_summaries")
    if not isinstance(raw_resources, list):
        raise OverviewReportError("resource_summaries must be an array")
    resources = [dict(item) for item in raw_resources]
    for resource in resources:
        resource["aggregates"] = _ordered_copies(
            resource.get("aggregates"),
            error="resource aggregates must be an array",
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


def _equal_command_option(provenance: Mapping[str, Any], name: str) -> int | None:
    concrete = [
        value
        for value in (
            _command_option(provenance.get(key), name)
            for key in _SHARED_COMMAND_KEYS
        )
        if value is not None
    ]
    if not concrete:
        return None
    if len(concrete) != 2 or concrete[0] != concrete[1]:
        raise OverviewReportError(f"prefill/decode command {name} does not match")
    return concrete[0]


def _build_workload(
    loaded: LoadedHybridRun,
    sections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reconcile coordinator provenance with the normalized request KPIs."""

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
    tokens = {
        field: _kpi_value(sections, f"request.{field}")
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }
    return {
        "request_count": request_count,
        **{
            field: int(value) if isinstance(value, int) else None
            for field, value in tokens.items()
        },
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



def _safe_model_id(value: str) -> str:
    """Publish only the final path component; the report carries no host path."""

    if not isinstance(value, str) or not value:
        raise OverviewReportError("model identity must be a non-empty string")
    if Path(value).is_absolute() or "/" in value or "\\" in value:
        result = Path(value.replace("\\", "/")).name
    else:
        result = value
    if not result or result in {".", ".."}:
        raise OverviewReportError("model identity cannot be safely redacted")
    return result


def _build_models(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "role": model.role,
                "model_id": _safe_model_id(model.model_id),
                "revision": model.revision,
                "dtype": model.dtype,
            }
            for model in loaded.manifest.models
        ),
        key=_canonical_key,
    )


def _build_hardware(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "device_type": device.device_type.value,
                "device_id": device.device_id,
                "vendor": device.vendor,
                "model": device.model,
                "memory_total_bytes": device.memory_total_bytes,
            }
            for device in loaded.manifest.devices
        ),
        key=_canonical_key,
    )


def _build_native_profiles(loaded: LoadedHybridRun) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for envelope in loaded.native_envelopes:
        rbln = envelope.profiler_type == _RBLN_PROFILER
        value = asdict(envelope)
        if rbln:
            value["opaque_rbln_pb"] = False
        value["native_event_alignment"] = "unaligned"
        value["structure_analysis"] = (
            "deferred_to_perfetto_conversion" if rbln else "not_applicable"
        )
        values.append(value)
    return sorted(values, key=_canonical_key)



def _metric_integer(loaded: LoadedHybridRun, name: str, *, default: int = 0) -> int:
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


def _build_data_quality(
    loaded: LoadedHybridRun,
    perfetto: LoadedPerfettoBundle,
) -> dict[str, Any]:
    """State what the capture proves, and what it explicitly does not."""

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
    attributes = loaded.manifest.attributes
    profiler_kind = attributes.get(LEGACY_PROFILE_KIND_ATTRIBUTE, "unknown")
    native_alignment = attributes.get(
        "hybrid.profiler_alignment_status",
        "not_available",
    )
    if not isinstance(profiler_kind, str) or not profiler_kind:
        raise OverviewReportError("profiler kind is invalid")
    if not isinstance(native_alignment, str) or not native_alignment:
        raise OverviewReportError("native profiler alignment status is invalid")
    has_rbln = any(
        envelope.profiler_type == _RBLN_PROFILER
        for envelope in loaded.native_envelopes
    )
    join_methods = sorted(
        {
            str(metric.dimensions["hybrid.join_method"])
            for metric in loaded.metrics
            if metric.metric_name == "latency.e2e"
            and metric.dimensions.get("hybrid.join_method") == "correlation_id"
        }
    )
    # Report ``aligned`` only with method, offset and uncertainty all present.
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
    method = loaded.manifest.configuration.get("alignment_method")
    if not isinstance(method, str) or not method:
        method = None
    alignment = {
        "status": (
            "aligned"
            if method is not None and offset is not None and uncertainty is not None
            else "not_available"
        ),
        "method": method,
        "offset_ns": offset,
        "uncertainty_ns": uncertainty,
    }
    limitations = set(_BASE_LIMITATIONS)
    if loaded.native_envelopes:
        limitations.add(_NATIVE_LIMITATION)
    if has_rbln:
        limitations.add(_RBLN_LIMITATION)
    fresh = perfetto.fresh_trace_validation
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
            "unjoined_count": _metric_integer(loaded, "hybrid.unjoined_requests"),
            "method": (
                join_methods[0] if len(join_methods) == 1 else "not_available"
            ),
        },
        "alignment": alignment,
        "resource_samples": {
            "total": len(resource_metrics),
            "available": available_resources,
            "unavailable": len(resource_metrics) - available_resources,
        },
        "profiler": {
            "kind": profiler_kind,
            "native_alignment_status": native_alignment,
        },
        "source_artifact_validation": normalized_input_metadata(loaded),
        "perfetto_sql_validation": {
            "valid": fresh["valid"],
            "query_count": len(fresh["queries"]),
            "mismatches": list(fresh["mismatches"]),
        },
        "trace_sha256": fresh["trace"]["sha256"],
        "per_sample_stream_preserved": (
            attributes.get("hybrid.per_sample_stream_preserved") is True
        ),
        "cleanup_complete": attributes.get("hybrid.cleanup_complete") is True,
        "rbln_pb_policy": {
            "classification": (
                "perfetto_compatible_rbln_trace" if has_rbln else "not_present"
            ),
            "structure_analysis": (
                "deferred_to_perfetto_conversion" if has_rbln else "not_applicable"
            ),
            "raw_bytes_embedded": False,
        },
        "sample_limitations": sorted(limitations),
    }


def _interpretation() -> dict[str, Any]:
    """Return the fixed interpretation cautions this product always states."""

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
    sections, resources = canonicalize_calculation(calculate_overview_kpis(loaded))
    profiler_kind = loaded.manifest.attributes.get(LEGACY_PROFILE_KIND_ATTRIBUTE)
    if not isinstance(profiler_kind, str) or not profiler_kind:
        profiler_kind = "unknown"
    return {
        "schema_version": OVERVIEW_SCHEMA_VERSION,
        "record_type": OVERVIEW_RECORD_TYPE,
        "run": {
            "run_id": loaded.manifest.run_id,
            "mode": loaded.manifest.mode.value,
            "profile_mode": loaded.manifest.profile_mode.value,
            "status": loaded.manifest.status.value,
            "profiler_kind": profiler_kind,
            "canonical_clock_domain_id": loaded.canonical_clock_domain_id,
        },
        "workload": _build_workload(loaded, sections),
        "models": _build_models(loaded),
        "hardware": _build_hardware(loaded),
        "kpis": sections,
        "resources": resources,
        "data_quality": _build_data_quality(loaded, perfetto),
        "perfetto": reconciliation_summary(perfetto),
        "native_profiles": _build_native_profiles(loaded),
        "interpretation": _interpretation(),
    }


__all__ = [
    "OverviewReportError",
    "build_overview_report",
]
