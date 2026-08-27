"""Repository-only semantic validation for comparison publications."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from perfetto_hetero_profiler.overview.publication import canonical_json_bytes
from perfetto_hetero_profiler.overview.validation import OverviewValidationError

from .comparison_schema import (
    canonical_comparison_json_bytes,
    overview_comparison_from_dict,
)


COMPARISON_VALIDATION_RECORD_TYPE = "overview_comparison_validation"


def build_comparison_validation(
    comparison: Mapping[str, Any],
    *,
    input_evidence: list[dict[str, Any]],
    html_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a deterministic comparison and immutable input evidence."""

    model = overview_comparison_from_dict(dict(comparison))
    model_bytes = canonical_comparison_json_bytes(model)
    mismatches: list[str] = []
    if html_validation.get("valid") is not True:
        issues = html_validation.get("issues")
        if isinstance(issues, list):
            mismatches.extend(f"HTML: {item}" for item in issues)
        else:
            mismatches.append("HTML offline validation failed")
    if not input_evidence or any(
        evidence.get("valid") is not True for evidence in input_evidence
    ):
        mismatches.append("one or more Overview inputs lack fresh integrity")
    run_ids = [evidence.get("run_id") for evidence in input_evidence]
    if run_ids != sorted(run_ids) or len(run_ids) != len(set(run_ids)):
        mismatches.append("Overview input evidence is not sorted and unique")
    comparison_runs = comparison.get("runs")
    expected_hashes = (
        {
            item.get("run_id"): item.get("overview_sha256")
            for item in comparison_runs
            if isinstance(item, Mapping)
        }
        if isinstance(comparison_runs, list)
        else {}
    )
    evidence_hashes = {
        item.get("run_id"): item.get("overview_sha256")
        for item in input_evidence
    }
    if evidence_hashes != expected_hashes:
        mismatches.append(
            "Overview input evidence hashes differ from comparison runs"
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": COMPARISON_VALIDATION_RECORD_TYPE,
        "valid": not mismatches,
        "schema_validation": {
            "valid": True,
            "schema_name": "overview_comparison.schema.json",
            "model_version": "1.0.0",
            "comparison_sha256": hashlib.sha256(model_bytes).hexdigest(),
        },
        "input_overviews": input_evidence,
        "comparability": dict(comparison["comparison"]),
        "html_validation": dict(html_validation),
        "publication_policy": {
            "semantic_payload_count": 3,
            "published_file_count": 5,
            "overwrite": False,
            "atomic_no_replace": True,
            "detached_manifest_self_reference": False,
        },
        "mismatches": sorted(set(mismatches)),
    }
    canonical_json_bytes(result)
    if result["valid"] is not True:
        raise OverviewValidationError(
            "Overview comparison validation found mismatch(es)"
        )
    return result


__all__ = [
    "COMPARISON_VALIDATION_RECORD_TYPE",
    "build_comparison_validation",
]
