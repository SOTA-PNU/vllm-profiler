"""Repository-only loader for published Overview comparison bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from perfetto_hetero_profiler.overview.bundle import (
    LoadedOverviewBundle,
    OverviewBundleIdentity,
    _identity,
    _stable_text,
)
from perfetto_hetero_profiler.overview.loader import (
    OverviewInputError,
    _read_json_object,
    _require_real_directory,
)
from perfetto_hetero_profiler.overview.publication import OVERVIEW_OUTPUT_ROOT_ID
from perfetto_hetero_profiler.overview.render import validate_offline_html
from perfetto_hetero_profiler.perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    verify_stored_sidecar,
)

from .comparison_render import render_comparison_html
from .comparison_schema import (
    canonical_comparison_json_bytes,
    comparison_to_dict,
    overview_comparison_from_dict,
)


COMPARISON_JSON_NAME = "comparison.json"
COMPARISON_HTML_NAME = "comparison.html"
COMPARISON_VALIDATION_NAME = "comparison_validation.json"
_COMPARISON_EXPECTED_FILES = frozenset(
    {
        COMPARISON_JSON_NAME,
        COMPARISON_HTML_NAME,
        COMPARISON_VALIDATION_NAME,
        ARTIFACT_MANIFEST_NAME,
        ARTIFACT_VALIDATION_NAME,
    }
)


@dataclass(frozen=True, slots=True)
class LoadedComparisonBundle:
    """Freshly validated deterministic multi-run comparison output."""

    root: Path
    comparison: dict[str, Any]
    validation: dict[str, Any]
    html_validation: dict[str, Any]
    artifact_validation: dict[str, Any]
    identity: OverviewBundleIdentity

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(item["run_id"]) for item in self.comparison["runs"])


def overview_input_evidence(item: LoadedOverviewBundle) -> dict[str, Any]:
    """Build path-free comparison evidence from a core Overview bundle."""

    overview_sha = next(
        file.sha256
        for file in item.identity.files
        if file.relative_path == "overview.json"
    )
    validation_sha = next(
        file.sha256
        for file in item.identity.files
        if file.relative_path == "overview_validation.json"
    )
    return {
        "run_id": item.run_id,
        "valid": True,
        "overview_sha256": overview_sha,
        "overview_validation_sha256": validation_sha,
        "bundle_inventory_sha256": item.identity.inventory_sha256,
        "artifact_manifest_sha256": item.artifact_validation["manifest_sha256"],
        "artifact_mismatch_count": len(item.artifact_validation["mismatches"]),
    }


def _validate_comparison_semantic_sidecar(
    comparison: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "record_type",
        "valid",
        "schema_validation",
        "input_overviews",
        "comparability",
        "html_validation",
        "publication_policy",
        "mismatches",
    }
    if set(validation) != required:
        raise OverviewInputError(
            "comparison semantic validation fields do not match contract"
        )
    if (
        validation.get("schema_version") != "1.0.0"
        or validation.get("record_type") != "overview_comparison_validation"
        or validation.get("valid") is not True
        or validation.get("mismatches") != []
        or validation.get("comparability") != comparison["comparison"]
    ):
        raise OverviewInputError("comparison semantic validation is not valid")
    schema = validation.get("schema_validation")
    if (
        not isinstance(schema, dict)
        or schema.get("valid") is not True
        or schema.get("schema_name") != "overview_comparison.schema.json"
        or schema.get("comparison_sha256")
        != hashlib.sha256(
            canonical_comparison_json_bytes(
                overview_comparison_from_dict(comparison)
            )
        ).hexdigest()
    ):
        raise OverviewInputError(
            "comparison semantic validation does not match comparison.json"
        )
    input_overviews = validation.get("input_overviews")
    if (
        not isinstance(input_overviews, list)
        or not input_overviews
        or any(
            not isinstance(item, dict)
            or item.get("valid") is not True
            or item.get("artifact_mismatch_count") != 0
            for item in input_overviews
        )
    ):
        raise OverviewInputError("comparison input Overview evidence is not valid")
    input_run_ids = [item.get("run_id") for item in input_overviews]
    comparison_run_ids = [item["run_id"] for item in comparison["runs"]]
    if input_run_ids != comparison_run_ids:
        raise OverviewInputError(
            "comparison input evidence does not match comparison runs"
        )
    evidence_hashes = {
        item["run_id"]: item.get("overview_sha256") for item in input_overviews
    }
    comparison_hashes = {
        item["run_id"]: item.get("overview_sha256") for item in comparison["runs"]
    }
    if evidence_hashes != comparison_hashes:
        raise OverviewInputError(
            "comparison input evidence hashes do not match comparison runs"
        )


def load_comparison_bundle(root: str | Path) -> LoadedComparisonBundle:
    """Load a published comparison with fresh semantic and integrity checks."""

    directory = _require_real_directory(
        root,
        description="Overview comparison output",
    )
    identity_before = _identity(
        directory,
        expected_files=_COMPARISON_EXPECTED_FILES,
    )
    comparison = _read_json_object(
        directory / COMPARISON_JSON_NAME,
        description="Overview comparison",
    )
    model = overview_comparison_from_dict(comparison)
    if comparison_to_dict(model) != comparison:
        raise OverviewInputError(
            "Overview comparison is not the canonical model representation"
        )
    validation = _read_json_object(
        directory / COMPARISON_VALIDATION_NAME,
        description="Overview comparison semantic validation",
    )
    _validate_comparison_semantic_sidecar(comparison, validation)

    html_text = _stable_text(
        directory / COMPARISON_HTML_NAME,
        description="Overview comparison HTML",
    )
    html_validation = validate_offline_html(html_text)
    if html_validation.get("valid") is not True:
        raise OverviewInputError(
            "stored Overview comparison HTML is not offline-safe"
        )
    if html_text != render_comparison_html(comparison):
        raise OverviewInputError(
            "stored comparison HTML differs from deterministic rendering"
        )
    if validation.get("html_validation") != html_validation:
        raise OverviewInputError(
            "stored comparison HTML validation differs from a fresh scan"
        )

    try:
        artifact_validation = verify_stored_sidecar(
            directory / ARTIFACT_MANIFEST_NAME,
            {OVERVIEW_OUTPUT_ROOT_ID: directory},
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise OverviewInputError(
            "Overview comparison detached artifact validation failed: "
            f"{error}"
        ) from error
    if (
        artifact_validation.get("valid") is not True
        or artifact_validation.get("mismatches") != []
    ):
        raise OverviewInputError(
            "Overview comparison detached artifact validation found mismatches"
        )
    identity_after = _identity(
        directory,
        expected_files=_COMPARISON_EXPECTED_FILES,
    )
    if identity_after != identity_before:
        raise OverviewInputError(
            "Overview comparison bundle changed while it was loaded"
        )
    return LoadedComparisonBundle(
        root=directory,
        comparison=comparison,
        validation=validation,
        html_validation=html_validation,
        artifact_validation=artifact_validation,
        identity=identity_after,
    )


__all__ = [
    "COMPARISON_HTML_NAME",
    "COMPARISON_JSON_NAME",
    "COMPARISON_VALIDATION_NAME",
    "LoadedComparisonBundle",
    "load_comparison_bundle",
    "overview_input_evidence",
]
