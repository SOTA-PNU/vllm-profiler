"""Repository-only Overview comparison API and publication orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from perfetto_hetero_profiler.overview.bundle import (
    LoadedOverviewBundle,
    load_overview_bundle,
    overview_bundle_identity,
)
from perfetto_hetero_profiler.overview.generator import OverviewGenerationError
from perfetto_hetero_profiler.overview.publication import (
    canonical_json_bytes,
    publish_bundle,
    validate_output_path,
)
from perfetto_hetero_profiler.overview.render import validate_offline_html

from .comparison_bundle import (
    COMPARISON_HTML_NAME,
    COMPARISON_JSON_NAME,
    COMPARISON_VALIDATION_NAME,
    LoadedComparisonBundle,
    load_comparison_bundle,
    overview_input_evidence,
)
from .comparison_model import (
    Comparability,
    ComparisonDelta,
    ComparisonKpi,
    ComparisonMetadata,
    ComparisonRun,
    ComparisonValue,
    DeltaValue,
    KpiDirection,
    OverviewComparison,
)
from .comparison_render import render_comparison_html
from .comparison_schema import (
    canonical_comparison_json_bytes,
    comparison_to_dict,
    load_comparison_schema,
    overview_comparison_from_dict,
    overview_document_from_json,
    validate_comparison_delta,
    validate_comparison_kpi,
    validate_comparison_metadata,
    validate_comparison_run,
    validate_comparison_value,
    validate_overview_comparison,
)
from .comparison_validation import build_comparison_validation
from .overview_comparison import OverviewComparisonError, build_comparison


@dataclass(frozen=True, slots=True)
class OverviewComparisonConfig:
    """Immutable inputs for one repository-only multi-run comparison."""

    input_directories: tuple[Path, ...]
    output_directory: Path | None = None
    baseline_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedComparison:
    config: OverviewComparisonConfig
    inputs: tuple[LoadedOverviewBundle, ...]
    output: Path
    comparison: dict[str, Any]
    comparison_bytes: bytes
    html_text: str
    validation: dict[str, Any]


def _default_comparison_output(
    inputs: tuple[LoadedOverviewBundle, ...],
) -> Path:
    parents = {item.root.parent.resolve(strict=True) for item in inputs}
    if len(parents) != 1:
        raise OverviewGenerationError(
            "comparison inputs in different parents require --output"
        )
    parent = next(iter(parents))
    return parent / "overview-comparison"


def _prepare_comparison(
    config: OverviewComparisonConfig,
) -> _PreparedComparison:
    if not isinstance(config, OverviewComparisonConfig):
        raise TypeError("config must be OverviewComparisonConfig")
    if len(config.input_directories) < 2:
        raise OverviewGenerationError(
            "comparison requires at least two Overview inputs"
        )
    loaded_inputs = tuple(
        sorted(
            (load_overview_bundle(path) for path in config.input_directories),
            key=lambda item: item.run_id,
        )
    )
    if len({item.run_id for item in loaded_inputs}) != len(loaded_inputs):
        raise OverviewGenerationError("comparison run IDs must be unique")
    requested_output = (
        _default_comparison_output(loaded_inputs)
        if config.output_directory is None
        else Path(config.output_directory)
    )
    output = validate_output_path(
        requested_output,
        immutable_roots=[item.root for item in loaded_inputs],
    )
    comparison_plain = build_comparison(
        [item.report for item in loaded_inputs],
        baseline_run_id=config.baseline_run_id,
    )
    model = overview_comparison_from_dict(comparison_plain)
    comparison = comparison_to_dict(model)
    comparison_bytes = canonical_comparison_json_bytes(model)
    html_text = render_comparison_html(comparison)
    if not html_text.endswith("\n"):
        raise OverviewGenerationError("comparison HTML must end with one newline")
    html_validation = validate_offline_html(html_text)
    validation = build_comparison_validation(
        comparison,
        input_evidence=[overview_input_evidence(item) for item in loaded_inputs],
        html_validation=html_validation,
    )
    return _PreparedComparison(
        config=config,
        inputs=loaded_inputs,
        output=output,
        comparison=comparison,
        comparison_bytes=comparison_bytes,
        html_text=html_text,
        validation=validation,
    )


def _comparison_input_check(prepared: _PreparedComparison) -> None:
    for item in prepared.inputs:
        if overview_bundle_identity(item.root) != item.identity:
            raise OverviewGenerationError(
                f"immutable Overview input changed: {item.run_id}"
            )


def plan_overview_comparison(
    config: OverviewComparisonConfig,
) -> dict[str, Any]:
    """Build and validate a comparison without creating output files."""

    prepared = _prepare_comparison(config)
    _comparison_input_check(prepared)
    return {
        "status": "planned",
        "dry_run": True,
        "run_ids": [item.run_id for item in prepared.inputs],
        "output_directory": os.fspath(prepared.output),
        "comparability": prepared.comparison["comparison"]["comparability"],
        "baseline_run_id": prepared.comparison["comparison"]["baseline_run_id"],
        "metric_count": len(prepared.comparison["metrics"]),
        "comparison_sha256": hashlib.sha256(
            prepared.comparison_bytes
        ).hexdigest(),
        "html_sha256": hashlib.sha256(
            prepared.html_text.encode("utf-8")
        ).hexdigest(),
        "validation_valid": prepared.validation["valid"],
        "overwrite": False,
        "hardware_execution": False,
    }


def compare_overviews(
    config: OverviewComparisonConfig,
) -> dict[str, Any]:
    """Generate and atomically publish one deterministic comparison."""

    prepared = _prepare_comparison(config)

    def validate_staging(root: Path) -> None:
        staged = load_comparison_bundle(root)
        if staged.comparison != prepared.comparison:
            raise OverviewGenerationError(
                "staged comparison differs from the validated report"
            )
        if staged.validation != prepared.validation:
            raise OverviewGenerationError(
                "staged comparison validation differs from expected evidence"
            )

    publication = publish_bundle(
        prepared.output,
        payloads={
            COMPARISON_JSON_NAME: prepared.comparison_bytes,
            COMPARISON_HTML_NAME: prepared.html_text.encode("utf-8"),
            COMPARISON_VALIDATION_NAME: canonical_json_bytes(
                prepared.validation
            ),
        },
        validate_staging=validate_staging,
        before_publish=lambda: _comparison_input_check(prepared),
    )
    published: LoadedComparisonBundle = load_comparison_bundle(prepared.output)
    if published.comparison != prepared.comparison:
        raise OverviewGenerationError(
            "published comparison differs from the validated report"
        )
    return {
        "status": "succeeded",
        "dry_run": False,
        "run_ids": list(published.run_ids),
        "output_directory": os.fspath(prepared.output),
        "comparability": prepared.comparison["comparison"]["comparability"],
        "baseline_run_id": prepared.comparison["comparison"]["baseline_run_id"],
        "comparison": {
            "relative_path": COMPARISON_JSON_NAME,
            "size_bytes": len(prepared.comparison_bytes),
            "sha256": hashlib.sha256(prepared.comparison_bytes).hexdigest(),
        },
        "html": {
            "relative_path": COMPARISON_HTML_NAME,
            "size_bytes": len(prepared.html_text.encode("utf-8")),
            "sha256": hashlib.sha256(
                prepared.html_text.encode("utf-8")
            ).hexdigest(),
        },
        "validation": {
            "valid": prepared.validation["valid"],
            "mismatches": prepared.validation["mismatches"],
        },
        "artifact_validation": publication["artifact_validation"],
        "files": publication["files"],
        "hardware_execution": False,
    }


__all__ = [
    "COMPARISON_HTML_NAME",
    "COMPARISON_JSON_NAME",
    "COMPARISON_VALIDATION_NAME",
    "Comparability",
    "ComparisonDelta",
    "ComparisonKpi",
    "ComparisonMetadata",
    "ComparisonRun",
    "ComparisonValue",
    "DeltaValue",
    "KpiDirection",
    "LoadedComparisonBundle",
    "OverviewComparison",
    "OverviewComparisonConfig",
    "OverviewComparisonError",
    "build_comparison",
    "build_comparison_validation",
    "canonical_comparison_json_bytes",
    "comparison_to_dict",
    "compare_overviews",
    "load_comparison_bundle",
    "load_comparison_schema",
    "overview_comparison_from_dict",
    "overview_document_from_json",
    "overview_input_evidence",
    "plan_overview_comparison",
    "render_comparison_html",
    "validate_comparison_delta",
    "validate_comparison_kpi",
    "validate_comparison_metadata",
    "validate_comparison_run",
    "validate_comparison_value",
    "validate_overview_comparison",
]
