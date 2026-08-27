"""Product orchestration for deterministic single-run Overview generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

from ..perfetto.loader import load_hybrid_run
from ..perfetto.tooling import (
    TRACE_PROCESSOR_FILENAME,
    TRACE_PROCESSOR_RELEASE,
    resolve_toolchain,
)
from .bundle import (
    load_overview_bundle,
)
from .loader import (
    FileIdentity,
    LoadedPerfettoBundle,
    _require_real_directory,
    _stable_regular_file,
    load_matching_perfetto,
    normalized_identity,
    perfetto_identity,
)
from .publication import (
    canonical_json_bytes,
    publish_bundle,
    validate_output_path,
)
from .render import render_overview_html, validate_offline_html
from .report import build_overview_report
from .schema import (
    canonical_json_bytes as canonical_model_json_bytes,
    overview_report_from_dict,
    overview_to_dict,
)
from .validation import build_overview_validation


OVERVIEW_JSON_NAME = "overview.json"
OVERVIEW_HTML_NAME = "overview.html"
OVERVIEW_VALIDATION_NAME = "overview_validation.json"


class OverviewGenerationError(RuntimeError):
    """An Overview product command could not satisfy the full contract."""


@dataclass(frozen=True, slots=True)
class OverviewGenerationConfig:
    """Immutable inputs for one single-run report."""

    run_directory: Path
    perfetto_directory: Path
    output_directory: Path | None = None
    trace_processor_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _PreparedGeneration:
    config: OverviewGenerationConfig
    loaded: Any
    perfetto: LoadedPerfettoBundle
    toolchain_identity: FileIdentity
    output: Path
    report: dict[str, Any]
    report_bytes: bytes
    html_text: str
    validation: dict[str, Any]


def _local_trace_processor_path(
    configured: Path | None,
) -> Path:
    """Select an existing local binary without invoking a downloader/cache."""

    if configured is not None:
        return Path(configured)
    candidate = (
        Path(sys.prefix)
        / "bin"
        / f"{TRACE_PROCESSOR_FILENAME}-{TRACE_PROCESSOR_RELEASE}"
    )
    if not candidate.is_file():
        raise OverviewGenerationError(
            "a local pinned Trace Processor was not found; provide "
            "--trace-processor explicitly (Overview never downloads tools)"
        )
    return candidate


def _prepare_generation(
    config: OverviewGenerationConfig,
) -> _PreparedGeneration:
    if not isinstance(config, OverviewGenerationConfig):
        raise TypeError("config must be OverviewGenerationConfig")
    run_directory = _require_real_directory(
        config.run_directory,
        description="normalized run",
    )
    loaded = load_hybrid_run(run_directory)
    perfetto = load_matching_perfetto(
        loaded,
        config.perfetto_directory,
        trace_processor_path=_local_trace_processor_path(
            config.trace_processor_path
        ),
    )
    toolchain_identity = _stable_regular_file(
        perfetto.toolchain.binary_path,
        relative_path=perfetto.toolchain.binary_path.name,
    )
    requested_output = (
        loaded.root.with_name(f"{loaded.manifest.run_id}-overview")
        if config.output_directory is None
        else Path(config.output_directory)
    )
    immutable_roots = [
        *(item.root for item in loaded.root_fingerprints),
        perfetto.root,
    ]
    output = validate_output_path(
        requested_output,
        immutable_roots=immutable_roots,
    )

    report_plain = build_overview_report(loaded, perfetto)
    model = overview_report_from_dict(report_plain)
    report = overview_to_dict(model)
    report_bytes = canonical_model_json_bytes(model)
    html_text = render_overview_html(report)
    if not html_text.endswith("\n"):
        raise OverviewGenerationError("Overview HTML must end with one newline")
    html_validation = validate_offline_html(html_text)
    validation = build_overview_validation(
        report,
        loaded=loaded,
        perfetto=perfetto,
        html_validation=html_validation,
    )
    return _PreparedGeneration(
        config=config,
        loaded=loaded,
        perfetto=perfetto,
        toolchain_identity=toolchain_identity,
        output=output,
        report=report,
        report_bytes=report_bytes,
        html_text=html_text,
        validation=validation,
    )


def _generation_input_check(prepared: _PreparedGeneration) -> None:
    after = load_hybrid_run(prepared.loaded.root)
    if normalized_identity(after) != normalized_identity(prepared.loaded):
        raise OverviewGenerationError(
            "immutable normalized input changed during Overview generation"
        )
    if perfetto_identity(prepared.perfetto.root) != prepared.perfetto.identity:
        raise OverviewGenerationError(
            "immutable Perfetto input changed during Overview generation"
        )
    current_toolchain = resolve_toolchain(prepared.perfetto.toolchain.binary_path)
    if current_toolchain != prepared.perfetto.toolchain:
        raise OverviewGenerationError(
            "pinned Trace Processor changed during Overview generation"
        )
    current_identity = _stable_regular_file(
        prepared.perfetto.toolchain.binary_path,
        relative_path=prepared.perfetto.toolchain.binary_path.name,
    )
    if current_identity != prepared.toolchain_identity:
        raise OverviewGenerationError(
            "pinned Trace Processor file identity changed during "
            "Overview generation"
        )


def plan_overview_generation(
    config: OverviewGenerationConfig,
) -> dict[str, Any]:
    """Perform all read-only calculation and validation without creating files."""

    prepared = _prepare_generation(config)
    _generation_input_check(prepared)
    kpis = prepared.report["kpis"]
    return {
        "status": "planned",
        "dry_run": True,
        "run_id": prepared.loaded.manifest.run_id,
        "output_directory": os.fspath(prepared.output),
        "perfetto_trace": dict(prepared.report["perfetto"]["trace"]),
        "kpi_counts": {
            name: len(values) for name, values in sorted(kpis.items())
        },
        "resource_stream_count": len(prepared.report["resources"]),
        "overview_sha256": hashlib.sha256(prepared.report_bytes).hexdigest(),
        "html_sha256": hashlib.sha256(
            prepared.html_text.encode("utf-8")
        ).hexdigest(),
        "validation_valid": prepared.validation["valid"],
        "overwrite": False,
        "hardware_execution": False,
    }


def generate_overview(
    config: OverviewGenerationConfig,
) -> dict[str, Any]:
    """Generate, validate, inventory, and atomically publish one Overview."""

    prepared = _prepare_generation(config)
    def validate_staging(root: Path) -> None:
        staged = load_overview_bundle(root)
        if staged.report != prepared.report:
            raise OverviewGenerationError(
                "staged Overview differs from the validated report"
            )
        if staged.validation != prepared.validation:
            raise OverviewGenerationError(
                "staged Overview validation differs from expected evidence"
            )

    publication = publish_bundle(
        prepared.output,
        payloads={
            OVERVIEW_JSON_NAME: prepared.report_bytes,
            OVERVIEW_HTML_NAME: prepared.html_text.encode("utf-8"),
            OVERVIEW_VALIDATION_NAME: canonical_json_bytes(
                prepared.validation
            ),
        },
        validate_staging=validate_staging,
        before_publish=lambda: _generation_input_check(prepared),
    )
    published = load_overview_bundle(prepared.output)
    if published.report != prepared.report:
        raise OverviewGenerationError(
            "published Overview differs from the validated report"
        )
    return {
        "status": "succeeded",
        "dry_run": False,
        "run_id": published.run_id,
        "output_directory": os.fspath(prepared.output),
        "overview": {
            "relative_path": OVERVIEW_JSON_NAME,
            "size_bytes": len(prepared.report_bytes),
            "sha256": hashlib.sha256(prepared.report_bytes).hexdigest(),
        },
        "html": {
            "relative_path": OVERVIEW_HTML_NAME,
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
    "OVERVIEW_HTML_NAME",
    "OVERVIEW_JSON_NAME",
    "OVERVIEW_VALIDATION_NAME",
    "OverviewGenerationConfig",
    "OverviewGenerationError",
    "generate_overview",
    "plan_overview_generation",
]
