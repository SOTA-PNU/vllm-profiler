"""Strict standalone loaders for published Overview outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

from ..perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    verify_stored_sidecar,
)
from .loader import (
    FileIdentity,
    OverviewInputError,
    _read_json_object,
    _require_real_directory,
    _stable_regular_file,
)
from .publication import OVERVIEW_OUTPUT_ROOT_ID
from .render import (
    render_comparison_html,
    render_overview_html,
    validate_offline_html,
)
from .schema import (
    canonical_json_bytes,
    overview_comparison_from_dict,
    overview_report_from_dict,
    overview_to_dict,
)


OVERVIEW_JSON_NAME = "overview.json"
OVERVIEW_HTML_NAME = "overview.html"
OVERVIEW_VALIDATION_NAME = "overview_validation.json"
COMPARISON_JSON_NAME = "comparison.json"
COMPARISON_HTML_NAME = "comparison.html"
COMPARISON_VALIDATION_NAME = "comparison_validation.json"
_OVERVIEW_EXPECTED_FILES = frozenset(
    {
        OVERVIEW_JSON_NAME,
        OVERVIEW_HTML_NAME,
        OVERVIEW_VALIDATION_NAME,
        ARTIFACT_MANIFEST_NAME,
        ARTIFACT_VALIDATION_NAME,
    }
)
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
class OverviewBundleIdentity:
    """Exact five-file path-free identity for comparison input mutation checks."""

    files: tuple[FileIdentity, ...]
    inventory_sha256: str

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "file_count": len(self.files),
            "inventory_sha256": self.inventory_sha256,
            "files": [item.metadata for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class LoadedOverviewBundle:
    """Freshly validated Overview input used by comparison publication."""

    root: Path
    report: dict[str, Any]
    validation: dict[str, Any]
    html_validation: dict[str, Any]
    artifact_validation: dict[str, Any]
    identity: OverviewBundleIdentity

    @property
    def run_id(self) -> str:
        return str(self.report["run"]["run_id"])

    @property
    def overview_sha256(self) -> str:
        return next(
            item.sha256
            for item in self.identity.files
            if item.relative_path == OVERVIEW_JSON_NAME
        )

    @property
    def evidence(self) -> dict[str, Any]:
        validation_sha = next(
            item.sha256
            for item in self.identity.files
            if item.relative_path == OVERVIEW_VALIDATION_NAME
        )
        return {
            "run_id": self.run_id,
            "valid": True,
            "overview_sha256": self.overview_sha256,
            "overview_validation_sha256": validation_sha,
            "bundle_inventory_sha256": self.identity.inventory_sha256,
            "artifact_manifest_sha256": self.artifact_validation[
                "manifest_sha256"
            ],
            "artifact_mismatch_count": len(
                self.artifact_validation["mismatches"]
            ),
        }


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
        return tuple(
            str(item["run_id"]) for item in self.comparison["runs"]
        )


def _identity(
    root: Path,
    *,
    expected_files: frozenset[str],
) -> OverviewBundleIdentity:
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    actual = {entry.name for entry in entries}
    if actual != expected_files or len(entries) != len(expected_files):
        raise OverviewInputError(
            "published Overview directory must contain exactly five files; "
            f"missing={sorted(expected_files - actual)}, "
            f"unexpected={sorted(actual - expected_files)}"
        )
    files = tuple(
        _stable_regular_file(entry, relative_path=entry.name)
        for entry in entries
    )
    payload = json.dumps(
        [item.metadata for item in files],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return OverviewBundleIdentity(
        files=files,
        inventory_sha256=hashlib.sha256(payload).hexdigest(),
    )


def overview_bundle_identity(
    root: str | Path,
) -> OverviewBundleIdentity:
    """Snapshot an exact Overview input without trusting stored JSON."""

    directory = _require_real_directory(root, description="Overview output")
    return _identity(
        directory,
        expected_files=_OVERVIEW_EXPECTED_FILES,
    )


def _stable_text(path: Path, *, description: str) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OverviewInputError(f"{description} must be a real regular file")
    try:
        text = path.read_text(encoding="utf-8")
        after = path.lstat()
    except (OSError, UnicodeError) as error:
        raise OverviewInputError(f"{description} cannot be read") from error
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise OverviewInputError(f"{description} changed while it was read")
    return text


def _validate_semantic_sidecar(
    report: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "valid",
        "schema_validation",
        "source_reconciliation",
        "perfetto_input_identity",
        "perfetto_reconciliation",
        "phase_duration_reconciliation",
        "step_reconciliation",
        "resource_reconciliation",
        "flow_reconciliation",
        "numeric_policy",
        "html_validation",
        "publication_policy",
        "mismatches",
    }
    if set(validation) != required:
        raise OverviewInputError(
            "Overview semantic validation fields do not match contract"
        )
    if (
        validation.get("schema_version") != "1.0.0"
        or validation.get("record_type") != "overview_validation"
        or validation.get("valid") is not True
        or validation.get("mismatches") != []
        or validation.get("run_id") != report["run"]["run_id"]
    ):
        raise OverviewInputError("Overview semantic validation is not valid")
    schema = validation.get("schema_validation")
    if (
        not isinstance(schema, dict)
        or schema.get("valid") is not True
        or schema.get("schema_name") != "overview_report.schema.json"
        or schema.get("overview_sha256")
        != hashlib.sha256(
            canonical_json_bytes(overview_report_from_dict(report))
        ).hexdigest()
    ):
        raise OverviewInputError(
            "Overview semantic validation does not match overview.json"
        )
    source = validation.get("source_reconciliation")
    perfetto = validation.get("perfetto_reconciliation")
    if (
        not isinstance(source, dict)
        or source.get("valid") is not True
        or not isinstance(perfetto, dict)
        or perfetto.get("valid") is not True
        or perfetto.get("mismatches") != []
    ):
        raise OverviewInputError(
            "Overview source or Perfetto reconciliation is not valid"
        )


def load_overview_bundle(root: str | Path) -> LoadedOverviewBundle:
    """Load a standalone published Overview with fresh integrity checks."""

    directory = _require_real_directory(root, description="Overview output")
    identity_before = _identity(
        directory,
        expected_files=_OVERVIEW_EXPECTED_FILES,
    )
    report = _read_json_object(
        directory / OVERVIEW_JSON_NAME,
        description="Overview report",
    )
    model = overview_report_from_dict(report)
    if overview_to_dict(model) != report:
        raise OverviewInputError(
            "Overview report is not the canonical model representation"
        )
    validation = _read_json_object(
        directory / OVERVIEW_VALIDATION_NAME,
        description="Overview semantic validation",
    )
    _validate_semantic_sidecar(report, validation)

    html_text = _stable_text(
        directory / OVERVIEW_HTML_NAME,
        description="Overview HTML",
    )
    html_validation = validate_offline_html(html_text)
    if html_validation.get("valid") is not True:
        raise OverviewInputError("stored Overview HTML is not offline-safe")
    expected_html = render_overview_html(report)
    if html_text != expected_html:
        raise OverviewInputError(
            "stored Overview HTML differs from deterministic rendering"
        )
    if validation.get("html_validation") != html_validation:
        raise OverviewInputError(
            "stored Overview HTML validation differs from a fresh scan"
        )

    try:
        artifact_validation = verify_stored_sidecar(
            directory / ARTIFACT_MANIFEST_NAME,
            {OVERVIEW_OUTPUT_ROOT_ID: directory},
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise OverviewInputError(
            f"Overview detached artifact validation failed: {error}"
        ) from error
    if (
        artifact_validation.get("valid") is not True
        or artifact_validation.get("mismatches") != []
    ):
        raise OverviewInputError(
            "Overview detached artifact validation found mismatches"
        )
    identity_after = _identity(
        directory,
        expected_files=_OVERVIEW_EXPECTED_FILES,
    )
    if identity_after != identity_before:
        raise OverviewInputError("Overview bundle changed while it was loaded")
    return LoadedOverviewBundle(
        root=directory,
        report=report,
        validation=validation,
        html_validation=html_validation,
        artifact_validation=artifact_validation,
        identity=identity_after,
    )


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
        or validation.get("record_type")
        != "overview_comparison_validation"
        or validation.get("valid") is not True
        or validation.get("mismatches") != []
        or validation.get("comparability") != comparison["comparison"]
    ):
        raise OverviewInputError(
            "comparison semantic validation is not valid"
        )
    schema = validation.get("schema_validation")
    if (
        not isinstance(schema, dict)
        or schema.get("valid") is not True
        or schema.get("schema_name")
        != "overview_comparison.schema.json"
        or schema.get("comparison_sha256")
        != hashlib.sha256(
            canonical_json_bytes(
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
        raise OverviewInputError(
            "comparison input Overview evidence is not valid"
        )
    input_run_ids = [item.get("run_id") for item in input_overviews]
    comparison_run_ids = [
        item["run_id"] for item in comparison["runs"]
    ]
    if input_run_ids != comparison_run_ids:
        raise OverviewInputError(
            "comparison input evidence does not match comparison runs"
        )
    evidence_hashes = {
        item["run_id"]: item.get("overview_sha256")
        for item in input_overviews
    }
    comparison_hashes = {
        item["run_id"]: item.get("overview_sha256")
        for item in comparison["runs"]
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
    if overview_to_dict(model) != comparison:
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
    "LoadedOverviewBundle",
    "OVERVIEW_HTML_NAME",
    "OVERVIEW_JSON_NAME",
    "OVERVIEW_VALIDATION_NAME",
    "OverviewBundleIdentity",
    "load_overview_bundle",
    "overview_bundle_identity",
]
