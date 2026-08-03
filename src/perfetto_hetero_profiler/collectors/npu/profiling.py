"""Pure Phase 3B planning for installed RBLN profiler APIs."""

from __future__ import annotations

from dataclasses import dataclass

from ...schema.validation import validate_relative_artifact_path


@dataclass(frozen=True)
class RblnProfilePlan:
    enabled: bool
    output_directory: str
    capture_reports_api: str
    activate_profiler: bool
    start_api: str
    stop_api: str
    environment: tuple[tuple[str, str], ...]
    expected_artifact_format: str | None
    expected_artifact_extension: str | None
    execution_verified: bool
    notes: tuple[str, ...]


def build_rbln_profile_plan(
    output_directory: str = "raw/npu/rbln-profiler",
) -> RblnProfilePlan:
    validate_relative_artifact_path(output_directory, "rbln.output_directory")
    return RblnProfilePlan(
        enabled=True,
        output_directory=output_directory,
        capture_reports_api="rebel.capture_reports()",
        activate_profiler=True,
        start_api="profiler_start(path: str) (installed source symbol)",
        stop_api="profiler_done() (installed source symbol)",
        environment=(("RBLN_PROFILER", "1"),),
        expected_artifact_format=None,
        expected_artifact_extension=None,
        execution_verified=False,
        notes=(
            "capture_reports is importable and used by vllm-rbln worker code.",
            "activate_profiler and RBLN_PROFILER are present in installed runtime code.",
            "profiler_start/profiler_done delegate to TVM global functions.",
            "Runtime connection, output format, extension, and overhead need Phase 3B validation.",
        ),
    )
