"""Pure detailed-profile command planning; no profiler execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from ...schema.validation import validate_relative_artifact_path


@dataclass(frozen=True)
class TorchProfilerPlan:
    profiler: str
    trace_directory: str
    start_endpoint: str
    stop_endpoint: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DetailedProfilePlan:
    nsys_argv: tuple[str, ...]
    torch: TorchProfilerPlan
    simultaneous_warning: str


def build_nsys_argv(
    command: tuple[str, ...], output_path: str = "raw/gpu/nsys-report"
) -> tuple[str, ...]:
    if not command:
        raise ValueError("target command is required")
    validate_relative_artifact_path(output_path, "nsys.output_path")
    return (
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--force-overwrite=false",
        "--output",
        str(PurePosixPath(output_path)),
        *command,
    )


def build_detailed_profile_plan(command: tuple[str, ...]) -> DetailedProfilePlan:
    return DetailedProfilePlan(
        nsys_argv=build_nsys_argv(command),
        torch=TorchProfilerPlan(
            profiler="torch",
            trace_directory="<absolute-run-directory>/raw/gpu/torch-profiler",
            start_endpoint="/start_profile",
            stop_endpoint="/stop_profile",
            notes=(
                "Configure vLLM profiler_config with profiler=torch.",
                "Set torch_profiler_dir before server startup.",
                "Call start/stop endpoints only during Phase 2B execution validation.",
            ),
        ),
        simultaneous_warning=(
            "Nsight Systems and PyTorch Profiler can add substantial combined overhead; "
            "measure them separately before enabling both."
        ),
    )
