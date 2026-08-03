"""Actual direct-RBLN workload child and normalized smoke runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from ..collectors.npu import NpuRunCollector, NpuRunConfig
from ..schema import (
    ArtifactKind,
    ArtifactReference,
    ProfileMode,
    RunPaths,
    RunStatus,
    WorkloadDescriptor,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .workload import (
    measured_window_metrics,
    observation_events,
    observation_metrics,
    parse_observations,
)


@dataclass(frozen=True)
class NpuRuntimeSmokeConfig:
    run_root: Path
    run_id: str
    artifact: Path
    runtime_python: Path
    profile_mode: ProfileMode = ProfileMode.MONITOR
    device_id: int = 0
    sample_interval_ms: int = 500
    warmup_inferences: int = 3
    measured_inferences: int = 3
    min_measured_seconds: float = 10.0
    timeout_sec: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "artifact", Path(self.artifact))
        object.__setattr__(self, "runtime_python", Path(self.runtime_python))
        if not self.run_root.is_absolute() or not self.artifact.is_absolute():
            raise ValueError("run_root and artifact must be absolute")
        if not self.runtime_python.is_absolute():
            raise ValueError("runtime_python must be absolute")
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.sample_interval_ms < 100:
            raise ValueError("sample_interval_ms must be >= 100")
        if self.warmup_inferences < 0 or self.measured_inferences < 1:
            raise ValueError("warmup must be >= 0 and measured must be >= 1")
        if self.min_measured_seconds < 0 or self.timeout_sec <= 0:
            raise ValueError("durations must be non-negative and timeout positive")

    @property
    def run_directory(self) -> Path:
        return self.run_root / self.run_id

    @property
    def summary_path(self) -> Path:
        return self.run_directory / "raw/npu/workload-summary.json"

    @property
    def profiler_directory(self) -> Path:
        return self.run_directory / "raw/npu/rbln_profiler"


@dataclass(frozen=True)
class NpuRuntimeSmokeResult:
    run_directory: Path
    status: RunStatus
    return_code: int
    warmup_count: int
    measured_count: int
    event_count: int
    metric_count: int
    artifact_count: int
    profiler_artifact_count: int
    errors: tuple[str, ...]


def build_runtime_smoke_plan(config: NpuRuntimeSmokeConfig) -> dict[str, object]:
    return {
        "executes": False,
        "run_id": config.run_id,
        "run_directory": str(config.run_directory),
        "profile_mode": config.profile_mode.value,
        "device_id": config.device_id,
        "artifact": str(config.artifact),
        "artifact_inspection": "rebel.RBLNCompiledModel.inspect",
        "child_argv": list(_child_argv(config)),
        "workload": {
            "warmup_inferences": config.warmup_inferences,
            "minimum_measured_inferences": config.measured_inferences,
            "minimum_measured_seconds": config.min_measured_seconds,
            "token_stream": False,
        },
        "profiler": {
            "enabled": config.profile_mode is ProfileMode.DETAILED_PROFILE,
            "public_api": (
                "rebel.Runtime(activate_profiler=True) + "
                "rebel.profiler.profile(output_dir=...)"
            ),
            "output_directory": str(config.profiler_directory),
        },
    }


class NpuRuntimeSmokeRunner:
    def __init__(self, config: NpuRuntimeSmokeConfig) -> None:
        self.config = config

    def run(self) -> NpuRuntimeSmokeResult:
        config = self.config
        paths = RunPaths(config.run_root, config.run_id)
        generic = NpuRunCollector(
            NpuRunConfig(
                run_root=config.run_root,
                run_id=config.run_id,
                profile_mode=config.profile_mode,
                sample_interval_ms=config.sample_interval_ms,
                command=_child_argv(config),
                cwd=Path.cwd(),
                timeout_sec=config.timeout_sec,
                model_id=config.artifact.name,
                device_ids=(config.device_id,),
                allow_detailed_execution=True,
            )
        ).run()
        errors: list[str] = []
        observations = ()
        summary: dict[str, object] | None = None
        if config.summary_path.is_file():
            try:
                loaded = json.loads(config.summary_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("workload summary root must be an object")
                summary = loaded
                observations = parse_observations(summary)
                if not summary.get("success"):
                    errors.append(str(summary.get("error") or "workload failed"))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"workload summary: {error}")
        else:
            errors.append("workload summary was not created")

        events = list(read_jsonl(paths.events))
        metrics = list(read_jsonl(paths.metrics))
        artifacts = list(read_jsonl(paths.artifacts))
        for observation in observations:
            events.extend(observation_events(config.run_id, observation))
            metrics.extend(observation_metrics(config.run_id, observation))
        metrics.extend(measured_window_metrics(config.run_id, observations))
        if config.summary_path.is_file():
            artifacts.append(
                _artifact_reference(
                    config.run_id,
                    "npu-workload-summary",
                    config.summary_path,
                    config.run_directory,
                    ArtifactKind.RAW_LOG,
                    "npu-runtime-smoke",
                    "json",
                )
            )

        profile_files = _profile_files(config.profiler_directory)
        profiler_report_count = 0
        for index, path in enumerate(profile_files):
            artifact_kind = _profile_artifact_kind(path)
            if artifact_kind is ArtifactKind.RBLN_REPORT:
                profiler_report_count += 1
            artifacts.append(
                _artifact_reference(
                    config.run_id,
                    f"rbln-profile-{index}",
                    path,
                    config.run_directory,
                    artifact_kind,
                    "rebel.profiler.profile",
                    _format_name(path),
                )
            )
        if (
            config.profile_mode is ProfileMode.DETAILED_PROFILE
            and profiler_report_count == 0
        ):
            errors.append("RBLN profiler created no non-empty report artifact")
        if generic.status is not RunStatus.SUCCEEDED:
            errors.append(f"generic NPU collection status: {generic.status.value}")

        _replace_jsonl(paths.events, events)
        _replace_jsonl(paths.metrics, metrics)
        _replace_jsonl(paths.artifacts, artifacts)
        status = RunStatus.SUCCEEDED if not errors else RunStatus.FAILED
        manifest = read_json(paths.manifest)
        manifest.configuration.update(
            {
                "artifact": str(config.artifact),
                "warmup_inferences": config.warmup_inferences,
                "minimum_measured_inferences": config.measured_inferences,
                "minimum_measured_seconds": config.min_measured_seconds,
                "profiler_output_directory": (
                    "raw/npu/rbln_profiler"
                    if config.profile_mode is ProfileMode.DETAILED_PROFILE
                    else None
                ),
            }
        )
        manifest.attributes.update(
            {
                "vendor.collector": "npu-runtime-smoke",
                "rbln.workload_success": bool(summary and summary.get("success")),
                "rbln.artifact_compiler_version": (
                    summary.get("artifact_metadata", {}).get("compiler_version")
                    if summary and isinstance(summary.get("artifact_metadata"), dict)
                    else None
                ),
                "rbln.errors": errors,
            }
        )
        manifest = replace(
            manifest,
            status=status,
            workload=WorkloadDescriptor(
                request_count=len(observations),
                concurrency=1,
                request_rate_per_s=next(
                    (
                        metric.value
                        for metric in reversed(metrics)
                        if metric.metric_name == "throughput.requests"
                    ),
                    None,
                ),
                input_tokens=None,
                output_tokens=None,
                max_model_len=None,
                warmup_requests=config.warmup_inferences,
            ),
        )
        _replace_json(paths.manifest, manifest)
        return NpuRuntimeSmokeResult(
            run_directory=config.run_directory,
            status=status,
            return_code=generic.return_code,
            warmup_count=int(summary.get("warmup_count", 0)) if summary else 0,
            measured_count=len(observations),
            event_count=len(events),
            metric_count=len(metrics),
            artifact_count=len(artifacts),
            profiler_artifact_count=profiler_report_count,
            errors=tuple(errors),
        )


def _child_argv(config: NpuRuntimeSmokeConfig) -> tuple[str, ...]:
    argv = [
        str(config.runtime_python),
        "-c",
        (
            "from perfetto_hetero_profiler.npu.runtime_smoke import child_main; "
            "raise SystemExit(child_main())"
        ),
        "--child",
        "--run-id",
        config.run_id,
        "--artifact",
        str(config.artifact),
        "--summary",
        str(config.summary_path),
        "--device-id",
        str(config.device_id),
        "--warmup",
        str(config.warmup_inferences),
        "--measured",
        str(config.measured_inferences),
        "--min-measured-seconds",
        str(config.min_measured_seconds),
    ]
    if config.profile_mode is ProfileMode.DETAILED_PROFILE:
        argv.extend(["--profiler-output", str(config.profiler_directory)])
    return tuple(argv)


def _run_child(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "run_id": args.run_id,
        "artifact": args.artifact,
        "device_id": args.device_id,
        "warmup_count": 0,
        "measured": [],
        "success": False,
        "profiler": {
            "enabled": args.profiler_output is not None,
            "public_api": "rebel.profiler.profile(output_dir=...)",
            "call_sequence": [],
            "new_files": [],
        },
    }
    try:
        import numpy as np
        import rebel

        artifact = Path(args.artifact)
        metadata = rebel.RBLNCompiledModel.inspect(artifact)
        summary["artifact_metadata"] = metadata
        inputs = metadata.get("inputs", [])
        if len(inputs) != 1:
            raise RuntimeError("smoke adapter requires exactly one input")
        input_spec = inputs[0]
        if input_spec.get("dtype") != "float32":
            raise RuntimeError("smoke adapter requires float32 input")
        input_array = np.zeros(tuple(input_spec["shape"]), dtype=np.float32)
        runtime = rebel.Runtime(
            artifact,
            device=args.device_id,
            tensor_type="np",
            activate_profiler=False,
        )
        output = None
        for _ in range(args.warmup):
            output = runtime.run(input_array)
            summary["warmup_count"] = int(summary["warmup_count"]) + 1

        measured: list[dict[str, object]] = []
        window_start_ns = time.monotonic_ns()

        def run_measured() -> None:
            nonlocal output
            index = len(measured)
            started_ns = time.monotonic_ns()
            output = runtime.run(input_array)
            ended_ns = time.monotonic_ns()
            measured.append(
                {
                    "request_id": f"{args.run_id}-measured-{index}",
                    "started_ns": started_ns,
                    "ended_ns": ended_ns,
                }
            )

        profile_output = (
            Path(args.profiler_output) if args.profiler_output is not None else None
        )
        before = set()
        if profile_output is not None:
            profile_output.mkdir(parents=True, exist_ok=True)
            before = _snapshot(profile_output)
            sidecars_before = set(Path.cwd().glob("profiler_error_*.log"))
            from rebel.profiler import profile

            del runtime
            gc.collect()
            summary["profiler"]["call_sequence"].append(
                "rebel.Runtime(activate_profiler=True)"
            )
            runtime = rebel.Runtime(
                artifact,
                device=args.device_id,
                tensor_type="np",
                activate_profiler=True,
            )
            summary["profiler"]["call_sequence"].append("profile.__enter__")
            with profile(output_dir=str(profile_output)):
                while (
                    len(measured) < args.measured
                    or (time.monotonic_ns() - window_start_ns) / 1e9
                    < args.min_measured_seconds
                ):
                    run_measured()
            summary["profiler"]["call_sequence"].append("profile.__exit__")
            _relocate_vendor_sidecars(
                Path.cwd(), profile_output, sidecars_before
            )
            after = _snapshot(profile_output)
            summary["profiler"]["new_files"] = sorted(after - before)
        else:
            while (
                len(measured) < args.measured
                or (time.monotonic_ns() - window_start_ns) / 1e9
                < args.min_measured_seconds
            ):
                run_measured()

        first_output = output[0] if isinstance(output, (list, tuple)) else output
        summary["measured"] = measured
        summary["measured_elapsed_ns"] = (
            measured[-1]["ended_ns"] - measured[0]["started_ns"]
        )
        summary["input"] = {
            "shape": list(input_array.shape),
            "dtype": str(input_array.dtype),
        }
        summary["output"] = {
            "shape": list(first_output.shape),
            "dtype": str(first_output.dtype),
        }
        summary["success"] = True
        print(
            f"NPU_RUNTIME_OK warmup={summary['warmup_count']} measured={len(measured)} "
            f"input={summary['input']} output={summary['output']}",
            flush=True,
        )
        del output, first_output, runtime
        gc.collect()
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        print(f"NPU_RUNTIME_ERROR {summary['error']}", file=sys.stderr, flush=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["success"] else 1


def _snapshot(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _profile_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and path.stat().st_size > 0))


def _relocate_vendor_sidecars(
    cwd: Path, output_directory: Path, before: set[Path]
) -> tuple[Path, ...]:
    """Keep newly created public-profiler diagnostics inside the owned run."""
    created = set(cwd.glob("profiler_error_*.log")) - before
    relocated: list[Path] = []
    for source in sorted(created):
        target = output_directory / source.name
        if target.exists():
            raise FileExistsError(f"profiler sidecar target already exists: {target}")
        source.replace(target)
        relocated.append(target)
    return tuple(relocated)


def _format_name(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if path.suffix:
        return path.suffix.lstrip(".").lower()
    return mime or "unknown"


def _profile_artifact_kind(path: Path) -> ArtifactKind:
    if path.suffix.lower() in {".log", ".txt"}:
        return ArtifactKind.RAW_LOG
    return ArtifactKind.RBLN_REPORT


def _artifact_reference(
    run_id: str,
    artifact_id: str,
    path: Path,
    run_root: Path,
    kind: ArtifactKind,
    producer: str,
    format_name: str,
) -> ArtifactReference:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactReference(
        run_id=run_id,
        artifact_id=artifact_id,
        artifact_kind=kind,
        relative_path=path.relative_to(run_root).as_posix(),
        format=format_name,
        producer=producer,
        created_at_unix_ns=time.time_ns(),
        size_bytes=path.stat().st_size,
        sha256=digest,
        attributes={},
    )


def _replace_jsonl(path: Path, records: list[object]) -> None:
    temporary = path.with_name(f".{path.name}.postprocess.tmp")
    write_jsonl(temporary, records)
    os.replace(temporary, path)


def _replace_json(path: Path, record: object) -> None:
    temporary = path.with_name(f".{path.name}.postprocess.tmp")
    write_json(temporary, record)
    os.replace(temporary, path)


def child_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    parser.add_argument("--min-measured-seconds", type=float, required=True)
    parser.add_argument("--profiler-output")
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("--child is required")
    return _run_child(args)


if __name__ == "__main__":
    raise SystemExit(child_main())
