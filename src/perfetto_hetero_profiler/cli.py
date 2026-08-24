"""Command-line interface for collection, Perfetto, and Overview products."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .schema import (
    ProfileMode,
    SCHEMA_VERSION,
    RecordType,
    RunStatus,
    SchemaValidationError,
    read_json,
    read_jsonl,
)
from .collectors.gpu import GpuRunCollector, GpuRunConfig, build_gpu_run_plan
from .collectors.npu import NpuRunCollector, NpuRunConfig, build_npu_run_plan
from .gpu.smoke import GpuVllmSmokeConfig, GpuVllmSmokeRunner, build_smoke_plan
from .hybrid import (
    AlignmentMethod,
    HybridBundleMerger,
    HybridMergeConfig,
    build_hybrid_plan,
    HybridRunner,
    build_hybrid_run_plan,
    load_hybrid_runner_config,
    validate_hybrid_invocation,
)
from .npu import (
    NpuRuntimeSmokeConfig,
    NpuRuntimeSmokeRunner,
    build_runtime_smoke_plan,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="hetero-profiler",
        description="Heterogeneous GPU/RBLN profiler.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="Print the package version.")

    schema_parser = subparsers.add_parser(
        "schema", help="Inspect and validate schema v1 records."
    )
    schema_parser.set_defaults(schema_parser=schema_parser)
    schema_subparsers = schema_parser.add_subparsers(dest="schema_command")
    schema_subparsers.add_parser("version", help="Print the schema version.")
    schema_subparsers.add_parser("list", help="List supported record types.")
    validate_parser = schema_subparsers.add_parser(
        "validate", help="Validate a JSON or JSONL record file."
    )
    validate_parser.add_argument("path", type=Path)

    collect_parser = subparsers.add_parser(
        "collect", help="Collect monitor telemetry for a child command."
    )
    collect_parser.set_defaults(collect_parser=collect_parser)
    collect_subparsers = collect_parser.add_subparsers(dest="collect_target")
    gpu_parser = collect_subparsers.add_parser(
        "gpu", help="Run or plan GPU-only collection."
    )
    gpu_parser.add_argument("--run-root", type=Path, required=True)
    gpu_parser.add_argument("--run-id", required=True)
    gpu_parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    gpu_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    gpu_parser.add_argument("--cwd", type=Path)
    gpu_parser.add_argument("--timeout-sec", type=float)
    gpu_parser.add_argument("--dry-run", action="store_true")
    gpu_parser.add_argument(
        "--command", dest="child_argv", nargs=argparse.REMAINDER, required=True
    )
    npu_parser = collect_subparsers.add_parser(
        "npu", help="Run or plan NPU-only collection."
    )
    npu_parser.add_argument("--run-root", type=Path, required=True)
    npu_parser.add_argument("--run-id", required=True)
    npu_parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    npu_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    npu_parser.add_argument("--cwd", type=Path)
    npu_parser.add_argument("--timeout-sec", type=float)
    npu_parser.add_argument("--host-id", default="host-0")
    npu_parser.add_argument("--device-id", type=int, action="append", default=[])
    npu_parser.add_argument("--dry-run", action="store_true")
    npu_parser.add_argument(
        "--command", dest="child_argv", nargs=argparse.REMAINDER, required=True
    )
    runtime_parser = collect_subparsers.add_parser(
        "npu-runtime", help="Run or plan a direct RBLN runtime smoke test."
    )
    runtime_parser.add_argument("--run-root", type=Path, required=True)
    runtime_parser.add_argument("--run-id", required=True)
    runtime_parser.add_argument("--artifact", type=Path, required=True)
    runtime_parser.add_argument("--runtime-python", type=Path, required=True)
    runtime_parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    runtime_parser.add_argument("--device-id", type=int, default=0)
    runtime_parser.add_argument("--sample-interval-ms", type=int, default=500)
    runtime_parser.add_argument("--warmup-inferences", type=int, default=3)
    runtime_parser.add_argument("--measured-inferences", type=int, default=3)
    runtime_parser.add_argument("--min-measured-seconds", type=float, default=10.0)
    runtime_parser.add_argument("--timeout-sec", type=float, default=120.0)
    runtime_parser.add_argument("--dry-run", action="store_true")
    vllm_parser = collect_subparsers.add_parser(
        "gpu-vllm", help="Run or plan a local GPU-only vLLM smoke test."
    )
    vllm_parser.add_argument("--run-root", type=Path, required=True)
    vllm_parser.add_argument("--run-id", required=True)
    vllm_parser.add_argument("--model", type=Path, required=True)
    vllm_parser.add_argument(
        "--profile-mode", choices=("monitor", "torch", "nsys"), default="monitor"
    )
    executable = vllm_parser.add_mutually_exclusive_group(required=True)
    executable.add_argument("--server-python", type=Path)
    executable.add_argument("--vllm-bin", type=Path)
    vllm_parser.add_argument("--host", default="127.0.0.1")
    vllm_parser.add_argument("--port", type=int, default=18080)
    vllm_parser.add_argument("--startup-timeout-sec", type=float, default=180)
    vllm_parser.add_argument("--request-timeout-sec", type=float, default=60)
    vllm_parser.add_argument("--shutdown-timeout-sec", type=float, default=60)
    vllm_parser.add_argument(
        "--sample-interval-ms", type=int, choices=(500, 1000), default=500
    )
    vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    vllm_parser.add_argument("--max-model-len", type=int, default=2048)
    vllm_parser.add_argument("--warmup-requests", type=int, default=1)
    vllm_parser.add_argument("--measured-requests", type=int, default=2)
    vllm_parser.add_argument("--max-output-tokens", type=int, default=8)
    offline = vllm_parser.add_mutually_exclusive_group()
    offline.add_argument("--offline", dest="offline", action="store_true", default=True)
    offline.add_argument("--allow-online", dest="offline", action="store_false")
    vllm_parser.add_argument("--dry-run", action="store_true")
    hybrid_collect = collect_subparsers.add_parser(
        "hybrid",
        help="Run reusable GPU-prefill/NPU-decode collection.",
    )
    hybrid_collect.add_argument("--config", type=Path, required=True)
    hybrid_collect.add_argument("--run-root", type=Path, required=True)
    hybrid_collect.add_argument("--run-id", required=True)
    hybrid_collect.add_argument(
        "--profile-mode",
        choices=("monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"),
        default="monitor",
    )
    prompts = hybrid_collect.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    hybrid_collect.add_argument("--warmup-requests", type=int)
    hybrid_collect.add_argument("--measured-requests", type=int)
    hybrid_collect.add_argument("--max-output-tokens", type=int)
    hybrid_collect.add_argument("--dry-run", action="store_true")
    phase7_parser = subparsers.add_parser(
        "phase7", help="Run and validate fixed Hybrid profiler experiments."
    )
    phase7_parser.set_defaults(phase7_parser=phase7_parser)
    phase7_subparsers = phase7_parser.add_subparsers(dest="phase7_command")
    phase7_run = phase7_subparsers.add_parser(
        "run", help="Run or resume the deterministic Phase 7B schedule."
    )
    phase7_run.add_argument("--config", type=Path, required=True)
    phase7_run.add_argument("--experiment-root", type=Path, required=True)
    phase7_run.add_argument("--resume", action="store_true")
    phase7_run.add_argument("--dry-run", action="store_true")
    for name, help_text in (
        ("status", "Inspect an existing Phase 7B checkpoint."),
        (
            "validate",
            "Freshly validate all successful Phase 7B trials without writes.",
        ),
    ):
        command_parser = phase7_subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--experiment-root", type=Path, required=True)
    phase7_report = phase7_subparsers.add_parser(
        "report", help="Deterministically generate Phase 7B JSON/HTML reports."
    )
    phase7_report.add_argument("--experiment-root", type=Path, required=True)
    phase7_report.add_argument(
        "--output-root",
        type=Path,
        help="Publish a corrected report to a new, non-overlapping directory.",
    )
    merge_parser = subparsers.add_parser(
        "merge", help="Merge immutable normalized source bundles."
    )
    merge_parser.set_defaults(merge_parser=merge_parser)
    merge_subparsers = merge_parser.add_subparsers(dest="merge_target")
    hybrid_parser = merge_subparsers.add_parser(
        "hybrid", help="Align and merge fake GPU/NPU source bundles."
    )
    hybrid_parser.add_argument("--run-root", type=Path, required=True)
    hybrid_parser.add_argument("--run-id", required=True)
    hybrid_parser.add_argument("--gpu-run", type=Path, required=True)
    hybrid_parser.add_argument("--npu-run", type=Path, required=True)
    hybrid_parser.add_argument(
        "--alignment-method",
        choices=tuple(method.value for method in AlignmentMethod),
        default=AlignmentMethod.SAME_CLOCK_DOMAIN.value,
    )
    hybrid_parser.add_argument("--max-uncertainty-ns", type=int, default=1_000_000)
    hybrid_parser.add_argument("--host-id", default="hybrid-coordinator")
    hybrid_parser.add_argument("--probe-count", type=int, default=7)
    hybrid_parser.add_argument("--minimum-probe-samples", type=int, default=5)
    hybrid_parser.add_argument("--fake-offset-ns", type=int, default=0)
    hybrid_parser.add_argument("--fake-delay-ns", type=int, default=100_000)
    hybrid_parser.add_argument("--fake-jitter-ns", type=int, default=0)
    hybrid_parser.add_argument("--fake-asymmetry-ns", type=int, default=0)
    hybrid_parser.add_argument("--dry-run", action="store_true")

    convert_parser = subparsers.add_parser(
        "convert", help="Convert an immutable normalized run bundle."
    )
    convert_parser.set_defaults(convert_parser=convert_parser)
    convert_subparsers = convert_parser.add_subparsers(dest="convert_target")
    perfetto_parser = convert_subparsers.add_parser(
        "perfetto",
        help=(
            "Generate and validate a deterministic trace with the timeline "
            "Heterogeneous LLM Summary (not Perfetto's built-in Overview)."
        ),
    )
    perfetto_parser.add_argument("--run", type=Path, required=True)
    perfetto_parser.add_argument("--output", type=Path)
    perfetto_parser.add_argument("--trace-processor", type=Path)
    perfetto_parser.add_argument(
        "--include-native-details",
        action="store_true",
        help=(
            "Convert supported GPU Torch/Nsight and NPU vLLM native events "
            "with explicit partial clock evidence. Supported RBLN captures "
            "are published as a separate unaligned native trace."
        ),
    )
    perfetto_parser.add_argument(
        "--request-focused",
        action="store_true",
        help=(
            "Also write trace.request-focused.pftrace with observed processing "
            "stages, canonical request/token boundaries, and source-backed "
            "resource samples selected for the client request_start/stream_end "
            "window. Full-window telemetry and capture envelopes are omitted. "
            "Original timestamps are not rebased."
        ),
    )
    perfetto_parser.add_argument("--dry-run", action="store_true")

    overview_parser = subparsers.add_parser(
        "overview",
        help=(
            "Generate or compare external KPI Overview reports (not the "
            "Perfetto UI)."
        ),
    )
    overview_parser.set_defaults(overview_parser=overview_parser)
    overview_subparsers = overview_parser.add_subparsers(dest="overview_command")
    overview_generate = overview_subparsers.add_parser(
        "generate",
        help="Generate a single-run external KPI JSON/HTML report.",
    )
    overview_generate.add_argument("--run", type=Path, required=True)
    overview_generate.add_argument("--perfetto", type=Path, required=True)
    overview_generate.add_argument("--output", type=Path)
    overview_generate.add_argument("--trace-processor", type=Path)
    overview_generate.add_argument("--dry-run", action="store_true")
    overview_compare = overview_subparsers.add_parser(
        "compare",
        help="Compare independently validated Overview outputs.",
    )
    overview_compare.add_argument(
        "--input",
        dest="overview_inputs",
        type=Path,
        action="append",
        required=True,
    )
    overview_compare.add_argument("--output", type=Path)
    overview_compare.add_argument("--baseline")
    overview_compare.add_argument("--dry-run", action="store_true")
    return parser


def _validate_path(path: Path) -> int:
    try:
        if path.suffix.lower() == ".json":
            count = 1
            read_json(path)
        elif path.suffix.lower() == ".jsonl":
            count = len(read_jsonl(path))
        else:
            raise SchemaValidationError(
                "path", "file extension must be .json or .jsonl"
            )
    except (SchemaValidationError, OSError) as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        print(f"internal error: {error}", file=sys.stderr)
        return 1

    print(f"valid: {path} ({count} record{'s' if count != 1 else ''})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"{parser.prog} {__version__}")
        return 0
    if args.command == "phase7":
        if args.phase7_command is None:
            args.phase7_parser.print_help()
            return 0
        try:
            from .phase7 import (
                experiment_status,
                generate_report,
                run_experiment,
                validate_experiment,
            )

            if args.phase7_command == "run":
                result = run_experiment(
                    config_path=args.config,
                    experiment_root=args.experiment_root,
                    resume=args.resume,
                    dry_run=args.dry_run,
                )
            elif args.phase7_command == "status":
                result = experiment_status(args.experiment_root)
            elif args.phase7_command == "validate":
                result = validate_experiment(args.experiment_root)
            else:
                result = generate_report(
                    args.experiment_root,
                    output_root=args.output_root,
                )
        except KeyboardInterrupt:
            print("phase7 interrupted; checkpoint preserved for --resume", file=sys.stderr)
            return 130
        except (OSError, ValueError, RuntimeError) as error:
            print(f"phase7 error: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "schema":
        if args.schema_command == "version":
            print(SCHEMA_VERSION)
            return 0
        if args.schema_command == "list":
            for record_type in RecordType:
                print(record_type.value)
            return 0
        if args.schema_command == "validate":
            return _validate_path(args.path)
        args.schema_parser.print_help()
        return 0
    if args.command == "convert":
        if args.convert_target != "perfetto":
            args.convert_parser.print_help()
            return 0
        try:
            from .perfetto.converter import (
                PerfettoConversionConfig,
                convert_perfetto,
                plan_perfetto_conversion,
            )

            config = PerfettoConversionConfig(
                run_directory=args.run,
                output_directory=args.output,
                trace_processor_path=args.trace_processor,
                include_native_details=args.include_native_details,
                request_focused=args.request_focused,
            )
            result = (
                plan_perfetto_conversion(config)
                if args.dry_run
                else convert_perfetto(config)
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"conversion error: {error}", file=sys.stderr)
            return 2
        except Exception as error:  # pragma: no cover - defensive CLI boundary
            print(f"internal error: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "overview":
        if args.overview_command is None:
            args.overview_parser.print_help()
            return 0
        try:
            from .overview.generator import (
                OverviewComparisonConfig,
                OverviewGenerationConfig,
                compare_overviews,
                generate_overview,
                plan_overview_comparison,
                plan_overview_generation,
            )

            if args.overview_command == "generate":
                config = OverviewGenerationConfig(
                    run_directory=args.run,
                    perfetto_directory=args.perfetto,
                    output_directory=args.output,
                    trace_processor_path=args.trace_processor,
                )
                result = (
                    plan_overview_generation(config)
                    if args.dry_run
                    else generate_overview(config)
                )
            elif args.overview_command == "compare":
                config = OverviewComparisonConfig(
                    input_directories=tuple(args.overview_inputs),
                    output_directory=args.output,
                    baseline_run_id=args.baseline,
                )
                result = (
                    plan_overview_comparison(config)
                    if args.dry_run
                    else compare_overviews(config)
                )
            else:  # pragma: no cover - argparse guards this branch
                args.overview_parser.print_help()
                return 0
        except (OSError, ValueError, RuntimeError) as error:
            print(f"overview error: {error}", file=sys.stderr)
            return 2
        except Exception as error:  # pragma: no cover - defensive CLI boundary
            print(f"internal error: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "merge":
        if args.merge_target != "hybrid":
            args.merge_parser.print_help()
            return 0
        try:
            config = HybridMergeConfig(
                run_root=args.run_root,
                run_id=args.run_id,
                gpu_run=args.gpu_run,
                npu_run=args.npu_run,
                alignment_method=AlignmentMethod(args.alignment_method),
                max_uncertainty_ns=args.max_uncertainty_ns,
                coordinator_host_id=args.host_id,
                probe_count=args.probe_count,
                minimum_probe_samples=args.minimum_probe_samples,
                fake_offset_ns=args.fake_offset_ns,
                fake_delay_ns=args.fake_delay_ns,
                fake_jitter_ns=args.fake_jitter_ns,
                fake_asymmetry_ns=args.fake_asymmetry_ns,
            )
            if args.dry_run:
                print(
                    json.dumps(
                        build_hybrid_plan(config),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            result = HybridBundleMerger(config).merge()
        except (OSError, ValueError, RuntimeError) as error:
            print(f"merge error: {error}", file=sys.stderr)
            return 2
        print(
            f"{result.status.value}: {result.run_directory} "
            f"(joined={result.joined_request_count}, "
            f"unjoined={result.unjoined_request_count}, "
            f"uncertainty_ns={result.uncertainty_ns}, "
            f"reasons={'; '.join(result.reasons) or 'none'})"
        )
        return 0 if result.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL} else 1
    if args.command == "collect":
        if args.collect_target not in {"gpu", "gpu-vllm", "hybrid", "npu", "npu-runtime"}:
            args.collect_parser.print_help()
            return 0
        if args.collect_target == "hybrid":
            try:
                config = load_hybrid_runner_config(args.config).with_overrides(
                    prompt=args.prompt,
                    prompt_file=args.prompt_file,
                    warmup_requests=args.warmup_requests,
                    measured_requests=args.measured_requests,
                    max_output_tokens=args.max_output_tokens,
                )
                validate_hybrid_invocation(
                    config,
                    run_root=args.run_root,
                    run_id=args.run_id,
                    profile_mode=args.profile_mode,
                )
                if args.dry_run:
                    print(
                        json.dumps(
                            build_hybrid_run_plan(
                                config,
                                run_root=args.run_root,
                                run_id=args.run_id,
                                profile_mode=args.profile_mode,
                            ),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    return 0
                result = HybridRunner(
                    config,
                    run_root=args.run_root,
                    run_id=args.run_id,
                    profile_mode=args.profile_mode,
                ).run()
            except (OSError, ValueError, RuntimeError) as error:
                print(f"collection error: {error}", file=sys.stderr)
                return 2
            print(
                json.dumps(
                    {
                        "status": result.status.value,
                        "hybrid": str(result.run_directory),
                        "gpu_source": str(result.gpu_run_directory),
                        "npu_source": str(result.npu_run_directory),
                        "coordinator": str(result.coordinator_directory),
                        "perfetto": (
                            str(result.perfetto_directory)
                            if result.perfetto_directory is not None
                            else None
                        ),
                        "request_focused_perfetto": (
                            str(result.request_focused_perfetto_directory)
                            if result.request_focused_perfetto_directory is not None
                            else None
                        ),
                        "external_html_overview": (
                            str(result.overview_directory)
                            if result.overview_directory is not None
                            else None
                        ),
                        "closeout_recovery": (
                            str(result.recovery_directory)
                            if result.recovery_directory is not None
                            else None
                        ),
                        "publication": str(result.publication_directory),
                        "warmup_completed": result.warmup_count,
                        "measured_completed": result.measured_count,
                        "errors": list(result.errors),
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result.status is RunStatus.SUCCEEDED else 1
        if args.collect_target == "npu-runtime":
            try:
                config = NpuRuntimeSmokeConfig(
                    run_root=args.run_root,
                    run_id=args.run_id,
                    artifact=args.artifact,
                    runtime_python=args.runtime_python,
                    profile_mode=ProfileMode(
                        args.profile_mode.replace("-", "_")
                    ),
                    device_id=args.device_id,
                    sample_interval_ms=args.sample_interval_ms,
                    warmup_inferences=args.warmup_inferences,
                    measured_inferences=args.measured_inferences,
                    min_measured_seconds=args.min_measured_seconds,
                    timeout_sec=args.timeout_sec,
                )
                if args.dry_run:
                    print(
                        json.dumps(
                            build_runtime_smoke_plan(config),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    return 0
                result = NpuRuntimeSmokeRunner(config).run()
            except (OSError, ValueError, RuntimeError) as error:
                print(f"collection error: {error}", file=sys.stderr)
                return 2
            print(
                f"{result.status.value}: {result.run_directory} "
                f"(return_code={result.return_code}, "
                f"warmup={result.warmup_count}, measured={result.measured_count}, "
                f"metrics={result.metric_count}, artifacts={result.artifact_count})"
            )
            return 0 if result.status is RunStatus.SUCCEEDED else 1
        if args.collect_target == "gpu-vllm":
            try:
                config = GpuVllmSmokeConfig(
                    run_root=args.run_root,
                    run_id=args.run_id,
                    model=args.model,
                    profile_mode=args.profile_mode,
                    host=args.host,
                    port=args.port,
                    startup_timeout_sec=args.startup_timeout_sec,
                    request_timeout_sec=args.request_timeout_sec,
                    shutdown_timeout_sec=args.shutdown_timeout_sec,
                    sample_interval_ms=args.sample_interval_ms,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    warmup_requests=args.warmup_requests,
                    measured_requests=args.measured_requests,
                    max_output_tokens=args.max_output_tokens,
                    server_python=args.server_python,
                    vllm_bin=args.vllm_bin,
                    offline=args.offline,
                )
                if args.dry_run:
                    print(
                        json.dumps(
                            build_smoke_plan(config),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    return 0
                result = GpuVllmSmokeRunner(config).run()
            except (OSError, ValueError, RuntimeError) as error:
                print(f"collection error: {error}", file=sys.stderr)
                return 2
            print(
                f"{result.status.value}: {result.run_directory} "
                f"(warmup={result.warmup_count}, measured={result.measured_count}, "
                f"metrics={result.metric_count}, artifacts={result.artifact_count})"
            )
            return 0 if result.status is RunStatus.SUCCEEDED else 1
        if args.collect_target == "npu":
            try:
                config = NpuRunConfig(
                    run_root=args.run_root,
                    run_id=args.run_id,
                    profile_mode=ProfileMode(args.profile_mode.replace("-", "_")),
                    sample_interval_ms=args.sample_interval_ms,
                    command=tuple(args.child_argv),
                    cwd=args.cwd,
                    host_id=args.host_id,
                    timeout_sec=args.timeout_sec,
                    device_ids=tuple(args.device_id),
                )
                if args.dry_run:
                    print(
                        json.dumps(
                            build_npu_run_plan(config),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    return 0
                result = NpuRunCollector(config).run()
            except (OSError, ValueError, RuntimeError) as error:
                print(f"collection error: {error}", file=sys.stderr)
                return 2
            print(
                f"{result.status.value}: {result.run_directory} "
                f"(return_code={result.return_code}, metrics={result.metric_count})"
            )
            return 0 if result.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL} else 1
        try:
            config = GpuRunConfig(
                run_root=args.run_root,
                run_id=args.run_id,
                profile_mode=ProfileMode(args.profile_mode.replace("-", "_")),
                sample_interval_ms=args.sample_interval_ms,
                command=tuple(args.child_argv),
                cwd=args.cwd,
                timeout_sec=args.timeout_sec,
            )
            if args.dry_run:
                print(
                    json.dumps(
                        build_gpu_run_plan(config),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            result = GpuRunCollector(config).run()
        except (OSError, ValueError, RuntimeError) as error:
            print(f"collection error: {error}", file=sys.stderr)
            return 2
        print(
            f"{result.status.value}: {result.run_directory} "
            f"(return_code={result.return_code}, metrics={result.metric_count})"
        )
        return 0 if result.status.value in {"succeeded", "partial"} else 1

    parser.print_help()
    return 0
