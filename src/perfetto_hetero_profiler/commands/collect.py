"""Collection command parsers and handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..collectors.gpu import GpuRunCollector, GpuRunConfig, build_gpu_run_plan
from ..collectors.npu import NpuRunCollector, NpuRunConfig, build_npu_run_plan
from ..gpu.smoke import GpuVllmSmokeConfig, GpuVllmSmokeRunner, build_smoke_plan
from ..hybrid import (
    HybridRunner,
    build_hybrid_run_plan,
    load_hybrid_runner_config,
    validate_hybrid_invocation,
)
from ..npu import (
    NpuRuntimeSmokeConfig,
    NpuRuntimeSmokeRunner,
    build_runtime_smoke_plan,
)
from ..schema import ProfileMode, RunStatus


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "collect", help="Collect monitor telemetry for a child command."
    )
    parser.set_defaults(collect_parser=parser)
    commands = parser.add_subparsers(dest="collect_target")
    _register_gpu(commands)
    _register_npu(commands)
    _register_npu_runtime(commands)
    _register_gpu_vllm(commands)
    _register_hybrid(commands)


def _register_gpu(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("gpu", help="Run or plan GPU-only collection.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--command", dest="child_argv", nargs=argparse.REMAINDER, required=True
    )


def _register_npu(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("npu", help="Run or plan NPU-only collection.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--host-id", default="host-0")
    parser.add_argument("--device-id", type=int, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--command", dest="child_argv", nargs=argparse.REMAINDER, required=True
    )


def _register_npu_runtime(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "npu-runtime", help="Run or plan a direct RBLN runtime smoke test."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument(
        "--profile-mode",
        choices=("monitor", "detailed-profile"),
        default="monitor",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    parser.add_argument("--warmup-inferences", type=int, default=3)
    parser.add_argument("--measured-inferences", type=int, default=3)
    parser.add_argument("--min-measured-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")


def _register_gpu_vllm(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "gpu-vllm", help="Run or plan a local GPU-only vLLM smoke test."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--profile-mode", choices=("monitor", "torch", "nsys"), default="monitor"
    )
    executable = parser.add_mutually_exclusive_group(required=True)
    executable.add_argument("--server-python", type=Path)
    executable.add_argument("--vllm-bin", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout-sec", type=float, default=180)
    parser.add_argument("--request-timeout-sec", type=float, default=60)
    parser.add_argument("--shutdown-timeout-sec", type=float, default=60)
    parser.add_argument(
        "--sample-interval-ms", type=int, choices=(500, 1000), default=500
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--measured-requests", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--offline", dest="offline", action="store_true", default=True)
    offline.add_argument("--allow-online", dest="offline", action="store_false")
    parser.add_argument("--dry-run", action="store_true")


def _register_hybrid(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "hybrid", help="Run reusable GPU-prefill/NPU-decode collection."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile-mode",
        choices=("monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"),
        default="monitor",
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--measured-requests", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true")


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _handle_hybrid(args: argparse.Namespace) -> int:
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
            _print_json(
                build_hybrid_run_plan(
                    config,
                    run_root=args.run_root,
                    run_id=args.run_id,
                    profile_mode=args.profile_mode,
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
    _print_json(
        {
            "status": result.status.value,
            "hybrid": str(result.run_directory),
            "gpu_source": str(result.gpu_run_directory),
            "npu_source": str(result.npu_run_directory),
            "coordinator": str(result.coordinator_directory),
            "perfetto": str(result.perfetto_directory) if result.perfetto_directory is not None else None,
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
        }
    )
    return 0 if result.status is RunStatus.SUCCEEDED else 1


def _handle_npu_runtime(args: argparse.Namespace) -> int:
    try:
        config = NpuRuntimeSmokeConfig(
            run_root=args.run_root,
            run_id=args.run_id,
            artifact=args.artifact,
            runtime_python=args.runtime_python,
            profile_mode=ProfileMode(args.profile_mode.replace("-", "_")),
            device_id=args.device_id,
            sample_interval_ms=args.sample_interval_ms,
            warmup_inferences=args.warmup_inferences,
            measured_inferences=args.measured_inferences,
            min_measured_seconds=args.min_measured_seconds,
            timeout_sec=args.timeout_sec,
        )
        if args.dry_run:
            _print_json(build_runtime_smoke_plan(config))
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


def _handle_gpu_vllm(args: argparse.Namespace) -> int:
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
            _print_json(build_smoke_plan(config))
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


def _handle_npu(args: argparse.Namespace) -> int:
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
            _print_json(build_npu_run_plan(config))
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


def _handle_gpu(args: argparse.Namespace) -> int:
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
            _print_json(build_gpu_run_plan(config))
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


_TARGET_HANDLERS = {
    "gpu": _handle_gpu,
    "gpu-vllm": _handle_gpu_vllm,
    "hybrid": _handle_hybrid,
    "npu": _handle_npu,
    "npu-runtime": _handle_npu_runtime,
}


def handle(args: argparse.Namespace, _: argparse.ArgumentParser) -> int:
    target = _TARGET_HANDLERS.get(args.collect_target)
    if target is None:
        args.collect_parser.print_help()
        return 0
    return target(args)
