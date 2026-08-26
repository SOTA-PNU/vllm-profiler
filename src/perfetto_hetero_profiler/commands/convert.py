"""Perfetto conversion command parser and handler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "convert", help="Convert an immutable normalized run bundle."
    )
    parser.set_defaults(convert_parser=parser)
    commands = parser.add_subparsers(dest="convert_target")
    perfetto = commands.add_parser(
        "perfetto",
        help=(
            "Generate and validate a deterministic trace with the timeline "
            "Heterogeneous LLM Summary (not Perfetto's built-in Overview)."
        ),
    )
    perfetto.add_argument("--run", type=Path, required=True)
    perfetto.add_argument("--output", type=Path)
    perfetto.add_argument("--trace-processor", type=Path)
    perfetto.add_argument(
        "--include-native-details",
        action="store_true",
        help=(
            "Convert supported GPU Torch/Nsight and NPU vLLM native events "
            "with explicit partial clock evidence. Supported RBLN captures "
            "are published as a separate unaligned native trace."
        ),
    )
    perfetto.add_argument(
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
    perfetto.add_argument("--dry-run", action="store_true")


def handle(args: argparse.Namespace, _: argparse.ArgumentParser) -> int:
    if args.convert_target != "perfetto":
        args.convert_parser.print_help()
        return 0
    try:
        from ..perfetto.converter import (
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
