"""External Overview command parser and handler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "overview",
        help=(
            "Generate or compare external KPI Overview reports (not the "
            "Perfetto UI)."
        ),
    )
    parser.set_defaults(overview_parser=parser)
    commands = parser.add_subparsers(dest="overview_command")
    generate = commands.add_parser(
        "generate", help="Generate a single-run external KPI JSON/HTML report."
    )
    generate.add_argument("--run", type=Path, required=True)
    generate.add_argument("--perfetto", type=Path, required=True)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--trace-processor", type=Path)
    generate.add_argument("--dry-run", action="store_true")
    compare = commands.add_parser(
        "compare", help="Compare independently validated Overview outputs."
    )
    compare.add_argument(
        "--input",
        dest="overview_inputs",
        type=Path,
        action="append",
        required=True,
    )
    compare.add_argument("--output", type=Path)
    compare.add_argument("--baseline")
    compare.add_argument("--dry-run", action="store_true")


def handle(args: argparse.Namespace, _: argparse.ArgumentParser) -> int:
    if args.overview_command is None:
        args.overview_parser.print_help()
        return 0
    try:
        from ..overview.generator import (
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
