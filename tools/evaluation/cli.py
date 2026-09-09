"""Repository-only CLI for repeatability and overhead evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from . import (
    experiment_status,
    generate_report,
    run_experiment,
    validate_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler-evaluation",
        description="Repository-only heterogeneous profiler evaluations.",
    )
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser(
        "run", help="Run or resume the deterministic evaluation schedule."
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--experiment-root", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    for name, help_text in (
        ("status", "Inspect an existing evaluation checkpoint."),
        ("validate", "Freshly validate successful evaluation trials without writes."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--experiment-root", type=Path, required=True)
    report = commands.add_parser(
        "report", help="Deterministically generate evaluation JSON/HTML reports."
    )
    report.add_argument("--experiment-root", type=Path, required=True)
    report.add_argument(
        "--output-root",
        type=Path,
        help="Publish a corrected report to a new, non-overlapping directory.",
    )
    overhead = commands.add_parser(
        "all-mode-overhead",
        help="Run the fixed 1 Hz reference-versus-profiler hardware campaign.",
    )
    overhead_commands = overhead.add_subparsers(dest="overhead_command")
    overhead_run = overhead_commands.add_parser(
        "run",
        help=(
            "Create or resume an immutable configured-round campaign without "
            "hardware retries."
        ),
    )
    overhead_run.add_argument("--config", type=Path, required=True)
    overhead_run.add_argument("--campaign-root", type=Path, required=True)
    overhead_run.add_argument("--resume", action="store_true")
    overhead_run.add_argument("--dry-run", action="store_true")
    overhead_run.add_argument("--preflight-only", action="store_true")
    overhead_status = overhead_commands.add_parser(
        "status", help="Inspect a campaign checkpoint without running hardware."
    )
    overhead_status.add_argument("--campaign-root", type=Path, required=True)
    overhead_recover = overhead_commands.add_parser(
        "recover-postprocess",
        help="Recover a postprocess-only failure without rerunning hardware.",
    )
    overhead_recover.add_argument("--config", type=Path, required=True)
    overhead_recover.add_argument("--campaign-root", type=Path, required=True)
    overhead_report = overhead_commands.add_parser(
        "report", help="Regenerate the deterministic campaign reports."
    )
    overhead_report.add_argument("--campaign-root", type=Path, required=True)
    overview = commands.add_parser(
        "overview", help="Evaluate independently validated Overview outputs."
    )
    overview_commands = overview.add_subparsers(dest="overview_command")
    compare = overview_commands.add_parser(
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
    wave = commands.add_parser(
        "concurrent-wave", help="Plan or run barrier-released Hybrid requests."
    )
    wave.add_argument("--matrix", type=Path, required=True)
    wave.add_argument("--hybrid-config", type=Path)
    wave.add_argument("--condition-id")
    wave.add_argument("--round", type=int, dest="round_index")
    wave.add_argument("--block-root", type=Path)
    wave.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            result = run_experiment(
                config_path=args.config,
                experiment_root=args.experiment_root,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        elif args.command == "status":
            result = experiment_status(args.experiment_root)
        elif args.command == "validate":
            result = validate_experiment(args.experiment_root)
        elif args.command == "report":
            result = generate_report(
                args.experiment_root,
                output_root=args.output_root,
            )
        elif args.command == "all-mode-overhead":
            from .all_mode_overhead import (
                campaign_status,
                generate_campaign_report,
                recover_failed_postprocess,
                run_all_mode_campaign,
            )

            if args.overhead_command == "run":
                result = run_all_mode_campaign(
                    config_path=args.config,
                    campaign_root=args.campaign_root,
                    resume=args.resume,
                    dry_run=args.dry_run,
                    preflight_only=args.preflight_only,
                )
            elif args.overhead_command == "status":
                result = campaign_status(args.campaign_root)
            elif args.overhead_command == "recover-postprocess":
                result = recover_failed_postprocess(
                    config_path=args.config,
                    campaign_root=args.campaign_root,
                )
            elif args.overhead_command == "report":
                result = generate_campaign_report(args.campaign_root)
            else:
                parser.print_help()
                return 0
        elif args.command == "concurrent-wave":
            from .concurrent_wave import plan, run_block

            if args.dry_run:
                result = plan(args.matrix)
            else:
                names = ("hybrid_config", "condition_id", "round_index", "block_root")
                required = {"--" + name.replace("_", "-"): getattr(args, name)
                            for name in names}
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    parser.error("concurrent-wave requires " + ", ".join(missing))
                result = run_block(matrix_path=args.matrix,
                                   hybrid_config_path=args.hybrid_config,
                                   condition_id=args.condition_id,
                                   round_index=args.round_index,
                                   block_root=args.block_root)
        elif args.overview_command == "compare":
            from .overview import (
                OverviewComparisonConfig,
                compare_overviews,
                plan_overview_comparison,
            )

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
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print(
            "evaluation interrupted; checkpoint preserved for --resume",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
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


__all__ = ["build_parser", "main"]
