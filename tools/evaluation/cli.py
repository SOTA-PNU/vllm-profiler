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
        else:
            result = generate_report(
                args.experiment_root,
                output_root=args.output_root,
            )
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
