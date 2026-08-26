"""Command-line entry point for the installed core profiler."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .commands import COMMAND_HANDLERS, CORE_COMMANDS


def build_parser() -> argparse.ArgumentParser:
    """Build the installed command parser from the explicit core registry."""

    parser = argparse.ArgumentParser(
        prog="hetero-profiler",
        description="Heterogeneous GPU/RBLN profiler.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in CORE_COMMANDS:
        command.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch one registered core command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args, parser)


__all__ = ["build_parser", "main"]
