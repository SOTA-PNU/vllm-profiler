"""Explicit registry for commands shipped by the installed package."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from . import collect, convert, merge, overview, schema


CommandHandler = Callable[[argparse.Namespace, argparse.ArgumentParser], int]
CommandRegistrar = Callable[[argparse._SubParsersAction], None]


@dataclass(frozen=True, slots=True)
class CommandRegistration:
    """One installed command's parser registration and execution boundary."""

    name: str
    register: CommandRegistrar
    handle: CommandHandler


def _register_version(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("version", help="Print the package version.")


def _handle_version(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    from .. import __version__

    print(f"{parser.prog} {__version__}")
    return 0


CORE_COMMANDS: tuple[CommandRegistration, ...] = (
    CommandRegistration("version", _register_version, _handle_version),
    CommandRegistration("schema", schema.register, schema.handle),
    CommandRegistration("collect", collect.register, collect.handle),
    CommandRegistration("merge", merge.register, merge.handle),
    CommandRegistration("convert", convert.register, convert.handle),
    CommandRegistration("overview", overview.register, overview.handle),
)

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    command.name: command.handle for command in CORE_COMMANDS
}

if len(COMMAND_HANDLERS) != len(CORE_COMMANDS):
    raise RuntimeError("duplicate installed CLI command")


__all__ = ["COMMAND_HANDLERS", "CORE_COMMANDS", "CommandRegistration"]
