"""Schema command parser and handler."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..schema import (
    SCHEMA_VERSION,
    RecordType,
    SchemaValidationError,
    read_json,
    read_jsonl,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "schema", help="Inspect and validate schema v1 records."
    )
    parser.set_defaults(schema_parser=parser)
    commands = parser.add_subparsers(dest="schema_command")
    commands.add_parser("version", help="Print the schema version.")
    commands.add_parser("list", help="List supported record types.")
    validate = commands.add_parser(
        "validate", help="Validate a JSON or JSONL record file."
    )
    validate.add_argument("path", type=Path)


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


def handle(args: argparse.Namespace, _: argparse.ArgumentParser) -> int:
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
