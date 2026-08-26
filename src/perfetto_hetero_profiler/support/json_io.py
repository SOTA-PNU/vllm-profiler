"""Deterministic JSON encoding and straightforward repository output I/O."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_text(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_pretty_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pretty_json_text(value), encoding="utf-8")


def write_jsonl_exclusive(
    path: str | Path, rows: Iterable[Mapping[str, object]]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row).decode("utf-8"))
