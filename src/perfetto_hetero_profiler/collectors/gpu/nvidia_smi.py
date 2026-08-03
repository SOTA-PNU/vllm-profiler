"""nvidia-smi CSV query and tolerant field parser."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import re
import subprocess
from typing import Callable

from ...schema import Availability


QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
)
NVIDIA_SMI_ARGV = (
    "nvidia-smi",
    f"--query-gpu={','.join(QUERY_FIELDS)}",
    "--format=csv,noheader,nounits",
)
_UNAVAILABLE = {"", "n/a", "[not supported]", "not supported", "na"}
_NUMBER_RE = re.compile(r"^[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")


class NvidiaSmiParseError(ValueError):
    pass


class NvidiaSmiCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedValue:
    value: int | float | None
    availability: Availability
    reason: str | None = None


@dataclass(frozen=True)
class NvidiaSmiRow:
    index: int
    name: str
    utilization_percent: ParsedValue
    memory_used_bytes: ParsedValue
    memory_total_bytes: ParsedValue
    power_watts: ParsedValue

    @property
    def device_id(self) -> str:
        return f"gpu-{self.index}"


@dataclass(frozen=True)
class NvidiaSmiQueryResult:
    rows: tuple[NvidiaSmiRow, ...]
    raw_output: str


def _parse_number(raw: str, field: str, *, integer: bool = False) -> ParsedValue:
    value = raw.strip()
    if value.lower() in _UNAVAILABLE:
        return ParsedValue(
            value=None,
            availability=Availability.NOT_AVAILABLE,
            reason=f"{field} is not available",
        )
    match = _NUMBER_RE.match(value)
    if match is None:
        return ParsedValue(
            value=None,
            availability=Availability.ERROR,
            reason=f"could not parse {field}: {value!r}",
        )
    number = float(match.group(1))
    if number < 0:
        return ParsedValue(
            value=None,
            availability=Availability.ERROR,
            reason=f"{field} must be non-negative",
        )
    return ParsedValue(
        value=int(number) if integer else number,
        availability=Availability.AVAILABLE,
    )


def _parse_memory(raw: str, field: str) -> ParsedValue:
    parsed = _parse_number(raw, field)
    if parsed.availability is not Availability.AVAILABLE:
        return parsed
    assert parsed.value is not None
    return ParsedValue(
        value=int(float(parsed.value) * 1024 * 1024),
        availability=Availability.AVAILABLE,
    )


def parse_nvidia_smi_csv(text: str) -> tuple[NvidiaSmiRow, ...]:
    rows: list[NvidiaSmiRow] = []
    for line_number, fields in enumerate(csv.reader(text.splitlines()), start=1):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != len(QUERY_FIELDS):
            raise NvidiaSmiParseError(
                f"line {line_number}: expected {len(QUERY_FIELDS)} fields, got {len(fields)}"
            )
        try:
            index = int(fields[0].strip())
        except ValueError as error:
            raise NvidiaSmiParseError(
                f"line {line_number}: GPU index must be an integer"
            ) from error
        name = fields[1].strip()
        if index < 0 or not name:
            raise NvidiaSmiParseError(
                f"line {line_number}: GPU index and name must be valid"
            )
        rows.append(
            NvidiaSmiRow(
                index=index,
                name=name,
                utilization_percent=_parse_number(fields[2], "utilization.gpu"),
                memory_used_bytes=_parse_memory(fields[3], "memory.used"),
                memory_total_bytes=_parse_memory(fields[4], "memory.total"),
                power_watts=_parse_number(fields[5], "power.draw"),
            )
        )
    if not rows:
        raise NvidiaSmiParseError("nvidia-smi returned no GPU rows")
    return tuple(rows)


class NvidiaSmiClient:
    def __init__(
        self,
        *,
        timeout_sec: float = 5.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        self.timeout_sec = timeout_sec
        self.runner = runner

    def query(self) -> NvidiaSmiQueryResult:
        try:
            result = self.runner(
                list(NVIDIA_SMI_ARGV),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise NvidiaSmiCommandError("nvidia-smi query timed out") from error
        except OSError as error:
            raise NvidiaSmiCommandError(f"nvidia-smi could not start: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise NvidiaSmiCommandError(f"nvidia-smi failed: {detail}")
        try:
            rows = parse_nvidia_smi_csv(result.stdout)
        except NvidiaSmiParseError as error:
            raise NvidiaSmiCommandError(f"nvidia-smi parse error: {error}") from error
        return NvidiaSmiQueryResult(rows=rows, raw_output=result.stdout)
