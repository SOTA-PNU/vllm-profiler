"""``rbln-smi --json`` command wrapper and tolerant field parser."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from typing import Any, Callable

from ...schema import Availability


RBLN_SMI_ARGV = ("rbln-smi", "--json")
_UNAVAILABLE = {"", "n/a", "na", "null", "none", "not supported", "[not supported]"}
_NUMBER_RE = re.compile(
    r"^[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))[ \t]*([A-Za-z%/]*)[ \t]*$"
)
_MEMORY_UNITS = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}
_POWER_UNITS = {"w": 1.0, "mw": 1e-3, "uw": 1e-6}


class RblnSmiParseError(ValueError):
    pass


class RblnSmiCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedValue:
    value: int | float | None
    availability: Availability
    reason: str | None = None
    structurally_unsupported: bool = False


@dataclass(frozen=True)
class RblnSmiRow:
    index: int
    name: str
    status: str
    utilization_percent: ParsedValue
    memory_used_bytes: ParsedValue
    memory_total_bytes: ParsedValue
    power_watts: ParsedValue
    temperature_celsius: ParsedValue
    firmware_version: str | None

    @property
    def device_id(self) -> str:
        return f"npu-{self.index}"


@dataclass(frozen=True)
class RblnSmiQueryResult:
    rows: tuple[RblnSmiRow, ...]
    raw_output: str
    kmd_version: str | None


def _missing(field: str) -> ParsedValue:
    return ParsedValue(
        value=None,
        availability=Availability.NOT_AVAILABLE,
        reason=f"installed rbln-smi 3.0.0 does not expose {field}",
        structurally_unsupported=True,
    )


def _parse_number(
    raw: Any,
    field: str,
    *,
    units: dict[str, float | int] | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> ParsedValue:
    if raw is None:
        return ParsedValue(None, Availability.NOT_AVAILABLE, f"{field} is not available")
    if isinstance(raw, bool):
        return ParsedValue(None, Availability.ERROR, f"could not parse {field}: {raw!r}")
    text = str(raw).strip()
    if text.lower() in _UNAVAILABLE:
        return ParsedValue(None, Availability.NOT_AVAILABLE, f"{field} is not available")
    match = _NUMBER_RE.match(text)
    if match is None:
        return ParsedValue(None, Availability.ERROR, f"could not parse {field}: {text!r}")
    number = float(match.group(1))
    unit = match.group(2).lower()
    if units is not None:
        if unit not in units:
            return ParsedValue(
                None, Availability.ERROR, f"unsupported {field} unit: {unit!r}"
            )
        number *= units[unit]
    elif unit not in {"", "%"}:
        return ParsedValue(None, Availability.ERROR, f"unexpected {field} unit: {unit!r}")
    if number < 0:
        return ParsedValue(None, Availability.ERROR, f"{field} must be non-negative")
    if maximum is not None and number > maximum:
        return ParsedValue(None, Availability.ERROR, f"{field} must be <= {maximum:g}")
    value: int | float = int(number) if integer else number
    return ParsedValue(value, Availability.AVAILABLE)


def _field(device: dict[str, Any], key: str, field: str) -> ParsedValue | Any:
    return _missing(field) if key not in device else device[key]


def _parse_device(device: Any, position: int) -> RblnSmiRow:
    if not isinstance(device, dict):
        raise RblnSmiParseError(f"device {position} must be an object")
    raw_index = device.get("npu")
    if isinstance(raw_index, bool):
        raise RblnSmiParseError(f"device {position}: npu must be an integer")
    try:
        index = int(raw_index)
    except (TypeError, ValueError) as error:
        raise RblnSmiParseError(f"device {position}: npu must be an integer") from error
    name = device.get("name")
    if index < 0 or not isinstance(name, str) or not name.strip():
        raise RblnSmiParseError(
            f"device {position}: npu and name must identify a valid device"
        )
    status = device.get("status", "unknown")
    if not isinstance(status, str) or not status.strip():
        status = "unknown"

    raw_memory = device.get("memory")
    memory = raw_memory if isinstance(raw_memory, dict) else {}
    memory_used = (
        _missing("memory.used")
        if "memory" not in device or "used" not in memory
        else _parse_number(
            memory["used"], "memory.used", units=_MEMORY_UNITS, integer=True
        )
    )
    memory_total = (
        _missing("memory.total")
        if "memory" not in device or "total" not in memory
        else _parse_number(
            memory["total"], "memory.total", units=_MEMORY_UNITS, integer=True
        )
    )
    raw_util = _field(device, "util", "util")
    utilization = (
        raw_util
        if isinstance(raw_util, ParsedValue)
        else _parse_number(raw_util, "util", maximum=100)
    )
    raw_power = _field(device, "card_power", "card_power")
    power = (
        raw_power
        if isinstance(raw_power, ParsedValue)
        else _parse_number(raw_power, "card_power", units=_POWER_UNITS)
    )
    raw_temperature = _field(device, "temperature", "temperature")
    temperature = (
        raw_temperature
        if isinstance(raw_temperature, ParsedValue)
        else _parse_number(raw_temperature, "temperature", units={"c": 1.0})
    )
    firmware = device.get("fw_ver")
    if not isinstance(firmware, str) or not firmware.strip():
        firmware = None
    return RblnSmiRow(
        index=index,
        name=name.strip(),
        status=status.strip(),
        utilization_percent=utilization,
        memory_used_bytes=memory_used,
        memory_total_bytes=memory_total,
        power_watts=power,
        temperature_celsius=temperature,
        firmware_version=firmware,
    )


def parse_rbln_smi_json(text: str) -> RblnSmiQueryResult:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise RblnSmiParseError(f"invalid JSON: {error.msg}") from error
    if not isinstance(document, dict):
        raise RblnSmiParseError("top-level JSON value must be an object")
    devices = document.get("devices")
    if not isinstance(devices, list) or not devices:
        raise RblnSmiParseError("rbln-smi returned no NPU devices")
    rows = tuple(_parse_device(device, position) for position, device in enumerate(devices))
    indices = [row.index for row in rows]
    if len(indices) != len(set(indices)):
        raise RblnSmiParseError("NPU device indices must be unique")
    kmd_version = document.get("KMD_version")
    if not isinstance(kmd_version, str) or not kmd_version.strip():
        kmd_version = None
    return RblnSmiQueryResult(
        rows=rows,
        raw_output=text,
        kmd_version=kmd_version,
    )


class RblnSmiClient:
    def __init__(
        self,
        *,
        device_ids: tuple[int, ...] = (),
        timeout_sec: float = 5.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        if any(index < 0 for index in device_ids):
            raise ValueError("device id must be a non-negative integer")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("device ids must be unique")
        self.device_ids = device_ids
        self.timeout_sec = timeout_sec
        self.runner = runner

    @property
    def argv(self) -> tuple[str, ...]:
        if not self.device_ids:
            return RBLN_SMI_ARGV
        return (*RBLN_SMI_ARGV, "--device", ",".join(map(str, self.device_ids)))

    def query(self) -> RblnSmiQueryResult:
        result = self._run(self.argv, "query")
        try:
            return parse_rbln_smi_json(result.stdout)
        except RblnSmiParseError as error:
            raise RblnSmiCommandError(f"rbln-smi parse error: {error}") from error

    def version(self) -> str:
        result = self._run(("rbln-smi", "--version"), "version")
        version = result.stdout.strip()
        if not version:
            raise RblnSmiCommandError("rbln-smi version output is empty")
        return version

    def _run(
        self, argv: tuple[str, ...], operation: str
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(
                list(argv),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RblnSmiCommandError(f"rbln-smi {operation} timed out") from error
        except OSError as error:
            raise RblnSmiCommandError(f"rbln-smi could not start: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RblnSmiCommandError(f"rbln-smi failed: {detail}")
        return result
