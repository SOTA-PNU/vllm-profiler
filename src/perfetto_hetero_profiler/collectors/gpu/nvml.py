"""NVML-backed NVIDIA GPU discovery and resource snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from numbers import Real
from typing import Any, Callable, Final

from ...schema import Availability


NVML_DISTRIBUTION: Final = "nvidia-ml-py"
NVML_DISTRIBUTION_VERSION: Final = "13.610.43"
NVML_SNAPSHOT_SCHEMA_VERSION: Final = "1.0.0"


class NvmlError(RuntimeError):
    """A sanitized NVML capability, lifecycle, or enumeration failure."""


@dataclass(frozen=True, slots=True)
class NvmlValue:
    value: int | float | None
    availability: Availability
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NvmlRow:
    index: int
    name: str
    name_availability: Availability
    name_reason: str | None
    utilization_percent: NvmlValue
    memory_used_bytes: NvmlValue
    memory_total_bytes: NvmlValue
    power_watts: NvmlValue

    @property
    def device_id(self) -> str:
        return f"gpu-{self.index}"


@dataclass(frozen=True, slots=True)
class NvmlQueryResult:
    rows: tuple[NvmlRow, ...]
    raw_snapshot: str


def _load_pynvml() -> Any:
    try:
        return importlib.import_module("pynvml")
    except (ImportError, ModuleNotFoundError) as error:
        raise NvmlError(
            "NVML Python binding is unavailable; install "
            "perfetto-hetero-profiler[gpu]"
        ) from error


def _is_error(binding: Any, error: Exception, name: str) -> bool:
    error_type = getattr(binding, name, None)
    return isinstance(error_type, type) and isinstance(error, error_type)


def _error_reason(binding: Any, error: Exception, operation: str) -> str:
    categories = (
        ("NVMLError_LibraryNotFound", "NVML library is unavailable"),
        ("NVMLError_DriverNotLoaded", "NVIDIA driver is not loaded"),
        ("NVMLError_NoPermission", "NVML access is denied"),
    )
    for name, reason in categories:
        if _is_error(binding, error, name):
            return reason
    return f"NVML {operation} failed"


def _error_value(binding: Any, error: Exception, field: str) -> NvmlValue:
    if _is_error(binding, error, "NVMLError_NotSupported"):
        return NvmlValue(
            None,
            Availability.NOT_AVAILABLE,
            f"NVML {field} is not supported",
        )
    return NvmlValue(None, Availability.ERROR, _error_reason(binding, error, field))


def _number(
    value: object,
    field: str,
    *,
    integer: bool = False,
    maximum: float | None = None,
) -> NvmlValue:
    valid_type = isinstance(value, int) if integer else isinstance(value, Real)
    if isinstance(value, bool) or not valid_type:
        return NvmlValue(None, Availability.ERROR, f"NVML {field} returned invalid data")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (
        maximum is not None and number > maximum
    ):
        return NvmlValue(None, Availability.ERROR, f"NVML {field} returned invalid data")
    return NvmlValue(int(value) if integer else number, Availability.AVAILABLE)


def _snapshot(rows: tuple[NvmlRow, ...]) -> str:
    def field(value: NvmlValue) -> dict[str, object]:
        return {
            "availability": value.availability.value,
            "reason": value.reason,
            "value": value.value,
        }

    payload = {
        "adapter": {
            "distribution": NVML_DISTRIBUTION,
            "version": NVML_DISTRIBUTION_VERSION,
        },
        "adapter_schema_version": NVML_SNAPSHOT_SCHEMA_VERSION,
        "devices": [
            {
                "index": row.index,
                "memory_total_bytes": field(row.memory_total_bytes),
                "memory_used_bytes": field(row.memory_used_bytes),
                "name": {
                    "availability": row.name_availability.value,
                    "reason": row.name_reason,
                    "value": (
                        row.name
                        if row.name_availability is Availability.AVAILABLE
                        else None
                    ),
                },
                "power_watts": field(row.power_watts),
                "utilization_percent": field(row.utilization_percent),
            }
            for row in rows
        ],
        "source": "nvml",
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def nvml_error_snapshot(indices: tuple[int, ...], reason: str) -> str:
    error = NvmlValue(None, Availability.ERROR, reason)
    rows = tuple(
        NvmlRow(
            index=index,
            name="unknown",
            name_availability=Availability.ERROR,
            name_reason=reason,
            utilization_percent=error,
            memory_used_bytes=error,
            memory_total_bytes=error,
            power_watts=error,
        )
        for index in sorted(indices)
    )
    return _snapshot(rows)


class NvmlClient:
    """Own one lazy NVML initialization for an entire collector lifecycle."""

    def __init__(
        self,
        *,
        binding: Any | None = None,
        module_loader: Callable[[], Any] = _load_pynvml,
    ) -> None:
        self._binding = binding
        self._module_loader = module_loader
        self._initialized = False
        self._closed = False
        self._initialization_error: NvmlError | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._closed:
            raise NvmlError("NVML client is closed")
        if self._initialization_error is not None:
            raise self._initialization_error
        binding = self._binding
        try:
            binding = binding or self._module_loader()
            binding.nvmlInit()
        except NvmlError as error:
            self._initialization_error = error
            raise
        except (ImportError, ModuleNotFoundError) as error:
            self._initialization_error = NvmlError(
                "NVML Python binding is unavailable; install "
                "perfetto-hetero-profiler[gpu]"
            )
            raise self._initialization_error from error
        except Exception as error:
            reason = (
                _error_reason(binding, error, "initialization")
                if binding is not None
                else "NVML initialization failed"
            )
            self._initialization_error = NvmlError(reason)
            raise self._initialization_error from error
        self._binding = binding
        self._initialized = True

    def query(self) -> NvmlQueryResult:
        self.initialize()
        assert self._binding is not None
        binding = self._binding
        try:
            count = binding.nvmlDeviceGetCount()
        except Exception as error:
            raise NvmlError(
                _error_reason(binding, error, "device enumeration")
            ) from error
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise NvmlError("NVML device enumeration returned no GPUs")
        rows = tuple(self._query_device(index) for index in range(count))
        return NvmlQueryResult(rows=rows, raw_snapshot=_snapshot(rows))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._initialized:
            return
        assert self._binding is not None
        try:
            self._binding.nvmlShutdown()
        except Exception as error:
            raise NvmlError(
                _error_reason(self._binding, error, "shutdown")
            ) from error
        finally:
            self._initialized = False

    def _query_device(self, index: int) -> NvmlRow:
        assert self._binding is not None
        binding = self._binding
        try:
            handle = binding.nvmlDeviceGetHandleByIndex(index)
        except Exception as error:
            reason = _error_reason(binding, error, "device handle")
            failed = NvmlValue(None, Availability.ERROR, reason)
            return NvmlRow(
                index, "unknown", Availability.ERROR, reason,
                failed, failed, failed, failed,
            )

        try:
            raw_name = binding.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("invalid device name")
            name, name_availability, name_reason = name.strip(), Availability.AVAILABLE, None
        except Exception as error:
            name = "unknown"
            name_availability = (
                Availability.NOT_AVAILABLE
                if _is_error(binding, error, "NVMLError_NotSupported")
                else Availability.ERROR
            )
            name_reason = (
                "NVML device name is not supported"
                if name_availability is Availability.NOT_AVAILABLE
                else _error_reason(binding, error, "device name")
            )

        utilization = self._field(
            "utilization", lambda: binding.nvmlDeviceGetUtilizationRates(handle).gpu,
            lambda value: _number(value, "utilization", maximum=100),
        )
        try:
            memory = binding.nvmlDeviceGetMemoryInfo(handle)
            memory_used = _number(memory.used, "memory used", integer=True)
            memory_total = _number(memory.total, "memory total", integer=True)
        except Exception as error:
            memory_used = memory_total = _error_value(binding, error, "memory")
        power = self._field(
            "power", lambda: binding.nvmlDeviceGetPowerUsage(handle),
            lambda value: _scale_power(value),
        )
        return NvmlRow(
            index,
            name,
            name_availability,
            name_reason,
            utilization,
            memory_used,
            memory_total,
            power,
        )

    def _field(
        self,
        name: str,
        getter: Callable[[], object],
        convert: Callable[[object], NvmlValue],
    ) -> NvmlValue:
        assert self._binding is not None
        try:
            return convert(getter())
        except Exception as error:
            return _error_value(self._binding, error, name)


def _scale_power(value: object) -> NvmlValue:
    milliwatts = _number(value, "power", integer=True)
    if milliwatts.availability is not Availability.AVAILABLE:
        return milliwatts
    assert isinstance(milliwatts.value, int)
    return NvmlValue(milliwatts.value / 1000.0, Availability.AVAILABLE)


__all__ = [
    "NVML_DISTRIBUTION",
    "NVML_DISTRIBUTION_VERSION",
    "NVML_SNAPSHOT_SCHEMA_VERSION",
    "NvmlClient",
    "NvmlError",
    "NvmlQueryResult",
    "NvmlRow",
    "NvmlValue",
    "nvml_error_snapshot",
]
