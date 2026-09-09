"""Shared fake NVML binding for GPU collector tests."""

from __future__ import annotations

from types import SimpleNamespace

from perfetto_hetero_profiler.collectors.gpu import NvmlClient


class FakeNvmlError(Exception):
    pass


class FakeNotSupported(FakeNvmlError):
    pass


class FakeLibraryNotFound(FakeNvmlError):
    pass


class FakeDriverNotLoaded(FakeNvmlError):
    pass


class FakeNoPermission(FakeNvmlError):
    pass


class FakeBinding:
    NVMLError = FakeNvmlError
    NVMLError_NotSupported = FakeNotSupported
    NVMLError_LibraryNotFound = FakeLibraryNotFound
    NVMLError_DriverNotLoaded = FakeDriverNotLoaded
    NVMLError_NoPermission = FakeNoPermission

    def __init__(
        self,
        devices=None,
        *,
        init_error=None,
        count_error=None,
        shutdown_error=None,
    ):
        self.devices = devices if devices is not None else [
            {
                "name": b"Test GPU",
                "utilization": 25,
                "memory_used": 100,
                "memory_total": 1000,
                "power_mw": 50_500,
            }
        ]
        self.init_error = init_error
        self.count_error = count_error
        self.shutdown_error = shutdown_error
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def nvmlDeviceGetCount(self):
        if self.count_error is not None:
            raise self.count_error
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def _value(self, handle, key):
        value = self.devices[handle][key]
        if isinstance(value, Exception):
            raise value
        return value

    def nvmlDeviceGetName(self, handle):
        return self._value(handle, "name")

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=self._value(handle, "utilization"))

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(
            used=self._value(handle, "memory_used"),
            total=self._value(handle, "memory_total"),
        )

    def nvmlDeviceGetPowerUsage(self, handle):
        return self._value(handle, "power_mw")


def client(binding=None):
    return NvmlClient(binding=binding or FakeBinding())
