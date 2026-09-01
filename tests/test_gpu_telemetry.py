"""NVML adapter and schema-v1 GPU telemetry tests."""

import json
import importlib
import math
from types import SimpleNamespace
import unittest

from perfetto_hetero_profiler.collectors.gpu import (
    GpuTelemetryCollector,
    NvmlClient,
    NvmlError,
)
from perfetto_hetero_profiler.schema import Availability, validate_record
import perfetto_hetero_profiler.collectors.gpu as gpu_package


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


class NvmlClientTests(unittest.TestCase):
    def test_removed_nvidia_smi_internals_are_not_public(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module(
                "perfetto_hetero_profiler.collectors.gpu.nvidia_smi"
            )
        for name in (
            "NvidiaSmiClient",
            "NvidiaSmiCommandError",
            "NvidiaSmiParseError",
            "parse_nvidia_smi_csv",
        ):
            self.assertFalse(hasattr(gpu_package, name), name)

    def test_one_gpu_values_units_and_name_bytes(self):
        row = client().query().rows[0]
        self.assertEqual(row.index, 0)
        self.assertEqual(row.name, "Test GPU")
        self.assertEqual(row.utilization_percent.value, 25.0)
        self.assertEqual(row.memory_used_bytes.value, 100)
        self.assertEqual(row.memory_total_bytes.value, 1000)
        self.assertEqual(row.power_watts.value, 50.5)

    def test_string_name_and_gpu_index_order(self):
        devices = [
            {
                "name": "GPU zero",
                "utilization": 1,
                "memory_used": 2,
                "memory_total": 3,
                "power_mw": 4000,
            },
            {
                "name": b"GPU one",
                "utilization": 5,
                "memory_used": 6,
                "memory_total": 7,
                "power_mw": 8000,
            },
        ]
        rows = client(FakeBinding(devices)).query().rows
        self.assertEqual([row.device_id for row in rows], ["gpu-0", "gpu-1"])
        self.assertEqual([row.name for row in rows], ["GPU zero", "GPU one"])

    def test_zero_power_is_available(self):
        binding = FakeBinding()
        binding.devices[0]["power_mw"] = 0
        value = client(binding).query().rows[0].power_watts
        self.assertIs(value.availability, Availability.AVAILABLE)
        self.assertEqual(value.value, 0.0)

    def test_not_supported_is_not_available(self):
        binding = FakeBinding()
        binding.devices[0]["power_mw"] = FakeNotSupported()
        row = client(binding).query().rows[0]
        self.assertIs(row.power_watts.availability, Availability.NOT_AVAILABLE)
        self.assertIs(row.utilization_percent.availability, Availability.AVAILABLE)

    def test_unsupported_device_name_does_not_discard_metrics(self):
        binding = FakeBinding()
        binding.devices[0]["name"] = FakeNotSupported()
        row = client(binding).query().rows[0]
        self.assertEqual(row.name, "unknown")
        self.assertIs(row.name_availability, Availability.NOT_AVAILABLE)
        self.assertIs(row.memory_used_bytes.availability, Availability.AVAILABLE)

    def test_field_failure_preserves_other_fields(self):
        binding = FakeBinding()
        binding.devices[0]["utilization"] = FakeNvmlError("private detail")
        row = client(binding).query().rows[0]
        self.assertIs(row.utilization_percent.availability, Availability.ERROR)
        self.assertEqual(row.utilization_percent.reason, "NVML utilization failed")
        self.assertIs(row.memory_used_bytes.availability, Availability.AVAILABLE)
        self.assertNotIn("private detail", row.utilization_percent.reason)

    def test_missing_python_binding_has_capability_error(self):
        def missing():
            raise ModuleNotFoundError("pynvml")

        with self.assertRaisesRegex(NvmlError, r"install .+\[gpu\]"):
            NvmlClient(module_loader=missing).query()

    def test_initialization_error_categories_are_sanitized(self):
        cases = (
            (FakeLibraryNotFound(), "library is unavailable"),
            (FakeDriverNotLoaded(), "driver is not loaded"),
            (FakeNoPermission(), "access is denied"),
        )
        for error, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(NvmlError, reason):
                    client(FakeBinding(init_error=error)).query()

    def test_no_gpu_is_enumeration_error(self):
        with self.assertRaisesRegex(NvmlError, "no GPUs"):
            client(FakeBinding(devices=[])).query()

    def test_enumeration_error_is_sanitized(self):
        binding = FakeBinding(count_error=FakeNvmlError("native object repr"))
        with self.assertRaisesRegex(NvmlError, "device enumeration failed") as caught:
            client(binding).query()
        self.assertNotIn("native object repr", str(caught.exception))

    def test_invalid_values_are_field_errors(self):
        for field, value in (
            ("utilization", -1),
            ("utilization", math.nan),
            ("utilization", math.inf),
            ("utilization", True),
            ("memory_used", -1),
            ("memory_total", False),
            ("power_mw", -1),
        ):
            with self.subTest(field=field, value=value):
                binding = FakeBinding()
                binding.devices[0][field] = value
                row = client(binding).query().rows[0]
                measured = {
                    "utilization": row.utilization_percent,
                    "memory_used": row.memory_used_bytes,
                    "memory_total": row.memory_total_bytes,
                    "power_mw": row.power_watts,
                }[field]
                self.assertIs(measured.availability, Availability.ERROR)
                self.assertIsNone(measured.value)

    def test_initialize_and_shutdown_once(self):
        binding = FakeBinding()
        nvml = client(binding)
        nvml.initialize()
        nvml.initialize()
        nvml.query()
        nvml.query()
        nvml.shutdown()
        nvml.shutdown()
        self.assertEqual(binding.init_calls, 1)
        self.assertEqual(binding.shutdown_calls, 1)

    def test_shutdown_failure_is_sanitized_and_not_retried(self):
        binding = FakeBinding(shutdown_error=FakeNvmlError("private shutdown"))
        nvml = client(binding)
        nvml.initialize()
        with self.assertRaisesRegex(NvmlError, "NVML shutdown failed") as caught:
            nvml.shutdown()
        self.assertNotIn("private shutdown", str(caught.exception))
        nvml.shutdown()
        self.assertEqual(binding.shutdown_calls, 1)

    def test_raw_snapshot_is_deterministic_json(self):
        nvml = client()
        first = nvml.query().raw_snapshot
        second = nvml.query().raw_snapshot
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        document = json.loads(first)
        self.assertEqual(document["adapter_schema_version"], "1.0.0")
        self.assertEqual(document["adapter"]["distribution"], "nvidia-ml-py")
        self.assertEqual(document["devices"][0]["memory_used_bytes"]["value"], 100)


class GpuTelemetryTests(unittest.TestCase):
    def make_collector(self, *, binding=None, monotonic_ns=lambda: 1000, indices=(0,)):
        return GpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=1000,
            client=client(binding),
            known_gpu_indices=indices,
            monotonic_ns=monotonic_ns,
        )

    def test_schema_metrics_and_contract(self):
        collector = self.make_collector()
        collector.prepare()
        collector.start()
        records = collector.sample()
        self.assertEqual(
            [(record.metric_name, record.unit, record.device_id) for record in records],
            [
                ("resource.gpu.utilization", "percent", "gpu-0"),
                ("resource.gpu.memory_used", "bytes", "gpu-0"),
                ("resource.gpu.power", "W", "gpu-0"),
            ],
        )
        self.assertTrue(
            all(record.attributes["nvml.gpu_index"] == 0 for record in records)
        )
        for record in records:
            validate_record(record)

    def test_enumeration_error_produces_known_device_error_metrics(self):
        collector = self.make_collector(
            binding=FakeBinding(devices=[]), indices=(0, 2)
        )
        collector.prepare()
        collector.start()
        records = collector.sample()
        self.assertEqual(len(records), 6)
        self.assertEqual(
            [record.device_id for record in records[::3]], ["gpu-0", "gpu-2"]
        )
        self.assertTrue(
            all(record.availability is Availability.ERROR for record in records)
        )
        self.assertEqual(json.loads(collector.last_raw_snapshot)["source"], "nvml")

    def test_initialization_failure_is_preserved_as_error_metrics(self):
        collector = self.make_collector(
            binding=FakeBinding(init_error=FakeDriverNotLoaded())
        )
        collector.prepare()
        collector.start()
        records = collector.sample()
        self.assertEqual(len(records), 3)
        self.assertTrue(
            all(record.availability is Availability.ERROR for record in records)
        )
        self.assertTrue(
            all(record.reason == "NVIDIA driver is not loaded" for record in records)
        )

    def test_actual_interval_and_timestamp_after_query(self):
        ticks = iter((1_000_000_000, 1_250_000_000))
        collector = self.make_collector(monotonic_ns=lambda: next(ticks))
        collector.prepare()
        collector.start()
        first = collector.sample()
        second = collector.sample()
        self.assertEqual(first[0].timestamp_ns, 1_000_000_000)
        self.assertEqual(first[0].interval_ns, 1_000_000_000)
        self.assertEqual(second[0].timestamp_ns, 1_250_000_000)
        self.assertEqual(second[0].interval_ns, 250_000_000)

    def test_full_lifecycle_owns_one_nvml_session(self):
        binding = FakeBinding()
        collector = self.make_collector(binding=binding)
        collector.prepare()
        collector.start()
        collector.sample()
        collector.sample()
        collector.stop()
        collector.stop()
        collector.finalize()
        self.assertEqual(binding.init_calls, 1)
        self.assertEqual(binding.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
