"""nvidia-smi parsing and schema metric conversion tests."""

import subprocess
import unittest

from perfetto_hetero_profiler.collectors.gpu import (
    GpuTelemetryCollector,
    NvidiaSmiClient,
    NvidiaSmiCommandError,
    NvidiaSmiParseError,
    parse_nvidia_smi_csv,
)
from perfetto_hetero_profiler.schema import Availability, validate_record


NORMAL = "0, Test GPU, 25, 100, 1000, 50.5\n"


class NvidiaSmiParserTests(unittest.TestCase):
    def test_normal_values(self):
        row = parse_nvidia_smi_csv(NORMAL)[0]
        self.assertEqual(row.index, 0)
        self.assertEqual(row.name, "Test GPU")
        self.assertEqual(row.utilization_percent.value, 25.0)
        self.assertEqual(row.power_watts.value, 50.5)

    def test_zero_values_are_available(self):
        row = parse_nvidia_smi_csv("0, GPU, 0, 0, 1000, 0\n")[0]
        for value in (
            row.utilization_percent,
            row.memory_used_bytes,
            row.power_watts,
        ):
            self.assertIs(value.availability, Availability.AVAILABLE)
            self.assertEqual(value.value, 0)

    def test_mib_to_bytes(self):
        row = parse_nvidia_smi_csv("0, GPU, 0 %, 1.5 MiB, 10 MiB, 1 W\n")[0]
        self.assertEqual(row.memory_used_bytes.value, int(1.5 * 1024 * 1024))
        self.assertEqual(row.memory_total_bytes.value, 10 * 1024 * 1024)

    def test_na_is_not_available(self):
        row = parse_nvidia_smi_csv("0, GPU, N/A, N/A, 10, N/A\n")[0]
        self.assertIs(
            row.utilization_percent.availability, Availability.NOT_AVAILABLE
        )
        self.assertIsNone(row.utilization_percent.value)

    def test_not_supported_is_not_available(self):
        row = parse_nvidia_smi_csv(
            "0, GPU, [Not Supported], 0, 10, [Not Supported]\n"
        )[0]
        self.assertIs(row.power_watts.availability, Availability.NOT_AVAILABLE)

    def test_empty_field_is_not_available(self):
        row = parse_nvidia_smi_csv("0, GPU, , 0, 10, \n")[0]
        self.assertIs(
            row.utilization_percent.availability, Availability.NOT_AVAILABLE
        )

    def test_malformed_value_is_field_error(self):
        row = parse_nvidia_smi_csv("0, GPU, broken, 0, 10, 1\n")[0]
        self.assertIs(row.utilization_percent.availability, Availability.ERROR)

    def test_malformed_row_rejected(self):
        with self.assertRaises(NvidiaSmiParseError):
            parse_nvidia_smi_csv("0, GPU, 1\n")

    def test_multiple_gpus(self):
        rows = parse_nvidia_smi_csv(NORMAL + "1, Other GPU, 0, 0, 2000, 0\n")
        self.assertEqual([row.device_id for row in rows], ["gpu-0", "gpu-1"])

    def test_no_rows_rejected(self):
        with self.assertRaises(NvidiaSmiParseError):
            parse_nvidia_smi_csv("\n")


class NvidiaSmiClientTests(unittest.TestCase):
    def test_nonzero_return_code(self):
        def runner(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 2, "", "driver unavailable")

        with self.assertRaisesRegex(NvidiaSmiCommandError, "driver unavailable"):
            NvidiaSmiClient(runner=runner).query()

    def test_timeout(self):
        def runner(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 1)

        with self.assertRaisesRegex(NvidiaSmiCommandError, "timed out"):
            NvidiaSmiClient(runner=runner).query()

    def test_shell_is_false(self):
        observed = {}

        def runner(*args, **kwargs):
            observed.update(kwargs)
            return subprocess.CompletedProcess(args[0], 0, NORMAL, "")

        NvidiaSmiClient(runner=runner).query()
        self.assertIs(observed["shell"], False)


class GpuTelemetryTests(unittest.TestCase):
    @staticmethod
    def client(text=NORMAL, returncode=0):
        def runner(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], returncode, text, "failure")

        return NvidiaSmiClient(runner=runner)

    def test_schema_metrics(self):
        collector = GpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=1000,
            client=self.client(),
            monotonic_ns=lambda: 1000,
        )
        collector.prepare()
        collector.start()
        records = collector.sample()
        self.assertEqual(len(records), 3)
        self.assertEqual(
            {record.metric_name for record in records},
            {
                "resource.gpu.utilization",
                "resource.gpu.memory_used",
                "resource.gpu.power",
            },
        )
        for record in records:
            validate_record(record)

    def test_error_produces_error_metrics(self):
        collector = GpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=1000,
            client=self.client(returncode=1),
            known_gpu_indices=(0, 1),
            monotonic_ns=lambda: 1000,
        )
        collector.prepare()
        collector.start()
        records = collector.sample()
        self.assertEqual(len(records), 6)
        self.assertTrue(
            all(record.availability is Availability.ERROR for record in records)
        )

    def test_actual_interval_after_first_sample(self):
        ticks = iter((1_000_000_000, 1_250_000_000))
        collector = GpuTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            sample_interval_ms=1000,
            client=self.client(),
            monotonic_ns=lambda: next(ticks),
        )
        collector.prepare()
        collector.start()
        first = collector.sample()
        second = collector.sample()
        self.assertEqual(first[0].interval_ns, 1_000_000_000)
        self.assertEqual(second[0].interval_ns, 250_000_000)


if __name__ == "__main__":
    unittest.main()
