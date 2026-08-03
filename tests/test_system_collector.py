"""Linux /proc parser and telemetry tests."""

from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.collectors import (
    ProcTelemetryCollector,
    parse_meminfo,
    parse_process_rss,
    parse_proc_stat,
)
from perfetto_hetero_profiler.schema import Availability, validate_record


class ProcParserTests(unittest.TestCase):
    def test_parse_cpu(self):
        value = parse_proc_stat("cpu  10 2 3 80 5 0 0 0\n")
        self.assertEqual((value.total, value.idle), (100, 85))

    def test_cpu_line_missing(self):
        with self.assertRaises(ValueError):
            parse_proc_stat("cpu0 1 2 3 4\n")

    def test_cpu_value_invalid(self):
        with self.assertRaises(ValueError):
            parse_proc_stat("cpu  1 x 3 4\n")

    def test_meminfo(self):
        self.assertEqual(
            parse_meminfo("MemTotal: 100 kB\nMemAvailable: 25 kB\n"),
            (102400, 25600),
        )

    def test_memavailable_missing(self):
        with self.assertRaises(ValueError):
            parse_meminfo("MemTotal: 100 kB\n")

    def test_available_exceeds_total(self):
        with self.assertRaises(ValueError):
            parse_meminfo("MemTotal: 1 kB\nMemAvailable: 2 kB\n")

    def test_process_rss(self):
        self.assertEqual(parse_process_rss("Name: x\nVmRSS: 12 kB\n"), 12288)

    def test_process_rss_missing(self):
        with self.assertRaises(ValueError):
            parse_process_rss("Name: x\n")


class ProcCollectorTests(unittest.TestCase):
    def make_proc(self, root: Path, stat: str):
        (root / "stat").write_text(stat, encoding="utf-8")
        (root / "meminfo").write_text(
            "MemTotal: 1000 kB\nMemAvailable: 400 kB\n",
            encoding="utf-8",
        )

    def test_first_cpu_sample_is_not_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_proc(root, "cpu 10 0 0 90\n")
            collector = ProcTelemetryCollector(
                run_id="run", host_id="host", clock_domain_id="clock",
                proc_root=root, monotonic_ns=lambda: 100,
            )
            collector.prepare()
            collector.start()
            records = collector.sample()
        self.assertIs(records[0].availability, Availability.NOT_AVAILABLE)
        self.assertEqual(records[1].value, 600 * 1024)

    def test_cpu_delta_calculation(self):
        ticks = iter((100, 200))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_proc(root, "cpu 10 0 0 90\n")
            collector = ProcTelemetryCollector(
                run_id="run", host_id="host", clock_domain_id="clock",
                proc_root=root, monotonic_ns=lambda: next(ticks),
            )
            collector.prepare()
            collector.start()
            collector.sample()
            self.make_proc(root, "cpu 30 0 0 170\n")
            record = collector.sample()[0]
        self.assertEqual(record.value, 20.0)
        validate_record(record)

    def test_idle_only_delta_is_zero_percent(self):
        ticks = iter((100, 200))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_proc(root, "cpu 10 0 0 90\n")
            collector = ProcTelemetryCollector(
                run_id="run", host_id="host", clock_domain_id="clock",
                proc_root=root, monotonic_ns=lambda: next(ticks),
            )
            collector.prepare()
            collector.start()
            collector.sample()
            self.make_proc(root, "cpu 10 0 0 190\n")
            record = collector.sample()[0]
        self.assertEqual(record.value, 0.0)

    def test_exited_process_is_not_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_proc(root, "cpu 10 0 0 90\n")
            collector = ProcTelemetryCollector(
                run_id="run", host_id="host", clock_domain_id="clock",
                pid_provider=lambda: 999, proc_root=root, monotonic_ns=lambda: 100,
            )
            collector.prepare()
            collector.start()
            record = collector.sample()[2]
        self.assertIs(record.availability, Availability.NOT_AVAILABLE)
        self.assertIn("exited", record.reason)

    def test_process_rss_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_proc(root, "cpu 10 0 0 90\n")
            (root / "7").mkdir()
            (root / "7/status").write_text("VmRSS: 5 kB\n", encoding="utf-8")
            collector = ProcTelemetryCollector(
                run_id="run", host_id="host", clock_domain_id="clock",
                pid_provider=lambda: 7, proc_root=root, monotonic_ns=lambda: 100,
            )
            collector.prepare()
            collector.start()
            record = collector.sample()[2]
        self.assertEqual(record.value, 5120)
        validate_record(record)


if __name__ == "__main__":
    unittest.main()
