"""psutil-backed system telemetry tests."""

from collections import namedtuple
import math
import unittest

import perfetto_hetero_profiler.collectors as collector_package
from perfetto_hetero_profiler.collectors import SystemTelemetryCollector
from perfetto_hetero_profiler.schema import (
    Availability,
    MetricKind,
    MetricScope,
    ValueOrigin,
    validate_record,
)


Cpu = namedtuple(
    "Cpu",
    "user nice system idle iowait irq softirq steal guest guest_nice",
)
Memory = namedtuple("Memory", "total available used")
ProcessMemory = namedtuple("ProcessMemory", "rss")


class FakePsutilError(Exception):
    pass


class FakeNoSuchProcess(FakePsutilError):
    pass


class FakeZombieProcess(FakeNoSuchProcess):
    pass


class FakeAccessDenied(FakePsutilError):
    pass


def cpu(
    *, user=10, nice=0, system=0, idle=90, iowait=0,
    irq=0, softirq=0, steal=0, guest=0, guest_nice=0,
):
    return Cpu(user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice)


class FakeProcess:
    def __init__(self, result, order):
        self.result = result
        self.order = order

    def memory_info(self):
        self.order.append("process")
        if isinstance(self.result, Exception):
            raise self.result
        return ProcessMemory(self.result)


class FakePsutil:
    Error = FakePsutilError
    NoSuchProcess = FakeNoSuchProcess
    ZombieProcess = FakeZombieProcess
    AccessDenied = FakeAccessDenied

    def __init__(
        self,
        cpu_values,
        *,
        memory=Memory(1000, 400, 999),
        process_rss=500,
    ):
        self.cpu_values = list(cpu_values)
        self.memory = memory
        self.process_rss = process_rss
        self.order = []

    def cpu_times(self):
        self.order.append("cpu")
        result = self.cpu_values.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def virtual_memory(self):
        self.order.append("memory")
        if isinstance(self.memory, Exception):
            raise self.memory
        return self.memory

    def Process(self, _pid):
        if isinstance(self.process_rss, Exception):
            raise self.process_rss
        return FakeProcess(self.process_rss, self.order)


class SystemTelemetryTests(unittest.TestCase):
    def make_collector(
        self,
        fake,
        *,
        pid_provider=None,
        ticks=(100,),
    ):
        times = iter(ticks)
        collector = SystemTelemetryCollector(
            run_id="run",
            host_id="host",
            clock_domain_id="clock",
            pid_provider=pid_provider,
            psutil_module=fake,
            monotonic_ns=lambda: next(times),
        )
        collector.prepare()
        collector.start()
        return collector

    def test_old_procfs_api_is_removed(self):
        for name in (
            "CpuTimes",
            "ProcTelemetryCollector",
            "parse_proc_stat",
            "parse_meminfo",
            "parse_process_rss",
        ):
            self.assertFalse(hasattr(collector_package, name), name)

    def test_first_cpu_sample_and_metric_contract(self):
        records = self.make_collector(FakePsutil([cpu()])).sample()
        self.assertEqual(
            [(item.metric_name, item.unit) for item in records],
            [
                ("resource.cpu.utilization", "percent"),
                ("resource.system.memory_used", "bytes"),
            ],
        )
        self.assertIs(records[0].availability, Availability.NOT_AVAILABLE)
        self.assertIsNone(records[0].value)
        self.assertEqual(records[1].value, 600)
        for record in records:
            self.assertIs(record.metric_kind, MetricKind.GAUGE)
            self.assertIs(record.scope, MetricScope.HOST)
            self.assertIs(record.origin, ValueOrigin.MEASURED)
            validate_record(record)

    def test_cpu_delta_timestamp_interval_and_order(self):
        fake = FakePsutil([cpu(), cpu(user=30, idle=170)])
        collector = self.make_collector(fake, ticks=(100, 250))
        first = collector.sample()
        second = collector.sample()
        self.assertEqual(second[0].value, 20.0)
        self.assertEqual(first[0].interval_ns, None)
        self.assertEqual(second[0].interval_ns, 150)
        self.assertEqual(second[0].timestamp_ns, 250)
        self.assertEqual(
            [item.metric_name for item in second],
            ["resource.cpu.utilization", "resource.system.memory_used"],
        )

    def test_idle_and_busy_cpu(self):
        cases = (
            (cpu(user=10, idle=190), 0.0),
            (cpu(user=110, idle=90), 100.0),
        )
        for second, expected in cases:
            with self.subTest(expected=expected):
                collector = self.make_collector(
                    FakePsutil([cpu(), second]), ticks=(1, 2)
                )
                collector.sample()
                self.assertEqual(collector.sample()[0].value, expected)

    def test_iowait_counts_as_idle(self):
        collector = self.make_collector(
            FakePsutil([cpu(), cpu(user=90, idle=90, iowait=20)]),
            ticks=(1, 2),
        )
        collector.sample()
        self.assertEqual(collector.sample()[0].value, 80.0)

    def test_guest_fields_are_not_double_counted(self):
        collector = self.make_collector(
            FakePsutil(
                [
                    cpu(user=10, idle=90, guest=5),
                    cpu(user=20, idle=180, guest=10),
                ]
            ),
            ticks=(1, 2),
        )
        collector.sample()
        self.assertEqual(collector.sample()[0].value, 10.0)

    def test_counter_decrease_and_zero_delta_are_errors(self):
        cases = (
            cpu(user=5, idle=110),
            cpu(),
        )
        for second in cases:
            with self.subTest(second=second):
                collector = self.make_collector(
                    FakePsutil([cpu(), second]), ticks=(1, 2)
                )
                collector.sample()
                record = collector.sample()[0]
                self.assertIs(record.availability, Availability.ERROR)
                self.assertIn("counter delta", record.reason)

    def test_counter_error_establishes_a_fresh_baseline(self):
        collector = self.make_collector(
            FakePsutil(
                [cpu(), cpu(user=5, idle=110), cpu(user=15, idle=200)]
            ),
            ticks=(1, 2, 3),
        )
        collector.sample()
        self.assertIs(collector.sample()[0].availability, Availability.ERROR)
        self.assertEqual(collector.sample()[0].value, 10.0)

    def test_invalid_cpu_values_are_errors_and_memory_survives(self):
        for value in (True, -1, math.nan, math.inf):
            with self.subTest(value=value):
                bad = cpu()._replace(user=value)
                records = self.make_collector(FakePsutil([bad])).sample()
                self.assertIs(records[0].availability, Availability.ERROR)
                self.assertIs(records[1].availability, Availability.AVAILABLE)
                self.assertEqual(records[1].value, 600)

    def test_memory_uses_total_minus_available_not_used(self):
        records = self.make_collector(
            FakePsutil([cpu()], memory=Memory(1000, 250, 1))
        ).sample()
        self.assertEqual(records[1].value, 750)
        self.assertEqual(records[1].attributes["psutil.mem_total_bytes"], 1000)
        self.assertEqual(records[1].attributes["psutil.mem_available_bytes"], 250)

    def test_invalid_memory_values_are_errors_and_cpu_survives(self):
        cases = (
            Memory(100, 101, 0),
            Memory(True, 0, 0),
            Memory(100, -1, 0),
            Memory(100, math.nan, 0),
            Memory(100, math.inf, 0),
        )
        for memory in cases:
            with self.subTest(memory=memory):
                records = self.make_collector(
                    FakePsutil([cpu()], memory=memory)
                ).sample()
                self.assertIs(records[0].availability, Availability.NOT_AVAILABLE)
                self.assertIs(records[1].availability, Availability.ERROR)

    def test_process_rss_and_dimensions(self):
        record = self.make_collector(
            FakePsutil([cpu()], process_rss=5120), pid_provider=lambda: 7
        ).sample()[2]
        self.assertEqual(record.value, 5120)
        self.assertIs(record.scope, MetricScope.PROCESS)
        self.assertEqual(record.dimensions, {"process_id": "7"})
        self.assertEqual(record.attributes, {"psutil.pid": 7})
        validate_record(record)

    def test_process_exit_zombie_and_permission_mapping(self):
        cases = (
            (FakeNoSuchProcess(), Availability.NOT_AVAILABLE, "exited"),
            (FakeZombieProcess(), Availability.NOT_AVAILABLE, "zombie"),
            (FakeAccessDenied(), Availability.ERROR, "denied"),
        )
        for error, availability, reason in cases:
            with self.subTest(error=type(error).__name__):
                record = self.make_collector(
                    FakePsutil([cpu()], process_rss=error),
                    pid_provider=lambda: 9,
                ).sample()[2]
                self.assertIs(record.availability, availability)
                self.assertIn(reason, record.reason)

    def test_generic_process_error_is_sanitized(self):
        record = self.make_collector(
            FakePsutil(
                [cpu()],
                process_rss=FakePsutilError("private /home/user detail"),
            ),
            pid_provider=lambda: 9,
        ).sample()[2]
        self.assertIs(record.availability, Availability.ERROR)
        self.assertEqual(record.reason, "psutil process RSS query failed")

    def test_external_cpu_and_memory_errors_are_sanitized(self):
        private = "private /home/user detail"
        records = self.make_collector(
            FakePsutil([ValueError(private)], memory=ValueError(private))
        ).sample()
        self.assertEqual(records[0].reason, "psutil CPU query failed")
        self.assertEqual(records[1].reason, "psutil virtual memory query failed")
        self.assertTrue(all(private not in item.reason for item in records))

    def test_invalid_process_rss_is_error(self):
        for value in (True, -1, math.nan, math.inf):
            with self.subTest(value=value):
                record = self.make_collector(
                    FakePsutil([cpu()], process_rss=value),
                    pid_provider=lambda: 9,
                ).sample()[2]
                self.assertIs(record.availability, Availability.ERROR)

    def test_timestamp_is_after_all_queries_and_stop_is_idempotent(self):
        fake = FakePsutil([cpu()])

        def clock():
            fake.order.append("clock")
            return 123

        collector = SystemTelemetryCollector(
            run_id="run", host_id="host", clock_domain_id="clock",
            pid_provider=lambda: 7, psutil_module=fake, monotonic_ns=clock,
        )
        collector.prepare()
        collector.start()
        records = collector.sample()
        collector.stop()
        collector.stop()
        collector.finalize()
        self.assertEqual(fake.order, ["cpu", "memory", "process", "clock"])
        self.assertTrue(all(record.timestamp_ns == 123 for record in records))


if __name__ == "__main__":
    unittest.main()
