"""Collector lifecycle contract tests."""

import unittest

from perfetto_hetero_profiler.collectors import (
    BaseCollector,
    CollectorError,
    CollectorState,
)


class RecordingCollector(BaseCollector):
    def __init__(self, fail_sample=False):
        super().__init__()
        self.calls = []
        self.fail_sample = fail_sample

    def _prepare(self):
        self.calls.append("prepare")

    def _start(self):
        self.calls.append("start")

    def _sample(self):
        self.calls.append("sample")
        if self.fail_sample:
            raise RuntimeError("sample failed")
        return 7

    def _stop(self):
        self.calls.append("stop")

    def _finalize(self):
        self.calls.append("finalize")
        return tuple(self.calls)


class CollectorLifecycleTests(unittest.TestCase):
    def test_initial_state(self):
        self.assertIs(RecordingCollector().state, CollectorState.CREATED)

    def test_prepare_start_sample_stop(self):
        collector = RecordingCollector()
        collector.prepare()
        collector.start()
        self.assertEqual(collector.sample(), 7)
        collector.stop()
        self.assertIs(collector.state, CollectorState.STOPPED)

    def test_sample_before_start_rejected(self):
        collector = RecordingCollector()
        with self.assertRaises(CollectorError):
            collector.sample()

    def test_duplicate_start_rejected(self):
        collector = RecordingCollector()
        collector.prepare()
        collector.start()
        with self.assertRaises(CollectorError):
            collector.start()

    def test_stop_is_idempotent(self):
        collector = RecordingCollector()
        collector.prepare()
        collector.start()
        collector.stop()
        collector.stop()
        self.assertEqual(collector.calls.count("stop"), 1)

    def test_stop_before_prepare_rejected(self):
        with self.assertRaises(CollectorError):
            RecordingCollector().stop()

    def test_sample_exception_marks_failed(self):
        collector = RecordingCollector(fail_sample=True)
        collector.prepare()
        collector.start()
        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            collector.sample()
        self.assertIs(collector.state, CollectorState.FAILED)
        self.assertIsInstance(collector.last_error, RuntimeError)
        collector.stop()
        collector.stop()
        self.assertEqual(collector.calls.count("stop"), 1)

    def test_finalize_after_stop(self):
        collector = RecordingCollector()
        collector.prepare()
        collector.start()
        collector.stop()
        self.assertEqual(collector.finalize()[-1], "finalize")


if __name__ == "__main__":
    unittest.main()
